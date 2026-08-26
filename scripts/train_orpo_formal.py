import os
import argparse
import json
import matplotlib.pyplot as plt
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training

try:
    from trl import ORPOConfig, ORPOTrainer
except ImportError:
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ORPO training script for math preference data.")

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Use first N shuffled samples for smoke test / pilot run.")

    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_completion_length", type=int, default=384)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1, help="Set >0 to override epochs.")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_bf16", action="store_true", default=False)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    return parser.parse_args()


def format_history(history: List[Any]) -> str:
    if not history:
        return ""

    chunks = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            chunks.append(f"{role.title()}:\n{content}")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            chunks.append(f"User:\n{item[0]}\nAssistant:\n{item[1]}")
        else:
            chunks.append(str(item))
    return "\n\n".join(chunks)


def build_prompt(example: Dict[str, Any]) -> str:
    system = (example.get("system") or "").strip()
    history = example.get("history") or []
    question = (example.get("question") or "").strip()

    parts = []

    if system:
        parts.append(system)

    hist_text = format_history(history)
    if hist_text:
        parts.append(hist_text)

    if question:
        parts.append("### Instruction:\n" + question)

    parts.append("### Response:\n")
    return "\n\n".join(parts)


def convert_example(example: Dict[str, Any]) -> Dict[str, str]:
    return {
        "prompt": build_prompt(example),
        "chosen": (example.get("response_chosen") or "").strip(),
        "rejected": (example.get("response_rejected") or "").strip(),
    }


def is_valid_example(example: Dict[str, Any]) -> bool:
    question = (example.get("question") or "").strip()
    chosen = (example.get("response_chosen") or "").strip()
    rejected = (example.get("response_rejected") or "").strip()
    return bool(question) and bool(chosen) and bool(rejected) and chosen != rejected


def load_preference_dataset(train_file: str, eval_ratio: float, seed: int, max_train_samples):
    raw = load_dataset("json", data_files=train_file, split="train")
    raw = raw.filter(is_valid_example)
    raw = raw.shuffle(seed=seed)

    if max_train_samples is not None:
        raw = raw.select(range(min(max_train_samples, len(raw))))

    mapped = raw.map(convert_example, remove_columns=raw.column_names)

    if len(mapped) >= 20 and eval_ratio > 0:
        split = mapped.train_test_split(test_size=eval_ratio, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = mapped
        eval_dataset = None

    return train_dataset, eval_dataset


def print_target_module_match_summary(model, target_modules):
    matched = {name: [] for name in target_modules}
    all_names = [n for n, _ in model.named_modules()]

    for mod_name in all_names:
        leaf = mod_name.split(".")[-1]
        if leaf in matched:
            matched[leaf].append(mod_name)

    print("=" * 80)
    print("Target module match summary")
    for tm in target_modules:
        names = matched.get(tm, [])
        print(f"[{tm}] count = {len(names)}")
        for sample in names[:8]:
            print("  -", sample)
        if len(names) > 8:
            print(f"  ... ({len(names) - 8} more)")
    missing = [tm for tm, names in matched.items() if len(names) == 0]
    if missing:
        print("WARNING: These target module names were not found:", missing)
    else:
        print("All target module names were found.")
    print("=" * 80)


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }

    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.use_bf16 else torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit quantization requires CUDA GPU.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        )
        model_kwargs["device_map"] = {"": 0}
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def build_peft_config(args: argparse.Namespace) -> LoraConfig:
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def build_orpo_args(args: argparse.Namespace, has_eval: bool):
    fp16_flag = torch.cuda.is_available() and not args.use_bf16
    bf16_flag = args.use_bf16

    common_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=fp16_flag,
        bf16=bf16_flag,
        do_train=True,
        do_eval=has_eval,
    )

    if args.max_steps and args.max_steps > 0:
        common_kwargs["max_steps"] = args.max_steps
    else:
        common_kwargs["max_steps"] = -1

    try:
        if has_eval:
            return ORPOConfig(
                **common_kwargs,
                eval_strategy="steps",
                eval_steps=args.eval_steps,
            )
        return ORPOConfig(
            **common_kwargs,
            eval_strategy="no",
        )
    except TypeError:
        if has_eval:
            return ORPOConfig(
                **common_kwargs,
                evaluation_strategy="steps",
                eval_steps=args.eval_steps,
            )
        return ORPOConfig(
            **common_kwargs,
            evaluation_strategy="no",
        )


def save_run_config(args: argparse.Namespace):
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def save_training_plots(trainer, output_dir: str):
    log_history = trainer.state.log_history

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    reward_steps = []
    rewards_chosen, rewards_rejected, rewards_margins = [], [], []

    for log in log_history:
        step = log.get("step", None)

        if "loss" in log:
            train_steps.append(step if step is not None else len(train_steps))
            train_losses.append(log["loss"])

        if "eval_loss" in log:
            eval_steps.append(step if step is not None else len(eval_steps))
            eval_losses.append(log["eval_loss"])

        if "rewards/chosen" in log or "rewards/rejected" in log or "rewards/margins" in log:
            reward_steps.append(step if step is not None else len(reward_steps))
            rewards_chosen.append(log.get("rewards/chosen"))
            rewards_rejected.append(log.get("rewards/rejected"))
            rewards_margins.append(log.get("rewards/margins"))

    with open(os.path.join(output_dir, "log_history.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    if train_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(train_steps, train_losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_loss.png"))
        plt.close()

    if eval_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(eval_steps, eval_losses)
        plt.xlabel("Step")
        plt.ylabel("Eval Loss")
        plt.title("Eval Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eval_loss.png"))
        plt.close()

    valid = [
        (s, c, r, m)
        for s, c, r, m in zip(reward_steps, rewards_chosen, rewards_rejected, rewards_margins)
        if c is not None or r is not None or m is not None
    ]

    if valid:
        rs = [x[0] for x in valid]
        rc = [x[1] for x in valid]
        rr = [x[2] for x in valid]
        rm = [x[3] for x in valid]

        plt.figure(figsize=(8, 5))
        if any(v is not None for v in rc):
            plt.plot(rs, rc, label="chosen")
        if any(v is not None for v in rr):
            plt.plot(rs, rr, label="rejected")
        if any(v is not None for v in rm):
            plt.plot(rs, rm, label="margin")

        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("ORPO Reward Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_curves.png"))
        plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_config(args)

    print("=" * 80)
    print("Loading dataset...")
    train_dataset, eval_dataset = load_preference_dataset(
        train_file=args.train_file,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
    )
    print(f"train size: {len(train_dataset)}")
    print(f"eval size : {0 if eval_dataset is None else len(eval_dataset)}")

    print("=" * 80)
    print("Loading model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(args)

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    print_target_module_match_summary(model, target_modules)

    peft_config = build_peft_config(args)
    training_args = build_orpo_args(args, has_eval=eval_dataset is not None)

    print("=" * 80)
    print("Creating trainer...")
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    try:
        trainer = ORPOTrainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = ORPOTrainer(
            **trainer_kwargs,
            tokenizer=tokenizer,
        )

    print("=" * 80)
    print("Start training...")
    trainer.train()

    save_training_plots(trainer, args.output_dir)

    print("=" * 80)
    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("=" * 80)
    print("Done.")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
