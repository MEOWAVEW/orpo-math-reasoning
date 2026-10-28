import os
import argparse
import json
import matplotlib.pyplot as plt
from typing import Any, Dict, List

# 开启显存碎片回收机制
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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


# =============================================================================
# 你最方便修改的参数区：以后主要改这里
# =============================================================================
CONFIG = {
    # 路径
    "model_name_or_path": "/kaggle/input/models/arthurchariyasathian/qwen3-5-9b/pytorch/fp16/1",
    "train_file": "/kaggle/input/datasets/anngaltiz/aimo-orpov-and-t/train.jsonl",
    "output_dir": "/kaggle/working/orpo_qwen35_output",

    # 随机种子 / 数据规模
    "seed": 42,
    "max_train_samples": None,   # 小规模试跑先用 100；正式训练可设为 None 或更大

    # 长度设置
    "max_length": 1536,
    "max_completion_length": 768,

    # batch / 累积
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,

    # 学习率与训练步数
    "learning_rate": 5e-6,
    "num_train_epochs": 3.0,
    "max_steps": 250,           # >0 时优先按 step 结束；若想按 epoch 训练，可改成 -1
    "logging_steps": 5,
    "save_steps": 50,
    "warmup_ratio": 0.05,      # 后面有定额
    "lr_scheduler_type": "cosine",
    "save_total_limit": 3,   #存档数量,后面还有一个要改的

    # ORPO
    "beta": 0.1,

    # 精度 / 显存
    "gradient_checkpointing": True,
    "load_in_4bit": False,
    "trust_remote_code": True,
    "use_bf16": True,           # H100 推荐 bf16=True

    # [修改点 1]：使用分页 8-bit 优化器，这是保住 BS=4 的核心
    "optim": "paged_adamw_8bit", 
    
    # [修改点 2]：增加这个参数，让数据按长度排序，减少填充（Padding）带来的显存浪费
    "group_by_length": True,
    
    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ORPO training script for math preference data.")

    parser.add_argument("--model_name_or_path", type=str, default=CONFIG["model_name_or_path"])
    parser.add_argument("--train_file", type=str, default=CONFIG["train_file"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"])

    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--max_train_samples", type=int, default=CONFIG["max_train_samples"],
                        help="Use first N shuffled samples for smoke test / pilot run.")

    parser.add_argument("--max_length", type=int, default=CONFIG["max_length"])
    parser.add_argument("--max_completion_length", type=int, default=CONFIG["max_completion_length"])

    parser.add_argument("--per_device_train_batch_size", type=int, default=CONFIG["per_device_train_batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=CONFIG["gradient_accumulation_steps"])

    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--num_train_epochs", type=float, default=CONFIG["num_train_epochs"])
    parser.add_argument("--max_steps", type=int, default=CONFIG["max_steps"], help="Set >0 to override epochs.")
    parser.add_argument("--logging_steps", type=int, default=CONFIG["logging_steps"])
    parser.add_argument("--save_steps", type=int, default=CONFIG["save_steps"])
    parser.add_argument("--warmup_ratio", type=float, default=CONFIG["warmup_ratio"])
    parser.add_argument("--lr_scheduler_type", type=str, default=CONFIG["lr_scheduler_type"])
    parser.add_argument("--save_total_limit", type=int, default=CONFIG["save_total_limit"])

    parser.add_argument("--beta", type=float, default=CONFIG["beta"])

    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true",
                        default=CONFIG["gradient_checkpointing"])
    parser.add_argument("--load_in_4bit", type=lambda x: str(x).lower() == "true",
                        default=CONFIG["load_in_4bit"])
    parser.add_argument("--trust_remote_code", type=lambda x: str(x).lower() == "true",
                        default=CONFIG["trust_remote_code"])
    parser.add_argument("--use_bf16", type=lambda x: str(x).lower() == "true",
                        default=CONFIG["use_bf16"])

    parser.add_argument("--lora_r", type=int, default=CONFIG["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=CONFIG["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=CONFIG["lora_dropout"])
    parser.add_argument("--target_modules", type=str, default=CONFIG["target_modules"])

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


def load_preference_dataset(train_file: str, seed: int, max_train_samples):
    raw = load_dataset("json", data_files=train_file, split="train")
    raw = raw.filter(is_valid_example)
    raw = raw.shuffle(seed=seed)

    if max_train_samples is not None:
        raw = raw.select(range(min(max_train_samples, len(raw))))

    mapped = raw.map(convert_example, remove_columns=raw.column_names)
    return mapped

    


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
        "attn_implementation": "sdpa",
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
   
    # 在 model = AutoModelForCausalLM.from_pretrained(...) 之后
    print("Enabling survival mode: Freezing first 12 layers...")
    for param in model.model.layers[:12].parameters():
        param.requires_grad = False
        
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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


def build_orpo_args(args):
    fp16_flag = torch.cuda.is_available() and not args.use_bf16
    bf16_flag = args.use_bf16

    return ORPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=50,                     # 每 100 步存一次
        eval_strategy="steps",
        eval_steps=50,
        per_device_eval_batch_size=1,     # 验证时一次只跑 1 条，保命第一
        eval_accumulation_steps=1,       # 极其重要：每步都把结果转到 CPU，不堆积在显存
        save_total_limit=3,   #修改存档数量 3
        warmup_ratio=0.05,    
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=0.5,
        remove_unused_columns=False,
        dataloader_drop_last=True,   # 丢弃不完整的最后一个batch，避免边界异常
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=fp16_flag,
        bf16=bf16_flag,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,       # 训练完后，自动加载这几个里表现最好的
        metric_for_best_model="eval_loss", # 判断标准是验证集 Loss
        greater_is_better=False,           # Loss 越小越好
        optim=CONFIG["optim"],            # [修改点 4]
    # ... 其他保持不变
    )


def save_run_config(args: argparse.Namespace):
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def save_training_plots(trainer, output_dir: str):
    log_history = trainer.state.log_history

    train_steps = []
    train_losses = []
    validation_steps = []
    validation_losses = []

    reward_steps = []
    rewards_chosen, rewards_rejected, rewards_margins = [], [], []

    for log in log_history:
        step = log.get("step", None)

        if "loss" in log:
            train_steps.append(step if step is not None else len(train_steps))
            train_losses.append(log["loss"])

        # 框架日志里这一项的键名仍然是 eval_loss，
        # 但我们读出来以后统一按 validation_loss 使用
        if "eval_loss" in log:
            validation_steps.append(step if step is not None else len(validation_steps))
            validation_losses.append(log["eval_loss"])

        if "rewards/chosen" in log or "rewards/rejected" in log or "rewards/margins" in log:
            reward_steps.append(step if step is not None else len(reward_steps))
            rewards_chosen.append(log.get("rewards/chosen"))
            rewards_rejected.append(log.get("rewards/rejected"))
            rewards_margins.append(log.get("rewards/margins"))

    with open(os.path.join(output_dir, "log_history.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    if train_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(train_steps, train_losses, label="train_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_loss.png"))
        plt.close()

    if validation_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(validation_steps, validation_losses, marker="o", label="validation_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "validation_loss.png"))
        plt.close()

    if train_losses or validation_losses:
        plt.figure(figsize=(8, 5))
        if train_losses:
            plt.plot(train_steps, train_losses, label="train_loss")
        if validation_losses:
            plt.plot(validation_steps, validation_losses, marker="o", label="validation_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
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
    train_dataset = load_preference_dataset(
    train_file=args.train_file,
    seed=args.seed,
    max_train_samples=args.max_train_samples,
    )
    print(f"train size: {len(train_dataset)}")
    
    # =========================
    # Load validation dataset（你自己的200条）
    # =========================
    validationdataset = load_dataset(
        "json",
        data_files="/kaggle/input/datasets/anngaltiz/aimo-orpov-and-t/val.jsonl",
        split="train"
    )

    validationdataset = validationdataset.filter(is_valid_example)
    validationdataset = validationdataset.map(convert_example, remove_columns=validationdataset.column_names)

    print(f"validation size: {len(validationdataset)}")
    
    
    print("=" * 80)
    print("Sample after preprocessing:")
    print(train_dataset[0]["prompt"])
    print("-" * 40)
    print("CHOSEN:")
    print(train_dataset[0]["chosen"])
    print("-" * 40)
    print("REJECTED:")
    print(train_dataset[0]["rejected"])
    print("=" * 80)

    print("=" * 80)
    print("Loading model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(args)

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    print_target_module_match_summary(model, target_modules)

    peft_config = build_peft_config(args)
    training_args = build_orpo_args(args)

    print("=" * 80)
    print("Creating trainer...")

            
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validationdataset,
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
    # 检查是否有存档文件夹
    import glob
    checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
    
    if checkpoints:
        print(f"Found checkpoints: {checkpoints}. Resuming...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoints found. Starting a fresh training...")
        trainer.train() # 删掉 resume_from_checkpoint 或设为 None

    print("=" * 80)
    print("Running validation...")

    validation_metrics = trainer.evaluate()

    print("Validation results:")
    print(validation_metrics)

    validation_path = os.path.join(args.output_dir, "validation_results.json")

    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_metrics, f, ensure_ascii=False, indent=2)

    print("Saved to:", validation_path)

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
