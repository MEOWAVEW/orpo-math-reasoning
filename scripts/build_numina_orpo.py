#!/usr/bin/env python3
"""Build ORPO preference data from NuminaMath-CoT.

The script keeps the original Numina solution as `response_chosen` and
generates a weaker candidate with a smaller Qwen-style model as
`response_rejected`. Only samples whose rejected answer is clearly worse are
kept in the final JSONL output, which matches this repo's ORPO training schema:

{
  "system": "",
  "history": [],
  "question": "...",
  "response_chosen": "...",
  "response_rejected": "..."
}
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT_SUFFIX = (
    "Let's think step by step and output the final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ORPO JSONL from NuminaMath-CoT.")
    parser.add_argument("--dataset_name", default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--local_dataset_path", default=None,
                        help="Optional local dataset path passed to datasets.load_dataset.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--generator_model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Smaller/weaker generator used to create rejected responses.")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--instruction_suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--disable_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--allow_missing_rejected_boxed", action="store_true",
                        help="Keep rejected responses with no boxed answer if text differs from chosen.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print_every", type=int, default=20)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def remove_boxed(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    if "\\boxed " in text:
        left = "\\boxed "
        if text.startswith(left):
            return text[len(left):]
        return None
    left = "\\boxed{"
    if text.startswith(left) and text.endswith("}"):
        return text[len(left):-1]
    return None


def last_boxed_only_string(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    brace_depth = 0
    right_brace_idx = None
    i = idx
    while i < len(text):
        if text[i] == "{":
            brace_depth += 1
        if text[i] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return text[idx:right_brace_idx + 1]


def extract_boxed_answer(text: Optional[str]) -> Optional[str]:
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def normalize_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return " ".join(text.strip().split())


def looks_truncated(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    bad_suffixes = (
        "\\",
        "{",
        "(",
        "[",
        "$",
        "=",
        ":",
        ",",
    )
    if stripped.endswith(bad_suffixes):
        return True

    open_braces = stripped.count("{") - stripped.count("}")
    open_parens = stripped.count("(") - stripped.count(")")
    open_brackets = stripped.count("[") - stripped.count("]")
    if open_braces > 0 or open_parens > 0 or open_brackets > 0:
        return True

    if len(stripped) >= 20 and stripped[-1].isalnum():
        return True

    return False


def load_numina_dataset(args: argparse.Namespace) -> Dataset:
    source = args.local_dataset_path or args.dataset_name
    dataset = load_dataset(
        source,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    end = min(len(dataset), args.start + args.max_samples)
    return dataset.select(range(args.start, end))


def build_user_prompt(problem: str, suffix: str) -> str:
    if suffix:
        return f"{problem.strip()}\n\n{suffix}"
    return problem.strip()


def build_messages(system_prompt: str, user_prompt: str) -> Iterable[Dict[str, str]]:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def render_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = list(build_messages(system_prompt, user_prompt))
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = ""
    if system_prompt.strip():
        rendered += f"System: {system_prompt.strip()}\n"
    rendered += f"User: {user_prompt}\nAssistant:"
    return rendered


def load_generator(args: argparse.Namespace, device: str, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        args.generator_model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.generator_model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_rejected(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    device: str,
    args: argparse.Namespace,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def is_valid_pair(
    chosen_text: str,
    rejected_text: str,
    allow_missing_rejected_boxed: bool,
) -> bool:
    if not rejected_text:
        return False
    if looks_truncated(rejected_text):
        return False

    chosen_answer = normalize_answer(extract_boxed_answer(chosen_text))
    rejected_answer = normalize_answer(extract_boxed_answer(rejected_text))

    if chosen_answer is None:
        return False
    if rejected_answer is None:
        if not allow_missing_rejected_boxed:
            return False
        rejected_norm = normalize_answer(rejected_text)
        chosen_norm = normalize_answer(chosen_text)
        if rejected_norm == chosen_norm:
            return False
        return True
    return rejected_answer != chosen_answer


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def to_orpo_record(example: Dict[str, Any], chosen_text: str, rejected_text: str, user_prompt: str) -> Dict[str, Any]:
    return {
        "system": "",
        "history": [],
        "question": user_prompt,
        "response_chosen": chosen_text.strip(),
        "response_rejected": rejected_text.strip(),
    }


def main() -> None:
    args = parse_args()
    if os.path.exists(args.output_file) and not args.overwrite:
        raise FileExistsError(f"{args.output_file} already exists. Use --overwrite to replace it.")

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.torch_dtype, device)
    print(f"[INFO] device={device} dtype={torch_dtype} model={args.generator_model}", flush=True)

    dataset = load_numina_dataset(args)
    print(f"[INFO] loaded {len(dataset)} Numina samples", flush=True)

    tokenizer, model = load_generator(args, device, torch_dtype)
    ensure_parent(args.output_file)

    kept = 0
    skipped = 0

    with open(args.output_file, "w", encoding="utf-8") as writer:
        for idx, example in enumerate(dataset, start=1):
            problem = example.get("problem", "").strip()
            solution = example.get("solution", "").strip()
            if not problem or not solution:
                skipped += 1
                continue

            user_prompt = build_user_prompt(problem, args.instruction_suffix)
            prompt_text = render_prompt(tokenizer, args.system_prompt, user_prompt)
            rejected = generate_rejected(model, tokenizer, prompt_text, device, args)

            if not is_valid_pair(solution, rejected, args.allow_missing_rejected_boxed):
                skipped += 1
            else:
                record = to_orpo_record(example, solution, rejected, user_prompt)
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

            if idx % args.print_every == 0:
                print(
                    f"[INFO] processed={idx} kept={kept} skipped={skipped}",
                    flush=True,
                )

    print(f"[DONE] wrote {kept} ORPO rows to {args.output_file}; skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
