"""
Kaggle-friendly ORPO preference-data builder for NuminaMath-CoT.

Why the original notebook showed CPU 1219% and GPU 0%:
1. `from_pretrained(model_name)` loads the model on CPU by default.
2. There was no `device_map="auto"` and no `model.to("cuda")`.
3. The loop generated one sample at a time, so even with GPU it would be slow.

This rewrite is designed for Kaggle H100:
- explicitly uses CUDA
- uses a smaller Qwen generator to create wrong answers
- batches generation
- writes ORPO-ready JSONL with:
  system / history / question / response_chosen / response_rejected
"""

import json
import os
import re
from typing import List, Optional

import torch
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Config
# -----------------------------
DATASET_NAME = "AI-MO/NuminaMath-CoT"
SPLIT = "train"
NUM_SAMPLES = 2000

# Use a weaker model to generate rejected answers.
# This is better for ORPO than using the strong 9B model itself.
GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

OUTPUT_FILE = "/kaggle/working/numina_orpo_train.jsonl"
DEBUG_FILE = "/kaggle/working/numina_orpo_debug.jsonl"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.95

INSTRUCTION_SUFFIX = (
    "Let's think step by step and output the final answer within \\boxed{}."
)

BAD_PATTERNS = [
    "system",
    "user",
    "assistant",
    "header",
    "cloud",
    "alibaba",
    "i am ai",
    "ai language",
    "language model",
    "artificial intelligence",
    "chatgpt",
    "you are the ai",
]
MIN_REJECTED_CHARS = 80
MIN_MATH_MARKERS = 2


def extract_boxed_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    boxed_spaced = re.findall(r"\\boxed\s+([^$\n]+)", text)
    if boxed_spaced:
        return boxed_spaced[-1].strip()
    boxed_braced = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_braced:
        return boxed_braced[-1].strip()
    return None


def normalize_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return " ".join(str(text).strip().split())


def looks_truncated(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(("\\", "{", "(", "[", "$", "=", ":", ",")):
        return True
    if stripped.count("{") > stripped.count("}"):
        return True
    if stripped.count("(") > stripped.count(")"):
        return True
    if stripped.count("[") > stripped.count("]"):
        return True
    return False


def has_bad_pattern(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in BAD_PATTERNS)


def count_math_markers(text: str) -> int:
    markers = [
        "\\boxed",
        "\\(",
        "\\[",
        "=",
        "+",
        "-",
        "*",
        "/",
        "^",
        "therefore",
        "so ",
        "thus",
        "equation",
    ]
    lowered = text.lower()
    return sum(1 for marker in markers if marker in lowered)


def is_clean_rejected(text: str) -> bool:
    if len(text.strip()) < MIN_REJECTED_CHARS:
        return False
    if looks_truncated(text):
        return False
    if has_bad_pattern(text):
        return False
    if count_math_markers(text) < MIN_MATH_MARKERS:
        return False
    return True


def score_answer(gold: str, pred: str) -> float:
    try:
        gold_parsed = parse(str(gold))
        pred_parsed = parse(str(pred))
        return 1.0 if verify(pred_parsed, gold_parsed) else 0.0
    except Exception:
        return 0.0


def build_prompt(problem: str) -> str:
    return f"{problem.strip()}\n\n{INSTRUCTION_SUFFIX}"


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. In Kaggle, please enable GPU accelerator first.")
print("GPU:", torch.cuda.get_device_name(0))


print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split=SPLIT)
dataset = dataset.select(range(NUM_SAMPLES))
print("Loaded rows:", len(dataset))


print("Loading tokenizer and model on GPU...")
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model device map loaded.")


def batch_generate(prompts: List[str]) -> List[str]:
    if tokenizer.chat_template:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
    else:
        texts = prompts

    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = model_inputs["input_ids"].shape[1]
    generations = outputs[:, input_len:]
    return tokenizer.batch_decode(generations, skip_special_tokens=True)


kept = 0
skipped = 0
debug_rows = []

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as writer:
    for start in range(0, len(dataset), BATCH_SIZE):
        batch = dataset.select(range(start, min(start + BATCH_SIZE, len(dataset))))
        prompts = [build_prompt(x["problem"]) for x in batch]
        rejecteds = batch_generate(prompts)

        for row, prompt, rejected in zip(batch, prompts, rejecteds):
            chosen = row["solution"].strip()
            rejected = rejected.strip()

            chosen_boxed = normalize_text(extract_boxed_answer(chosen))
            rejected_boxed = normalize_text(extract_boxed_answer(rejected))

            if not chosen_boxed:
                skipped += 1
                continue
            if not is_clean_rejected(rejected):
                skipped += 1
                continue
            if normalize_text(rejected) == normalize_text(chosen):
                skipped += 1
                continue
            if rejected_boxed and rejected_boxed == chosen_boxed:
                skipped += 1
                continue

            # math_verify says wrong => best negative sample
            score = score_answer(chosen, rejected)
            if score >= 0.5:
                skipped += 1
                continue

            record = {
                "system": "",
                "history": [],
                "question": prompt,
                "response_chosen": chosen,
                "response_rejected": rejected,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if len(debug_rows) < 20:
                debug_rows.append({
                    "problem": row["problem"],
                    "chosen_boxed": chosen_boxed,
                    "rejected_boxed": rejected_boxed,
                    "rejected_preview": rejected[:400],
                })

        print(f"Processed {min(start + BATCH_SIZE, len(dataset))}/{len(dataset)} | kept={kept} skipped={skipped}")

with open(DEBUG_FILE, "w", encoding="utf-8") as f:
    for item in debug_rows:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done. Wrote {kept} rows to {OUTPUT_FILE}")
print(f"Debug samples saved to {DEBUG_FILE}")
