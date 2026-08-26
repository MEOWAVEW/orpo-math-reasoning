# -*- coding: utf-8 -*-
"""
@description:
Evaluate math reasoning accuracy on GSM8K Distilled and Calc-GSM8K benchmarks.
Supports evaluating  ORPO-trained LoRA models.
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    logger.warning("math_verify not installed, falling back to regex-based answer extraction")


# ---------------------------------------------------------------------------
# Answer extraction (fallback when math_verify is unavailable)
# ---------------------------------------------------------------------------

def extract_number_regex(text):
    """Extract numerical answer from text using multiple strategies."""
    if text is None:
        return None

    # Strip <think>...</think> blocks (Qwen3.5 reasoning mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Strategy 1: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        try:
            return float(boxed[-1].replace(',', '').strip())
        except ValueError:
            pass

    # Strategy 2: #### N (GSM8K format)
    hash_match = re.findall(r'####\s*([+-]?\d[\d,]*\.?\d*)', text)
    if hash_match:
        try:
            return float(hash_match[-1].replace(',', ''))
        except ValueError:
            pass

    # Strategy 3: "The answer is N"
    answer_match = re.findall(
        r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?([+-]?\d[\d,]*\.?\d*)\$?',
        text
    )
    if answer_match:
        try:
            return float(answer_match[-1].replace(',', ''))
        except ValueError:
            pass

    # Strategy 4: Last number in text
    numbers = re.findall(r'[+-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass

    return None


def verify_answer_math_verify(prediction_text, ground_truth_text):
    """Use math_verify library for robust answer comparison."""
    try:
        if '####' in ground_truth_text:
            gold_parsed = parse(ground_truth_text.split("####", 1)[-1].strip())
        else:
            gold_parsed = parse(
                ground_truth_text,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )

        # Strip thinking blocks before parsing prediction
        clean_prediction = re.sub(r'<think>.*?</think>', '', prediction_text, flags=re.DOTALL)

        answer_parsed = parse(
            clean_prediction,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        return float(verify(answer_parsed, gold_parsed))
    except Exception as e:
        logger.debug(f"math_verify failed: {e}, falling back to regex")
        return None


def verify_answer_regex(prediction_text, ground_truth_value):
    """Fallback: compare extracted numbers with tolerance."""
    pred = extract_number_regex(prediction_text)
    if pred is None:
        return 0.0
    try:
        gt = float(ground_truth_value)
    except (ValueError, TypeError):
        return 0.0
    return 1.0 if abs(pred - gt) < 1e-3 else 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    """Load base model and optionally attach LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.base_model,
        trust_remote_code=True,
        padding_side='left',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    # Use Flash Attention 2 if available (2-3x speedup on attention)
    try:
        import flash_attn  # noqa: F401
        config_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        logger.info("Flash Attention 2 not available; using default attention")
    if args.load_in_4bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **config_kwargs)
    try:
        model.generation_config = GenerationConfig.from_pretrained(
            args.base_model, trust_remote_code=True
        )
    except OSError:
        pass

    model = PeftModel.from_pretrained(model, args.lora_model, device_map="auto")
    logger.info(f"Loaded ORPO LoRA adapter from {args.lora_model}")

    model.eval()
    return model, tokenizer


def load_vllm_model(args):
    """Load model via vLLM for high-throughput inference."""
    kwargs = dict(
        dtype="bfloat16",
        max_model_len=3072,
        trust_remote_code=True,
    )
    if args.lora_model:
        kwargs["enable_lora"] = True
    llm = LLM(model=args.base_model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.base_model,
        trust_remote_code=True,
    )
    return llm, tokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_dataset(dataset_name, num_samples=-1):
    """Load and normalize evaluation dataset.

    Returns list of dicts with keys: question, ground_truth, ground_truth_value
    """
    samples = []

    if dataset_name == "gsm8k_distilled":
        ds = load_dataset("camel-ai/gsm8k_distilled", split="train", trust_remote_code=True)
        for item in ds:
            question = item.get("problem", item.get("question", ""))
            solution = item.get("ground_truth_solution", item.get("ground_truth_solution", ""))
            # Extract numeric answer from \boxed{N} or #### N
            gt_value = extract_number_regex(solution)
            samples.append({
                "question": question,
                "ground_truth": solution,
                "ground_truth_value": gt_value,
            })

    elif dataset_name == "calc_gsm8k":
        ds = load_dataset("MU-NLPC/Calc-gsm8k", split="test", trust_remote_code=True)
        for item in ds:
            question = item.get("question", "")
            gt_value = item.get("result_float", None)
            if gt_value is not None:
                gt_value = float(gt_value)
            samples.append({
                "question": question,
                "ground_truth": str(item.get("result", "")),
                "ground_truth_value": gt_value,
            })

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'gsm8k_distilled' or 'calc_gsm8k'.")

    if num_samples > 0:
        samples = samples[:num_samples]

    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
    return samples


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful and proficient mathematical assistant. "
    "Let's think step by step. Put the final numerical answer within \\boxed{}."
)


def batch_generate_vllm(llm, tokenizer, questions, max_new_tokens, temperature, lora_model=None):
    """High-throughput generation using vLLM (PagedAttention + continuous batching)."""
    from vllm.lora.request import LoRARequest

    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0.0 else 0.0,
        top_p=1.0,
    )
    lora_req = LoRARequest("lora", 1, lora_model) if lora_model else None
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
    return [o.outputs[0].text.strip() for o in outputs]


@torch.inference_mode()
def batch_generate(model, tokenizer, questions, batch_size, max_new_tokens, temperature):
    """Generate answers for a list of questions in batches.

    Inputs are sorted by prompt length before batching to minimise padding
    waste, then results are restored to the original order.
    """
    n = len(questions)
    device = next(model.parameters()).device

    # Build all prompts first so we can sort by length
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # Sort by prompt character length to reduce intra-batch padding
    order = sorted(range(n), key=lambda i: len(prompts[i]))
    sorted_prompts = [prompts[i] for i in order]
    sorted_responses = []

    for start in tqdm(range(0, n, batch_size), desc="Generating"):
        batch_prompts = sorted_prompts[start:start + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )

        for i, gen_seq in enumerate(outputs):
            prompt_len = input_ids.shape[1]
            gen_tokens = gen_seq[prompt_len:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            sorted_responses.append(response)

    # Restore original order
    all_responses = [None] * n
    for sorted_idx, orig_idx in enumerate(order):
        all_responses[orig_idx] = sorted_responses[sorted_idx]

    return all_responses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, samples, args):
    """Run evaluation and return results."""
    questions = [s["question"] for s in samples]

    logger.info(f"Generating answers for {len(questions)} questions...")
    start_time = time.time()
    predictions = batch_generate(
        model, tokenizer, questions,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    elapsed = time.time() - start_time
    logger.info(f"Generation completed in {elapsed:.1f}s ({elapsed / len(questions):.2f}s/sample)")

    correct = 0
    total = 0
    results = []

    for idx, (sample, prediction) in enumerate(zip(samples, predictions), start=1):
        is_correct = 0.0

        # Try math_verify first if available
        if HAS_MATH_VERIFY and sample["ground_truth"]:
            result = verify_answer_math_verify(prediction, sample["ground_truth"])
            if result is not None:
                is_correct = result

        # Fallback to regex comparison
        if is_correct == 0.0 and sample["ground_truth_value"] is not None:
            is_correct = verify_answer_regex(prediction, sample["ground_truth_value"])

        correct += int(is_correct > 0.5)
        total += 1
        running_accuracy = correct / total * 100 if total > 0 else 0.0
        print(f"[{idx}/{len(samples)}] correct={correct} total={total} accuracy={running_accuracy:.2f}%")

        results.append({
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "ground_truth_value": sample["ground_truth_value"],
            "prediction": prediction,
            "correct": bool(is_correct > 0.5),
        })

    accuracy = correct / total * 100 if total > 0 else 0.0
    return results, accuracy, correct, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate math reasoning accuracy")
    parser.add_argument('--base_model', type=str, required=True, help='Base model path or HF model id')
    parser.add_argument('--lora_model', type=str, required=True, help='ORPO LoRA adapter path')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Tokenizer path (defaults to base_model)')
    parser.add_argument('--eval_dataset', type=str, required=True,
                        choices=['gsm8k_distilled', 'calc_gsm8k'],
                        help='Evaluation dataset name')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples (-1 for all)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0.0 for greedy)')
    parser.add_argument('--output_file', type=str, default='results/eval_results.jsonl',
                        help='Output file for detailed results')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit quantization')
    parser.add_argument('--use_vllm', action='store_true',
                        help='Use vLLM for high-throughput inference (requires: pip install vllm)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Resume from this sample index (skips already-processed samples, appends to output_file)')
    args = parser.parse_args()

    if args.use_vllm and not HAS_VLLM:
        logger.error("vLLM requested but not installed. Run: pip install vllm")
        raise SystemExit(1)

    # Load model
    model_label = "ORPO-adapted"
    logger.info(f"Evaluating {model_label} model on {args.eval_dataset} ({'vLLM' if args.use_vllm else 'HF Transformers'})")
    if args.use_vllm:
        model, tokenizer = load_vllm_model(args)
    else:
        model, tokenizer = load_model_and_tokenizer(args)

    # Load dataset
    samples = load_eval_dataset(args.eval_dataset, args.num_samples)

    # Resume from checkpoint
    if args.start_index > 0:
        logger.info(f"Resuming from sample index {args.start_index} (skipping {args.start_index} already-processed samples)")
        samples = samples[args.start_index:]

    # Evaluate
    if args.use_vllm:
        questions = [s["question"] for s in samples]
        logger.info(f"Generating answers for {len(questions)} questions via vLLM...")
        start_time = time.time()
        predictions = batch_generate_vllm(
            model, tokenizer, questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            lora_model=args.lora_model,
        )
        elapsed = time.time() - start_time
        logger.info(f"vLLM generation completed in {elapsed:.1f}s ({elapsed / len(questions):.2f}s/sample)")
        # Reuse the scoring loop from evaluate()
        correct, total_count = 0, 0
        results = []
        for idx, (sample, prediction) in enumerate(zip(samples, predictions), start=1):
            is_correct = 0.0
            if HAS_MATH_VERIFY and sample["ground_truth"]:
                result = verify_answer_math_verify(prediction, sample["ground_truth"])
                if result is not None:
                    is_correct = result
            if is_correct == 0.0 and sample["ground_truth_value"] is not None:
                is_correct = verify_answer_regex(prediction, sample["ground_truth_value"])
            correct += int(is_correct > 0.5)
            total_count += 1
            running_accuracy = correct / total_count * 100 if total_count > 0 else 0.0
            print(f"[{idx}/{len(samples)}] correct={correct} total={total_count} accuracy={running_accuracy:.2f}%")
            results.append({
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "ground_truth_value": sample["ground_truth_value"],
                "prediction": prediction,
                "correct": bool(is_correct > 0.5),
            })
        accuracy = correct / total_count * 100 if total_count > 0 else 0.0
        correct_out, total = correct, total_count
        results_out = results
    else:
        results_out, accuracy, correct_out, total = evaluate(model, tokenizer, samples, args)

    # Save detailed results (append when resuming, overwrite otherwise)
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    write_mode = 'a' if args.start_index > 0 else 'w'
    with open(args.output_file, write_mode, encoding='utf-8') as f:
        for r in results_out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # When resuming, recompute accuracy over all samples in the output file
    if args.start_index > 0:
        all_results = []
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
        total = len(all_results)
        correct_out = sum(1 for r in all_results if r['correct'])
        accuracy = correct_out / total * 100 if total > 0 else 0.0

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Model:    {args.base_model}")
    if args.lora_model:
        print(f"  LoRA:     {args.lora_model}")
    print(f"  Dataset:  {args.eval_dataset}")
    print(f"  Samples:  {total}")
    print(f"  Correct:  {correct_out}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    print(f"  Results saved to: {args.output_file}")

    # Also save summary as JSON
    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    summary = {
        "lora_model": args.lora_model or None,
        "eval_dataset": args.eval_dataset,
        "total": total,
        "correct": correct_out,
        "accuracy": round(accuracy, 2),
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
