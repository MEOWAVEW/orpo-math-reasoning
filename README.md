# ORPO Fine-Tuning of LLMs for Mathematical Reasoning

Preference-optimization (ORPO) training of a Qwen3.5-9B model for mathematical reasoning, with 4-bit QLoRA on a Kaggle H100. This is a self-initiated project in the spirit of the Kaggle AIMO (AI Mathematical Olympiad) competition.

## Highlights

- Built a preference dataset from **AI-MO/NuminaMath-CoT**: a weaker generator (Qwen2.5-1.5B-Instruct) produces candidate solutions, which are filtered with rule-based checks and `math_verify` into chosen/rejected pairs.
- Fine-tuned **Qwen3.5-9B** with **ORPO + 4-bit QLoRA** (TRL / PEFT), 1,000 steps, lr 5e-6, max length 768.
- Evaluated the LoRA adapter on math benchmarks:
  - **GSM8K-distilled: 79.98% accuracy** (2,981 / 3,727)
  - **Calc-GSM8K: 77.79% accuracy** (1,026 / 1,319)

## Repository layout

```
scripts/     preference-data construction, ORPO training, evaluation, figure regeneration
notebooks/   Kaggle-ready versions of the pipeline
data/        built preference pairs (train 2,603 / val 200) + smoke-test samples
results/     evaluation summary JSONs
```

## Pipeline

1. **Build preference data** — `scripts/kaggle_aimo_orpo_rewrite.py` (batched CUDA rewrite) or `scripts/build_numina_orpo.py`. Output is ORPO-ready JSONL with `system / history / question / response_chosen / response_rejected`.
2. **Train** — `scripts/train_orpo_formal.py` with TRL `ORPOTrainer` and 4-bit QLoRA. Start with a smoke test, then a pilot run, then the full dataset:

   ```bash
   python scripts/train_orpo_formal.py \
     --model_name_or_path Qwen/Qwen3.5-9B \
     --train_file data/train.jsonl \
     --output_dir outputs/orpo_qwen35_full \
     --max_steps 1000 --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 8 --learning_rate 5e-6 \
     --max_length 768 --max_completion_length 384 \
     --load_in_4bit --use_bf16
   ```
3. **Evaluate** — `scripts/eval_math_accuracy.py` (or `notebooks/aimoeval-math-accurac.ipynb`) on GSM8K-distilled / Calc-GSM8K.

## Requirements

See `requirements.txt` (torch, transformers, trl, peft, datasets, math-verify, loguru, ...).

## Notes

- Model checkpoints are **not included** in this repository (adapter weights were several hundred MB). Reproduce with the commands above, or contact me for details.
- `scripts/regenerate_orpo_figures.py` expects the training output directory `results-final/orpo_qwen35_output` from the original run.

## Acknowledgement

- Dataset: [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- Method: [ORPO: Monolithic Preference Optimization without Reference Model](https://huggingface.co/papers/2403.07691) (TRL implementation)
