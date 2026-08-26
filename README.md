# 🧠 ORPO Fine-Tuning of LLMs for Mathematical Reasoning

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![TRL](https://img.shields.io/badge/TRL-ORPO-8A2BE2) ![QLoRA](https://img.shields.io/badge/4--bit-QLoRA-4B0082) ![License](https://img.shields.io/badge/License-MIT-green)

**Preference-optimization (ORPO) + 4-bit QLoRA training of Qwen3.5-9B for mathematical reasoning** — a self-initiated project in the spirit of the Kaggle AIMO (AI Mathematical Olympiad) competition, built end-to-end on a single Kaggle H100.

---

## ✨ Highlights

- **Data** — built a 2,800+ preference-pair dataset from [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT): a weaker generator (`Qwen2.5-1.5B-Instruct`) produces candidate solutions, which are filtered with rule-based quality checks and `math_verify` into chosen / rejected pairs.
- **Training** — fine-tuned **Qwen3.5-9B** with **ORPO + 4-bit QLoRA** (TRL / PEFT), 1,000 steps, lr 5e-6, bf16.
- **Evaluation** — answer-verification-based eval on two math benchmarks, reaching **~80% accuracy**.
- **Full ownership** — preference-data construction → training → evaluation → result analysis, all in this repository.

## 📊 Results

| Benchmark | Accuracy | Correct / Total |
|---|---|---|
| GSM8K-distilled | **79.98%** | 2,981 / 3,727 |
| Calc-GSM8K | **77.79%** | 1,026 / 1,319 |

## 🧩 Pipeline

1. **Build preference data** — `scripts/kaggle_aimo_orpo_rewrite.py` (batched CUDA rewrite) or `scripts/build_numina_orpo.py`. Output is ORPO-ready JSONL with `system / history / question / response_chosen / response_rejected`.
2. **Train** — `scripts/train_orpo_formal.py` with TRL `ORPOTrainer` and 4-bit QLoRA.
3. **Evaluate** — `scripts/eval_math_accuracy.py` or `notebooks/aimoeval-math-accurac.ipynb`.

## 📁 Repository structure

```
.
├── scripts/        # data construction, ORPO training, evaluation, figure regeneration
├── notebooks/      # Kaggle-ready pipeline notebooks
├── data/           # built preference pairs (train 2,603 / val 200) + smoke-test samples
├── results/        # evaluation summary JSONs
└── requirements.txt
```

## 🚀 Quick start

### 1 · Smoke test (confirm format & loss)

```bash
python scripts/train_orpo_formal.py \
  --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
  --train_file data/numina_orpo_smoke.jsonl \
  --output_dir outputs/smoke \
  --max_train_samples 100 --max_steps 50 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
  --learning_rate 5e-6 --max_length 768 --max_completion_length 384 \
  --no-load_in_4bit
```

### 2 · Full ORPO training

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

### 3 · Evaluate

```bash
python scripts/eval_math_accuracy.py \
  --model_name_or_path Qwen/Qwen3.5-9B \
  --lora_model_path <your-checkpoint>
```

## 📝 Notes

- Model checkpoints are **not included** in this repository (adapter weights were several hundred MB). Reproduce with the commands above, or reach out for details.
- `scripts/regenerate_orpo_figures.py` expects the original training output directory (`results-final/orpo_qwen35_output`) for figure regeneration.

## 🙏 Acknowledgement

- Dataset: [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- Method: [ORPO: Monolithic Preference Optimization without Reference Model](https://huggingface.co/papers/2403.07691), implemented via [TRL](https://github.com/huggingface/trl)
- Evaluation: GSM8K-distilled & Calc-GSM8K benchmarks
