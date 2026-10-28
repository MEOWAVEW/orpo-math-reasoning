# ORPO Mathematical Reasoning Experiments

基于 Qwen3.5-9B 的数学偏好优化项目，包含 NuminaMath 偏好数据构造、TRL ORPO + PEFT LoRA 训练与数学答案验证评估。

**证据状态（2026-10-28）：已找回训练配置、日志、适配器及逐题预测，并完成离线完整性核验；尚未重新训练、重新推理或独立重新判分。** 代码支持可选的 4-bit QLoRA，但找回的正式训练记录是 **250 steps、BF16 LoRA，`load_in_4bit=false`**。此前的“1,000 steps、4-bit 训练”缺少对应运行证据，不作为已完成成果。

## 实现范围

- 从 NuminaMath-CoT 构建偏好对，使用较小模型生成 rejected 候选，并提供规则／答案验证过滤路径。不同构造脚本的过滤强度不同。
- 偏好文件共 2,803 条，划分为训练 2,603 条、验证 200 条；原始实验数据与展示仓库的数据字节一致。
- 使用 ORPOTrainer 与 LoRA 训练适配器，保留训练损失、验证损失和偏好指标记录。
- 实现答案生成、数学验证和数值正则回退评分。当前历史成绩不证明相对基座的提升。

## 已找回的训练运行

| 项目 | 记录值 |
|---|---|
| 基座 | Kaggle `arthurchariyasathian/qwen3-5-9b/pytorch/fp16/1`；上游模型 revision 未记录 |
| 训练 | 250 steps；学习率 5e-6；seed 42；BF16；不启用 4-bit 权重量化 |
| LoRA / ORPO | r=8，alpha=16，dropout=0.05，beta=0.1 |
| 批量／长度 | batch=4，梯度累积=8；max_length=1536，max_completion_length=768 |
| 优化器 | `paged_adamw_8bit`；8-bit 优化器不等于 4-bit QLoRA |
| 日志耗时 | 7,168.745 秒，约 1.99 小时；GPU 型号、峰值显存未获运行日志确认 |
| 验证 | loss=0.399604；偏好排序准确率约 93%，**不是数学问答正确率** |

配置、原始源码和日志见 [训练证据](evidence/training/run250/)。另有旧版 500-sample、200-step Notebook，属于不同运行。原始代码冻结基座前 12 层的语句发生在 PEFT 注入前，不能据此声称前 12 层 LoRA 适配器也被冻结。

## 历史评估结果

以下是对原始预测文件内已保存 `correct` 标志的重新汇总；**未重新运行模型或验证评分器判断**。

| 记录 | 正确／总数 | 历史准确率 | 与训练／验证集重叠的题目数 |
|---|---:|---:|---:|
| GSM8K-distilled D1 | 2,981 / 3,727 | 79.98% | 11 / 1 |
| GSM8K-distilled D2 | 2,962 / 3,728 | 79.45% | 6 / 2 |
| Calc-GSM8K | 1,026 / 1,319 | 77.79% | 0 / 0 |

D1 与 D2 的题目无重叠，应作为两个归档子集展示，不能解释为同一测试集的重复实验。两者对应的数据集 revision 相同，但子集选择规则尚未找回。GSM8K-distilled 脚本使用 `camel-ai/gsm8k_distilled` 的 `train` split；Calc 使用 `MU-NLPC/Calc-gsm8k` 的 `test` split。前者不是官方 GSM8K test 成绩。

历史评估路径指向名称含 `200` 的适配器目录；目前虽找回 checkpoint-200 和 checkpoint-250，但没有历史评估权重哈希，**不能把上述分数直接归因于找回的任一权重**。

## 离线核验

仅需要 Python 标准库，无需 GPU：

```bash
python3 scripts/audit_evidence.py
```

只读检查逐题数量、汇总一致性、数据重叠、训练步数及文件哈希，并对照已保存的 [核验报告](evidence/audit_report.json) 和 [证据清单](evidence/manifest.json)。逐题记录、原始评估脚本及数据集版本元数据见 [评估证据](evidence/evaluation/)。

## 为后续重跑准备的数据

已排除 20 条与归档评估题重叠的记录，并移除 4 条重复题记录，得到训练 **2,579 条**、验证 **200 条**。新划分保存在 [data/clean](data/clean/)，保留原始数据用于历史核验。当前规则下训练／验证及其与归档评估题的交集均为 0；这不等于完成语义近重复或预训练污染审计，也没有产生新的训练成绩。

```bash
python3 scripts/prepare_clean_data.py --output outputs/new-clean-data
python3 -m unittest discover -s tests -v
```

划分固定 seed=42，并保存输入和输出哈希；脚本拒绝覆盖非空目录。

## 训练与评估入口

恢复后的训练入口保留原始 250-step 实现，仅适配数据和输出路径：

```bash
python scripts/train_recovered_run.py \
  --model_name_or_path /path/to/verified/base-model \
  --train_file data/clean/train.jsonl \
  --validation_file data/clean/val.jsonl \
  --output_dir outputs/new_clean_orpo
```

新的严格数字评估入口使用相同的冻结题目清单比较基座与适配器。以下以 Calc 归档题目为例：

```bash
# 基座对照：不加载 LoRA
python scripts/evaluate.py \
  --base_model /path/to/verified/base-model \
  --samples_file data/evaluation/calc.jsonl \
  --output_dir outputs/base_calc

# 同一基座与同一份题目，加载新训练的适配器
python scripts/evaluate.py \
  --base_model /path/to/verified/base-model \
  --lora_model outputs/new_clean_orpo \
  --samples_file data/evaluation/calc.jsonl \
  --output_dir outputs/orpo_calc
```

`strict_numeric_v1` 仅接受显式数字答案；不取推理文本最后一个数字，不在判错后改用宽松回退。解析失败仍计入总数，逐题保存提取值和状态。每次新运行保存权重、输入文件与代码哈希、实际 chat template、生成配置、环境及显存记录；拒绝覆盖已有运行目录。该评分器不支持一般符号数学，结果不能直接与历史回退评分相比。

GPU 环境依赖见 `requirements.txt`；[恢复版本记录](requirements-recovered.txt) 仅包含部分历史固定版本，不是完整锁定环境。以上 GPU 命令尚未执行，不承诺重现历史分数。具体步骤见 [复现说明](docs/REPRODUCIBILITY.md)。`scripts/eval_math_accuracy.py` 与旧 Notebook 保留为历史入口；4-bit 训练代码能力与归档的非 4-bit 运行分别说明。

## 局限与下一步

- 训练集有 3 条额外的重复题记录，训练／验证集有 1 道重叠题；GSM8K 评估也存在上述重叠。已提供新的去重划分，仍需重新训练与评估；当前检查只覆盖规范化文本完全匹配。
- 历史评分在数学验证判错后仍尝试数值正则，可能改变得分；未记录每题回退路径及依赖版本。
- 尚缺基座／SFT 同口径对照、训练数据构造 revision、完整环境锁定和评估权重映射。
- AIMO 为项目动机，未提供比赛成绩。项目范围以代码和归档证据为准，个人分工仍需作者确认。

代码采用 MIT 许可；数据和模型分别遵循上游许可，仓库的 MIT 不自动覆盖它们。恢复权重通过 [v0.1.0-evidence Release](https://github.com/MEOWAVEW/orpo-math-reasoning/releases/tag/v0.1.0-evidence) 单独提供下载，包含模型说明和 SHA256；详见 [附件下载及校验](docs/MODEL_ARTIFACTS.md)。权重二进制不进入主分支。详见 [核验发现与待补项](docs/EVIDENCE_REVIEW.md)。

## 项目来源

数据：[AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)、[GSM8K-distilled](https://huggingface.co/datasets/camel-ai/gsm8k_distilled)、[Calc-GSM8K](https://huggingface.co/datasets/MU-NLPC/Calc-gsm8k)。方法：[ORPO](https://arxiv.org/abs/2403.07691)，训练框架：TRL / PEFT。
