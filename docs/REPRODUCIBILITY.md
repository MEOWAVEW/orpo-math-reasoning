# 复现说明

## 核验范围

2026-10-28 完成严格 JSONL 解析、已保存正确标志的汇总、文件 SHA256、规范化题目重叠检查及源码语法检查。没有重新训练、生成答案或重新执行数学评分器。离线核验通过不代表模型效果已独立复现。

## 训练与权重

- `evidence/training/run250/original_training.py` 是原始源码快照，保留 Kaggle 路径；`scripts/train_recovered_run.py` 仅适配训练／输出路径、增加验证文件参数。其余实现保留，仍需兼容 CUDA 环境及正确基座。
- 配置、checkpoint-250 状态和日志均记录 250 steps，`load_in_4bit=false`，BF16。模型卡提供部分框架版本；`requirements-recovered.txt` 区分已记录与未知版本，不能称为完整锁定环境。
- 原始 `training_args.bin` 经静态 pickle 指令检查确认优化器为 `paged_adamw_8bit`，未执行反序列化；源码也记录此值。8-bit 优化器不表示 4-bit 权重量化。
- 本地 checkpoint-200 权重 SHA256：`f91d018efeed7885cacb8c7ca9276613eebc3fd14180845c8cff412e159669c2`。
- 本地 checkpoint-250／最终权重 SHA256：`c435ed5f71441f17e8fdeee9776efaf77d67685e77d5be19508ac6505d56b4a3`。
- 两个适配器各 58,229,536 bytes；`artifacts/` 被 Git 忽略，包含适配器、tokenizer 与模板。本地保存不等于提供了公开下载。
- 历史评估文件未记录权重哈希，不能仅凭路径名称中的 `200` 认定它就是找回的 checkpoint-200。

## 数据和评分

偏好训练 2,603 条、验证 200 条，与原始工作目录同名文件 SHA256 一致。源数据为 NuminaMath-CoT，但构造时的源 revision、完整筛选流水及偏好标签人工质量抽检未恢复。`val-json.py` 使用 seed 42 打乱后取 200 条，没有按题目分组去重。

评估目录的 `dataset_revisions.json` 记录：

- `camel-ai/gsm8k_distilled`：`8bcb2469f88500aa997f70624aad7db76c2a1021`，脚本使用 train split。
- `MU-NLPC/Calc-gsm8k`：`a73d4e22e7e7c4c0d2294d4747b783de481866d4`，脚本使用 test split。

这些是随归档保存的版本元数据，本次未在线重新下载确认。D1、D2 题目无重叠，具体分片选择命令未恢复；历史评估脚本默认加载整个 split，不会自动重建 D1/D2；新入口通过冻结 JSONL 保留题目及其顺序。

离线核验只统计保存的 `correct`。历史评估器在数学验证判错后仍可能尝试正则，其中一个策略是取最后一个数字；需固定规则、记录回退原因并抽检误判。新评分规则的结果应另列，不能覆盖历史分数。

训练的偏好排序准确率约 93% 与数学答案正确率不同。loss 和 reward 曲线不能代替问答性能评估；带 illustrative 的趋势图不得作为实测数学成绩证据。

## 完成模型级复现的步骤

1. 确认基座身份、文件哈希及许可，将其与评估适配器绑定；无法确认历史映射时另开有唯一 run ID 的新实验。
2. 以题目为组去重训练／验证文件，排除评估题和近重复题，保留历史文件。
3. 在干净 CUDA 环境验证安装，保存完整包版本、Python/CUDA、GPU 和峰值显存。
4. 固定 dataset revision、样本清单、提示词、chat template、推理模式、种子和生成参数。
5. 在同一协议下分别评估基座、ORPO；如需比较 SFT，再单独运行 SFT 对照。
6. 保存逐题输出、答案提取、评分路径、失败原因及完整运行清单，再发布已验证结论。

## 新增的重跑工具

- `scripts/prepare_clean_data.py`：排除归档评估题、按规范化题目去重，生成 2,579／200 的新划分。输入／输出哈希和计数见 `data/clean/manifest.json`；保留第一条同题偏好记录，不声称标签冲突已人工复核。
- `scripts/prepare_eval_samples.py`：从指定归档提取 question/gold，不携带旧预测或判分，生成冻结题目及来源清单。已提供 `data/evaluation/` 三套样本。
- `scripts/evaluate.py`：`--lora_model` 可省略，省略即为真正基座对照；只加载本地模型。模型类根据基座 config 的 architectures 选择，GPU 兼容性尚待实测。
- `strict_numeric_v1`：显式 boxed 或 #### 数字；标准答案额外允许裸数字。支持整数、小数和普通 a/b 分数，使用精确有理数比较；不支持一般 LaTeX 分式、符号表达式或单位转换。解析失败按错计入全样本分母，单独报告金标准和预测解析失败数。
- 每次评估写入模型／适配器文件哈希、样本清单、提示词、实际 chat template、生成配置、包版本、GPU、代码哈希、逐题提取及解析状态。失败运行保留 failed 状态，不追加到旧结果。
- `scripts/audit_evidence.py` 默认只读，验证归档与已提交 manifest 一致；仅在审核了有意变更后用 `--write` 重建清单。本地没有权重也能核验固定权重元数据；本地有权重时额外检查哈希。

本次新增的 12 项标准库测试验证离线数据和协议行为；模型加载、训练及 GPU 生成未执行。GitHub Actions 在 Python 3.11／3.12 上运行证据核验、测试及语法检查，不代表模型级复现。
