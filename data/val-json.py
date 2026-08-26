import json
import random

# 1. 读取数据
with open("numina_orpo_fixed.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 2. 固定随机种子并打乱
random.seed(42)
random.shuffle(data)

# 3. 划分数据集
val_data = data[:200]
train_data = data[200:]

# 4. 定义保存函数
def save_jsonl(data_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data_list:
            # ensure_ascii=False 可以保证中文不被转义成 Unicode 编码
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"成功保存 {len(data_list)} 条数据至 {filename}")

# 5. 执行保存
save_jsonl(val_data, "val.jsonl")
save_jsonl(train_data, "train.jsonl")
