import argparse
import torch
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )
    leaf_names = {}
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        leaf_names.setdefault(leaf, 0)
        leaf_names[leaf] += 1

    print("=" * 80)
    print("Leaf module name counts")
    for k in sorted(leaf_names):
        print(f"{k}: {leaf_names[k]}")
    print("=" * 80)

if __name__ == "__main__":
    main()
