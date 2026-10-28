"""Run a NEW base-model or LoRA evaluation with strict numeric scoring.

The environment, local weights, sample list and outputs are recorded together.
This does not reproduce or overwrite archived historical results.
"""
import argparse
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

from eval_protocol import (SCORER, SYSTEM_PROMPT, local_fingerprint, question_id,
                           read_jsonl, score, sha256)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_model', type=Path, required=True, help='Verified local base-model directory')
    parser.add_argument('--lora_model', type=Path, help='Omit for a true base-model baseline')
    parser.add_argument('--samples_file', type=Path, required=True, help='Frozen JSONL: question, ground_truth')
    parser.add_argument('--output_dir', type=Path, required=True, help='A new experiment directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--max_input_tokens', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser


def validate(args):
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_input_tokens < 1:
        raise ValueError('Token limits and batch size must be positive.')
    if args.output_dir.exists():
        raise FileExistsError('Output directory already exists; choose a new run ID.')
    rows = read_jsonl(args.samples_file)
    if not rows:
        raise ValueError('Sample file is empty.')
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ['question', 'ground_truth']):
            raise ValueError('Samples require nonempty question and ground_truth strings.')
    ids = [question_id(row['question']) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate question IDs; freeze a deduplicated sample list.')
    return rows, ids


def main():
    args = build_parser().parse_args()
    rows, ids = validate(args)
    base_files = local_fingerprint(args.base_model)
    adapter_files = local_fingerprint(args.lora_model) if args.lora_model else None
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('This entry point requires a CUDA GPU; offline evidence audit does not.')
    set_seed(args.seed)
    versions = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'safetensors']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    manifest = {'status': 'running', 'scorer': SCORER, 'python': platform.python_version(),
                'packages': versions, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'code_commit': revision.stdout.strip() or None,
                'entrypoint_sha256': sha256(__file__),
                'protocol_sha256': sha256(Path(__file__).with_name('eval_protocol.py')),
                'sample_file_sha256': sha256(args.samples_file), 'sample_ids': ids,
                'base_files': base_files, 'adapter_files': adapter_files,
                'seed': args.seed, 'batch_size': args.batch_size,
                'generation': {'do_sample': False, 'num_beams': 1,
                               'max_new_tokens': args.max_new_tokens,
                               'max_input_tokens': args.max_input_tokens},
                'system_prompt': SYSTEM_PROMPT,
                'note': 'New strict numeric evaluation; not comparable to legacy fallback scores.'}
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / 'run_manifest.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    save_manifest()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        manifest['chat_template'] = tokenizer.chat_template
        config = AutoConfig.from_pretrained(args.base_model, local_files_only=True)
        architecture = (config.architectures or [''])[0]
        model_class = getattr(transformers, architecture, AutoModelForCausalLM)
        manifest['model_class'] = model_class.__name__
        model = model_class.from_pretrained(args.base_model, dtype=torch.bfloat16,
                                            device_map='auto', local_files_only=True)
        if args.lora_model:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=True)
        model.eval()
        manifest['model_generation_config'] = model.generation_config.to_dict()
        save_manifest()
        torch.cuda.reset_peak_memory_stats()
        results, parsing = [], Counter()
        with (args.output_dir / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                prompts = [tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': SYSTEM_PROMPT},
                     {'role': 'user', 'content': row['question']}],
                    tokenize=False, add_generation_prompt=True) for row in batch]
                # Fail rather than silently truncate a question.
                encoded = tokenizer(prompts, padding=True, return_tensors='pt', truncation=False)
                if encoded['input_ids'].shape[1] > args.max_input_tokens:
                    raise ValueError('Input token limit exceeded; use a documented larger limit.')
                encoded = encoded.to(next(model.parameters()).device)
                with torch.inference_mode():
                    generated = model.generate(**encoded, do_sample=False, num_beams=1,
                        num_return_sequences=1, max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(generated[:, encoded['input_ids'].shape[1]:],
                                                      skip_special_tokens=True)
                for index, (row, prediction) in enumerate(zip(batch, predictions), start=start):
                    result = {'sample_id': ids[index], 'question': row['question'],
                              'ground_truth': row['ground_truth'], 'prediction': prediction,
                              **score(prediction, row['ground_truth'])}
                    parsing[result['answer_parse_status']] += 1
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
        summary = {'total': len(results), 'correct': sum(x['correct'] for x in results),
                   'accuracy_percent': 100 * sum(x['correct'] for x in results) / len(results),
                   'answer_parse_statuses': dict(parsing),
                   'gold_parse_failures': sum(x['extracted_gold'] is None for x in results),
                   'scorer': SCORER, 'denominator': 'All frozen samples, including parse failures.'}
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        manifest.update(status='complete', predictions_sha256=sha256(args.output_dir / 'predictions.jsonl'),
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        print(json.dumps(summary, indent=2))
    except Exception as error:
        manifest.update(status='failed', failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        save_manifest()


if __name__ == '__main__':
    main()
