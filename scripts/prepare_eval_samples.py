"""Freeze question/gold-only evaluation inputs from an immutable archive."""
import argparse
import json
from pathlib import Path

from eval_protocol import question_id, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, choices=['gsm8k_d1', 'gsm8k_d2', 'calc'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'evidence/evaluation' / args.archive / 'predictions_raw.jsonl'
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        raise FileExistsError('Choose a new output path; frozen samples are not overwritten.')
    rows = [{'sample_id': question_id(row['question']), 'question': row['question'],
             'ground_truth': row['ground_truth']} for row in read_jsonl(source)]
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Archive contains duplicate question IDs.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    manifest = {'archive': args.archive, 'source_sha256': sha256(source),
                'samples_sha256': sha256(args.output), 'records': len(rows),
                'note': 'Archived question/gold content only; no historic predictions or correctness flags.'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
