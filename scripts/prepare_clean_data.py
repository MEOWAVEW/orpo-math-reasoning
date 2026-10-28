"""Create a deterministic, text-deduplicated split for NEW training runs.

Historical files and predictions are never modified. Exact text exclusion is
limited and cannot rule out semantic duplicates or pretraining contamination.
"""
import argparse
import json
import random
from pathlib import Path

from audit_evidence import question_key
from eval_protocol import read_jsonl, sha256


def prepare(root, output, seed=42, validation_size=200):
    source = root / 'data/numina_orpo_fixed.jsonl'
    exclusions = sorted((root / 'evidence/evaluation').glob('*/predictions_raw.jsonl'))
    if not exclusions:
        raise ValueError('Archived evaluation questions are required for exclusion.')
    blocked = {question_key(row['question']) for path in exclusions for row in read_jsonl(path)}
    original = read_jsonl(source)
    kept, duplicates, overlaps = {}, 0, 0
    for row in original:
        for key in ['question', 'response_chosen', 'response_rejected']:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('Missing preference field ' + key)
        key = question_key(row['question'])
        if key in blocked:
            overlaps += 1
        elif key in kept:
            duplicates += 1
        else:
            # Preserve the first source record for a repeated question.
            kept[key] = row
    keys = sorted(kept)
    random.Random(seed).shuffle(keys)
    if not 0 < validation_size < len(keys):
        raise ValueError('Invalid validation size.')
    parts = {'val': keys[:validation_size], 'train': keys[validation_size:]}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new or empty output directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    for name, subset in parts.items():
        (output / (name + '.jsonl')).write_text(''.join(
            json.dumps(kept[key], ensure_ascii=False) + '\n' for key in subset), encoding='utf-8')
    report = {'purpose': 'New experiment inputs; no training or new accuracy claimed.',
              'method': 'Known suffix removal, whitespace collapse, casefold; first record per question.',
              'seed': seed, 'source_rows': len(original), 'excluded_eval_overlap_rows': overlaps,
              'duplicate_rows_removed': duplicates, 'train_rows': len(parts['train']),
              'validation_rows': len(parts['val']), 'source_sha256': sha256(source),
              'excluded_sources': [{'path': str(p.relative_to(root)), 'sha256': sha256(p)} for p in exclusions],
              'output_sha256': {name: sha256(output / (name + '.jsonl')) for name in parts}}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation_size', type=int, default=200)
    args = parser.parse_args()
    prepare(args.root, args.output, args.seed, args.validation_size)
