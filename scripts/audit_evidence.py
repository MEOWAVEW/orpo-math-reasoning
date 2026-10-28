"""Audit archived records without loading a model or changing historical scores.

Only Python's standard library is required. The raw archives are immutable inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
        return h.hexdigest()


def question_key(question):
    # Only remove the two documented construction suffixes, collapse whitespace,
    # and case-fold. This is a limited text-overlap check, not semantic deduplication.
    suffixes = [
        "Please explicitly output the final answer within \\boxed{}.",
        "Let's think step by step and output the final answer within \\boxed{}.",
    ]
    value = question.strip()
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[:-len(suffix)].rstrip()
    return ' '.join(value.split()).casefold()


def read_archive(path):
    text = path.read_text(encoding='utf-8')
    bad_lines = []
    for number, line in enumerate(text.split('\n'), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            bad_lines.append({'line': number, 'error': error.msg})
    if bad_lines:
        raise ValueError(f"Invalid JSONL: {path}: {bad_lines}")
    # JSONL delimiters are LF, not Python splitlines()'s wider Unicode set.
    # U+2028 inside a question is legal JSON and must remain inside its record.
    rows = [json.loads(line) for line in text.split('\n') if line.strip()]
    for row in rows:
        assert isinstance(row, dict) and isinstance(row['correct'], bool)
        for key in ['question', 'ground_truth', 'prediction']:
            assert isinstance(row[key], str)
    return rows, bad_lines


def audit(root, write=False):
    evaluations = {}
    report = {'scope': 'Offline archive aggregation; no inference or rescoring performed.',
              'evaluations': {}, 'data': {}, 'overlap': {}, 'weights': {}}
    data = {}
    for name in ['train', 'val']:
        path = root / 'data' / (name + '.jsonl')
        rows = [json.loads(line) for line in path.read_text().split('\n') if line.strip()]
        data[name] = {question_key(row['question']) for row in rows}
        report['data'][name] = {'rows': len(rows), 'unique_questions': len(data[name]),
                               'sha256': digest(path)}
    for name in ['gsm8k_d1', 'gsm8k_d2', 'calc']:
        folder = root / 'evidence/evaluation' / name
        rows, bad = read_archive(folder / 'predictions_raw.jsonl')
        keys = {question_key(row['question']) for row in rows}
        evaluations[name] = keys
        original = json.loads((folder / 'summary_original.json').read_text())
        correct = sum(row['correct'] for row in rows)
        assert (correct, len(rows)) == (original['correct'], original['total']), name
        assert round(correct / len(rows) * 100, 2) == original['accuracy'], name
        report['evaluations'][name] = {
            'records': len(rows), 'correct_saved_flags': correct,
            'accuracy_percent_saved_flags': round(correct / len(rows) * 100, 2),
            'unique_questions': len(keys), 'strict_jsonl_errors': bad,
            'original_summary_matches': True,
            'training_overlap_questions': len(keys & data['train']),
            'validation_overlap_questions': len(keys & data['val']),
        }
        excluded = data['train'] | data['val']
        clean = [row for row in rows if question_key(row['question']) not in excluded]
        report['evaluations'][name]['after_text_overlap_exclusion'] = {
            'records': len(clean), 'correct_saved_flags': sum(row['correct'] for row in clean),
            'accuracy_percent_saved_flags': round(sum(row['correct'] for row in clean) / len(clean) * 100, 2),
            'note': 'Diagnostic reuse of saved flags; not a newly validated benchmark result.'}
        hits = [{'question_sha256': hashlib.sha256(key.encode()).hexdigest(),
                 'in_train': key in data['train'], 'in_val': key in data['val']}
                for key in sorted(keys & excluded)]
        if write:
            (folder / 'overlap_ids.json').write_text(json.dumps(hits, indent=2) + '\n')
        elif json.loads((folder / 'overlap_ids.json').read_text()) != hits:
            raise ValueError('Overlap evidence changed: ' + name)
    report['overlap'] = {
        'method': 'Strip two known instruction suffixes, collapse whitespace, casefold; no fuzzy matching.',
        'train_val_questions': len(data['train'] & data['val']),
        'gsm8k_d1_d2_questions': len(evaluations['gsm8k_d1'] & evaluations['gsm8k_d2'])}
    run = root / 'evidence/training/run250'
    config = json.loads((run / 'run_config.json').read_text())
    state = json.loads((run / 'checkpoint-250/trainer_state.json').read_text())
    logs = json.loads((run / 'log_history.json').read_text())
    assert config['max_steps'] == state['global_step'] == 250
    assert config['load_in_4bit'] is False
    report['training'] = {'step': state['global_step'], 'load_in_4bit': config['load_in_4bit'],
                          'train_runtime_seconds': next(x['train_runtime'] for x in logs if 'train_runtime' in x),
                          'evaluation_adapter_mapping': 'unresolved: historical eval paths name 200; recovered final is 250'}
    report['weights'] = json.loads((run / 'weight_manifest.json').read_text())
    for path in sorted((root / 'artifacts').glob('checkpoint-*/adapter_model.safetensors')):
        expected = report['weights'][path.parent.name]
        if digest(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('Local adapter does not match the recovered manifest: ' + str(path))
    manifest = [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': digest(path)}
                for path in sorted((root / 'evidence').rglob('*'))
                if path.is_file() and path.name not in ['manifest.json', 'audit_report.json']]
    if write:
        (root / 'evidence/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (root / 'evidence/audit_report.json').write_text(json.dumps(report, indent=2) + '\n')
    else:
        if json.loads((root / 'evidence/manifest.json').read_text()) != manifest:
            raise ValueError('Evidence manifest mismatch; investigate before regenerating it.')
        if json.loads((root / 'evidence/audit_report.json').read_text()) != report:
            raise ValueError('Evidence report mismatch.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write', action='store_true', help='Explicitly regenerate manifests after reviewing intentional changes')
    args = parser.parse_args()
    audit(args.root, write=args.write)
