"""Dependency-free numeric scoring and experiment provenance for new runs.

Historical archived scores deliberately do not use this scorer.
"""
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

SCORER = 'strict_numeric_v1'
NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?'
SYSTEM_PROMPT = ('Solve the mathematical problem. Give the final numerical answer '
                 'inside \\boxed{}. Do not include units inside the box.')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def question_id(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def extract_numeric(text, allow_bare=False):
    """Accept explicit answers only; never use an arbitrary final number."""
    text = re.sub(r'<think>.*?</think>', '', str(text), flags=re.DOTALL).strip()
    # An unclosed thinking block cannot supply a final answer.
    if '<think>' in text:
        return None, 'unfinished_thinking'
    boxes = re.findall(r'\\boxed\{([^{}]*)\}', text)
    markers = re.findall(r'^\s*####\s*(.+?)\s*$', text, flags=re.MULTILINE)
    if boxes:
        candidate, method = boxes[-1].strip(), 'boxed'
    elif markers:
        candidate, method = markers[-1].strip(), 'hash_marker'
    elif allow_bare:
        candidate, method = text, 'bare_gold'
    else:
        return None, 'missing_explicit_answer'
    if not re.fullmatch(NUMBER, candidate):
        return None, 'unsupported_numeric_format'
    try:
        value = Fraction(candidate.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None, 'invalid_number'
    return value, method


def score(prediction, ground_truth):
    gold, gold_method = extract_numeric(ground_truth, allow_bare=True)
    answer, method = extract_numeric(prediction)
    return {'correct': gold is not None and answer is not None and answer == gold,
            'extracted_answer': str(answer) if answer is not None else None,
            'extracted_gold': str(gold) if gold is not None else None,
            'answer_parse_status': method, 'gold_parse_status': gold_method,
            'scorer': SCORER}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def local_fingerprint(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError('Model and adapter inputs must be existing local directories.')
    files = sorted(f for f in path.rglob('*') if f.is_file() and
                   (f.suffix in {'.json', '.safetensors', '.jinja', '.model', '.bin'}))
    if not files:
        raise ValueError('No model or adapter files found in ' + str(path))
    return [{'file': str(f.relative_to(path)), 'bytes': f.stat().st_size,
             'sha256': sha256(f)} for f in files]
