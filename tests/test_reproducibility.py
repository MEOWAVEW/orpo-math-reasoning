import json
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_evidence import question_key, read_archive
from eval_protocol import extract_numeric, read_jsonl, score, sha256
from evaluate import build_parser, validate
from prepare_clean_data import prepare


class ScoringTests(unittest.TestCase):
    def test_arbitrary_final_number_is_not_an_answer(self):
        result = score('I counted 42 items while working.', '42')
        self.assertFalse(result['correct'])
        self.assertEqual(result['answer_parse_status'], 'missing_explicit_answer')

    def test_explicit_wrong_answer_cannot_be_rescued(self):
        self.assertFalse(score('\\boxed{7} I also considered 42.', '42')['correct'])

    def test_numeric_equivalence(self):
        self.assertTrue(score('\\boxed{1/2}', '#### 0.5')['correct'])
        self.assertTrue(score('\\boxed{1,200}', '1200')['correct'])

    def test_unsupported_gold_is_visible(self):
        result = score('\\boxed{1}', 'The answer is a circle.')
        self.assertFalse(result['correct'])
        self.assertIsNone(result['extracted_gold'])

    def test_thinking_is_excluded(self):
        self.assertTrue(score('<think>\\boxed{9}</think>\\boxed{2}', '2')['correct'])
        self.assertIsNone(extract_numeric('<think>\\boxed{2}')[0])


class ArchiveTests(unittest.TestCase):
    def test_unicode_separator_is_not_a_jsonl_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'records.jsonl'
            row = {'question': 'first\u2028second', 'ground_truth': '2',
                   'prediction': '\\boxed{2}', 'correct': True}
            path.write_text(json.dumps(row, ensure_ascii=False) + '\n')
            rows, errors = read_archive(path)
            self.assertEqual(rows, [row])
            self.assertEqual(errors, [])

    def test_tampered_archived_labels_fail_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'records.jsonl'
            path.write_text('{"question":"x","ground_truth":"2","prediction":"2","correct":"false"}\n')
            with self.assertRaises(AssertionError):
                read_archive(path)


class PipelineTests(unittest.TestCase):
    def test_frozen_samples_match_archives_without_saved_predictions(self):
        for name in ['calc', 'gsm8k_d1', 'gsm8k_d2']:
            samples = ROOT / 'data/evaluation' / (name + '.jsonl')
            manifest = json.loads(samples.with_suffix('.manifest.json').read_text())
            source = ROOT / 'evidence/evaluation' / name / 'predictions_raw.jsonl'
            self.assertEqual(sha256(source), manifest['source_sha256'])
            self.assertEqual(sha256(samples), manifest['samples_sha256'])
            rows, original = read_jsonl(samples), read_jsonl(source)
            self.assertEqual(len(rows), manifest['records'])
            self.assertEqual([(r['question'], r['ground_truth']) for r in rows],
                             [(r['question'], r['ground_truth']) for r in original])
            self.assertTrue(all(set(row) == {'question', 'ground_truth', 'sample_id'} for row in rows))

    def test_clean_split_is_reproducible_and_excludes_eval(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'clean'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(ROOT, output)
            self.assertEqual(json.loads((output / 'manifest.json').read_text()),
                             json.loads((ROOT / 'data/clean/manifest.json').read_text()))
            train = {question_key(r['question']) for r in read_jsonl(output / 'train.jsonl')}
            val = {question_key(r['question']) for r in read_jsonl(output / 'val.jsonl')}
            blocked = {question_key(r['question']) for p in (ROOT / 'evidence/evaluation').glob('*/predictions_raw.jsonl')
                       for r in read_jsonl(p)}
            self.assertFalse(train & val)
            self.assertFalse((train | val) & blocked)
            with self.assertRaises(FileExistsError):
                prepare(ROOT, output)

    def test_baseline_needs_no_adapter_and_duplicate_samples_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'samples.jsonl'
            path.write_text('{"question":"2+2?","ground_truth":"4"}\n' * 2)
            args = build_parser().parse_args(['--base_model', folder, '--samples_file', str(path),
                                             '--output_dir', str(Path(folder) / 'new')])
            self.assertIsNone(args.lora_model)
            with self.assertRaises(ValueError):
                validate(args)

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            args = build_parser().parse_args(['--base_model', folder, '--samples_file', 'unused',
                                             '--output_dir', folder])
            with self.assertRaises(FileExistsError):
                validate(args)

    def test_offline_audit_is_read_only(self):
        paths = [ROOT / 'evidence/audit_report.json', ROOT / 'evidence/manifest.json']
        before = [path.read_bytes() for path in paths]
        subprocess.run([sys.executable, str(ROOT / 'scripts/audit_evidence.py')],
                       cwd=ROOT, check=True, capture_output=True)
        self.assertEqual(before, [path.read_bytes() for path in paths])


if __name__ == '__main__':
    unittest.main()
