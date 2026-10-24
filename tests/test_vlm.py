"""Guard evaluation denominators, prompt boundaries and resume sample identity."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from evaluate_vlm import EMOTIONS, log_wandb, messages, parse_label, score, validate_progress


class VLMEvaluationTest(unittest.TestCase):
    def test_parser_does_not_guess_from_explanations_or_multiple_labels(self):
        self.assertEqual(parse_label(' "happy".\n'), 4)
        for response in ('Happy or Sad', 'The answer is Happy', '', 'happiness', '4'):
            self.assertEqual(parse_label(response), -1)

    def test_invalid_response_remains_a_false_negative(self):
        rows = [{'label': i, 'prediction': i} for i in range(7)]
        rows[0]['prediction'] = -1
        result = score(rows)
        self.assertEqual(result['accuracy'], 6 / 7)
        self.assertAlmostEqual(result['macro_f1'], 6 / 7)
        self.assertEqual(result['invalid_outputs'], 1)
        self.assertEqual(result['confusion_matrix'][0][-1], 1)
        self.assertEqual(sum(map(sum, result['confusion_matrix'])), 7)

    def test_resume_rejects_reordered_or_modified_predictions(self):
        rows = [{'id': 'one', 'label': 4}, {'id': 'two', 'label': 6}]
        saved = {'predictions': [{**r, 'prediction': r['label'], 'response': EMOTIONS[r['label']]} for r in rows],
                 'batches': [{'samples': 2}]}
        validate_progress(saved, rows, 2)
        with self.assertRaises(ValueError):
            validate_progress(saved, list(reversed(rows)), 2)
        saved['predictions'][0]['prediction'] = 6
        with self.assertRaises(ValueError):
            validate_progress(saved, rows, 2)

    def test_model_prompt_has_only_the_image_placeholder_and_caption(self):
        message = messages('An ordinary day.')[0]
        self.assertEqual(message['content'][0], {'type': 'image'})
        self.assertIn('<post_text>\nAn ordinary day.\n</post_text>', message['content'][1]['text'])
        self.assertEqual(set(message), {'role', 'content'})

    def test_offline_wandb_records_completed_validation_metrics(self):
        args = SimpleNamespace(wandb_mode='offline', wandb_project='tumemo-vlm-test', wandb_entity=None, model='4b')
        result = {'config': {'model_id': 'test-only'},
                  'metrics': score([{'label': i, 'prediction': i} for i in range(7)]),
                  'efficiency': {'evaluation_seconds': 1.0}}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            identity = log_wandb(args, out, result)
            self.assertEqual(identity['mode'], 'offline')
            self.assertEqual(len(identity['id']), 12)
            self.assertEqual(json.loads((out / 'wandb_run.json').read_text()), identity)
            self.assertTrue(list(out.glob('wandb/offline-run-*/*.wandb')))


if __name__ == '__main__':
    unittest.main()
