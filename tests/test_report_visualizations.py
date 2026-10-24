"""Reject misleading model joins and undefined class-center geometry."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_model_outputs import QWEN, center_similarity, join_predictions, select_cases


class ReportVisualizationsTest(unittest.TestCase):
    def records(self):
        return [{'model': name, 'id': sid, 'true_label': truth, 'predicted_label': pred}
                for name in QWEN for sid, truth, pred in [('b', 'Bored', 'Invalid'), ('a', 'Angry', 'Angry')]]

    def test_id_join_preserves_invalid_outputs_and_reference_order(self):
        result = join_predictions(['a', 'b'], [0, 1], self.records())
        for values in result.values():
            self.assertEqual(values.tolist(), [0, -1])

    def test_mismatched_missing_and_duplicate_predictions_are_rejected(self):
        rows = self.records()
        for invalid in [rows[:-1], rows + rows[:1], [{**rows[0], 'true_label': 'Sad'}] + rows[1:],
                        [{**rows[0], 'id': 'foreign'}] + rows[1:]]:
            with self.assertRaises(ValueError):
                join_predictions(['a', 'b'], [0, 1], invalid)

    def test_center_cosine_ignores_sample_magnitude_and_rejects_undefined_means(self):
        labels = np.repeat(np.arange(7), 2)
        features = np.eye(7)[labels] * np.arange(1, 15)[:, None]
        similarity, lengths = center_similarity(features, labels)
        np.testing.assert_allclose(similarity, np.eye(7), atol=1e-12)
        np.testing.assert_allclose(lengths, 1, atol=1e-12)
        for broken in [features * 0, np.full_like(features, np.nan)]:
            with self.assertRaises(ValueError):
                center_similarity(broken, labels)
        with self.assertRaises(ValueError):
            center_similarity(features[:-2], labels[:-2])

    def test_highest_confidence_error_is_reserved_without_duplicated_cases(self):
        ids = ['a', 'b', 'c', 'd']
        correct = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=bool)
        cases = select_cases(ids, correct, np.array([.99, .8, .6, .5]))
        self.assertEqual(cases[-1]['id'], 'a')
        self.assertEqual(cases[0]['id'], 'b')
        self.assertEqual(len(cases), len({c['id'] for c in cases}))


if __name__ == '__main__':
    unittest.main()
