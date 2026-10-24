"""Binary checkpoints must keep their head while plots keep seven-class ground truth."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tumemo-embedding-test-matplotlib')
import numpy as np
from embedding_analysis.extract import default_checkpoint, evaluation_targets, validate_checkpoint, verify_metrics
from embedding_analysis.plot import EMOTIONS, load_files


class EmbeddingAnalysisTest(unittest.TestCase):
    def test_binary_and_seven_class_checkpoints_keep_distinct_output_maps(self):
        binary = {'config': {'variant': 'shared_binary', 'num_classes': 2, 'architecture': 'multimodal'},
                  'epoch': 3, 'best_epoch': 3, 'label_mapping': {'negative': 0, 'positive': 1}}
        validate_checkpoint(binary, 'Binary')
        with self.assertRaises(ValueError):
            validate_checkpoint(binary, 'V4')
        changed = copy.deepcopy(binary)
        changed['label_mapping'] = {'positive': 0, 'negative': 1}
        with self.assertRaises(ValueError):
            validate_checkpoint(changed, 'Binary')
        rows = [{'emotion': name, 'label': i} for i, name in enumerate(EMOTIONS)]
        self.assertEqual(evaluation_targets(rows, 'Binary').tolist(), [0, 0, 1, 0, 1, 1, 0])
        self.assertEqual(evaluation_targets(rows, 'V4').tolist(), list(range(7)))

    def test_validation_reproduction_failure_blocks_export(self):
        metrics = {'accuracy': .565, 'macro_f1': .5423889}
        self.assertTrue(all(x['matches'] for x in verify_metrics(metrics, metrics).values()))
        with self.assertRaises(ValueError):
            verify_metrics(metrics, {**metrics, 'accuracy': .55})
        with self.assertRaises(ValueError):
            verify_metrics(metrics, {})

    def test_ambiguous_binary_sources_require_an_explicit_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                default_checkpoint(root, 'Binary')
            for name in ['a', 'b']:
                path = root / 'shared_binary' / name / 'seed_42/best.pt'
                path.parent.mkdir(parents=True)
                path.touch()
                if name == 'a':
                    self.assertEqual(default_checkpoint(root, 'Binary'), path)
            with self.assertRaises(ValueError):
                default_checkpoint(root, 'Binary')

    def test_plot_accepts_binary_head_with_seven_labels_and_rejects_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.arange(7, dtype=np.int64)
            meta = {'class_names': EMOTIONS, 'sample_count': 7, 'split': 'validation',
                    'validation_reproduction': {'accuracy': {'matches': True}, 'macro_f1': {'matches': True}},
                    'input_signature': 'same-inputs', 'text_max_length': 96}
            shared = {'ids': np.array([str(i) for i in labels]), 'labels': labels,
                      'polarity': np.isin(labels, [2, 4, 5]).astype(np.int64),
                      'image_sha256': np.array(['image'+str(i) for i in labels]),
                      'fusion': np.zeros((7, 256), dtype=np.float32)}
            paths = [root / 'Binary.npz', root / 'V4.npz']
            np.savez(paths[0], **shared, probabilities=np.zeros((7, 2)),
                     metadata=np.array(json.dumps({**meta, 'model': 'Binary', 'num_classes': 2})))
            np.savez(paths[1], **shared, probabilities=np.zeros((7, 7)),
                     metadata=np.array(json.dumps({**meta, 'model': 'V4', 'num_classes': 7})))
            self.assertEqual(len(load_files(paths)), 2)
            np.savez(paths[1], **{**shared, 'ids': shared['ids'][::-1]}, probabilities=np.zeros((7, 7)),
                     metadata=np.array(json.dumps({**meta, 'model': 'V4', 'num_classes': 7})))
            with self.assertRaisesRegex(ValueError, 'same samples and order'):
                load_files(paths)
            np.savez(paths[0], **shared, probabilities=np.zeros((7, 7)),
                     metadata=np.array(json.dumps({**meta, 'model': 'Binary', 'num_classes': 2})))
            with self.assertRaisesRegex(ValueError, 'Probability columns'):
                load_files(paths[:1])


if __name__ == '__main__':
    unittest.main()
