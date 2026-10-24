"""Portable data identity must ignore paths while detecting changed model inputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from verify_data import records_fingerprint


class ReproductionTest(unittest.TestCase):
    def test_moving_cache_preserves_identity_but_inputs_and_order_do_not(self):
        row = {'id': '1', 'emotion': 'Happy', 'label': 4, 'binary_label': 1, 'split': 'target',
               'image_sha256': 'a'*64, 'image': '/old/cache/image.jpg', 'text': 'example'}
        second = {**row, 'id': '2', 'split': 'test'}
        original = records_fingerprint([row, second])
        self.assertEqual(original, records_fingerprint([{**row, 'image': '/new/cache/image.jpg'}, second]))
        self.assertNotEqual(original, records_fingerprint([second, row]))
        for key, changed in [('text', 'different'), ('image_sha256', 'b'*64), ('label', 5), ('split', 'source')]:
            self.assertNotEqual(original, records_fingerprint([{**row, key: changed}, second]), key)


if __name__ == '__main__':
    unittest.main()
