"""Protect the reported downstream cost scope and matched comparison inputs."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from plot_main_experiment import summarize


class MainExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads((ROOT / 'reports/data/seed42.json').read_text())

    def test_source_and_full_lifecycle_costs_cannot_change_downstream_results(self):
        expected = summarize(self.snapshot)
        changed = copy.deepcopy(self.snapshot)
        for run in changed['runs']:
            if run['variant'] == 'shared_binary':
                run['efficiency']['stage_wall_seconds'] = 1e12
                run['efficiency']['train_seconds'] = 1e12
            run['comparison'] = {'total_stage_wall_seconds': 1e15, 'stage2_stage_wall_seconds': None}
        self.assertEqual(summarize(changed), expected)
        self.assertEqual([r['model'] for r in expected['runs']], ['V4', 'V5', 'V6'])
        self.assertAlmostEqual(expected['runs'][2]['stage_time_reduction_vs_v4_pct'], 67.57354983255033)
        self.assertAlmostEqual(expected['runs'][2]['test_accuracy_delta_vs_v4_pp'], -2.15)
        self.assertAlmostEqual(expected['runs'][2]['validation_accuracy_delta_vs_v4_pp'], -.2)

    def test_mismatched_data_incomplete_timing_and_wrong_selection_are_rejected(self):
        mutations = [('config', 'dataset_id', 'different-data'),
                     ('config', 'init_checkpoint_sha256', 'different-source'),
                     ('efficiency', 'timing_complete', False),
                     ('best_validation', 'accuracy', .99)]
        for section, key, value in mutations:
            changed = copy.deepcopy(self.snapshot)
            run = next(r for r in changed['runs'] if r['variant'] == 'tune_fusion')
            run[section][key] = value
            with self.assertRaises(ValueError):
                summarize(changed)


if __name__ == '__main__':
    unittest.main()
