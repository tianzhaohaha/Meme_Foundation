"""CPU integration checks with tiny encoders; no downloads or full training runs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))

import experiment
import model
import training_engine
from experiment_data import EMOTIONS, POLARITY, fingerprint, read_list, stage_rows
from training_engine import run_stage, load_checkpoint


class UnsafeCheckpoint:
    def __reduce__(self):
        return eval, ('1 + 1',)


class TinyImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3, 768)
        self.dropout = nn.Dropout(0.2)
        self.heads = nn.Identity()

    def forward(self, images):
        return self.dropout(self.project(images.mean(dim=(2, 3))))


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 768)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


def tokenize(texts, **kwargs):
    return {'input_ids': torch.tensor([[int(t) % 16, 1] for t in texts]),
            'attention_mask': torch.ones((len(texts), 2), dtype=torch.long)}


class ExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tumemo-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image_patch = patch.object(model, 'vit_b_16', side_effect=lambda **kw: TinyImage())
        self.text_patch = patch.object(model.AutoModel, 'from_pretrained', side_effect=lambda *a, **kw: TinyText())
        self.image_patch.start()
        self.text_patch.start()
        self.addCleanup(self.image_patch.stop)
        self.addCleanup(self.text_patch.stop)
        self.rows = []
        for split in ('source', 'target', 'validation', 'test'):
            for i, emotion in enumerate(EMOTIONS):
                path = self.root / f'{split}_{i}.jpg'
                Image.new('RGB', (12, 12), (i * 30, 90, 150)).save(path)
                self.rows.append({'id': f'{split}_{i}', 'emotion': emotion, 'label': i,
                                  'binary_label': POLARITY[emotion], 'split': split, 'image': str(path),
                                  'image_sha256': f'{split}_{i}', 'text': str(i)})
        self.dataset = {'records': self.rows, 'dataset_id': fingerprint(self.rows), 'audit': {}}

    def args(self, output, **overrides):
        args = experiment.parser().parse_args(['--variant', 'direct', '--device', 'cpu', '--wandb-mode', 'disabled',
                                               '--output', str(output), '--epochs', '2', '--binary-epochs', '1',
                                               '--batch-size', '7'])
        for key, value in overrides.items():
            setattr(args, key, value)
        return experiment.stage_arguments(args, self.dataset)

    def test_frozen_encoders_and_fusion_stay_identical(self):
        torch.manual_seed(42)
        net = model.EmotionModel(num_classes=7)
        net.freeze_backbone()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        net.train()
        self.assertFalse(net.image_encoder.training)
        self.assertFalse(net.text_encoder.training)
        self.assertFalse(net.fusion.training)
        self.assertTrue(net.head.training)
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=.01)
        loss = net(torch.rand(7, 3, 12, 12), torch.ones(7, 2, dtype=torch.long), torch.ones(7, 2)).sum()
        loss.backward()
        optimizer.step()
        for name, value in net.state_dict().items():
            if not name.startswith('head.'):
                self.assertTrue(torch.equal(before[name], value), name)
        self.assertFalse(torch.equal(before['head.1.weight'], net.head[1].weight))
        self.assertEqual({n for n, p in net.named_parameters() if p.requires_grad}, {'head.1.weight', 'head.1.bias'})

    def test_three_workflows_reuse_pretraining_and_compare(self):
        output = self.root / 'runs'
        for variant in ('direct', 'full', 'frozen'):
            argv = ['experiment.py', '--variant', variant, '--device', 'cpu', '--wandb-mode', 'disabled',
                    '--output', str(output), '--cache', str(self.root / 'cache'), '--epochs', '1',
                    '--binary-epochs', '1', '--batch-size', '7']
            with patch.object(sys, 'argv', argv), patch.object(experiment, 'prepare_experiment', return_value=self.dataset), \
                 patch.object(experiment.AutoTokenizer, 'from_pretrained', return_value=tokenize):
                experiment.main()
        result = {v: json.loads((output / v / 'seed_42/results.json').read_text()) for v in ('direct', 'full', 'frozen')}
        self.assertEqual(result['direct']['config']['sample_counts']['train'], 14)
        self.assertEqual(result['full']['config']['sample_counts']['train'], 7)
        self.assertEqual(result['frozen']['config']['trainable_modules'], ['head'])
        self.assertEqual(result['full']['config']['init_checkpoint_sha256'], result['frozen']['config']['init_checkpoint_sha256'])
        self.assertTrue(result['frozen']['comparison']['pretraining_reused'])
        self.assertEqual(len(list((output / 'shared_binary').glob('*/seed_42/results.json'))), 1)
        comparison = json.loads((output / 'comparison_seed_42.json').read_text())
        self.assertEqual(len(comparison), 3)
        for variant, r in result.items():
            self.assertEqual(sum(sum(row) for row in r['test']['confusion_matrix']), 7)
            self.assertGreater(r['efficiency']['samples_per_second'], 0)
            self.assertGreater(r['efficiency']['checkpoint_seconds'], 0)
            self.assertAlmostEqual(r['comparison']['total_train_seconds'], r['efficiency']['train_seconds'] + r['comparison']['pretraining_train_seconds'])
        pretrained = load_checkpoint(next((output / 'shared_binary').glob('*/seed_42/best.pt')))
        frozen = load_checkpoint(output / 'frozen/seed_42/last.pt')
        full = load_checkpoint(output / 'full/seed_42/last.pt')
        for name, tensor in pretrained['model_state_dict'].items():
            if not name.startswith('head.'):
                self.assertTrue(torch.equal(tensor, frozen['model_state_dict'][name]), name)
        self.assertFalse(torch.equal(pretrained['model_state_dict']['image_encoder.project.weight'],
                                     full['model_state_dict']['image_encoder.project.weight']))
        for name in ('text_encoder.embedding.weight', 'fusion.0.weight'):
            self.assertFalse(torch.equal(pretrained['model_state_dict'][name], full['model_state_dict'][name]), name)
        self.assertEqual(result['direct']['comparison']['seven_class_labeled_train_samples'], 14)
        self.assertEqual(result['full']['comparison']['seven_class_labeled_train_samples'], 7)
        # Changing a learning rate between strategies must not silently enter one comparison.
        with patch.object(sys, 'argv', argv + ['--head-lr', '0.1']), \
             patch.object(experiment, 'prepare_experiment', return_value=self.dataset):
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                experiment.main()

    def test_resume_matches_uninterrupted_weights_and_accumulates_time(self):
        rows = stage_rows(self.rows, 'direct')
        reference = self.args(self.root / 'reference')
        run_stage(reference, rows, tokenize, 42)
        resumed = self.args(self.root / 'resumed')
        out = resumed.output / 'seed_42'
        out.mkdir(parents=True)
        stop = out / 'STOP'
        stop.touch()
        first = run_stage(resumed, rows, tokenize, 42)
        self.assertEqual(first['status'], 'stopped')
        stop.unlink()
        resumed.resume = out / 'last.pt'
        second = run_stage(resumed, rows, tokenize, 42)
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(second['efficiency']['optimizer_steps'], 4)
        self.assertGreater(second['efficiency']['train_seconds'], first['efficiency']['train_seconds'])
        a = load_checkpoint(reference.output / 'seed_42/last.pt')['model_state_dict']
        b = load_checkpoint(out / 'last.pt')['model_state_dict']
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)
        resumed.batch_size = 1
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            run_stage(resumed, rows, tokenize, 42)

    def test_restricted_checkpoint_loading_rejects_code_objects(self):
        path = self.root / 'unsafe.pt'
        torch.save(UnsafeCheckpoint(), path)
        with self.assertRaisesRegex(ValueError, 'restricted format'):
            load_checkpoint(path)

    def test_recover_crash_between_last_and_best_publication(self):
        args = self.args(self.root / 'interrupted', epochs=1)
        rows = stage_rows(self.rows, 'direct')
        with patch.object(training_engine, 'publish_checkpoint', side_effect=RuntimeError('simulated crash')):
            with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
                run_stage(args, rows, tokenize, 42)
        out = args.output / 'seed_42'
        self.assertTrue((out / 'last.pt').exists())
        self.assertFalse((out / 'best.pt').exists())
        args.resume = out / 'last.pt'
        result = run_stage(args, rows, tokenize, 42)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue((out / 'best.pt').exists())
        self.assertFalse(result['efficiency']['timing_complete'])

    def test_tracking_finalize_failure_is_not_marked_completed(self):
        args = self.args(self.root / 'tracking_failure', epochs=1)
        with patch.object(training_engine.Tracker, 'finish', side_effect=RuntimeError('sync failed')):
            with self.assertRaisesRegex(RuntimeError, 'sync failed'):
                run_stage(args, stage_rows(self.rows, 'direct'), tokenize, 42)
        result = json.loads((args.output / 'seed_42/results.json').read_text())
        self.assertEqual(result['status'], 'finalizing_tracking')

    def test_stop_is_nonzero_and_does_not_launch_second_stage(self):
        argv = ['experiment.py', '--variant', 'full', '--device', 'cpu', '--wandb-mode', 'disabled',
                '--output', str(self.root / 'stopped'), '--cache', str(self.root / 'cache')]
        with patch.object(sys, 'argv', argv), patch.object(experiment, 'prepare_experiment', return_value=self.dataset), \
             patch.object(experiment.AutoTokenizer, 'from_pretrained', return_value=tokenize), \
             patch.object(experiment, 'execute_stage', return_value=({'status': 'stopped'}, False)) as execute:
            self.assertEqual(experiment.main(), 130)
            self.assertEqual(execute.call_count, 1)

    def test_image_and_text_features_and_mask(self):
        net = model.EmotionModel(num_classes=7).eval()
        images = torch.rand(1, 3, 12, 12)
        mask = torch.tensor([[1, 0]])
        with torch.no_grad():
            first = net(images, torch.tensor([[1, 2]]), mask)
            padded = net(images, torch.tensor([[1, 3]]), mask)
            changed_text = net(images, torch.tensor([[4, 2]]), mask)
            changed_image = net(images + .5, torch.tensor([[1, 2]]), mask)
        self.assertTrue(torch.equal(first, padded))
        self.assertFalse(torch.equal(first, changed_text))
        self.assertFalse(torch.equal(first, changed_image))

    def test_list_validation_and_test_exclusion(self):
        with self.assertRaises(ValueError):
            read_list('id\tlabel\tpolarity\n1\tHappy\tnegative\n')
        with self.assertRaises(ValueError):
            read_list('id\tlabel\tpolarity\n1\tHappy\tpositive\n1\tHappy\tpositive\n')
        binary = stage_rows(self.rows, 'full', binary=True)
        self.assertEqual({r['split'] for r in binary}, {'train', 'validation'})
        self.assertEqual({r['label'] for r in binary}, {0, 1})
        for variant in ('direct', 'full', 'frozen'):
            selected = stage_rows(self.rows, variant)
            self.assertEqual({r['id'] for r in selected if r['split'] == 'test'}, {r['id'] for r in self.rows if r['split'] == 'test'})

    def test_wandb_offline_logs_metrics_and_summary(self):
        args = self.args(self.root / 'offline', epochs=2, wandb_mode='offline')
        out = args.output / 'seed_42'
        out.mkdir(parents=True)
        (out / 'STOP').touch()
        first = run_stage(args, stage_rows(self.rows, 'direct'), tokenize, 42)
        self.assertEqual(first['status'], 'stopped')
        first_identity = json.loads((out / 'wandb_run.json').read_text())
        (out / 'STOP').unlink()
        args.resume = out / 'last.pt'
        result = run_stage(args, stage_rows(self.rows, 'direct'), tokenize, 42)
        identity = json.loads((out / 'wandb_run.json').read_text())
        self.assertEqual(identity['mode'], 'offline')
        self.assertNotEqual(first_identity['id'], identity['id'])
        self.assertEqual(result['status'], 'completed')
        run_files = list((args.output / 'seed_42/wandb').glob('offline-run-*/run-*.wandb'))
        self.assertEqual(len(run_files), 2)
        self.assertTrue(all(path.stat().st_size > 0 for path in run_files))


if __name__ == '__main__':
    unittest.main()
