"""Compare direct training and transfer ablations using shared binary pretraining."""
import argparse
import copy
import csv
import fcntl
import gc
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from transformers import AutoTokenizer

from experiment_data import fingerprint, prepare_experiment, stage_rows, write_json
from model import TEXT_MODEL
from tracking import add_tracking_arguments
from training_engine import run_stage, file_sha256

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
VARIANTS = ('direct', 'full', 'frozen', 'direct_5k', 'reset_fusion', 'tune_fusion')
TRANSFER_VARIANTS = ('full', 'frozen', 'reset_fusion', 'tune_fusion')


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another experiment is using {path}; run the scripts sequentially.') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--data', type=Path, default=PROJECT / 'data/subset_50000.zip')
    p.add_argument('--source-list', type=Path, default=PROJECT / 'data/manifests/source_binary_20000.tsv')
    p.add_argument('--target-list', type=Path, default=PROJECT / 'data/manifests/finetune_5000.tsv')
    p.add_argument('--cache', type=Path, default=HERE / 'prepared_comparison_v2')
    p.add_argument('--output', type=Path, default=HERE / 'runs_comparison')
    p.add_argument('--architecture', choices=['multimodal', 'vit'], default='multimodal')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--validation-size', type=int, default=2000)
    p.add_argument('--test-size', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=20, help='Direct seven-class or stage-two maximum epochs')
    p.add_argument('--binary-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--binary-patience', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-length', type=int, default=96)
    p.add_argument('--encoder-lr', type=float, default=5e-6)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-ratio', type=float, default=0.05)
    p.add_argument('--class-weight-power', type=float, default=0.0)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--selection-metric', choices=['accuracy', 'macro_f1'], default='accuracy')
    p.add_argument('--target-accuracy', type=float, default=0.8)
    p.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default='cuda')
    p.add_argument('--resume', action='store_true', help='Resume interrupted stages from their last complete epoch')
    p.add_argument('--stop-file', type=Path, help='Stop the whole experiment queue after the current epoch')
    p.add_argument('--trust-legacy-checkpoint', action='store_true', help='Allow pickle loading ONLY for a trusted old checkpoint')
    p.add_argument('--prepare-only', action='store_true')
    add_tracking_arguments(p)
    return p


def stage_arguments(args, dataset, binary=False):
    stage = copy.copy(args)
    stage.dataset_id = dataset['dataset_id']
    stage.num_classes = 2 if binary else 7
    stage.stage = 'binary' if binary else 'direct' if args.variant in ('direct', 'direct_5k') else 'stage2'
    stage.variant = 'shared_binary' if binary else args.variant
    stage.freeze_backbone = not binary and args.variant == 'frozen'
    stage.freeze_encoders = not binary and args.variant in ('frozen', 'tune_fusion')
    stage.reset_fusion = not binary and args.variant == 'reset_fusion'
    stage.epochs = args.binary_epochs if binary else args.epochs
    stage.patience = args.binary_patience if binary else args.patience
    stage.selection_metric = 'macro_f1' if binary else args.selection_metric
    stage.require_cuda = args.device == 'cuda'
    stage.init_checkpoint = None
    stage.resume = None
    return stage


def execute_stage(stage, rows, tokenizer, resume_allowed):
    if stage.stop_file and stage.stop_file.exists():
        return {'status': 'stopped'}, False
    out = stage.output / f'seed_{stage.seed}'
    # Operational flags (W&B mode, output, resume) must not change model identity.
    keys = ('dataset_id', 'architecture', 'num_classes', 'freeze_backbone', 'freeze_encoders', 'reset_fusion',
            'epochs', 'patience', 'batch_size',
            'max_length', 'encoder_lr', 'head_lr', 'dropout', 'weight_decay', 'warmup_ratio',
            'class_weight_power', 'label_smoothing', 'selection_metric', 'seed', 'target_accuracy', 'device')
    expected = {k: getattr(stage, k) for k in keys}
    expected['rows_sha256'] = fingerprint(rows)
    expected['init_checkpoint_sha256'] = file_sha256(stage.init_checkpoint) if stage.init_checkpoint else None
    identity = out / 'stage_identity.json'
    if identity.exists():
        recorded = json.loads(identity.read_text())
        # Old identities predate these ablations; their behavior is unchanged.
        recorded.setdefault('freeze_encoders', recorded['freeze_backbone'])
        recorded.setdefault('reset_fusion', False)
        if recorded != expected:
            raise ValueError(f'Experiment configuration changed in {out}; choose a new --output directory')
    if not identity.exists() and (out / 'last.pt').exists():
        raise ValueError(f'Existing checkpoint has no experiment identity: {out}')
    if not identity.exists():
        write_json(identity, expected)
    result_file = out / 'results.json'
    if result_file.exists():
        result = json.loads(result_file.read_text())
        if result['status'] == 'completed':
            if not (out / 'best.pt').is_file():
                raise FileNotFoundError(out / 'best.pt')
            print(json.dumps({'event': 'reuse_completed_stage', 'output': str(out)}), flush=True)
            return result, True
    if (out / 'last.pt').exists():
        if not resume_allowed:
            raise FileExistsError(f'Interrupted stage in {out}. Remove its STOP file, then pass --resume.')
        if (out / 'STOP').exists():
            raise ValueError(f'Remove {out / "STOP"} before resuming')
        stage.resume = out / 'last.pt'
    result = run_stage(stage, rows, tokenizer, stage.seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, False


def comparison_table(output, dataset_id, architecture, seed):
    entries = []
    for variant in VARIANTS:
        path = output / variant / f'seed_{seed}' / 'results.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result['status'] != 'completed':
            continue
        config = result['config']
        if (config['dataset_id'], config['architecture']) != (dataset_id, architecture):
            raise ValueError('Cannot compare different datasets/architectures in one output directory')
        entries.append({**result['comparison'], 'seed': seed, 'architecture': architecture,
                        'dataset_id': dataset_id, 'batch_size': config['batch_size'],
                        'selection_metric': config['selection_metric'], 'gpu_name': config['gpu_name'],
                        'seconds_to_target_accuracy': result['efficiency']['seconds_to_target_accuracy'],
                        'result_file': str(path)})
    write_json(output / f'comparison_seed_{seed}.json', entries)
    if entries:
        with (output / f'comparison_seed_{seed}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    return entries


def main():
    p = parser()
    args = p.parse_args()
    for field in ('data', 'source_list', 'target_list', 'cache', 'output'):
        setattr(args, field, getattr(args, field).resolve())
    if args.stop_file:
        args.stop_file = args.stop_file.resolve()
        if args.stop_file.exists():
            return 130
    if min(args.epochs, args.binary_epochs, args.patience, args.binary_patience, args.batch_size,
           args.max_length, args.log_every, args.validation_size, args.test_size) <= 0:
        p.error('Epochs, patience, batch/sequence sizes, log interval and split sizes must be positive')
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.target_accuracy <= 1:
        p.error('Require 0 <= warmup-ratio < 1 and 0 <= target-accuracy <= 1')
    if not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing <= 1:
        p.error('Invalid dropout or label smoothing')
    if min(args.encoder_lr, args.head_lr) <= 0 or args.weight_decay < 0:
        p.error('Learning rates must be positive and weight decay nonnegative')
    if not all(math.isfinite(getattr(args, name)) for name in (
        'encoder_lr', 'head_lr', 'weight_decay', 'class_weight_power', 'dropout', 'warmup_ratio',
        'label_smoothing', 'target_accuracy')):
        p.error('Numeric hyperparameters must be finite')
    if not args.prepare_only and args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Explicitly choose --device cpu only for smoke checks.')
    started = time.perf_counter()
    with exclusive(args.output / '.experiment.lock'):
        with exclusive(args.cache / '.prepare.lock'):
            dataset = prepare_experiment(args)
        preparation_seconds = time.perf_counter() - started
        print(json.dumps({'event': 'dataset_ready', 'dataset_id': dataset['dataset_id'],
                          'audit': dataset['audit'], 'preparation_seconds': preparation_seconds}), flush=True)
        if args.prepare_only:
            return
        cohort_path = args.output / 'comparison_identity.json'
        cohort_keys = ('architecture', 'batch_size', 'max_length', 'selection_metric', 'device',
                       'epochs', 'binary_epochs', 'patience', 'binary_patience', 'encoder_lr', 'head_lr',
                       'dropout', 'weight_decay', 'warmup_ratio', 'class_weight_power', 'label_smoothing',
                       'target_accuracy', 'wandb_mode', 'log_every')
        cohort = {**{key: getattr(args, key) for key in cohort_keys}, 'dataset_id': dataset['dataset_id'],
                  'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
                  'gpu_name': torch.cuda.get_device_name() if args.device == 'cuda' else None}
        if cohort_path.exists() and json.loads(cohort_path.read_text()) != cohort:
            raise ValueError('Comparison data/architecture/batch/evaluation settings changed; use a new --output directory')
        write_json(cohort_path, cohort)
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL) if args.architecture == 'multimodal' else None
        final = stage_arguments(args, dataset)
        final.output = args.output / args.variant
        final.data_preparation_seconds = preparation_seconds
        if args.variant in TRANSFER_VARIANTS:
            binary = stage_arguments(args, dataset, binary=True)
            binary_rows = stage_rows(dataset['records'], args.variant, binary=True)
            binary_keys = ('architecture', 'epochs', 'patience', 'batch_size', 'max_length', 'encoder_lr', 'head_lr',
                           'dropout', 'weight_decay', 'warmup_ratio', 'class_weight_power', 'label_smoothing', 'seed',
                           'target_accuracy', 'device')
            binary_id = fingerprint({**{k: getattr(binary, k) for k in binary_keys}, 'rows': fingerprint(binary_rows)})[:16]
            binary.output = args.output / 'shared_binary' / binary_id
            result, reused = execute_stage(binary, binary_rows, tokenizer, args.resume)
            if result['status'] != 'completed':
                return 130
            final.init_checkpoint = binary.output / f'seed_{args.seed}' / 'best.pt'
            final.pretraining_efficiency = result['efficiency']
            final.pretraining_sample_count = result['config']['sample_counts']['train']
            final.pretraining_reused = reused
            write_json(final.output / f'seed_{args.seed}' / 'pretraining.json', {
                'checkpoint': str(final.init_checkpoint), 'sha256': file_sha256(final.init_checkpoint),
                'reused': reused, 'efficiency': result['efficiency'],
            })
        result, _ = execute_stage(final, stage_rows(dataset['records'], args.variant), tokenizer, args.resume)
        if result['status'] == 'completed':
            entries = comparison_table(args.output, dataset['dataset_id'], args.architecture, args.seed)
            print(json.dumps({'event': 'comparison_updated', 'csv': str(args.output / f'comparison_seed_{args.seed}.csv'),
                              'completed_variants': [r['variant'] for r in entries]}), flush=True)
        else:
            return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
