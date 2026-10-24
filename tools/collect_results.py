"""Export six completed local runs for plotting, without credentials or checkpoints."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('direct', 'full', 'frozen', 'direct_5k', 'reset_fusion', 'tune_fusion')


def read(path):
    return json.loads(path.read_text())


def collect(runs, seed):
    records = []
    common = None
    source_directories = set()
    source_hashes = set()
    keys = ('dataset_id', 'architecture', 'seed', 'batch_size', 'max_length', 'epochs', 'patience',
            'encoder_lr', 'head_lr', 'dropout', 'weight_decay', 'warmup_ratio', 'class_weight_power',
            'label_smoothing', 'selection_metric', 'gpu_name', 'torch_version', 'wandb_mode')
    directories = [runs / v / f'seed_{seed}' for v in VARIANTS]
    for directory in directories:
        result = read(directory / 'results.json')
        if result['status'] != 'completed':
            raise ValueError(f'Run is not completed: {directory}')
        config = result['config']
        settings = {k: config[k] for k in keys}
        if common is not None and settings != common:
            raise ValueError(f'Cannot compare different settings: {directory}')
        common = settings
        if config.get('init_checkpoint'):
            source = Path(config['init_checkpoint']).resolve().parent
            if not source.is_relative_to(runs):
                raise ValueError('Source stage must belong to the same runs directory.')
            source_directories.add(source)
            source_hashes.add(config['init_checkpoint_sha256'])
    if len(source_directories) != 1 or len(source_hashes) != 1:
        raise ValueError('Transfer variants must share one source model.')
    directories += sorted(source_directories)
    for directory in directories:
        result = read(directory / 'results.json')
        if result['status'] != 'completed':
            raise ValueError(f'Run is not completed: {directory}')
        config = result['config']
        wandb_file = directory / 'wandb_run.json'
        wandb = read(wandb_file) if wandb_file.exists() else None
        records.append({'variant': config['variant'], 'stage': config['stage'], 'config': config,
                        'artifact_directory': str(directory.relative_to(runs)), 'wandb': wandb,
                        **{k: result.get(k) for k in ('best_epoch', 'stopped_epoch', 'best_validation', 'test', 'efficiency', 'comparison')},
                        'history': read(directory / 'history.json')})
    return {'seed': seed, 'verification': 'Local completed runs; not independently verified against W&B.', 'runs': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'training/runs_comparison')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, help='Defaults to RUNS/analysis')
    args = parser.parse_args()
    runs = args.runs.resolve()
    snapshot = collect(runs, args.seed)
    output = args.output or runs / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'seed{args.seed}.json'
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    rows = [{**r['comparison'], 'epochs': r['stopped_epoch'], 'best_epoch': r['best_epoch']}
            for r in snapshot['runs'] if r['variant'] in VARIANTS]
    with (output / f'comparison_seed_{args.seed}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'snapshot': str(path), 'completed_variants': list(VARIANTS)}))


if __name__ == '__main__':
    main()
