"""V4/V5/V6 quality and downstream-only adaptation cost, from verified aggregate runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from report_visuals import ROOT, plt, save, style, write_csv, write_json
from matplotlib.lines import Line2D

VARIANTS = {'direct_5k': 'V4', 'reset_fusion': 'V5', 'tune_fusion': 'V6'}
COLORS = ['#7c858f', '#4677b5', '#168577']
NAMES = ['V4\nDirect 5k', 'V5\nFoundation + full tuning', 'V6\nFoundation + fusion/head']
SCOPE = '5k downstream adaptation only | Foundation pretraining excluded (model already available)'
COMMON_KEYS = ['architecture', 'num_classes', 'seed', 'split_seed', 'batch_size', 'epochs',
               'patience', 'encoder_lr', 'head_lr', 'dropout', 'weight_decay', 'warmup_ratio',
               'max_length', 'selection_metric', 'sample_counts', 'total_parameters',
               'gpu_name', 'torch_version', 'cuda_version', 'dataset_id']


def summarize(snapshot):
    selected = []
    for variant in VARIANTS:
        found = [r for r in snapshot['runs'] if r['variant'] == variant]
        if len(found) != 1:
            raise ValueError('Expected one verified run for ' + variant)
        selected.append(found[0])
    baseline = selected[0]
    if baseline['config']['sample_counts'] != {'train': 5000, 'validation': 2000, 'test': 2000}:
        raise ValueError('Expected the fixed 5k/2k/2k comparison')
    for run in selected:
        if any(run['config'][key] != baseline['config'][key] for key in COMMON_KEYS):
            raise ValueError('Runs differ in common data, training, or hardware settings')
        if run['stage'] != ('direct' if run['variant'] == 'direct_5k' else 'stage2'):
            raise ValueError('Expected a downstream seven-class stage')
        if not run['efficiency']['timing_complete'] or run['wandb']['state'] != 'finished':
            raise ValueError('Only completed runs with complete timing can be compared')
        if not run['verified_wandb_epoch_history']:
            raise ValueError('Unverified epoch history')
    source_hash = selected[1]['config']['init_checkpoint_sha256']
    if not source_hash or selected[2]['config']['init_checkpoint_sha256'] != source_hash:
        raise ValueError('V5 and V6 must reuse the same source checkpoint')
    if baseline['config']['init_checkpoint_sha256'] is not None:
        raise ValueError('V4 must not inherit the domain source checkpoint')
    if not selected[1]['config']['reset_fusion'] or selected[1]['config']['freeze_encoders']:
        raise ValueError('V5 must reset fusion and train both encoders')
    if selected[2]['config']['reset_fusion'] or not selected[2]['config']['freeze_encoders']:
        raise ValueError('V6 must retain fusion and freeze both encoders')
    base = baseline['efficiency']
    rows = []
    histories = {}
    for run in selected:
        name = VARIANTS[run['variant']]
        eff, config, history = run['efficiency'], run['config'], run['history']
        if [r['epoch'] for r in history] != list(range(1, eff['epochs'] + 1)):
            raise ValueError('Epoch history is incomplete')
        best = next(r for r in history if r['epoch'] == run['best_epoch'])
        if abs(best['val_accuracy'] - run['best_validation']['accuracy']) > 1e-9:
            raise ValueError('Selected validation score differs from epoch history')
        if run['best_epoch'] != max(history, key=lambda h: h['val_accuracy'])['epoch']:
            raise ValueError('Checkpoint is not the first best validation accuracy')
        if eff['samples_seen'] != 5000 * eff['epochs']:
            raise ValueError('Unexpected downstream sample exposures')
        if eff['optimizer_steps'] != math.ceil(5000/config['batch_size']) * eff['epochs']:
            raise ValueError('Unexpected number of downstream optimizer steps')
        if not math.isclose(sum(h['train_seconds'] for h in history), eff['train_seconds'], abs_tol=1e-6):
            raise ValueError('Training-loop time does not match complete history')
        components = {key: float(eff[key]) for key in ['train_seconds', 'validation_seconds',
                      'checkpoint_seconds', 'setup_seconds', 'test_seconds']}
        residual = eff['stage_wall_seconds'] - sum(components.values())
        if residual < -1e-6 or not all(math.isfinite(v) and v >= 0 for v in components.values()):
            raise ValueError('Invalid downstream timing components')
        wall = eff['stage_wall_seconds']  # Deliberately never use comparison/total_* costs.
        row = {'model': name, 'variant': run['variant'], 'validation_accuracy': run['best_validation']['accuracy'],
               'validation_macro_f1': run['best_validation']['macro_f1'],
               'test_accuracy': run['test']['accuracy'], 'test_macro_f1': run['test']['macro_f1'],
               'downstream_stage_seconds': wall, 'downstream_stage_minutes': wall / 60,
               **components, 'other_stage_seconds': max(0., residual),
               'peak_allocated_gpu_gib': eff['peak_allocated_gpu_gib'],
               'peak_reserved_gpu_gib': eff['peak_reserved_gpu_gib'],
               'trainable_parameters': config['trainable_parameters'], 'total_parameters': config['total_parameters'],
               'trainable_percent': config['trainable_parameters'] / config['total_parameters'] * 100,
               'epochs': eff['epochs'], 'best_epoch': run['best_epoch'],
               'optimizer_steps': eff['optimizer_steps'], 'samples_seen': eff['samples_seen'],
               'train_seconds_per_epoch': eff['train_seconds'] / eff['epochs'],
               'milliseconds_per_step': eff['seconds_per_step'] * 1000,
               'samples_per_second': eff['samples_per_second'],
               'selected_epoch_stage_seconds_retrospective': best['cumulative_stage_seconds'],
               'stage_speedup_vs_v4': base['stage_wall_seconds'] / wall,
               'stage_time_reduction_vs_v4_pct': (1 - wall / base['stage_wall_seconds']) * 100,
               'allocated_memory_reduction_vs_v4_pct': (1 - eff['peak_allocated_gpu_gib'] / base['peak_allocated_gpu_gib']) * 100,
               'throughput_ratio_vs_v4': eff['samples_per_second'] / base['samples_per_second'],
               'validation_accuracy_delta_vs_v4_pp': (run['best_validation']['accuracy'] - baseline['best_validation']['accuracy']) * 100,
               'test_accuracy_delta_vs_v4_pp': (run['test']['accuracy'] - baseline['test']['accuracy']) * 100,
               'wandb_url': run['wandb']['url']}
        rows.append(row)
        histories[name] = [{key: h[key] for key in ['epoch', 'val_accuracy', 'val_macro_f1',
                             'cumulative_train_seconds', 'cumulative_stage_seconds']} for h in history]
    return {'scope': SCOPE, 'source_pretraining_included': False,
            'primary_time_field': 'efficiency.stage_wall_seconds of the selected seven-class run',
            'stage_time_definition': 'Completed downstream stage: setup, training loop, validation, checkpoint writes, final test, residual overhead; includes early-stopping patience epochs.',
            'memory_definition': 'Recorded peak PyTorch allocated GPU memory in the downstream train/validation epochs; not total device memory or inference memory.',
            'quality_definition': 'Validation-selected best.pt; its validation and final test scores are kept separate, not the last epoch scores.',
            'common_config': {k: baseline['config'][k] for k in COMMON_KEYS},
            'source_checkpoint_sha256': source_hash, 'runs': rows, 'histories': histories,
            'checks': {'common_inputs_and_settings': True, 'same_v5_v6_source': True,
                       'completed_downstream_timing': True, 'selected_metrics_match_history': True}}


def bar(ax, values, title, ylabel, fmt, *, percent=False):
    artists = ax.bar(range(3), values, color=COLORS, width=.58)
    ax.bar_label(artists, labels=[fmt.format(v) for v in values], padding=5, fontsize=11)
    ax.set(xticks=range(3), xticklabels=NAMES, ylabel=ylabel, title=title,
           ylim=(0, 70 if percent else max(values) * 1.23))
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)


def main_results(summary, output):
    rows = summary['runs']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    bar(axes[0, 0], [r['test_accuracy'] * 100 for r in rows], 'Final test accuracy | same 2,000 test samples',
        'Accuracy (%)', '{:.2f}%', percent=True)
    bar(axes[0, 1], [r['validation_accuracy'] * 100 for r in rows], 'Validation accuracy | checkpoint selection set',
        'Accuracy (%)', '{:.2f}%', percent=True)
    bar(axes[1, 0], [r['downstream_stage_minutes'] for r in rows], 'Completed downstream stage | includes early stopping',
        'Minutes', '{:.2f} min')
    bar(axes[1, 1], [r['peak_allocated_gpu_gib'] for r in rows], 'Downstream peak GPU memory | allocated',
        'GiB', '{:.3f} GiB')
    for i, row in enumerate(rows[1:], 1):
        axes[1, 0].text(i, .25, f"{row['stage_time_reduction_vs_v4_pct']:.1f}% less time", ha='center', color='white', fontsize=10)
    fig.suptitle('Main experiment | Same 5,000 target samples | ViT + RoBERTa\n' + SCOPE, fontsize=12)
    save(fig, output, 'main_results')


def tradeoff(summary, output):
    rows = summary['runs']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout='constrained')
    for ax, field, title in [(axes[0], 'test_accuracy', 'Final test accuracy vs. adaptation time'),
                              (axes[1], 'validation_accuracy', 'Validation accuracy vs. adaptation time')]:
        for row, color, offset in zip(rows, COLORS, [(0, 12), (0, -34), (0, 12)]):
            x, y = row['downstream_stage_minutes'], row[field] * 100
            ax.scatter(x, y, color=color, s=150, edgecolors='white', linewidths=1, zorder=3)
            ax.annotate(f"{row['model']}  {y:.2f}%\n{x:.2f} min", (x, y), xytext=offset,
                        textcoords='offset points', ha='center', fontsize=10)
        ax.set(xlim=(0, 10), ylim=(50, 60), xlabel='Completed downstream stage (minutes)',
               ylabel='Accuracy (%) | zoomed 50–60% axis', title=title)
        ax.grid(alpha=.18)
    fig.suptitle('Quality–cost trade-off | lower time, higher accuracy are preferred\n' + SCOPE, fontsize=12)
    save(fig, output, 'quality_cost_tradeoff')


def efficiency(summary, output):
    rows = summary['runs']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    bar(axes[0, 0], [r['milliseconds_per_step'] for r in rows], 'Training-loop cost per optimizer step',
        'Milliseconds / step', '{:.2f} ms')
    bar(axes[0, 1], [r['samples_per_second'] for r in rows], 'Training throughput | same batch 16, FP32',
        'Samples / second', '{:.2f}')
    values = [r['trainable_parameters'] for r in rows]
    axes[1, 0].scatter(range(3), values, s=100, c=COLORS, zorder=3)
    for i, value in enumerate(values):
        axes[1, 0].annotate(f'{value:,}', (i, value), xytext=(0, 12), textcoords='offset points', ha='center')
    axes[1, 0].set(yscale='log', ylim=(1e5, 1e9), xlim=(-.5, 2.5), xticks=range(3), xticklabels=NAMES,
                   ylabel='Trainable parameters (log scale)', title='Parameters updated | total size unchanged')
    axes[1, 0].grid(axis='y', alpha=.16)
    left = np.zeros(3)
    parts = [('train_seconds', 'Training loop', '#4577b2'),
             ('validation_seconds', 'Validation', '#b5cce2'),
             ('checkpoint_seconds', 'Checkpoint writes', '#f1b86d')]
    for key, name, color in parts:
        values = np.array([r[key] for r in rows])
        axes[1, 1].barh(range(3), values, left=left, label=name, color=color)
        left += values
    overhead = np.array([r['setup_seconds'] + r['test_seconds'] + r['other_stage_seconds'] for r in rows])
    axes[1, 1].barh(range(3), overhead, left=left, color='#b9bec2', label='Setup + final test + other')
    for i, r in enumerate(rows):
        axes[1, 1].text(r['downstream_stage_seconds'] + 5, i, f"{r['downstream_stage_seconds']:.2f}s", va='center', fontsize=9)
    axes[1, 1].set(yticks=range(3), yticklabels=[r['model'] for r in rows],
                   xlabel='Downstream stage seconds', title='Completed stage cost breakdown',
                   xlim=(0, max(r['downstream_stage_seconds'] for r in rows) * 1.23))
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2)
    fig.suptitle('Where the adaptation savings come from | ' + SCOPE, fontsize=11)
    save(fig, output, 'efficiency_breakdown')


def convergence(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    for row, color in zip(summary['runs'], COLORS):
        history = summary['histories'][row['model']]
        best = next(h for h in history if h['epoch'] == row['best_epoch'])
        for ax, key, divisor in [(axes[0], 'cumulative_stage_seconds', 60), (axes[1], 'epoch', 1)]:
            ax.plot([h[key] / divisor for h in history], [h['val_accuracy'] * 100 for h in history],
                    marker='o', markersize=4, color=color, label=row['model'])
            ax.scatter(best[key] / divisor, best['val_accuracy'] * 100, s=150,
                       facecolors='none', edgecolors=color, linewidths=2, zorder=4)
        axes[0].annotate(f"{row['model']} best: epoch {row['best_epoch']}",
                         (best['cumulative_stage_seconds']/60, best['val_accuracy']*100),
                         xytext=(8, 10 if row['model'] != 'V5' else -34),
                         textcoords='offset points', color=color, fontsize=9)
    axes[0].set(xlabel='Elapsed downstream stage at epoch end (minutes)', title='Convergence in measured stage time', xlim=(0, 9))
    axes[1].set(xlabel='Downstream epoch', title='Convergence with equal data passes', xticks=range(1, 13), xlim=(.5, 12.5))
    for ax in axes:
        ax.set(ylabel='Validation accuracy (%) | zoomed axis', ylim=(30, 60))
        ax.grid(alpha=.18)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], marker='o', color='black', markerfacecolor='none', linestyle='', markersize=9))
        ax.legend(handles, labels + ['Selected checkpoint (retrospective)'], fontsize=8, loc='lower right')
    fig.suptitle('Full observed curves through early stopping | seed 42\n' + SCOPE + '\nEpoch-end time includes setup/train/validation/checkpoints; final test follows the curves.', fontsize=11)
    save(fig, output, 'convergence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'reports/data/seed42.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports')
    args = parser.parse_args()
    source = args.input.read_bytes()
    summary = summarize(json.loads(source))
    summary['source_file'] = args.input.name
    summary['source_sha256'] = hashlib.sha256(source).hexdigest()
    style()
    output = args.output / 'figures/main_experiment'
    for render in [main_results, tradeoff, efficiency, convergence]:
        render(summary, output)
    write_json(args.output / 'data/main_experiment.json', summary)
    write_csv(args.output / 'data/main_experiment.csv', summary['runs'])
    print(json.dumps([{k: r[k] for k in ['model', 'test_accuracy', 'validation_accuracy',
          'downstream_stage_seconds', 'stage_speedup_vs_v4', 'stage_time_reduction_vs_v4_pct',
          'allocated_memory_reduction_vs_v4_pct', 'throughput_ratio_vs_v4', 'trainable_percent']}
          for r in summary['runs']], indent=2))


if __name__ == '__main__':
    main()
