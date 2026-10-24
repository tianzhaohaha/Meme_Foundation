"""Plot validation comparisons without model weights, raw samples, W&B or a GPU."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tumemo-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'reports/data/vlm_validation.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/figures')
    args = parser.parse_args()
    snapshot = json.loads(args.input.read_text())
    runs = snapshot['runs']
    vlms = [r for r in runs if r['kind'] == 'zero_shot']
    names = [r['name'].replace('Qwen3-VL ', 'Qwen\n') for r in runs]
    labels = snapshot['labels']
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    x = np.arange(len(runs))
    for shift, field, title, color in [(-.18, 'accuracy', 'Accuracy', '#2463a8'), (.18, 'macro_f1', 'Macro-F1', '#9cb8d5')]:
        values = [100 * r['metrics'][field] for r in runs]
        axes[0, 0].bar(x + shift, values, width=.36, label=title, color=color)
        for i, value in enumerate(values):
            axes[0, 0].text(i + shift, value + .6, f'{value:.1f}', ha='center', fontsize=8)
    axes[0, 0].axvline(5.5, color='#777777', linestyle='--', linewidth=1)
    axes[0, 0].set(xticks=x, xticklabels=names, ylim=(0, 70), ylabel='Percent',
                   title='Validation quality | V1–V6 selected on validation; Qwen zero-shot')
    axes[0, 0].legend(frameon=False)
    recall = np.array([[r['metrics']['report'][name]['recall'] for name in labels] for r in runs])
    artist = axes[0, 1].imshow(recall, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for row in range(len(runs)):
        for col in range(7):
            axes[0, 1].text(col, row, f'{100*recall[row, col]:.0f}', ha='center', va='center',
                            fontsize=8, color='white' if recall[row, col] > .55 else '#222222')
    axes[0, 1].set(xticks=range(7), xticklabels=labels, yticks=range(len(runs)), yticklabels=names,
                   title='Validation recall by class (%)')
    fig.colorbar(artist, ax=axes[0, 1], shrink=.7)
    for ax, field, title, ylabel in [(axes[1, 0], 'evaluation_seconds', 'Qwen inference over 2,000 validation samples', 'Seconds'),
                                    (axes[1, 1], 'peak_allocated_gpu_gib', 'Qwen peak allocated GPU memory', 'GiB')]:
        values = [r['efficiency'][field] for r in vlms]
        ax.bar([r['name'] for r in vlms], values, color=['#168579', '#8c4fa3'])
        for i, value in enumerate(values):
            ax.text(i, value + max(values)*.02, f'{value:.2f}', ha='center')
        ax.set(ylim=(0, max(values)*1.2), ylabel=ylabel, title=title)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    config = vlms[0]['config']
    fig.suptitle(f"TumEmo | same {snapshot['samples']:,} validation samples | 224×224 images\n"
                 f"Qwen: BF16, batch {config['batch_size']}, {config['gpu_name']} | no comparison of training time with inference time", fontsize=12)
    fig.savefig(args.output / 'vlm_validation.png', dpi=180)
    fig.savefig(args.output / 'vlm_validation.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.7), layout='constrained')
    for ax, run in zip(axes, vlms):
        cm = np.array(run['metrics']['confusion_matrix'])
        normalized = cm / cm.sum(axis=1, keepdims=True)
        artist = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        for i in range(7):
            for j in range(8):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                        color='white' if normalized[i, j] > .55 else '#222222')
        ax.set(xticks=range(8), xticklabels=labels + ['Invalid'], yticks=range(7), yticklabels=labels,
               xlabel='Predicted label', ylabel='True label', title=run['name'])
        ax.tick_params(axis='x', rotation=45)
    fig.colorbar(artist, ax=axes, shrink=.7, label='Fraction of true class')
    fig.suptitle('Zero-shot Qwen | validation confusion matrices | counts include invalid outputs')
    fig.savefig(args.output / 'vlm_confusion_matrices.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
