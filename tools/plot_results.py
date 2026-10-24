"""Render shareable figures from the curated, W&B-verified experiment snapshot."""
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
COLORS = {'direct': '#2463a8', 'full': '#d67522', 'frozen': '#168579', 'direct_5k': '#8c4fa3'}
NAMES = {'direct': 'V1 Direct 25k', 'full': 'V2 Full fine-tune', 'frozen': 'V3 Frozen',
         'direct_5k': 'V4 Direct 5k'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'reports/data/seed42.json')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/figures')
    args = p.parse_args()
    runs = {r['variant']: r for r in json.loads(args.input.read_text())['runs']}
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    variants = [v for v in NAMES if v in runs]
    short_names = [NAMES[v].split()[0] for v in variants]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout='constrained')
    x = np.arange(len(variants))
    accuracy = [100 * runs[v]['test']['accuracy'] for v in variants]
    f1 = [100 * runs[v]['test']['macro_f1'] for v in variants]
    axes[0, 0].bar(x - .18, accuracy, .36, label='Test accuracy', color='#2463a8')
    axes[0, 0].bar(x + .18, f1, .36, label='Test macro-F1', color='#a6bddb')
    for i, (a, f) in enumerate(zip(accuracy, f1)):
        axes[0, 0].text(i-.18, a+1, f'{a:.2f}', ha='center', fontsize=9)
        axes[0, 0].text(i+.18, f+1, f'{f:.2f}', ha='center', fontsize=9)
    axes[0, 0].set(xticks=x, xticklabels=short_names, ylim=(0, max(70, max(accuracy + f1)+8)), ylabel='Percent', title='Final quality on the same 2,000 test samples')
    axes[0, 0].legend(frameon=False)
    binary_minutes = runs['shared_binary']['efficiency']['stage_wall_seconds'] / 60
    stage_minutes = [runs[v]['efficiency']['stage_wall_seconds']/60 for v in variants]
    bases = [binary_minutes if v in ('full', 'frozen') else 0 for v in variants]
    axes[0, 1].bar(x, bases, color='#bdc5cc', label='Binary pretraining cost')
    axes[0, 1].bar(x, stage_minutes, bottom=bases, color=[COLORS[v] for v in variants], label='Seven-class stage')
    for i, (b, s) in enumerate(zip(bases, stage_minutes)):
        axes[0, 1].text(i, b+s+.4, f'{b+s:.2f} min', ha='center', fontsize=9)
    axes[0, 1].set(xticks=x, xticklabels=short_names, ylim=(0, max(28, max(b+s for b,s in zip(bases,stage_minutes))+4)), ylabel='Minutes', title='Full method cost (shared pretraining charged to both)')
    axes[0, 1].legend(frameon=False, fontsize=9)
    for v in variants:
        h = runs[v]['history']
        axes[1, 0].plot([z['epoch'] for z in h], [100*z['val_accuracy'] for z in h], marker='o', markersize=3, color=COLORS[v], label=NAMES[v])
        best = next(z for z in h if z['epoch'] == runs[v]['best_epoch'])
        axes[1, 0].scatter(best['epoch'], 100*best['val_accuracy'], s=85, facecolors='none', edgecolors=COLORS[v], linewidths=1.5)
    axes[1, 0].set(xlabel='Seven-class epoch', ylabel='Validation accuracy (%)', title='Convergence (rings mark selected checkpoints)')
    axes[1, 0].legend(frameon=False, fontsize=9)
    for v in (v for v in variants if v != 'direct'):
        h = runs[v]['history']
        axes[1, 1].plot([z['cumulative_train_seconds'] for z in h], [100*z['val_accuracy'] for z in h], marker='o', markersize=3, color=COLORS[v], label=NAMES[v])
    axes[1, 1].set(xlabel='Seven-class training-loop seconds (excludes pretraining)', ylabel='Validation accuracy (%)', title='Same 5,000 training samples: quality versus time')
    axes[1, 1].legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('TumEmo | ViT + RoBERTa | seed 42 | NVIDIA H200 NVL', fontsize=14)
    fig.savefig(args.output / 'comparison.png', dpi=180)
    fig.savefig(args.output / 'comparison.pdf')
    plt.close(fig)
    labels = ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad']
    columns = 2 if len(variants) == 4 else len(variants)
    rows = (len(variants) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5*columns, 4.6*rows), layout='constrained', squeeze=False)
    for ax, variant in zip(axes.flat, variants):
        cm = np.array(runs[variant]['test']['confusion_matrix'])
        normalized = cm / cm.sum(axis=1, keepdims=True)
        artist = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if normalized[i, j] > .5 else '#222222')
        ax.set(xticks=range(7), xticklabels=labels, yticks=range(7), yticklabels=labels,
               title=NAMES[variant], xlabel='Predicted class', ylabel='True class')
        ax.tick_params(axis='x', rotation=45)
    fig.colorbar(artist, ax=axes, shrink=.7, label='Fraction within each true class')
    fig.suptitle('Test confusion matrices | annotations are sample counts', fontsize=14)
    fig.savefig(args.output / 'confusion_matrices.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
