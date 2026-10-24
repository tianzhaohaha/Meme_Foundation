"""Plot matched Binary/V4/V5/V6 features, colored by true seven-class or polarity labels."""
import argparse
import csv
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
EMOTIONS = ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad']
CAPTIONS = {'V1': 'Direct 25k (reference)', 'V2': 'Full fine-tuning', 'V3': 'Head only',
            'V4': 'Direct 5k', 'V5': 'Reset fusion / full tuning', 'V6': 'Tune fusion / frozen encoders',
            'Binary': 'Shared source (2-class head)'}


def load_files(paths):
    bundles = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            bundle = {key: archive[key].copy() for key in archive.files}
        bundle['meta'] = json.loads(str(bundle.pop('metadata')))
        if bundle['meta']['class_names'] != EMOTIONS:
            raise ValueError('Class order differs: ' + str(path))
        if bundle['meta']['sample_count'] != len(bundle['ids']):
            raise ValueError('Sample count differs from metadata')
        if len(set(bundle['ids'].tolist())) != len(bundle['ids']):
            raise ValueError('Duplicate sample IDs')
        expected_binary = np.isin(bundle['labels'], [2, 4, 5]).astype(np.int64)
        n = len(bundle['ids'])
        if bundle['labels'].shape != (n,) or not np.isin(bundle['labels'], np.arange(7)).all():
            raise ValueError('Plot labels must remain seven-class ground truth, even for Binary')
        if bundle['probabilities'].shape != (n, bundle['meta'].get('num_classes', 7)):
            raise ValueError('Probability columns do not match the actual classifier head')
        reproduction = bundle['meta'].get('validation_reproduction', {})
        if bundle['meta']['split'] == 'validation' and not (
            {'accuracy', 'macro_f1'} <= reproduction.keys() and all(x['matches'] for x in reproduction.values())
        ):
            raise ValueError('Validation reproduction must pass before plotting')
        if not np.array_equal(expected_binary, bundle['polarity']):
            raise ValueError('Incompatible polarity mapping')
        if bundles:
            for key in ('ids', 'labels', 'polarity', 'image_sha256'):
                if not np.array_equal(bundle[key], bundles[0][key]):
                    raise ValueError('Models must use exactly the same samples and order: ' + key)
            if bundle['meta']['split'] != bundles[0]['meta']['split']:
                raise ValueError('Cannot mix validation and test features')
            for key in ('input_signature', 'text_max_length', 'text_model_revision', 'dataset_id'):
                if bundle['meta'].get(key) != bundles[0]['meta'].get(key):
                    raise ValueError('Incompatible model inputs: ' + key)
        bundles.append(bundle)
    names = [bundle['meta']['model'] for bundle in bundles]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate model names')
    return bundles


def save_panel(bundles, projections, method, space, binary, output, seed):
    columns = min(2, len(bundles))
    rows = (len(bundles) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    labels = ['Negative', 'Positive'] if binary else EMOTIONS
    colors = ['#2878B5', '#E78326'] if binary else list(plt.get_cmap('tab10').colors[:7])
    order = np.random.default_rng(seed).permutation(len(bundles[0]['ids']))
    for ax, bundle, projection in zip(axes.flat, bundles, projections):
        coords, explained = projection
        y = bundle['polarity'] if binary else bundle['labels']
        # Same shuffled draw order for all models; avoid systematically drawing one class last.
        ax.scatter(coords[order, 0], coords[order, 1], s=8, alpha=.55,
                   c=[colors[i] for i in y[order]], linewidths=0, rasterized=True)
        name = bundle['meta']['model']
        ax.set_title(name + ': ' + CAPTIONS.get(name, name), fontsize=11)
        if method == 'PCA':
            ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)')
        else:
            ax.set_xlabel('t-SNE 1 (model-local coordinates)')
            ax.set_ylabel('t-SNE 2')
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(bundles):]:
        ax.set_visible(False)
    handles = [Line2D([], [], marker='o', linestyle='', color=color, label=label, markersize=6)
               for label, color in zip(labels, colors)]
    fig.legend(handles=handles, loc='lower center', ncol=len(labels), frameon=False)
    split = bundles[0]['meta']['split']
    coloring = 'true polarity labels' if binary else 'true seven-class labels'
    fig.suptitle(f'{method} | {space} features | same {len(bundles[0]["ids"]):,} {split} samples\n{coloring}; independently fitted model-local coordinates', fontsize=13)
    fig.tight_layout(rect=(0, .08, 1, .94))
    stem = f'{space}_{method.lower().replace("-", "")}_' + ('binary' if binary else 'seven')
    fig.savefig(output / (stem + '.png'), dpi=250, bbox_inches='tight')
    fig.savefig(output / (stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/plots')
    parser.add_argument('--spaces', nargs='+', choices=['fusion', 'encoded', 'image', 'text'], default=['fusion'])
    parser.add_argument('--normalization', choices=['l2', 'none'], default='l2')
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=1500)
    args = parser.parse_args(argv)
    threadpool_limits(limits=2)
    bundles = load_files(args.features)
    n = len(bundles[0]['ids'])
    if n < 4 or not 0 < args.perplexity < n:
        parser.error('Need at least 4 samples and perplexity below sample count')
    if args.iterations < 250:
        parser.error('--iterations must be at least 250')
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {'seed': args.seed, 'normalization': args.normalization,
        'perplexity': args.perplexity, 'iterations': args.iterations, 'sample_count': n,
        'split': bundles[0]['meta']['split'], 'spaces': args.spaces,
        'projection_policy': 'Fit independently per model; orientations and coordinates cannot be compared across panels.',
        'tsne_policy': 'L2 normalize (default), independent PCA to <=50 dimensions, then Euclidean t-SNE; same seed/settings.',
        'source_files': [p.name for p in args.features], 'models': [],
        'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'matplotlib': matplotlib.__version__,
        'cpu_threads': 2, 'display_shuffle_seed': args.seed,
        'coloring': 'Ground truth only; the Binary model retains its original two-class head.'}
    for space in args.spaces:
        projections = {'PCA': [], 't-SNE': []}
        for bundle in bundles:
            name = bundle['meta']['model']
            if space == 'fusion':
                x = bundle['fusion']
            else:
                if 'encoded' not in bundle:
                    raise ValueError('Re-export with --include-encoded to plot ' + space)
                x = bundle['encoded']
                if space == 'image':
                    x = x[:, :768]
                elif space == 'text':
                    x = x[:, 768:]
            if x.ndim != 2 or x.shape[0] != n or not np.isfinite(x).all():
                raise ValueError('Invalid features: ' + name + ' / ' + space)
            if args.normalization == 'l2':
                x = normalize(x, norm='l2')
            pca = PCA(n_components=2, svd_solver='full')
            pca_coords = pca.fit_transform(x)
            projections['PCA'].append((pca_coords, pca.explained_variance_ratio_))
            reduced = PCA(n_components=min(50, n - 1, x.shape[1]), svd_solver='randomized',
                          random_state=args.seed).fit_transform(x)
            kwargs = {'n_components': 2, 'perplexity': args.perplexity, 'init': 'pca',
                      'learning_rate': 'auto', 'random_state': args.seed, 'metric': 'euclidean', 'n_jobs': 2}
            iteration_key = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
            kwargs[iteration_key] = args.iterations
            print('Projecting ' + name + ' / ' + space, flush=True)
            tsne = TSNE(**kwargs)
            tsne_coords = tsne.fit_transform(reduced)
            projections['t-SNE'].append((tsne_coords, None))
            with (args.output / f'{name}_{space}_coordinates.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'emotion', 'polarity', 'pc1', 'pc2', 'tsne1', 'tsne2'])
                for index, sample_id in enumerate(bundle['ids']):
                    writer.writerow([sample_id, EMOTIONS[bundle['labels'][index]],
                        int(bundle['polarity'][index]), *pca_coords[index], *tsne_coords[index]])
            settings['models'].append({'model': name, 'space': space,
                'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'tsne_kl_divergence': float(tsne.kl_divergence_), 'checkpoint': bundle['meta']})
        for method in projections:
            for binary in (False, True):
                save_panel(bundles, projections[method], method, space, binary, args.output, args.seed)
    (args.output / 'projection_settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    paragraphs = ['# Embedding 可视化说明', '',
        f'使用相同的 {n} 条 {settings["split"]} 样本，所有模型严格核对 ID、顺序、标签和图片哈希。', '',
        '融合特征取分类头之前的 256 维表示；encoded 是图像 768 维和文本 768 维的拼接。',
        'PCA/t-SNE 都对每个模型独立拟合。跨图的方向、坐标和簇间距离不代表模型之间的直接距离。',
        '默认先对每条特征做 L2 归一化；t-SNE 先通过 PCA 降到最多 50 维。',
        '七分类图和二分类图使用完全相同的投影坐标，仅颜色不同。', '',
        'Binary保留原二分类头，其七类着色来自真实标签，不代表模型输出七分类预测。', '',
        '重点观察：七类是否混合；混合是否主要发生在同一极性内；V6 的融合表示是否保留类别结构。',
        '二维投影仅用于描述，不能单凭簇的形状证明信息丢失、负迁移或模型性能优劣。',
        '本分析不训练模型、不更换 checkpoint，也不用测试集选择可视化参数。', '']
    for space in args.spaces:
        for method in ('pca', 'tsne'):
            paragraphs.extend([f'![{space} {method}]({space}_{method}_seven.png)', ''])
    (args.output / 'README.md').write_text('\n'.join(paragraphs), encoding='utf-8')
    print('Plots saved to ' + str(args.output))


if __name__ == '__main__':
    main()
