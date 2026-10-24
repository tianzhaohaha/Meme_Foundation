"""Audit actual inputs and render label, sample, and input-quality visualizations."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from report_visuals import (ROOT, LABELS, SPLITS, SPLIT_NAMES, COLORS, plt, style, save,
                            write_json, write_csv, stable_order, raw_image,
                            raw_text, prepared_image, excerpt, TOKENIZER, REVISION,
                            load_tokenizer, retained_prefix)
from verify_data import verify

sys.path.insert(0, str(ROOT / 'training'))
from experiment_data import clean_text

def describe(rows):
    def quantiles(key):
        return dict(zip(['min', 'p25', 'median', 'p75', 'p95', 'max'],
                        map(float, np.percentile([r[key] for r in rows], [0, 25, 50, 75, 95, 100]))))
    return {'n': len(rows), 'empty_cleaned': sum(r['empty_cleaned'] for r in rows),
            'truncated': sum(r['truncated'] for r in rows),
            'smaller_than_224_long_side': sum(max(r['width'], r['height']) < 224 for r in rows),
            'extreme_aspect_ratio': sum(r['aspect_ratio'] < .5 or r['aspect_ratio'] > 2 for r in rows),
            'clean_tokens': quantiles('clean_tokens'), 'raw_tokens': quantiles('raw_tokens'),
            'width': quantiles('width'), 'height': quantiles('height'), 'aspect_ratio': quantiles('aspect_ratio')}


def collect(records, archive, tokenizer):
    rows = []
    for start in range(0, len(records), 256):
        chunk = records[start:start + 256]
        original = [raw_text(archive, r['id']) for r in chunk]
        if [clean_text(t) for t in original] != [r['text'] for r in chunk]:
            raise ValueError('Raw text differs from the prepared manifest')
        raw_lengths = [len(t) for t in tokenizer(original, truncation=False, padding=False)['input_ids']]
        clean_lengths = [len(t) for t in tokenizer([r['text'] for r in chunk], truncation=False, padding=False)['input_ids']]
        for record, raw_len, clean_len in zip(chunk, raw_lengths, clean_lengths):
            data = archive.read(f"subset_50000/all_data/{record['id']}.jpg")
            if hashlib.sha256(data).hexdigest() != record['image_sha256']:
                raise ValueError('Image hash differs: ' + record['id'])
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if image.getexif().get(274) in (5, 6, 7, 8):
                    width, height = height, width
            rows.append({'id': record['id'], 'split': record['split'], 'emotion': record['emotion'],
                         'width': width, 'height': height, 'aspect_ratio': width / height,
                         'raw_tokens': raw_len, 'clean_tokens': clean_len,
                         'empty_cleaned': int(not record['text']), 'truncated': int(clean_len > 96)})
        if start % 5120 == 0:
            print(f'Audited {len(rows):,}/{len(records):,} inputs', flush=True)
    return rows


def distribution_plots(rows, output):
    counts = np.array([[sum(r['split'] == s and r['emotion'] == label for r in rows)
                        for label in LABELS] for s in SPLITS])
    combined = np.vstack([counts, counts[0] + counts[1]])
    names = SPLIT_NAMES + ['V1 combined 25k']
    fractions = combined / combined.sum(1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), layout='constrained',
                             gridspec_kw={'width_ratios': [1.1, 1.3, .8]})
    for i, name in enumerate(SPLIT_NAMES):
        axes[0].bar(np.arange(7) + (i - 1.5) * .19, counts[i], .19, label=name)
    axes[0].set(xticks=range(7), xticklabels=LABELS, ylabel='Records', title='Class counts')
    axes[0].tick_params(axis='x', rotation=40)
    axes[0].legend(fontsize=8)
    left = np.zeros(5)
    for j, label in enumerate(LABELS):
        axes[1].barh(names, fractions[:, j], left=left, color=COLORS[j], label=label)
        for i in range(5):
            axes[1].text(left[i] + fractions[i, j] / 2, i, f'{fractions[i, j]:.1f}',
                         ha='center', va='center', fontsize=7, color='white' if j < 6 else 'black')
        left += fractions[:, j]
    axes[1].set(xlim=(0, 100), xlabel='Percent', title='Seven-class proportions (%)')
    axes[1].invert_yaxis()
    axes[1].legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.15))
    positive = fractions[:, [2, 4, 5]].sum(1)
    axes[2].barh(names, 100 - positive, label='Negative', color='#2878b5')
    axes[2].barh(names, positive, left=100 - positive, label='Positive', color='#e78326')
    for i, value in enumerate(positive):
        axes[2].text((100-value)/2, i, f'{100-value:.1f}%', ha='center', va='center', color='white')
        axes[2].text(100-value/2, i, f'{value:.1f}%', ha='center', va='center')
    axes[2].set(xlim=(0, 100), xlabel='Percent', title='Binary polarity')
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.15))
    fig.suptitle('TumEmo | Fixed experimental lists (29,000 records; training duplicates retained)')
    save(fig, output, 'label_distribution')
    return counts


def quality_plots(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    groups = [[r for r in rows if r['split'] == s] for s in SPLITS]
    for group, name in zip(groups, SPLIT_NAMES):
        for ax, key in [(axes[0, 0], 'clean_tokens'), (axes[1, 0], 'aspect_ratio')]:
            values = np.sort([r[key] for r in group])
            ax.plot(values, np.arange(1, len(values) + 1) / len(values) * 100, label=name)
    axes[0, 0].axvline(96, color='black', ls='--', label='96-token limit')
    axes[0, 0].set(xscale='log', xlabel='Cleaned RoBERTa tokens (including special tokens)',
                   ylabel='Cumulative percent', title='Text length: actual tokenizer')
    axes[0, 0].legend(fontsize=8)
    x = np.arange(4)
    for offset, key, label, color in [(-.18, 'empty_cleaned', 'Empty cleaned text', '#b64b4b'),
                                     (.18, 'truncated', '>96 tokens / truncated', '#376da3')]:
        values = [sum(r[key] for r in group) / len(group) * 100 for group in groups]
        bars = axes[0, 1].bar(x + offset, values, .36, label=label, color=color)
        axes[0, 1].bar_label(bars, fmt='%.2f%%', fontsize=8, padding=3)
    axes[0, 1].set(xticks=x, xticklabels=SPLIT_NAMES, ylabel='Percent of split',
                   title='Empty inputs and truncation', ylim=(0, max(values) * 1.25))
    axes[0, 1].legend(fontsize=8)
    for value in [.5, 1, 2]:
        axes[1, 0].axvline(value, ls='--', alpha=.3, color='black')
    axes[1, 0].set(xscale='log', xlabel='Original width / height (EXIF-oriented)',
                   ylabel='Cumulative percent', title='Original image aspect ratios')
    axes[1, 0].legend(fontsize=8)
    art = axes[1, 1].hexbin([r['width'] for r in rows], [r['height'] for r in rows],
                            gridsize=35, xscale='log', yscale='log', bins='log', mincnt=1, cmap='Blues')
    axes[1, 1].axvline(224, color='black', ls='--', alpha=.5)
    axes[1, 1].axhline(224, color='black', ls='--', alpha=.5)
    axes[1, 1].set(xlabel='Original width (px)', ylabel='Original height (px)',
                   title='All 29,000 image records; dashed = 224 px')
    fig.colorbar(art, ax=axes[1, 1], label='Record count (log color scale)')
    save(fig, output, 'input_quality')
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout='constrained')
    for ax, key, title in zip(axes, ['truncated', 'empty_cleaned'],
                              ['Text truncation by class (%)', 'Empty cleaned text by class (%)']):
        values = np.array([[np.mean([r[key] for r in group if r['emotion'] == label]) * 100
                            for label in LABELS] for group in groups])
        art = ax.imshow(values, cmap='YlOrRd', vmin=0, vmax=max(1, values.max()))
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f'{values[i,j]:.1f}', ha='center', va='center',
                    color='white' if values[i,j] > values.max() * .6 else 'black')
        ax.set(xticks=range(7), xticklabels=LABELS, yticks=range(4), yticklabels=SPLIT_NAMES, title=title)
        fig.colorbar(art, ax=ax, shrink=.65, label='Percent within split and true class')
    save(fig, output, 'input_quality_by_label')


def sample_plots(records, stats, archive, tokenizer, output):
    index = {r['id']: r for r in records}
    numeric = {r['id']: r for r in stats}
    ids = [sid for label in LABELS for sid in stable_order(
        [r['id'] for r in records if r['split'] == 'target' and r['emotion'] == label])[:2]]
    fig, axes = plt.subplots(2, 7, figsize=(17, 6), layout='constrained')
    for j, label in enumerate(LABELS):
        for i in range(2):
            sid = ids[2 * j + i]
            axes[i, j].imshow(raw_image(archive, sid))
            axes[i, j].set_title(f'{label} | ID {sid}', fontsize=10)
            axes[i, j].axis('off')
    fig.suptitle('Target 5k | Two fixed random examples per class | SHA-256 ordering, seed 42')
    save(fig, output, 'sample_gallery')
    table = ['| 标签 | ID | 原始文本节选 | 清洗后文本节选 | 清洗后token数 |',
             '|---|---|---|---|---:|']
    for sid in ids:
        table.append(f"| {index[sid]['emotion']} | {sid} | {excerpt(raw_text(archive, sid))} | {excerpt(index[sid]['text'])} | {numeric[sid]['clean_tokens']} |")
    target = [r for r in stats if r['split'] == 'target']
    cases = [('Empty after cleaning', stable_order([r['id'] for r in target if r['empty_cleaned']])[0]),
             ('Text truncated at 96 tokens', stable_order([r['id'] for r in target if r['truncated']])[0]),
             ('Most extreme aspect ratio', max(target, key=lambda r: (max(r['aspect_ratio'], 1/r['aspect_ratio']), r['id']))['id'])]
    fig, axes = plt.subplots(3, 2, figsize=(9, 11), layout='constrained')
    details = ['| 示例 | ID / 标签 | 原始文本节选 | 模型保留的清洗文本前缀 | token数（截断前→输入） |',
               '|---|---|---|---|---:|']
    for i, (reason, sid) in enumerate(cases):
        for ax, image, title in [(axes[i, 0], raw_image(archive, sid), 'Original'),
                                 (axes[i, 1], prepared_image(archive, sid), '224 x 224 / JPEG q95')]:
            ax.imshow(image)
            ax.set_title(f'{reason}\n{title} | ID {sid}', fontsize=10)
            ax.axis('off')
        prefix, length = retained_prefix(index[sid]['text'], tokenizer)
        details.append(f"| {reason} | {sid} / {index[sid]['emotion']} | {excerpt(raw_text(archive, sid), 240)} | {excerpt(prefix, 320)} | {numeric[sid]['clean_tokens']}→{length} |")
    for sid in set(ids) | {sid for _, sid in cases}:
        with Image.open(index[sid]['image']) as cached:
            if not np.array_equal(np.array(cached), np.array(prepared_image(archive, sid))):
                raise ValueError('Displayed preprocessing differs from cached training image: ' + sid)
    save(fig, output, 'preprocessing_examples')
    return {'gallery_ids': ids, 'diagnostic_cases': [{'reason': reason, 'id': sid} for reason, sid in cases],
            'gallery_table': '\n'.join(table), 'preprocessing_table': '\n'.join(details)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports')
    args = parser.parse_args()
    reference = json.loads((ROOT / 'data/dataset_signature.json').read_text())
    verification = verify(args.cache, reference, ROOT / 'data/manifests', args.data)
    records = json.loads((args.cache / 'experiment_manifest.json').read_text())['records']
    tokenizer = load_tokenizer()
    style()
    with zipfile.ZipFile(args.data) as archive:
        rows = collect(records, archive, tokenizer)
        counts = distribution_plots(rows, args.output / 'figures/dataset')
        quality_plots(rows, args.output / 'figures/dataset')
        samples = sample_plots(records, rows, archive, tokenizer, args.output / 'figures/dataset')
    old = json.loads((ROOT / 'reports/data/dataset_audit.json').read_text())
    for i, split in enumerate(SPLITS):
        if counts[i].tolist() != [old['class_counts'][split][label] for label in LABELS]:
            raise ValueError('Class counts differ from historical audit')
    summary = {'verification': verification, 'tokenizer': TOKENIZER, 'revision': REVISION,
               'token_definition': 'RoBERTa token count including BOS/EOS; truncation iff cleaned count > 96',
               'image_definition': 'Original dimensions after EXIF orientation; records, not unique images',
               'splits': {s: describe([r for r in rows if r['split'] == s]) for s in SPLITS},
               'by_class': {s: {label: describe([r for r in rows if r['split'] == s and r['emotion'] == label])
                                for label in LABELS} for s in SPLITS},
               'class_counts': {s: dict(zip(LABELS, map(int, counts[i]))) for i, s in enumerate(SPLITS)},
               'sample_selection': {k: v for k, v in samples.items() if not k.endswith('_table')},
               'selection_policy': 'Gallery: first two IDs per target class sorted by SHA256(42:ID). Diagnostics: predetermined conditions, not representative sampling.',
               'displayed_preprocessing_matches_training_cache': True,
               'versions': {'numpy': np.__version__}}
    for s in SPLITS:
        if summary['splits'][s]['empty_cleaned'] != old['empty_text_counts'][s]:
            raise ValueError('Empty text counts differ from historical audit')
    write_csv(args.output / 'data/input_quality.csv', rows)
    write_json(args.output / 'data/dataset_visualization.json', summary)
    # Small captions belong to the report, not to a bulk text export.
    (args.output / 'data/sample_captions.md').write_text(samples['gallery_table'] + '\n\n' + samples['preprocessing_table'] + '\n')
    print(json.dumps(summary['splits'], indent=2), flush=True)


if __name__ == '__main__':
    main()
