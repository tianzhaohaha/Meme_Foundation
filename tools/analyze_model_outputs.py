"""Compare matched validation errors and original-space class-center similarities."""
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from report_visuals import (ROOT, LABELS, plt, style, save, write_json, write_csv,
                            stable_order, prepared_image, raw_text, excerpt,
                            load_tokenizer, retained_prefix)
from verify_data import file_sha256

sys.path.insert(0, str(ROOT))
from embedding_analysis.plot import load_files
from embedding_analysis._data import clean_text

MODELS = ['V4', 'V5', 'V6', 'Qwen 4B', 'Qwen 8B']
QWEN = {'Qwen/Qwen3-VL-4B-Instruct': 'Qwen 4B', 'Qwen/Qwen3-VL-8B-Instruct': 'Qwen 8B'}


def join_predictions(ids, labels, rows):
    """Join by ID, reject missing/duplicate/foreign records, and preserve invalid outputs."""
    truth = dict(zip(ids, labels))
    if len(truth) != len(ids) or len(ids) != len(labels):
        raise ValueError('Reference IDs must be unique and aligned')
    joined = {name: {} for name in QWEN.values()}
    for row in rows:
        if row['model'] not in QWEN:
            raise ValueError('Unknown VLM model')
        name, sid = QWEN[row['model']], row['id']
        if sid not in truth or sid in joined[name]:
            raise ValueError('Foreign or duplicate VLM prediction ID')
        if row['true_label'] != LABELS[truth[sid]]:
            raise ValueError('VLM truth label differs from validation manifest')
        value = row['predicted_label']
        if value not in LABELS + ['Invalid']:
            raise ValueError('Unknown predicted label')
        joined[name][sid] = LABELS.index(value) if value != 'Invalid' else -1
    if any(set(values) != set(ids) for values in joined.values()):
        raise ValueError('Missing VLM predictions')
    return {name: np.array([values[sid] for sid in ids]) for name, values in joined.items()}


def center_similarity(features, labels):
    """Mean unit sample vectors per true class, then cosine between class means."""
    x = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if x.ndim != 2 or labels.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('Invalid features or labels')
    if set(labels.tolist()) != set(range(7)):
        raise ValueError('All seven true classes must be present')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('Zero feature vector has no cosine direction')
    unit = x / norms
    means = np.stack([unit[labels == j].mean(0) for j in range(7)])
    lengths = np.linalg.norm(means, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('Zero class mean has no cosine direction')
    centers = means / lengths[:, None]
    return np.clip(centers @ centers.T, -1, 1), lengths


def centroid_plot(bundles, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout='constrained')
    result, rows = {}, []
    for ax, bundle in zip(axes.flat, bundles):
        name = bundle['meta']['model']
        similarity, concentration = center_similarity(bundle['fusion'], bundle['labels'])
        if (bundle['fusion'] < 0).any():
            raise ValueError('Expected post-ReLU nonnegative fusion features')
        art = ax.imshow(similarity, cmap='YlGnBu', vmin=0, vmax=1)
        for i, j in np.ndindex(similarity.shape):
            ax.text(j, i, f'{similarity[i,j]:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if similarity[i,j] > .65 else 'black')
            rows.append({'model': name, 'true_class_a': LABELS[i], 'true_class_b': LABELS[j],
                         'cosine_similarity': float(similarity[i,j])})
        ax.set(xticks=range(7), xticklabels=LABELS, yticks=range(7), yticklabels=LABELS,
               title=name + (' (original binary head)' if name == 'Binary' else ''))
        ax.tick_params(axis='x', rotation=35)
        same, different = [], []
        positive = {2, 4, 5}
        pairs = []
        for i in range(7):
            for j in range(i+1, 7):
                (same if (i in positive) == (j in positive) else different).append(similarity[i,j])
                pairs.append({'a': LABELS[i], 'b': LABELS[j], 'cosine': float(similarity[i,j])})
        result[name] = {'matrix': similarity.tolist(), 'class_mean_unit_vector_length': concentration.tolist(),
                        'mean_same_polarity_pairs': float(np.mean(same)),
                        'mean_cross_polarity_pairs': float(np.mean(different)),
                        'closest_pairs': sorted(pairs, key=lambda r: -r['cosine'])[:5],
                        'checkpoint_sha256': bundle['meta']['checkpoint_sha256']}
    fig.colorbar(art, ax=axes, shrink=.65, label='Cosine similarity of class centers (fixed 0–1 scale)')
    fig.suptitle('Same 2,000 validation samples | 256-D fusion space, no PCA/t-SNE\nL2-normalize samples, average by true class, compare mean directions')
    save(fig, output / 'figures/embedding', 'fusion_centroid_similarity')
    write_csv(output / 'data/fusion_centroid_similarity.csv', rows)
    write_json(output / 'data/fusion_centroid_similarity.json', {
        'labels': LABELS, 'split': 'validation', 'sample_count': len(bundles[0]['ids']),
        'definition': 'Cosine between per-class means of L2-normalized 256-D sample features; no global centering or projection.',
        'interpretation': 'Within-model angular proximity, not accuracy, cluster overlap, or causal evidence. ReLU and shared feature directions raise cosine values.',
        'models': result})
    return result


def error_summary_plot(correct, output):
    n = len(correct)
    wrong = (~correct).astype(np.int64)
    right = correct.astype(np.int64)
    shared = wrong.T @ wrong
    wins = right.T @ wrong
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), layout='constrained',
                             gridspec_kw={'width_ratios': [1.05, 1.05, 1]})
    for ax, matrix, title in [(axes[0], shared, 'Both models wrong (count / 2,000)'),
                               (axes[1], wins, 'Row correct, column wrong (count / 2,000)')]:
        ax.imshow(matrix / n, cmap='Blues', vmin=0, vmax=.6)
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, str(matrix[i,j]), ha='center', va='center',
                    color='white' if matrix[i,j] / n > .35 else 'black')
        ax.set(xticks=range(5), xticklabels=MODELS, yticks=range(5), yticklabels=MODELS, title=title)
        ax.tick_params(axis='x', rotation=35)
    counts = np.bincount(correct.sum(1), minlength=6)
    bars = axes[2].bar(range(6), counts, color=['#b64b4b'] + ['#6e95bd'] * 4 + ['#328779'])
    axes[2].bar_label(bars, padding=3)
    axes[2].set(xlabel='Number of correct models (out of 5)', ylabel='Validation samples',
                title='Shared errors and complementary predictions', ylim=(0, max(counts) * 1.15), xticks=range(6))
    fig.suptitle('V4 / V5 / V6 / Qwen3-VL 4B / 8B | Same 2,000 validation samples')
    save(fig, output / 'figures/errors', 'error_overlap')
    return shared, wins, counts


def select_cases(ids, correct, v4_confidence):
    sup, vlm = correct[:, :3], correct[:, 3:]
    masks = [('All five wrong', ~correct.any(1)),
             ('All supervised correct; both VLMs wrong', sup.all(1) & ~vlm.any(1)),
             ('Both VLMs correct; all supervised wrong', vlm.all(1) & ~sup.any(1)),
             ('Supervised models disagree in correctness', (sup.sum(1) > 0) & (sup.sum(1) < 3)),
             ('Highest-confidence V4 error', ~correct[:, 0])]
    positions = {sid: i for i, sid in enumerate(ids)}
    ranked_errors = sorted([sid for sid, keep in zip(ids, ~correct[:, 0]) if keep],
                           key=lambda sid: (-v4_confidence[positions[sid]], sid))
    reserved = ranked_errors[0] if ranked_errors else None
    selected, used = [], set()
    for reason, mask in masks:
        candidates = [sid for sid, keep in zip(ids, mask) if keep and sid not in used
                      and (reason.startswith('Highest') or sid != reserved)]
        if not candidates:
            continue
        order = (sorted(candidates, key=lambda sid: (-v4_confidence[positions[sid]], sid))
                 if reason.startswith('Highest') else stable_order(candidates))
        selected.append({'reason': reason, 'id': order[0], 'index': positions[order[0]],
                         'candidate_count': int(mask.sum())})
        used.add(order[0])
    return selected


def case_plot(cases, labels, predictions, confidence, archive, output):
    tokenizer = load_tokenizer()
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 3 * len(cases)),
                             gridspec_kw={'width_ratios': [1, 2]}, layout='constrained', squeeze=False)
    table = ['| 选择条件 | ID | 真值 | 原始文本节选 | 模型保留的清洗文本前缀 |', '|---|---|---|---|---|']
    for row, case in enumerate(cases):
        i, sid = case['index'], case['id']
        axes[row, 0].imshow(prepared_image(archive, sid))
        axes[row, 0].set_title(f'ID {sid} | true: {LABELS[labels[i]]}')
        axes[row, 0].axis('off')
        ax = axes[row, 1]
        ax.axis('off')
        cells, colors = [], []
        for name in MODELS:
            prediction = predictions[name][i]
            value = LABELS[prediction] if prediction >= 0 else 'Invalid'
            score = f'{confidence[name][i]:.3f}' if name in confidence else 'not recorded'
            good = prediction == labels[i]
            cells.append([name, value, 'correct' if good else 'wrong', score])
            colors.append(['#e3f2e9' if good else '#fbe6e3'] * 4)
        artist = ax.table(cellText=cells, colLabels=['Model', 'Prediction', 'Result', 'Max softmax'],
                          cellColours=colors, bbox=[0, .05, 1, .8], colWidths=[.2, .2, .18, .3])
        artist.auto_set_font_size(False)
        artist.set_fontsize(10)
        ax.set_title(case['reason'] + f"\n{case['candidate_count']} qualifying samples; selected ID {sid}", fontsize=11)
        text = raw_text(archive, sid)
        prefix, _ = retained_prefix(clean_text(text), tokenizer)
        table.append(f"| {case['reason']} | {sid} | {LABELS[labels[i]]} | {excerpt(text, 220)} | {excerpt(prefix, 320)} |")
    fig.suptitle('Diagnostic examples, not a random accuracy estimate\nImages are the exact 224 x 224 input form; confidence is uncalibrated')
    save(fig, output / 'figures/errors', 'error_cases')
    (output / 'data/error_case_captions.md').write_text('\n'.join(table) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=ROOT / 'embedding_analysis/results/features')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports')
    args = parser.parse_args()
    names = ['Binary', 'V4', 'V5', 'V6']
    paths = [args.features / f'{name}_validation.npz' for name in names]
    bundles = load_files(paths)
    reference = {r['model']: r for r in json.loads((ROOT / 'reports/data/embedding_visualization.json').read_text())['models']}
    for path, bundle in zip(paths, bundles):
        if file_sha256(path) != reference[bundle['meta']['model']]['feature_file_sha256']:
            raise ValueError('Features differ from the audited historical extraction: ' + str(path))
    with (ROOT / 'data/manifests/validation.tsv').open() as stream:
        manifest = list(csv.DictReader(stream, delimiter='\t'))
    ids, labels = bundles[0]['ids'].tolist(), bundles[0]['labels']
    expected = [(r['id'], LABELS.index(r['emotion']), r['image_sha256']) for r in manifest]
    if expected != list(zip(ids, labels.tolist(), bundles[0]['image_sha256'].tolist())):
        raise ValueError('Feature rows differ from the published validation manifest')
    predictions, confidence = {}, {}
    for bundle in bundles[1:]:
        name, prob = bundle['meta']['model'], bundle['probabilities']
        if not np.isfinite(prob).all() or np.any(prob < 0) or not np.allclose(prob.sum(1), 1, atol=1e-6):
            raise ValueError('Invalid class probabilities')
        predictions[name], confidence[name] = prob.argmax(1), prob.max(1)
    with (ROOT / 'reports/data/vlm_predictions.csv').open() as stream:
        predictions.update(join_predictions(ids, labels.tolist(), list(csv.DictReader(stream))))
    metrics = {}
    old = {r['name'].replace('Qwen3-VL', 'Qwen'): r['metrics'] for r in json.loads(
        (ROOT / 'reports/data/vlm_validation.json').read_text())['runs']}
    for name, pred in predictions.items():
        metrics[name] = {'accuracy': float(accuracy_score(labels, pred)),
                         'macro_f1': float(f1_score(labels, pred, labels=list(range(7)), average='macro', zero_division=0))}
        if any(abs(metrics[name][k] - old[name][k]) > 1e-9 for k in metrics[name]):
            raise ValueError('Predictions do not reproduce saved validation metrics: ' + name)
    correct = np.stack([predictions[name] == labels for name in MODELS], axis=1)
    style()
    centers = centroid_plot(bundles, args.output)
    shared, wins, counts = error_summary_plot(correct, args.output)
    cases = select_cases(ids, correct, confidence['V4'])
    with zipfile.ZipFile(args.data) as archive:
        for case in cases:
            raw = archive.read(f"subset_50000/all_data/{case['id']}.jpg")
            if hashlib.sha256(raw).hexdigest() != manifest[case['index']]['image_sha256']:
                raise ValueError('Case image differs from evaluation image')
        case_plot(cases, labels, predictions, confidence, archive, args.output)
    rows = [{'model': name, 'id': sid, 'true_label': LABELS[labels[i]],
             'predicted_label': LABELS[predictions[name][i]] if predictions[name][i] >= 0 else 'Invalid',
             'correct': int(predictions[name][i] == labels[i]),
             'max_softmax': float(confidence[name][i]) if name in confidence else ''}
            for name in MODELS for i, sid in enumerate(ids)]
    write_csv(args.output / 'data/model_error_predictions.csv', rows)
    summary = {'split': 'validation', 'n': len(ids), 'models': MODELS, 'metrics': metrics,
               'shared_error_counts': shared.tolist(), 'row_correct_column_wrong': wins.tolist(),
               'correct_models_histogram': counts.tolist(), 'cases': cases,
               'selection_policy': 'Reserve the highest V4 max-softmax error for the last case. Other cases use fixed SHA256(42:ID) ordering within predefined conditions, excluding the reserved and already selected IDs.',
               'source_feature_sha256': {name: file_sha256(path) for name, path in zip(names, paths)},
               'source_vlm_predictions_sha256': file_sha256(ROOT / 'reports/data/vlm_predictions.csv'),
               'checks': {'ids_labels_images_match_manifest': True, 'all_metrics_match_historical': True},
               'scope': 'No new inference/training. Validation selected supervised checkpoints. Qwen Invalid counts as wrong. Softmax not calibrated; no VLM confidence available.'}
    write_json(args.output / 'data/model_error_analysis.json', summary)
    print(json.dumps({'metrics': metrics, 'correct_models_histogram': counts.tolist(), 'cases': cases,
                      'centroid_closest_pairs': {name: value['closest_pairs'][:2] for name, value in centers.items()}}, indent=2))


if __name__ == '__main__':
    main()
