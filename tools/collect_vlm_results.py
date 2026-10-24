"""Verify completed VLM predictions and export a validation-only comparison."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from evaluate_vlm import EMOTIONS, MODELS, score, validate_progress

VARIANTS = ('direct', 'full', 'frozen', 'direct_5k', 'reset_fusion', 'tune_fusion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'training/runs_vlm')
    parser.add_argument('--output', type=Path, help='Default: RUNS/analysis; use reports/data only to publish verified results')
    args = parser.parse_args()
    output = args.output or args.runs / 'analysis'
    with (ROOT / 'data/manifests/validation.tsv').open() as stream:
        expected = [{'id': r['id'], 'label': EMOTIONS.index(r['emotion'])} for r in csv.DictReader(stream, delimiter='\t')]
    original = {r['variant']: r for r in json.loads((ROOT / 'reports/data/seed42.json').read_text())['runs']}
    runs = [{'variant': v, 'name': f'V{i}', 'kind': 'supervised_validation_selected',
             'metrics': original[v]['best_validation'], 'best_epoch': original[v]['best_epoch']}
            for i, v in enumerate(VARIANTS, 1)]
    common = None
    predictions = []
    for key, (model_id, revision) in MODELS.items():
        folder = args.runs / f'qwen3_vl_{key}'
        result = json.loads((folder / 'results.json').read_text())
        config = result['config']
        if result['status'] != 'completed' or not config['full_validation'] or config['samples'] != 2000:
            raise ValueError('Both full 2,000-sample evaluations must be completed')
        if (config['model_id'], config['revision']) != (model_id, revision):
            raise ValueError('Model does not match the fixed protocol')
        settings = {k: v for k, v in config.items() if k not in ('model_id', 'revision')}
        if common is not None and settings != common:
            raise ValueError('VLM models used different data/prompts/inference settings')
        common = settings
        progress = json.loads((folder / 'progress.json').read_text())
        validate_progress(progress, expected, config['batch_size'])
        if len(progress['predictions']) != 2000 or score(progress['predictions']) != result['metrics']:
            raise ValueError('Metrics do not match saved predictions')
        runs.append({'variant': f'qwen3_vl_{key}', 'name': f'Qwen3-VL {key.upper()}', 'kind': 'zero_shot', **result})
        predictions.extend({'model': model_id, 'id': r['id'], 'true_label': EMOTIONS[r['label']],
                            'predicted_label': EMOTIONS[r['prediction']] if r['prediction'] >= 0 else 'Invalid'}
                           for r in progress['predictions'])
    output.mkdir(parents=True, exist_ok=True)
    payload = {'split': 'validation', 'samples': 2000, 'labels': EMOTIONS, 'runs': runs,
               'verification': 'VLM predictions rescored and IDs/order/labels checked against committed validation.tsv; baseline metrics from historical selected checkpoints.',
               'comparison_limit': 'Supervised models selected epochs on this validation set; zero-shot models did not. No independent test-set ranking or equal training/inference budget is claimed.'}
    (output / 'vlm_validation.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    summary = [{'name': r['name'], 'kind': r['kind'], 'validation_accuracy': r['metrics']['accuracy'],
                'validation_macro_f1': r['metrics']['macro_f1'], 'invalid_outputs': r['metrics'].get('invalid_outputs', 0),
                **{k: r.get('efficiency', {}).get(k) for k in ('evaluation_seconds', 'samples_per_second', 'peak_allocated_gpu_gib')}} for r in runs]
    for filename, rows in [('vlm_validation.csv', summary), ('vlm_predictions.csv', predictions)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({'output': str(output), 'models': len(runs), 'vlm_predictions_verified': len(predictions)}))


if __name__ == '__main__':
    main()
