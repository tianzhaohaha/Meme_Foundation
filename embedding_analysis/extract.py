"""Export held-out features from existing checkpoints. No model is trained."""
import argparse
import csv
import gc
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
MODULE = Path(__file__).resolve().parent
ROOT = MODULE.parent
VARIANTS = {'V1': 'direct', 'V2': 'full', 'V3': 'frozen', 'V4': 'direct_5k',
            'V5': 'reset_fusion', 'V6': 'tune_fusion', 'Binary': 'shared_binary'}
DEFAULT_VARIANTS = ['Binary', 'V4', 'V5', 'V6']


def checkpoint_classes(name):
    from ._data import EMOTIONS
    return ['negative', 'positive'] if name == 'Binary' else EMOTIONS


def validate_checkpoint(checkpoint, name):
    """Preserve the source's binary head while keeping seven-class plotting labels."""
    config = checkpoint['config']
    classes = checkpoint_classes(name)
    if config.get('variant') != VARIANTS[name] or config.get('num_classes') != len(classes):
        raise ValueError(f'{name}: wrong checkpoint variant/classes')
    if checkpoint['epoch'] != checkpoint['best_epoch']:
        raise ValueError(f'{name}: use validation-selected best.pt')
    if checkpoint.get('label_mapping') != {label: i for i, label in enumerate(classes)}:
        raise ValueError(f'{name}: incompatible class order')
    if config.get('architecture') != 'multimodal':
        raise ValueError('This comparison requires ViT + RoBERTa')
    return config


def default_checkpoint(runs_root, name):
    if name != 'Binary':
        return runs_root / VARIANTS[name] / 'seed_42/best.pt'
    candidates = sorted((runs_root / 'shared_binary').glob('*/seed_42/best.pt'))
    if len(candidates) != 1:
        raise ValueError('Expected one shared binary source; explicitly specify --checkpoint Binary=/path/best.pt')
    return candidates[0]


def evaluation_targets(rows, name):
    import numpy as np
    from ._data import POLARITY
    return np.array([POLARITY[r['emotion']] if name == 'Binary' else r['label'] for r in rows], dtype=np.int64)


def verify_metrics(actual, saved):
    checks = {key: {'checkpoint': saved.get(key), 'actual': value,
                   'matches': saved.get(key) is not None and abs(value - saved[key]) < 1e-6}
              for key, value in actual.items()}
    if not all(item['matches'] for item in checks.values()):
        raise ValueError('Validation metrics differ from checkpoint; refusing to export unverified features: ' + json.dumps(checks))
    return checks


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_rows(data, manifest):
    from PIL import Image, ImageOps
    from ._data import EMOTIONS, clean_text, read_list
    with manifest.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain unique IDs')
    prepared = []
    with zipfile.ZipFile(data) as archive:
        metadata = {r['id']: r for r in read_list(archive.read(
            'subset_50000/all_data_id_and_label_with_polarity.txt').decode('utf-8-sig'))}
        for row in rows:
            sample_id = row['id']
            if row['emotion'] != metadata[sample_id]['label']:
                raise ValueError('Label mismatch: ' + sample_id)
            raw = archive.read(f'subset_50000/all_data/{sample_id}.jpg')
            if hashlib.sha256(raw).hexdigest() != row['image_sha256']:
                raise ValueError('Image hash mismatch: ' + sample_id)
            # Same padding AND JPEG quality=95 roundtrip as experiment_data.py.
            with Image.open(io.BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image).convert('RGBA')
                background = Image.new('RGBA', image.size, (128, 128, 128, 255))
                background.alpha_composite(image)
                padded = ImageOps.pad(background.convert('RGB'), (224, 224),
                    method=Image.Resampling.BILINEAR, color=(128, 128, 128))
                jpeg = io.BytesIO()
                padded.save(jpeg, format='JPEG', quality=95)
            text = clean_text(archive.read(f'subset_50000/all_data/{sample_id}.txt').decode(
                'utf-8-sig', errors='replace'))
            prepared.append({**row, 'jpeg': jpeg.getvalue(), 'text': text,
                             'label': EMOTIONS.index(row['emotion'])})
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'training/runs_comparison')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    parser.add_argument('--checkpoint', action='append', default=[], help='Repeat Binary=/path/best.pt, V4=/path/best.pt etc.')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--output', type=Path, default=MODULE / 'results/features')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--include-encoded', action='store_true', help='Also export pre-fusion 1536-D features')
    parser.add_argument('--trust-legacy-checkpoint', action='store_true', help='Only for your trusted historical checkpoint')
    args = parser.parse_args(argv)
    selected = {}
    for item in args.checkpoint:
        name, sep, path = item.partition('=')
        if not sep or name not in VARIANTS or name in selected:
            raise ValueError('Use unique --checkpoint Binary=/path/best.pt, V4=/path/best.pt, etc.')
        selected[name] = Path(path).expanduser().resolve()
    if not selected:
        selected = {v: default_checkpoint(args.runs_root, v) for v in args.variants}
    missing = [str(path) for path in selected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Existing best checkpoints required; no training is started: ' + ', '.join(missing))
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    import numpy as np
    import torch
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from transformers import AutoConfig, AutoTokenizer
    from ._data import EMOTIONS, POLARITY
    from ._model import EmotionModel, TEXT_MODEL, TEXT_MODEL_REVISION

    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic CPU fallback is performed')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    device = torch.device(args.device)
    manifest = MODULE / f'manifests/{args.split}.tsv'
    rows = prepare_rows(args.data, manifest)
    # Includes processed image bytes and cleaned text, without exposing their contents in metadata.
    input_signature = hashlib.sha256(json.dumps([
        {'id': r['id'], 'label': r['label'], 'image_sha256': r['image_sha256'],
         'prepared_image_sha256': hashlib.sha256(r['jpeg']).hexdigest(),
         'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest()} for r in rows
    ], sort_keys=True).encode()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_MODEL_REVISION)
    text_config = AutoConfig.from_pretrained(TEXT_MODEL, revision=TEXT_MODEL_REVISION)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes = {name: sha256(path) for name, path in selected.items()}

    class Examples(Dataset):
        def __init__(self, length):
            self.tokens = tokenizer([r['text'] for r in rows], padding='max_length',
                truncation=True, max_length=length, return_tensors='pt')
            self.transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(io.BytesIO(rows[index]['jpeg'])) as image:
                tensor = self.transform(image.convert('RGB'))
            return tensor, self.tokens['input_ids'][index], self.tokens['attention_mask'][index]

    for name, checkpoint_path in selected.items():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu',
                weights_only=not args.trust_legacy_checkpoint, mmap=True)
        except Exception as exc:
            raise RuntimeError('Cannot load checkpoint. For YOUR trusted old checkpoint only, '
                'use --trust-legacy-checkpoint. Original error: ' + str(exc)) from exc
        config = validate_checkpoint(checkpoint, name)
        if 'Binary' in selected and name in ('V2', 'V3', 'V5', 'V6'):
            if config.get('init_checkpoint_sha256') != checkpoint_hashes['Binary']:
                raise ValueError(f'{name}: does not descend from the selected binary checkpoint')
        if config.get('text_model_revision', TEXT_MODEL_REVISION) != TEXT_MODEL_REVISION:
            raise ValueError('Tokenizer revision differs from the training configuration')
        classes = checkpoint_classes(name)
        model = EmotionModel(pretrained=False, dropout=config.get('dropout', .5), num_classes=len(classes),
                             text_config=text_config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        epoch = checkpoint['epoch']
        saved_validation = checkpoint.get('validation', {})
        del checkpoint
        gc.collect()
        model.to(device).eval()
        loader = DataLoader(Examples(config.get('max_length', 96)), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        fused, encoded, probabilities = [], [], []
        with torch.inference_mode():
            for index, (image, ids, mask) in enumerate(loader, 1):
                z = model.encode(image.to(device), ids.to(device), mask.to(device))
                features = model.fusion(z)
                fused.append(features.float().cpu().numpy())
                probabilities.append(model.head(features).softmax(-1).float().cpu().numpy())
                if args.include_encoded:
                    encoded.append(z.float().cpu().numpy())
                if index % 25 == 0 or index == len(loader):
                    print(json.dumps({'model': name, 'batch': index, 'batches': len(loader)}), flush=True)
        labels = np.array([r['label'] for r in rows], dtype=np.int64)
        prob = np.concatenate(probabilities)
        prediction = prob.argmax(1)
        targets = evaluation_targets(rows, name)
        metrics = {'accuracy': float(accuracy_score(targets, prediction)), 'macro_f1': float(f1_score(
            targets, prediction, labels=list(range(len(classes))), average='macro', zero_division=0))}
        reproduction = {}
        if args.split == 'validation':
            reproduction = verify_metrics(metrics, saved_validation)
        metadata = {'model': name, 'variant': VARIANTS[name], 'epoch': epoch, 'split': args.split,
            'sample_count': len(rows), 'class_names': EMOTIONS, 'manifest_sha256': sha256(manifest),
            'checkpoint_sha256': checkpoint_hashes[name],
            'init_checkpoint_sha256': config.get('init_checkpoint_sha256'),
            'num_classes': len(classes), 'output_class_names': classes,
            'metrics_task': 'binary_polarity' if name == 'Binary' else 'seven_class_emotion',
            'plot_label_policy': 'Always true seven-class labels; never predicted labels, including Binary',
            'input_signature': input_signature, 'text_max_length': config.get('max_length', 96),
            'text_model_revision': TEXT_MODEL_REVISION, 'dataset_id': config.get('dataset_id'),
            'feature_definition': '256-D Linear -> LayerNorm -> ReLU, before dropout/head',
            'encoded_definition': 'ViT CLS 768 + masked-mean RoBERTa 768; before fusion',
            'metrics': metrics, 'validation_reproduction': reproduction,
            'device': str(device), 'torch': str(torch.__version__)}
        arrays = {'fusion': np.concatenate(fused), 'labels': labels,
            'polarity': np.array([POLARITY[r['emotion']] for r in rows], dtype=np.int64),
            'ids': np.array([r['id'] for r in rows]),
            'image_sha256': np.array([r['image_sha256'] for r in rows]),
            'probabilities': prob, 'metadata': np.array(json.dumps(metadata))}
        if args.include_encoded:
            arrays['encoded'] = np.concatenate(encoded)
        destination = args.output / f'{name}_{args.split}.npz'
        temporary = destination.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
        destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(json.dumps({'event': 'exported', 'model': name, 'file': str(destination),
            'bytes': destination.stat().st_size, 'metrics': metrics,
            'validation_reproduction': reproduction}), flush=True)
        del model, loader, arrays, fused, encoded, probabilities
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('Send only the .npz and .json files in ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
