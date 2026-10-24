"""Fixed 20k/5k training lists and shared, image-disjoint validation/test sets."""
import collections
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

EMOTIONS = ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad']
POLARITY = {label: int(label in ('Calm', 'Happy', 'Love')) for label in EMOTIONS}
PATTERN = r'(?i)#(?:angry|anger|bored|boredom|calm|fear|fearful|happy|happiness|love|loving|sad|sadness)\b'


def clean_text(text):
    return ' '.join(re.sub(r'https?://\S+', ' ', re.sub(PATTERN, ' ', text)).split())


def text_key(text):
    # Empty text carries no label-specific content; do not group all missing texts.
    return ' '.join(text.casefold().split())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def read_list(text, expected=None):
    rows = list(csv.DictReader(io.StringIO(text), delimiter='\t'))
    if expected is not None and len(rows) != expected:
        raise ValueError(f'Expected {expected} rows, got {len(rows)}')
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty list or duplicate sample IDs')
    for row in rows:
        if not row['id'].isdigit() or row['label'] not in EMOTIONS:
            raise ValueError(f'Invalid ID/label: {row}')
        if row.get('polarity') != ('positive' if POLARITY[row['label']] else 'negative'):
            raise ValueError(f'Inconsistent polarity: {row}')
    return rows


def prepare_experiment(args):
    cache = args.cache
    source_text = args.source_list.read_text(encoding='utf-8-sig')
    target_text = args.target_list.read_text(encoding='utf-8-sig')
    source = read_list(source_text, 20000)
    target = read_list(target_text, 5000)
    source_ids, target_ids = {r['id'] for r in source}, {r['id'] for r in target}
    if source_ids & target_ids:
        raise ValueError('Source and fine-tuning lists overlap by ID')
    stat = args.data.stat()
    inputs = {'schema': 2, 'zip': str(args.data), 'zip_bytes': stat.st_size,
              'zip_mtime_ns': stat.st_mtime_ns, 'source_sha256': hashlib.sha256(source_text.encode()).hexdigest(),
              'target_sha256': hashlib.sha256(target_text.encode()).hexdigest(),
              'split_seed': args.split_seed, 'validation_size': args.validation_size,
              'test_size': args.test_size, 'cache': str(cache)}
    manifest = cache / 'experiment_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['inputs'] != inputs:
            raise ValueError('Dataset/cache configuration changed; choose a new --cache directory')
        if saved['dataset_id'] != fingerprint(saved['records']):
            raise ValueError('Cached manifest fingerprint mismatch; rebuild the cache')
        if any(not Path(r['image']).is_file() for r in saved['records']):
            raise FileNotFoundError('Cached images missing; rebuild with a new --cache directory')
        return saved
    with zipfile.ZipFile(args.data) as archive:
        all_rows = read_list(archive.read('subset_50000/all_data_id_and_label_with_polarity.txt').decode('utf-8-sig'))
        metadata = {r['id']: r for r in all_rows}
        for row in source + target:
            if metadata.get(row['id']) != row:
                raise ValueError(f"List does not match ZIP metadata for ID {row['id']}")
        # Hash actual ZIP bytes; do not trust a stale precomputed checksum file.
        hashes, texts = {}, {}
        for index, row in enumerate(all_rows, 1):
            hashes[row['id']] = hashlib.sha256(archive.read(f"subset_50000/all_data/{row['id']}.jpg")).hexdigest()
            texts[row['id']] = clean_text(archive.read(f"subset_50000/all_data/{row['id']}.txt").decode('utf-8-sig', errors='replace'))
            if index % 10000 == 0:
                print(json.dumps({'event': 'audit_images', 'done': index, 'total': len(all_rows)}), flush=True)
        train_hashes = {hashes[i] for i in source_ids | target_ids}
        seen = set(train_hashes)
        train_texts = {text_key(texts[i]) for i in source_ids | target_ids if text_key(texts[i])}
        seen_texts = set(train_texts)
        eligible = []
        for row in all_rows:
            digest = hashes[row['id']]
            text_digest = text_key(texts[row['id']])
            if digest not in seen and (not text_digest or text_digest not in seen_texts):
                eligible.append(row)
                seen.add(digest)
                if text_digest:
                    seen_texts.add(text_digest)
        if len(eligible) <= args.validation_size + args.test_size:
            raise ValueError('Not enough image/text-disjoint held-out samples')
        _, heldout = train_test_split(eligible, test_size=args.validation_size + args.test_size,
                                     stratify=[r['label'] for r in eligible], random_state=args.split_seed)
        validation, test = train_test_split(heldout, test_size=args.test_size,
                                           stratify=[r['label'] for r in heldout], random_state=args.split_seed)
        records = []
        for split, rows in [('source', source), ('target', target), ('validation', validation), ('test', test)]:
            for row in rows:
                digest = hashes[row['id']]
                path = cache / 'images' / f'{digest}.jpg'
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    raw = archive.read(f"subset_50000/all_data/{row['id']}.jpg")
                    with Image.open(io.BytesIO(raw)) as image:
                        image = ImageOps.exif_transpose(image).convert('RGBA')
                        background = Image.new('RGBA', image.size, (128, 128, 128, 255))
                        background.alpha_composite(image)
                        ImageOps.pad(background.convert('RGB'), (224, 224), method=Image.Resampling.BILINEAR,
                                     color=(128, 128, 128)).save(path, quality=95)
                text = texts[row['id']]
                records.append({'id': row['id'], 'emotion': row['label'], 'label': EMOTIONS.index(row['label']),
                                'binary_label': POLARITY[row['label']], 'split': split,
                                'image_sha256': digest, 'image': str(path), 'text': text})
            print(json.dumps({'event': 'prepared_split', 'split': split, 'count': len(rows)}), flush=True)
    audit = {'counts': dict(collections.Counter(r['split'] for r in records)),
             'class_counts': {s: dict(collections.Counter(r['emotion'] for r in records if r['split'] == s))
                              for s in ('source', 'target', 'validation', 'test')},
             'source_target_shared_image_hashes': len({hashes[i] for i in source_ids} & {hashes[i] for i in target_ids}),
             'empty_text_counts': {s: sum(not r['text'] for r in records if r['split'] == s)
                                   for s in ('source', 'target', 'validation', 'test')},
             'unique_heldout_candidates': len(eligible),
             'scope': 'Validation/test are disjoint from all training by ID, exact image bytes, and normalized nonempty text, '
                      'and from each other. Text normalization is cleaning, casefold and whitespace normalization. '
                      'Training duplicates are retained to preserve the specified lists. No near-duplicate guarantee.'}
    saved = {'inputs': inputs, 'records': records, 'audit': audit, 'dataset_id': fingerprint(records)}
    for split in audit['counts']:
        path = cache / f'{split}.tsv'
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'emotion', 'image_sha256'], delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(r for r in records if r['split'] == split)
    write_json(cache / 'audit.json', audit)
    write_json(manifest, saved)
    return saved


def stage_rows(records, variant, binary=False):
    training = {'source'} if binary else {'source', 'target'} if variant == 'direct' else {'target'}
    return [{**r, 'split': 'train' if r['split'] in training else r['split'],
             'label': r['binary_label'] if binary else r['label']}
            for r in records if r['split'] in training | ({'validation'} if binary else {'validation', 'test'})]
