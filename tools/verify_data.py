"""Verify the original archive and portable prepared-data fingerprint (stdlib only)."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def records_fingerprint(records):
    # Include order, raw-image identity, cleaned text and labels, excluding machine paths.
    portable = [{k: r[k] for k in ('id', 'emotion', 'label', 'binary_label', 'split', 'image_sha256')}
                | {'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest()} for r in records]
    return hashlib.sha256(json.dumps(portable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify(cache, reference, manifests, archive=None):
    if archive is not None:
        if archive.stat().st_size != reference['archive']['bytes'] or file_sha256(archive) != reference['archive']['sha256']:
            raise ValueError('Archive differs from the reported subset_50000.zip; obtain the original data package.')
    saved = json.loads((cache / 'experiment_manifest.json').read_text())
    records = saved['records']
    if dict(collections.Counter(r['split'] for r in records)) != reference['counts']:
        raise ValueError('Prepared split sizes differ from the reference experiment.')
    if records_fingerprint(records) != reference['records_sha256']:
        raise ValueError('Prepared IDs/order/labels/image hashes/cleaned text differ from the reference experiment.')
    for split in ('validation', 'test'):
        with (manifests / f'{split}.tsv').open() as stream:
            expected = list(csv.DictReader(stream, delimiter='\t'))
        actual = [{k: r[k] for k in ('id', 'emotion', 'image_sha256')} for r in records if r['split'] == split]
        if actual != expected:
            raise ValueError(f'{split} rows differ from the committed evaluation list.')
    return {'status': 'passed', 'counts': reference['counts'], 'records_sha256': reference['records_sha256'],
            'archive_verified': archive is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'training/prepared_comparison_v2')
    parser.add_argument('--data', type=Path, help='Also verify the downloaded archive bytes')
    args = parser.parse_args()
    reference = json.loads((ROOT / 'data/dataset_signature.json').read_text())
    print(json.dumps(verify(args.cache, reference, ROOT / 'data/manifests', args.data), indent=2))


if __name__ == '__main__':
    main()
