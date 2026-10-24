"""Shared, CPU-only helpers for reproducible report figures."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tumemo-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad']
SPLITS = ['source', 'target', 'validation', 'test']
SPLIT_NAMES = ['Source 20k', 'Target 5k', 'Validation 2k', 'Test 2k']
COLORS = list(plt.get_cmap('tab10').colors[:7])
TOKENIZER = 'FacebookAI/roberta-base'
REVISION = 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'


def load_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION, local_files_only=True)
    # Length statistics intentionally include texts longer than the encoder limit.
    tokenizer.model_max_length = 10**9
    return tokenizer


def retained_prefix(text, tokenizer):
    encoded = tokenizer(text, truncation=True, max_length=96, return_offsets_mapping=True)
    end = max(stop for _, stop in encoded['offset_mapping'])
    return text[:end], len(encoded['input_ids'])


def style():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def stable_order(ids, seed=42):
    return sorted(ids, key=lambda sample_id: (hashlib.sha256(
        f'{seed}:{sample_id}'.encode()).hexdigest(), sample_id))


def raw_text(archive, sample_id):
    return archive.read(f'subset_50000/all_data/{sample_id}.txt').decode('utf-8-sig', errors='replace')


def raw_image(archive, sample_id):
    with Image.open(io.BytesIO(archive.read(f'subset_50000/all_data/{sample_id}.jpg'))) as image:
        image = ImageOps.exif_transpose(image).convert('RGBA')
        background = Image.new('RGBA', image.size, (128, 128, 128, 255))
        background.alpha_composite(image)
        return background.convert('RGB')


def prepared_image(archive, sample_id):
    image = ImageOps.pad(raw_image(archive, sample_id), (224, 224),
                         method=Image.Resampling.BILINEAR, color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    with Image.open(buffer) as saved:
        return saved.copy()


def excerpt(text, limit=160):
    # Escape user-authored text before embedding it in Markdown tables.
    import html
    text = ' '.join(text.split())
    if len(text) > limit:
        text = text[:limit] + '…'
    text = html.escape(text).replace('|', '&#124;')
    for character in ('\\', '*', '_', '[', ']', '`', '~'):
        text = text.replace(character, '\\' + character)
    return text or '（空文本）'
