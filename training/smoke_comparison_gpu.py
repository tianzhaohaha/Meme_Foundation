"""One real GPU optimizer step per strategy. Metrics are not model quality estimates."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_HOME', str(PROJECT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))
os.environ.setdefault('TORCH_HOME', str(PROJECT / '.cache/torch'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import EmotionModel, TEXT_MODEL, TEXT_MODEL_REVISION
from train import Memes, dump
from training_engine import transfer_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=HERE / 'prepared_comparison_v2')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real-model smoke check')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_MODEL_REVISION)
    rows = json.loads((args.cache / 'experiment_manifest.json').read_text())['records'][:2]
    batch = next(iter(DataLoader(Memes(rows, tokenizer, 16), batch_size=2)))
    images, ids, mask, _ = [x.cuda() for x in batch]
    records = []
    pretrained = None
    for variant in ('binary', 'direct', 'full', 'frozen', 'direct_5k', 'reset_fusion', 'tune_fusion'):
        torch.manual_seed(42)
        classes = 2 if variant == 'binary' else 7
        net = EmotionModel(num_classes=classes)
        if variant in ('full', 'frozen', 'reset_fusion', 'tune_fusion'):
            transfer_backbone(net, pretrained, reset_fusion=variant == 'reset_fusion')
        if variant == 'frozen':
            net.freeze_backbone()
        elif variant == 'tune_fusion':
            net.freeze_backbone(train_fusion=True)
        net.cuda().train()
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-5)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = net(images, ids, mask)
        assert logits.shape == (2, classes)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device='cuda'))
        loss.backward()
        assert torch.isfinite(loss)
        assert net.head[1].weight.grad is not None
        if variant == 'frozen':
            assert all(p.grad is None for name, p in net.named_parameters() if not name.startswith('head.'))
            assert not net.image_encoder.training and not net.text_encoder.training and not net.fusion.training
        elif variant == 'tune_fusion':
            assert all(p.grad is None for name, p in net.named_parameters() if name.startswith(('image_encoder.', 'text_encoder.')))
            assert net.fusion[0].weight.grad is not None
            assert not net.image_encoder.training and not net.text_encoder.training and net.fusion.training
        else:
            assert net.image_encoder.conv_proj.weight.grad is not None
            assert net.text_encoder.embeddings.word_embeddings.weight.grad is not None
        optimizer.step()
        torch.cuda.synchronize()
        record = {'variant': variant, 'loss': float(loss.detach()), 'step_seconds': time.perf_counter() - started,
                  'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
                  'peak_allocated_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
        records.append(record)
        print(json.dumps(record), flush=True)
        if variant == 'binary':
            pretrained = {'model_state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}}
        del net, optimizer, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    report = {'status': 'passed', 'gpu': torch.cuda.get_device_name(0), 'records': records,
              'note': 'One optimizer step with two real images/texts per variant; not final accuracy or a benchmark.'}
    dump(HERE / 'runs_comparison_smoke/verification.json', report)


if __name__ == '__main__':
    main()
