"""Shared training loop for binary pretraining and seven-class experiments."""
import copy
import hashlib
import json
import os
import pickle
import platform
import random
import resource
import signal
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_data import EMOTIONS, fingerprint
from model import EmotionModel, TEXT_MODEL, IMAGE_MODEL, IMAGE_WEIGHTS
from tracking import Tracker


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, trust_legacy=False):
    try:
        return torch.load(path, map_location='cpu', weights_only=not trust_legacy)
    except pickle.UnpicklingError as exc:
        raise ValueError('Checkpoint is not in the restricted format. Only for a trusted local old checkpoint, '
                         'explicitly pass --trust-legacy-checkpoint; never use it for an unknown file.') from exc


def publish_checkpoint(source, target):
    """Publish the same immutable checkpoint without serializing gigabytes twice."""
    temporary = target.with_suffix('.pt.link.tmp')
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def repair_best_checkpoints(saved, last_path, out):
    from experiment_data import write_json
    # last.pt is the commit point. A crash may interrupt publication of best.pt.
    for filename, epoch in [('best', saved['best_epoch']),
                            ('best_accuracy', saved.get('best_accuracy_epoch', -1))]:
        if epoch == saved['epoch']:
            publish_checkpoint(last_path, out / f'{filename}.pt')
            write_json(out / f'{filename}_validation_metrics.json', {'epoch': epoch, **saved['validation']})
    if not (out / 'best.pt').exists():
        raise FileNotFoundError('Keep the selected best.pt with last.pt in the original output directory')


def transfer_backbone(model, checkpoint, reset_fusion=False):
    excluded = ('head.', 'fusion.') if reset_fusion else ('head.',)
    source = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith(excluded)}
    missing, unexpected = model.load_state_dict(source, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith(excluded)}
    if unexpected or set(missing) != expected_missing:
        raise ValueError(f'Incompatible pretrained backbone: missing={missing}, unexpected={unexpected}')


def build_config(args, rows, seed):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(seed=seed, architecture=getattr(args, 'architecture', 'multimodal'),
                  num_classes=getattr(args, 'num_classes', 2),
                  freeze_backbone=getattr(args, 'freeze_backbone', False),
                  freeze_encoders=getattr(args, 'freeze_encoders', getattr(args, 'freeze_backbone', False)),
                  reset_fusion=getattr(args, 'reset_fusion', False),
                  stage=getattr(args, 'stage', 'binary'), variant=getattr(args, 'variant', 'legacy_binary'),
                  selection_metric=getattr(args, 'selection_metric', 'macro_f1'),
                  manifest_sha256=fingerprint(rows), optimizer='Adam', precision='fp32',
                  image_model=IMAGE_MODEL, image_weights=IMAGE_WEIGHTS,
                  text_model=TEXT_MODEL if getattr(args, 'architecture', 'multimodal') == 'multimodal' else None)
    config['labels'] = EMOTIONS if config['num_classes'] == 7 else ['negative', 'positive']
    return config


def validate_resume(saved, config):
    keys = ('image_model', 'image_weights', 'text_model', 'seed', 'epochs', 'batch_size', 'max_length',
            'encoder_lr', 'head_lr', 'warmup_ratio', 'dropout', 'weight_decay', 'class_weight_power',
            'label_smoothing', 'total_steps', 'manifest_sha256')
    for key in keys:
        if saved[key] != config[key]:
            raise ValueError(f'Resume configuration mismatch: {key}')
    for key in ('architecture', 'num_classes', 'freeze_backbone', 'selection_metric', 'patience', 'init_checkpoint_sha256'):
        if key in saved:
            old = saved[key]
            if key == 'architecture' and old.startswith('ViT-B/16 + RoBERTa'):
                old = 'multimodal'
            if old != config.get(key):
                raise ValueError(f'Resume configuration mismatch: {key}')
    for key, default in (('reset_fusion', False), ('freeze_encoders', saved.get('freeze_backbone', False))):
        if saved.get(key, default) != config[key]:
            raise ValueError(f'Resume configuration mismatch: {key}')


def run_stage(args, rows, tokenizer, seed):
    from train import Memes, dump, evaluate, make_scheduler, restore_rng, rng_state, save_checkpoint

    session_start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(getattr(args, 'device', None) or
                          ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    if getattr(args, 'require_cuda', False) and device.type != 'cuda':
        raise RuntimeError('CUDA is required for comparison runs. Use --device cpu only for explicit smoke checks.')
    torch.set_num_threads(4)
    out = args.output / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out / 'last.pt').exists():
        raise FileExistsError(f'Run exists: resume {out / "last.pt"} or use a new output directory')
    config = build_config(args, rows, seed)
    labels = config['labels']
    sets = {s: Memes([r for r in rows if r['split'] == s], tokenizer, args.max_length)
            for s in ('train', 'validation', 'test') if any(r['split'] == s for r in rows)}
    if not all(s in sets and len(sets[s]) for s in ('train', 'validation')):
        raise ValueError('Both train and validation data are required')
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == 'train', num_workers=0,
                             generator=generator if s == 'train' else None) for s, d in sets.items()}
    model = EmotionModel(dropout=args.dropout, num_classes=config['num_classes'], architecture=config['architecture'])
    init = getattr(args, 'init_checkpoint', None)
    trust_legacy = getattr(args, 'trust_legacy_checkpoint', False)
    config['init_checkpoint_sha256'] = file_sha256(init) if init else None
    if init:
        checkpoint = load_checkpoint(init, trust_legacy)
        transfer_backbone(model, checkpoint, reset_fusion=config['reset_fusion'])
        del checkpoint
    if config['freeze_backbone']:
        model.freeze_backbone()
    elif config['freeze_encoders']:
        model.freeze_backbone(train_fusion=True)
    model.to(device)
    head_parameters = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith('head.') or n.startswith('fusion.'))]
    encoder_parameters = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith('head.') or n.startswith('fusion.'))]
    groups = [{'params': head_parameters, 'lr': args.head_lr}]
    if encoder_parameters:
        groups.append({'params': encoder_parameters, 'lr': args.encoder_lr})
    optimizer = torch.optim.Adam(groups, weight_decay=args.weight_decay)
    config.update(device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda, python_version=platform.python_version(),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  trainable_modules=sorted({n.split('.')[0] for n, p in model.named_parameters() if p.requires_grad}),
                  total_steps=args.epochs * len(loaders['train']), sample_counts={s: len(d) for s, d in sets.items()},
                  image_preprocess='aspect-preserving pad to 224x224, ImageNet normalization')
    scheduler = make_scheduler(optimizer, config['total_steps'], args.warmup_ratio)
    counts = np.bincount([r['label'] for r in sets['train'].rows], minlength=len(labels))
    if np.any(counts == 0):
        raise ValueError('Every training class must have samples')
    weights = torch.tensor((len(sets['train']) / (len(labels) * counts)) ** args.class_weight_power,
                           dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    best = best_accuracy = best_selection = -1.
    bad = global_step = best_epoch = best_accuracy_epoch = 0
    start_epoch = 1
    history = []
    efficiency = {k: 0.0 for k in ('train_seconds', 'validation_seconds', 'checkpoint_seconds', 'setup_seconds',
                                  'stage_wall_seconds', 'test_seconds', 'peak_allocated_gpu_gib', 'peak_reserved_gpu_gib')}
    efficiency.update(epochs=0, optimizer_steps=0, samples_seen=0, timing_complete=True)
    if args.resume:
        saved = load_checkpoint(args.resume, trust_legacy)
        validate_resume(saved['config'], config)
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        start_epoch, global_step = saved['epoch'] + 1, saved['global_step']
        best, best_accuracy = saved['best_macro_f1'], saved['best_accuracy']
        best_selection = saved.get('best_selection', best)
        bad, history, best_epoch = saved['bad_epochs'], saved['history'], saved['best_epoch']
        best_accuracy_epoch = saved.get('best_accuracy_epoch', 0)
        efficiency.update(saved.get('efficiency', {'timing_complete': False}))
        # Sidecar is written after checkpoint I/O, so it includes its measured cost.
        sidecar = out / 'efficiency.json'
        complete_sidecar = False
        if sidecar.exists():
            measured = json.loads(sidecar.read_text())
            if measured.get('epochs') == saved['epoch'] and (out / 'history.json').exists():
                efficiency.update(measured)
                history = json.loads((out / 'history.json').read_text())
                complete_sidecar = True
        if not complete_sidecar:
            efficiency['timing_complete'] = False
        repair_best_checkpoints(saved, args.resume, out)
        restore_rng(saved['rng_state'], generator)
        del saved
    dump(out / 'config.json', config)
    previous_wall = efficiency['stage_wall_seconds']
    tracker = Tracker(args, out, config)
    stop = {'requested': False}
    def request_stop(signum, frame):
        stop['requested'] = True
        print(json.dumps({'event': 'stop_requested', 'action': 'finish current epoch and checkpoint'}), flush=True)
    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    failed = True
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    stop_file = getattr(args, 'stop_file', None)
    try:
        if args.resume and tracker.mode == 'offline':
            for entry in history:
                if 'cumulative_train_seconds' in entry:
                    tracker.epoch(entry)
        synchronize(device)
        efficiency['setup_seconds'] += time.perf_counter() - session_start
        print(json.dumps({'event': 'start', 'seed': seed, 'stage': config['stage'], 'device': str(device),
                          'trainable': config['trainable_parameters'], 'counts': counts.tolist(),
                          'start_epoch': start_epoch}), flush=True)
        for epoch in range(start_epoch, args.epochs + 1):
            if bad >= args.patience:
                break
            model.train()
            synchronize(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.perf_counter()
            weighted_sum = weight_sum = plain_sum = 0.
            n = 0
            for step, (image, ids, mask, y) in enumerate(loaders['train'], 1):
                optimizer.zero_grad(set_to_none=True)
                target = y.to(device)
                logits = model(image.to(device), ids.to(device), mask.to(device))
                loss = criterion(logits, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.)
                used_lrs = [group['lr'] for group in optimizer.param_groups]
                optimizer.step()
                scheduler.step()
                global_step += 1
                batch_weight = weights[target].sum().item()
                weighted_sum += loss.item() * batch_weight
                weight_sum += batch_weight
                plain_sum += torch.nn.functional.cross_entropy(logits.detach(), target, reduction='sum').item()
                n += len(y)
                if step == 1 or step % getattr(args, 'log_every', 25) == 0:
                    metrics = {'global_step': global_step, 'epoch': epoch,
                               'train/objective_loss': weighted_sum / weight_sum,
                               'train/head_lr': used_lrs[0], 'train/encoder_lr': used_lrs[1] if len(used_lrs) > 1 else 0.0}
                    tracker.log(metrics)
                    print(json.dumps({'event': 'step', 'step': step, 'steps': len(loaders['train']), **metrics}), flush=True)
            synchronize(device)
            train_seconds = time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            val, _, _ = evaluate(model, loaders['validation'], device, labels)
            synchronize(device)
            validation_seconds = time.perf_counter() - val_start
            allocated = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.
            reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.
            efficiency['train_seconds'] += train_seconds
            efficiency['validation_seconds'] += validation_seconds
            efficiency['optimizer_steps'] += len(loaders['train'])
            efficiency['samples_seen'] += n
            efficiency['epochs'] = epoch
            efficiency['peak_allocated_gpu_gib'] = max(efficiency['peak_allocated_gpu_gib'], allocated)
            efficiency['peak_reserved_gpu_gib'] = max(efficiency['peak_reserved_gpu_gib'], reserved)
            entry = {'epoch': epoch, 'global_step': global_step, 'train_objective_loss': weighted_sum / weight_sum,
                     'train_loss': plain_sum / n, 'val_loss': val['loss'], 'val_macro_f1': val['macro_f1'],
                     'val_accuracy': val['accuracy'], 'per_class_recall': {k: val['report'][k]['recall'] for k in labels},
                     'head_lr': optimizer.param_groups[0]['lr'],
                     'encoder_lr': optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.,
                     'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                     'samples_per_second': n / train_seconds, 'seconds_per_step': train_seconds / len(loaders['train']),
                     'peak_allocated_gpu_gib': allocated, 'peak_reserved_gpu_gib': reserved,
                     'process_peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                     'checkpoint_seconds': 0., 'epoch_seconds': train_seconds + validation_seconds,
                     'cumulative_train_seconds': efficiency['train_seconds'],
                     'cumulative_stage_seconds': previous_wall + time.perf_counter() - session_start}
            history.append(entry)
            selection = val[config['selection_metric']]
            improved = selection > best_selection
            if improved:
                best_selection, bad, best_epoch = selection, 0, epoch
            else:
                bad += 1
            best = max(best, val['macro_f1'])
            accuracy_improved = val['accuracy'] > best_accuracy
            best_accuracy = max(best_accuracy, val['accuracy'])
            if accuracy_improved:
                best_accuracy_epoch = epoch
            checkpoint_start = time.perf_counter()
            efficiency['stage_wall_seconds'] = previous_wall + time.perf_counter() - session_start
            state = {'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                     'epoch': epoch, 'global_step': global_step, 'best_epoch': best_epoch, 'best_macro_f1': best,
                     'best_selection': best_selection, 'bad_epochs': bad, 'best_accuracy': best_accuracy,
                     'best_accuracy_epoch': best_accuracy_epoch,
                     'history': history, 'config': config, 'validation': val, 'efficiency': copy.deepcopy(efficiency),
                     'label_mapping': {label: i for i, label in enumerate(labels)}, 'rng_state': rng_state(generator)}
            save_checkpoint(out / 'last.pt', state)
            if improved:
                publish_checkpoint(out / 'last.pt', out / 'best.pt')
                dump(out / 'best_validation_metrics.json', {'epoch': epoch, **val})
            if accuracy_improved:
                publish_checkpoint(out / 'last.pt', out / 'best_accuracy.pt')
                dump(out / 'best_accuracy_validation_metrics.json', {'epoch': epoch, **val})
            del state
            entry['checkpoint_seconds'] = time.perf_counter() - checkpoint_start
            entry['epoch_seconds'] = time.perf_counter() - epoch_start
            entry['seconds'] = entry['epoch_seconds']
            efficiency['checkpoint_seconds'] += entry['checkpoint_seconds']
            efficiency['stage_wall_seconds'] = previous_wall + time.perf_counter() - session_start
            entry['cumulative_stage_seconds'] = efficiency['stage_wall_seconds']
            dump(out / 'latest_validation_metrics.json', val)
            dump(out / 'history.json', history)
            dump(out / 'efficiency.json', efficiency)
            tracker.epoch(entry)
            print(json.dumps({'event': 'epoch_completed', **entry, 'best_epoch': best_epoch}), flush=True)
            if stop['requested'] or (out / 'STOP').exists() or (stop_file and stop_file.exists()):
                result = {'seed': seed, 'status': 'stopped', 'epoch': epoch, 'efficiency': efficiency}
                dump(out / 'results.json', result)
                tracker.summary({'status': 'stopped'})
                failed = False
                return result
        best_validation = json.loads((out / 'best_validation_metrics.json').read_text())
        result = {'seed': seed, 'status': 'completed', 'stopped_epoch': history[-1]['epoch'],
                  'best_epoch': best_epoch, 'best_validation': best_validation, 'config': config}
        if 'test' in loaders:
            # Test exactly once after selection on validation, never use it for early stopping.
            synchronize(device)
            test_start = time.perf_counter()
            selected = load_checkpoint(out / 'best.pt', trust_legacy)
            model.load_state_dict(selected['model_state_dict'])
            del selected
            metrics, _, _ = evaluate(model, loaders['test'], device, labels)
            synchronize(device)
            efficiency['test_seconds'] += time.perf_counter() - test_start
            result['test'] = metrics
            dump(out / 'test_metrics.json', {'checkpoint': 'best.pt', 'selected_epoch': best_epoch, **metrics})
        efficiency['stage_wall_seconds'] = previous_wall + time.perf_counter() - session_start
        efficiency.update(mean_epoch_seconds=sum(h.get('epoch_seconds', h.get('seconds', 0.)) for h in history) / len(history),
                          samples_per_second=efficiency['samples_seen'] / max(efficiency['train_seconds'], 1e-12),
                          seconds_per_step=efficiency['train_seconds'] / max(efficiency['optimizer_steps'], 1))
        target_accuracy = getattr(args, 'target_accuracy', 0.8)
        reached = next((h for h in history if h['val_accuracy'] >= target_accuracy and 'cumulative_train_seconds' in h), None)
        efficiency['target_accuracy'] = target_accuracy
        efficiency['seconds_to_target_accuracy'] = reached['cumulative_train_seconds'] if reached else None
        efficiency['stage_seconds_to_target_accuracy'] = reached['cumulative_stage_seconds'] if reached else None
        pretraining = getattr(args, 'pretraining_efficiency', None) or {}
        comparison = {
            'variant': config['variant'], 'stage': config['stage'], 'test_accuracy': result.get('test', {}).get('accuracy'),
            'test_macro_f1': result.get('test', {}).get('macro_f1'), 'best_validation_accuracy': best_validation['accuracy'],
            'total_train_seconds': efficiency['train_seconds'] + pretraining.get('train_seconds', 0.),
            'total_stage_wall_seconds': efficiency['stage_wall_seconds'] + pretraining.get('stage_wall_seconds', 0.),
            'stage2_train_seconds': efficiency['train_seconds'] if init else None,
            'stage2_wall_seconds': efficiency['stage_wall_seconds'] if init else None,
            'stage2_samples_per_second': efficiency['samples_per_second'] if init else None,
            'stage2_seconds_per_step': efficiency['seconds_per_step'] if init else None,
            'stage2_mean_epoch_seconds': efficiency['mean_epoch_seconds'] if init else None,
            'stage2_peak_allocated_gpu_gib': efficiency['peak_allocated_gpu_gib'] if init else None,
            'peak_allocated_gpu_gib': max(efficiency['peak_allocated_gpu_gib'], pretraining.get('peak_allocated_gpu_gib', 0.)),
            'trainable_parameters': config['trainable_parameters'],
            'pretraining_reused': getattr(args, 'pretraining_reused', False),
            'pretraining_train_seconds': pretraining.get('train_seconds', 0.),
            'timing_complete': efficiency['timing_complete'] and pretraining.get('timing_complete', True),
            'seven_class_labeled_train_samples': len(sets['train']) if config['num_classes'] == 7 else 0,
            'binary_labeled_train_samples': getattr(args, 'pretraining_sample_count',
                                                   len(sets['train']) if config['num_classes'] == 2 else 0),
            'total_train_seconds_to_target_accuracy': (efficiency['seconds_to_target_accuracy'] +
                pretraining.get('train_seconds', 0.)) if reached else None,
        }
        result.update(efficiency=efficiency, comparison=comparison)
        dump(out / 'efficiency.json', efficiency)
        # A tracking failure must not leave a reusable 'completed' marker.
        dump(out / 'results.json', {**result, 'status': 'finalizing_tracking'})
        tracker.summary({'status': 'completed', 'best_epoch': best_epoch,
                         'selected_validation': best_validation,
                         **({'test': result['test']} if 'test' in result else {}),
                         **{f'comparison/{k}': v for k, v in comparison.items() if v is not None},
                         **{f'stage_total/{k}': v for k, v in efficiency.items() if v is not None}})
        tracker.finish()
        dump(out / 'results.json', result)
        print(json.dumps({'event': 'completed', 'seed': seed, 'comparison': comparison}), flush=True)
        failed = False
        return result
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        tracker.finish(failed=failed)
