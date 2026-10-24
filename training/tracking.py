"""Local metrics plus optional W&B tracking. No credentials are stored in configs."""
import json
import os
import uuid
from pathlib import Path
from experiment_data import write_json


def add_tracking_arguments(parser):
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                        default=os.environ.get('WANDB_MODE', 'online'))
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'tumemo-training-comparison'))
    parser.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--wandb-group', default=os.environ.get('WANDB_RUN_GROUP', 'three-strategies'))
    parser.add_argument('--log-every', type=int, default=25)


class Tracker:
    def __init__(self, args, out, config):
        self.run = None
        self.out = Path(out)
        self.mode = getattr(args, 'wandb_mode', 'disabled')
        if self.mode == 'disabled':
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError('Install requirements.txt to enable W&B, or pass --wandb-mode disabled.') from exc
        identity = self.out / 'wandb_run.json'
        previous = json.loads(identity.read_text()) if identity.exists() and args.resume else None
        project = args.wandb_project
        entity = args.wandb_entity
        if previous and (previous['project'], previous['entity']) != (project, entity):
            raise ValueError('Resume must keep the same W&B project/entity')
        # Offline runs cannot resume the W&B history. Start a linked segment and
        # replay completed epochs; local checkpoint counters remain cumulative.
        run_id = previous['id'] if previous and self.mode == 'online' else uuid.uuid4().hex[:12]
        extra = {'continued_from_run': previous['id']} if previous else {}
        self.run = wandb.init(
            project=project, entity=entity, group=args.wandb_group,
            name=f"{config['variant']}-{config['stage']}-seed{config['seed']}",
            job_type=config['stage'], tags=[config['variant'], config['architecture']],
            config={**config, **extra}, dir=str(self.out), mode=self.mode,
            id=run_id, resume='allow' if previous and self.mode == 'online' else None,
        )
        write_json(identity, {'id': self.run.id, 'project': project, 'entity': entity,
                              'mode': self.mode, 'url': self.run.url})
        for axis in ('epoch', 'global_step', 'time/train_seconds', 'time/stage_seconds'):
            self.run.define_metric(axis)
        for pattern, axis in [('train/*', 'global_step'), ('validation/*', 'epoch'),
                              ('efficiency/*', 'epoch'), ('accuracy_vs_train_time/*', 'time/train_seconds'),
                              ('accuracy_vs_stage_time/*', 'time/stage_seconds')]:
            self.run.define_metric(pattern, step_metric=axis)
        self.run.define_metric('validation/accuracy', summary='max')
        self.run.define_metric('validation/macro_f1', summary='max')

    def log(self, metrics):
        if self.run:
            self.run.log(metrics)

    def epoch(self, entry):
        self.log({
            'epoch': entry['epoch'], 'global_step': entry['global_step'],
            'time/train_seconds': entry['cumulative_train_seconds'],
            'time/stage_seconds': entry['cumulative_stage_seconds'],
            'train/loss': entry['train_loss'], 'train/objective_loss': entry['train_objective_loss'],
            'train/head_lr': entry['head_lr'], 'train/encoder_lr': entry['encoder_lr'],
            'validation/loss': entry['val_loss'], 'validation/accuracy': entry['val_accuracy'],
            'validation/macro_f1': entry['val_macro_f1'],
            'accuracy_vs_train_time/accuracy': entry['val_accuracy'],
            'accuracy_vs_stage_time/accuracy': entry['val_accuracy'],
            **{f'efficiency/{k}': entry[k] for k in (
                'train_seconds', 'validation_seconds', 'epoch_seconds', 'checkpoint_seconds',
                'samples_per_second', 'seconds_per_step', 'peak_allocated_gpu_gib', 'peak_reserved_gpu_gib',
                'process_peak_rss_gib')},
            **{f'validation/recall_{k}': v for k, v in entry['per_class_recall'].items()},
        })

    def summary(self, values):
        if self.run:
            self.run.summary.update(values)

    def finish(self, failed=False):
        if self.run:
            run, self.run = self.run, None
            run.finish(exit_code=1 if failed else 0)
