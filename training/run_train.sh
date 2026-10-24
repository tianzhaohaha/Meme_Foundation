#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -lt 1 ]; then echo 'Usage: bash run_train.sh /absolute/path/subset_50000.zip [--resume runs_binary/seed_42/last.pt] [--wandb-mode online|offline|disabled]'; exit 2; fi
DATA="$1"
shift
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-tumemo-training-comparison}"
export WANDB_MODE="${WANDB_MODE:-online}"
python -u train.py --data "$DATA" --cache prepared --output runs_binary --seeds 42 --epochs 20 --patience 4 --batch-size 16 --max-length 96 --encoder-lr 5e-6 --head-lr 1e-4 --dropout 0.5 --weight-decay 0.01 --warmup-ratio 0.05 "$@" 2>&1 | tee -a training.log
