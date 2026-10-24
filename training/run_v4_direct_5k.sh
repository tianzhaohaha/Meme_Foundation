#!/usr/bin/env bash
# ViT + RoBERTa: direct seven-class training on the fixed 5,000 target samples.
# Uses generic pretrained encoders, with no binary pretraining or frozen layers.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment_env.bash"
exec python -u "$SCRIPT_DIR/experiment.py" "$@" --variant direct_5k
