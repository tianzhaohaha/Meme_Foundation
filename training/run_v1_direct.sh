#!/usr/bin/env bash
# ViT + RoBERTa: seven classes, all 20,000 + 5,000 training samples.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment_env.bash"
exec python -u "$SCRIPT_DIR/experiment.py" "$@" --variant direct
