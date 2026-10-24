#!/usr/bin/env bash
# V5: retain source ViT/RoBERTa encoders; fresh fusion/head; fine-tune all parameters.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment_env.bash"
exec python -u "$SCRIPT_DIR/experiment.py" "$@" --variant reset_fusion
