#!/usr/bin/env bash
# Evaluate both model sizes on one GPU, sequentially, with the same protocol.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment_env.bash"
for model in 4b 8b; do
  python -u "$SCRIPT_DIR/evaluate_vlm.py" --model "$model" "$@"
done
