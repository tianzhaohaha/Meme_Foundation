#!/usr/bin/env bash
# Shared binary pretraining on 20,000; full seven-class fine-tuning on 5,000.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment_env.bash"
exec python -u "$SCRIPT_DIR/experiment.py" "$@" --variant full
