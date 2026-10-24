#!/usr/bin/env bash
# Shared binary pretraining; freeze encoders/fusion, train seven-class head on 5,000.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment_env.bash"
exec python -u "$SCRIPT_DIR/experiment.py" "$@" --variant frozen
