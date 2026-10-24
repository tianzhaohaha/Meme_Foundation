#!/usr/bin/env bash
# V6: retain source encoders/fusion, freeze encoders, train fusion and fresh head.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment_env.bash"
exec python -u "$SCRIPT_DIR/experiment.py" "$@" --variant tune_fusion
