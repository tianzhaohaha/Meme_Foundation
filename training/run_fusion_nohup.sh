#!/usr/bin/env bash
# Detach the V5 -> V6 queue; supports the same status/stop/resume commands.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action=start
case "${1:-}" in
  start|status|stop|-h|--help) action="$1"; shift ;;
esac
exec bash "$SCRIPT_DIR/run_all_nohup.sh" "$action" --suite fusion "$@"
