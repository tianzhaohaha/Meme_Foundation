#!/usr/bin/env bash
# One detached queue, sequential experiments. Arguments are arrays, never eval.
set -euo pipefail
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/run_all_nohup.sh"

usage() {
  cat <<'HELP'
Usage:
  bash run_all_nohup.sh [start] [--suite baseline|fusion|all] [--log-dir DIR] [--] [experiment options]
  bash run_all_nohup.sh status [--suite baseline|fusion|all] [--log-dir DIR]
  bash run_all_nohup.sh stop [--suite baseline|fusion|all] [--log-dir DIR]

Examples:
  bash run_all_nohup.sh
  bash run_all_nohup.sh -- --resume
  WANDB_MODE=offline bash run_all_nohup.sh -- --batch-size 16
  bash run_all_nohup.sh --log-dir /path/to/logs -- --output /path/to/results

Default suite runs v1 -> v2 -> v3; fusion suite runs v5 -> v6.
The all suite runs v1 -> v2 -> v3 -> v4 -> v5 -> v6 for a fresh reproduction.
Runs sequentially with nohup + setsid. Stops on any failure or
graceful training stop. Reconnect using 'status' and tail the printed log paths.
Use the same options and --resume to continue an interrupted queue.
HELP
}

write_state() {
  printf '%s\n' "$*" > "$LOG_DIR/status.tmp"
  mv -f -- "$LOG_DIR/status.tmp" "$LOG_DIR/status"
}

# PID identity includes the Linux process start time, guarding against PID reuse.
process_stamp() { awk '$3 == "Z" {exit 1} {print $22}' "/proc/$1/stat" 2>/dev/null; }
queue_alive() {
  [[ -s "$LOG_DIR/queue.pid" ]] || return 1
  read -r QUEUE_PID QUEUE_STAMP < "$LOG_DIR/queue.pid"
  [[ "$QUEUE_PID" =~ ^[0-9]+$ && "$QUEUE_STAMP" =~ ^[0-9]+$ ]] || return 1
  [[ "$(process_stamp "$QUEUE_PID")" == "$QUEUE_STAMP" ]] && kill -0 "$QUEUE_PID" 2>/dev/null
}

if [[ "${1:-}" == --worker ]]; then
  LOG_DIR="$2"
  SUITE="$3"
  shift 3
  child=''
  stopped=0
  stop_queue() {
    stopped=1
    touch "$LOG_DIR/STOP"
    write_state stopping
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi
  }
  trap stop_queue TERM INT
  # SIGHUP remains ignored, inherited from nohup.
  printf '%s %s\n' "$$" "$(process_stamp "$$")" > "$LOG_DIR/queue.pid.tmp"
  mv -f -- "$LOG_DIR/queue.pid.tmp" "$LOG_DIR/queue.pid"
  write_state running
  case "$SUITE" in
    baseline) scripts=(run_v1_direct.sh run_v2_finetune.sh run_v3_frozen.sh) ;;
    fusion) scripts=(run_v5_reset_fusion.sh run_v6_tune_fusion.sh) ;;
    all) scripts=(run_v1_direct.sh run_v2_finetune.sh run_v3_frozen.sh run_v4_direct_5k.sh run_v5_reset_fusion.sh run_v6_tune_fusion.sh) ;;
    *) echo "Unknown suite: $SUITE" >&2; exit 2 ;;
  esac
  for script in "${scripts[@]}"; do
    if [[ "$stopped" == 1 || -e "$LOG_DIR/STOP" ]]; then
      write_state stopped
      exit 130
    fi
    stage_log="$LOG_DIR/${script%.sh}.log"
    write_state "running $script"
    printf '[%s] Starting %s\n' "$(date -u +%FT%TZ)" "$script"
    printf 'Command: '
    printf '%q ' bash "$SCRIPT_DIR/$script" "$@" --stop-file "$LOG_DIR/STOP"
    printf '\n'
    printf '\n[%s] New attempt\n' "$(date -u +%FT%TZ)" >> "$stage_log"
    bash "$SCRIPT_DIR/$script" "$@" --stop-file "$LOG_DIR/STOP" >> "$stage_log" 2>&1 < /dev/null &
    child=$!
    while true; do
      if wait "$child"; then rc=0; break; else rc=$?; fi
      # A trapped signal interrupts wait; keep the lock until the child saves/exits.
      if [[ "$stopped" == 1 ]] && kill -0 "$child" 2>/dev/null; then continue; fi
      break
    done
    child=''
    if [[ "$stopped" == 1 || "$rc" == 130 || -e "$LOG_DIR/STOP" ]]; then
      write_state "stopped $script"
      exit 130
    fi
    if [[ "$rc" != 0 ]]; then
      write_state "failed $script exit=$rc"
      printf 'Failed: %s (exit %s). See %s\n' "$script" "$rc" "$stage_log" >&2
      exit "$rc"
    fi
    printf '[%s] Completed %s\n' "$(date -u +%FT%TZ)" "$script"
  done
  write_state completed
  exit 0
fi

action=start
case "${1:-}" in
  start|status|stop) action="$1"; shift ;;
  -h|--help) usage; exit 0 ;;
esac
LOG_DIR=''
SUITE=baseline
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --log-dir|--suite)
      [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
      if [[ "$1" == --log-dir ]]; then LOG_DIR="$2"; else SUITE="$2"; fi
      shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
case "$SUITE" in
  baseline) LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/background_comparison}" ;;
  fusion) LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/background_fusion}" ;;
  all) LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/background_all}" ;;
  *) echo "Unknown suite: $SUITE" >&2; exit 2 ;;
esac
mkdir -p -- "$LOG_DIR"
LOG_DIR="$(cd -- "$LOG_DIR" && pwd)"
if [[ "$action" == status ]]; then
  if queue_alive; then printf 'Queue running, PID %s\n' "$QUEUE_PID"; else printf 'Queue is not running.\n'; fi
  if [[ -f "$LOG_DIR/status" ]]; then cat "$LOG_DIR/status"; fi
  printf 'Logs: %s\n' "$LOG_DIR"
  exit 0
fi
if [[ "$action" == stop ]]; then
  if ! queue_alive; then echo 'No active queue.'; exit 1; fi
  touch "$LOG_DIR/STOP"
  kill -TERM "$QUEUE_PID"
  echo 'Stop requested. Waiting in the background for the current epoch to save; no next experiment will start.'
  exit 0
fi
for command in nohup setsid flock; do
  command -v "$command" > /dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
exec 9>"$LOG_DIR/queue.lock"
if ! flock -n 9; then echo "A queue is already active. Logs: $LOG_DIR" >&2; exit 1; fi
resume=0
for arg in "$@"; do
  [[ "$arg" != --resume ]] || resume=1
  case "$arg" in
    --variant|--variant=*|--stop-file|--stop-file=*) echo "Queue controls $arg; remove this option." >&2; exit 2 ;;
  esac
done
if [[ -e "$LOG_DIR/STOP" ]]; then
  [[ "$resume" == 1 ]] || { echo 'Queue has a STOP marker; use --resume to continue.' >&2; exit 1; }
  rm -f -- "$LOG_DIR/STOP"
fi
: > "$LOG_DIR/queue.pid"
write_state starting
# FD 9 is inherited by the worker and remains locked for the entire queue.
nohup setsid bash "$SCRIPT_PATH" --worker "$LOG_DIR" "$SUITE" "$@" >> "$LOG_DIR/queue.log" 2>&1 < /dev/null &
for ((attempt=0; attempt<50; attempt++)); do
  [[ ! -s "$LOG_DIR/queue.pid" ]] || break
  sleep 0.1
done
if [[ ! -s "$LOG_DIR/queue.pid" ]]; then echo "Worker failed to start. See $LOG_DIR/queue.log" >&2; exit 1; fi
if ! queue_alive; then
  state="$(cat "$LOG_DIR/status")"
  [[ "$state" == completed ]] || { echo "Startup failed: $state. See $LOG_DIR" >&2; exit 1; }
fi
printf 'Detached queue started. Logs: %s\n' "$LOG_DIR"
printf 'Status: bash %q status --suite %q --log-dir %q\n' "$SCRIPT_PATH" "$SUITE" "$LOG_DIR"
printf 'Follow: tail -f %q\n' "$LOG_DIR/queue.log"
