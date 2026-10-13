#!/usr/bin/env bash
# Usage: bash scripts/launch_recap.sh --model ... --input ... [evaluate options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == "--worker" ]]; then
  RUN_TAG="$2"
  STATUS_FILE="$3"
  shift 3
  on_exit() {
    code=$?
    trap - EXIT
    printf 'exit_code=%s\nfinished_at=%s\nrun_tag=%s\n' \
      "$code" "$(date +%Y-%m-%dT%H:%M:%S%z)" "$RUN_TAG" > "$STATUS_FILE"
    exit "$code"
  }
  trap on_exit EXIT
  echo "started_at=$(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "run_tag=$RUN_TAG"
  echo "status_file=$STATUS_FILE"
  printf 'evaluate_args='; printf '%q ' "$@"; printf '\n'
  set +e
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN:-python}" -m recap evaluate "$@"
  code=$?
  set -e
  exit "$code"
fi
command -v tmux >/dev/null
RUN_TAG="$(date +%Y%m%d-%H%M%S)-recap-$$"
SESSION="vlmprid-${RUN_TAG}"
mkdir -p outputs/logs/eval
LOG="outputs/logs/eval/${RUN_TAG}.launcher.log"
STATUS="outputs/logs/eval/${RUN_TAG}.status"
printf -v cmd 'cd %q && PYTHON_BIN=%q bash scripts/launch_recap.sh --worker %q %q' \
  "$PWD" "${PYTHON_BIN:-python}" "$RUN_TAG" "$STATUS"
for arg in "$@"; do printf -v cmd '%s %q' "$cmd" "$arg"; done
printf -v cmd '%s > %q 2>&1' "$cmd" "$LOG"
tmux new-session -d -s "$SESSION" "$cmd"
printf 'tmux_session=%s\nrun_tag=%s\nlauncher_log=%s\nstatus_file=%s\n' \
  "$SESSION" "$RUN_TAG" "$LOG" "$STATUS"
sleep 1
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tail -n 10 "$LOG"
elif [[ -f "$STATUS" ]]; then
  cat "$STATUS"
  if ! grep -qx 'exit_code=0' "$STATUS"; then tail -n 40 "$LOG"; exit 1; fi
else
  echo "Launch failed before status was recorded: $LOG" >&2
  exit 1
fi
