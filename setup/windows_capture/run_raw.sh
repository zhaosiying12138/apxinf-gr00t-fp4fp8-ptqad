#!/usr/bin/env bash
# Capture provenance without replacing, selecting, or reformatting command output.
set -uo pipefail
script=$1
marker=$2
log=$3
bash -x "$script" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$marker"
sleep 120
exit "$status"
