#!/usr/bin/env bash
# Caller must exclusively own the GPU. --dry-run only prints the possible plan.
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${PTQAD_MEDIA_LIB:-$HOME/miniforge3/envs/media7/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 "$PROJECT/exp/run_development.py" "$@"
