#!/usr/bin/env bash
# After the main five-arm round; caller must exclusively own the GPU.
# Arguments: --plan FROZEN.json --round ROUND_DIR --out NEW_DIR [--port 5595]
# --dry-run validates provenance and prints commands without launching a GPU job.
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${PTQAD_MEDIA_LIB:-$HOME/miniforge3/envs/media7/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 "$PROJECT/eval/compare_ptq_frontier.py" run "$@"
