#!/usr/bin/env bash
# Download HF weights (restorable data). ~15-25 GB total. Resumable; re-run to fix.
# weights/ layout mirrors the paths used in our configs:
#   weights/GR00T-N1.7-LIBERO        (primary VLA, LIBERO-finetuned N1.7)
#   weights/Cosmos-Reason2-2B        (backbone processor snapshot, required by gr00t loader)
#   weights/pi05_libero_base         (second model: pi0.5)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
W="$ROOT/weights"; mkdir -p "$W"

HF() {  # HF <repo-id> <local-dir>   (no excludes: .pth IS the checkpoint for GR00T repos)
  local tries=0
  until [ $tries -ge 5 ]; do
    if uvx --from huggingface_hub hf download "$1" --local-dir "$2"; then return 0; fi
    tries=$((tries+1)); echo "retry $tries for $1 (switching endpoint)"; sleep 3
    export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  done
  return 1
}

HF nvidia/GR00T-N1.7-LIBERO   "$W/GR00T-N1.7-LIBERO"
HF nvidia/Cosmos-Reason2-2B   "$W/Cosmos-Reason2-2B"
HF lerobot/pi05_libero_base   "$W/pi05_libero_base"

echo; echo "== norm stats (openpi) for pi0.5 =="
mkdir -p "$W/pi05_norm_stats"
curl -sL -o "$W/pi05_norm_stats/norm_stats.json" \
  https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/assets/normalized_stats.json

du -sh "$W"/*
echo "WEIGHTS DONE"
