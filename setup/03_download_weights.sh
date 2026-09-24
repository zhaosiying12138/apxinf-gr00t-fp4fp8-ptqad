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

HF() {  # HF <repo-id> <local-dir> [extra include/exclude args...]
  local tries=0
  until [ $tries -ge 5 ]; do
    if uvx --from huggingface_hub hf download "$1" --local-dir "$2" "${@:3}"; then return 0; fi
    tries=$((tries+1)); echo "retry $tries for $1 (switching endpoint)"; sleep 3
    export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  done
  return 1
}

# GR00T-N1.7-LIBERO repo ships ALL suites + optimizer states (~100GB).
# Inference needs only libero_10 model shards + configs (~6.5GB):
HF nvidia/GR00T-N1.7-LIBERO "$W/GR00T-N1.7-LIBERO" \
   --include "libero_10/*" \
   --exclude "libero_10/global_step18000/*" "libero_10/rng_state_*.pth"
# restorable extras (only if fine-tuning from this ckpt): drop the --exclude flags.
HF nvidia/Cosmos-Reason2-2B   "$W/Cosmos-Reason2-2B"
HF lerobot/pi05_libero_base   "$W/pi05_libero_base"

echo; echo "== norm stats + tokenizer for pi0.5 =="
curl -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  -o "$W/pi05_libero_base/norm_stats.json"
# PaliGemma SentencePiece tokenizer is NOT distributed with pi05 checkpoints (openpi fetches
# it from big_vision at runtime); apxinf never downloads -> put it in place once:
curl -fL https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o "$W/pi05_libero_base/paligemma_tokenizer.model"

du -sh "$W"/*
echo "WEIGHTS DONE"
