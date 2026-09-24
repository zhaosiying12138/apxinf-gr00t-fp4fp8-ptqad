#!/usr/bin/env bash
# Environment snapshot -> results/env/ (committed: small text files)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/results/env"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/results/env/snapshot_${STAMP}.txt"
{
  echo "== date: $(date -Is)"
  echo "== uname: $(uname -a)"
  echo "== gpu:"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit,clocks.max.sm --format=csv
  echo "== nvcc:"; /usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2 || nvcc --version | tail -2
  echo "== rust:"; (~/.cargo/bin/rustc --version 2>/dev/null || rustc --version)
  echo "== cpu: $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
  echo "== mem:"; free -h | head -2
  echo "== disk:"; df -h "$ROOT" | tail -1
  echo "== python deps (repo venv):"
  if [ -f "$ROOT/third_party/apxinf-robo/.venv/bin/python" ]; then
    "$ROOT/third_party/apxinf-robo/.venv/bin/python" -m pip list 2>/dev/null | grep -Ei "apxinf|torch|numpy|lerobot|libero|openpi" || true
  fi
  echo "== engine commit:"
  git -C "$ROOT/third_party/apxinf-robo" log --oneline -1
  git -C "$ROOT/third_party/apxinf-robo/apxinf" log --oneline -1
} > "$OUT" 2>&1
echo "wrote $OUT"
cat "$OUT"
