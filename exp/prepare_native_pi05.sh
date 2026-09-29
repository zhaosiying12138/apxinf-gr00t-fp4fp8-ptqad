#!/usr/bin/env bash
# CPU-only pi0.5 native artifact preparation. No CUDA import or model inference.
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PTQAD_NATIVE_PYTHON:-"$PROJECT/third_party/apxinf-robo/.venv/bin/python"}
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec "$PYTHON" "$PROJECT/exp/prepare_native_pi05.py" "$@"
