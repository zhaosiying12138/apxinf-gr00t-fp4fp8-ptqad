#!/usr/bin/env bash
# Rebuild the seven CategorySpecificLinear NVFP4 banks on the current PTQ
# parent.  The calibration cache is deliberately bound to that parent's
# recipe and bake manifest; an older cache is rejected by bake_category.py.
set -euo pipefail

PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GR00T_REPO=${GR00T_REPO:-$HOME/codebase/groot-fsdp2/Isaac-GR00T}
PY=${PTQAD_PYTHON:-$GR00T_REPO/.venv/bin/python}
PTQ_PARENT=${PTQ_PARENT:?Set the current pure NVFP4 PTQ parent}
CATEGORY_CALIB_OUT=${CATEGORY_CALIB_OUT:?Set a new category calibration directory}
CATEGORY_OUT=${CATEGORY_OUT:?Set a new full-category W4A4 checkpoint directory}
DATASET=${QAD_DATASET:-$GR00T_REPO/demo_data/libero_demo}
WINDOWS=${CATEGORY_CALIB_WINDOWS:-128}
BATCH=${CATEGORY_CALIB_BATCH:-1}
SEED=${CATEGORY_CALIB_SEED:-20261003}

test ! -e "$CATEGORY_CALIB_OUT" || { echo "CATEGORY_CALIB_OUT already exists" >&2; exit 1; }
test ! -e "$CATEGORY_OUT" || { echo "CATEGORY_OUT already exists" >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PTQAD_LOCAL_HF_METADATA=1
export LD_LIBRARY_PATH=${PTQAD_MEDIA_LIB:-${LD_LIBRARY_PATH:-$HOME/miniforge3/envs/media7/lib}}

cd "$GR00T_REPO"
"$PY" "$PROJECT/quant/ptq/collector_category.py" \
  --parent "$PTQ_PARENT" --out "$CATEGORY_CALIB_OUT" --dataset "$DATASET" \
  --windows "$WINDOWS" --batch "$BATCH" --seed "$SEED" --device cuda

"$PY" "$PROJECT/quant/ptq/bake_category.py" \
  --parent "$PTQ_PARENT" --calib "$CATEGORY_CALIB_OUT" --out "$CATEGORY_OUT" \
  --expected-windows "$WINDOWS" --gptq-damp 0.01

echo "Full-category W4A4 checkpoint: $CATEGORY_OUT"
