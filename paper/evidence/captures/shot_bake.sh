#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
cd /home/zhaosiying/codebase/fp4vla
set -x
python3 paper/build_recipe_inventory_v11.py \
  --checkpoint results/ptqad_20261003/w4a4_full_category \
  --recovery-manifest results/ptqad_20261003/v11_recovery_r4/artifacts/train_qad_lr_0.0001/recovery_manifest.json \
  --out paper/_build/capture_v11_20261004/inventory.json
sha256sum exp/recovery_protocol_v11_w4a4_category.json \
  results/ptqad_20261003/w4a4_full_category/category_ptq_recipe.json
