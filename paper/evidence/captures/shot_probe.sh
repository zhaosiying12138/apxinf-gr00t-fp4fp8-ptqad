#!/usr/bin/env bash
# Real CPU tests of the current probe implementation; no model evaluation.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export GR00T_REPO=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
cd /home/zhaosiying/codebase/fp4vla
/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python -m unittest tests.test_probe_distill tests.test_gr00t_runtime
