#!/usr/bin/env bash
export SHOT_GPU_RELEASED=0
#!/usr/bin/env bash
# Prepared smoke evidence only; formal experiments and their outputs are read-only.
set -euo pipefail
PROJECT=/home/zhaosiying/codebase/fp4vla
SCRATCH=/home/zhaosiying/codebase/fp4vla/paper/_build/capture_batch/scratch/20260929_r4
FIGURE=shot_qat
GR00T_REPO=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
PTQAD_PYTHON="$GR00T_REPO/.venv/bin/python"
LIBERO_PYTHON="$GR00T_REPO/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
NATIVE_PY="$PROJECT/third_party/apxinf-robo/.venv/bin/python"
BF16_TEACHER="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
NATIVE_PI05_MODEL="$PROJECT/weights/native_pi05_20260929/model"
export GR00T_REPO PTQAD_PYTHON LIBERO_PYTHON
export GR00T_BACKBONE_MODEL="$PROJECT/weights/nvidia/Cosmos-Reason2-2B"
export PTQAD_MEDIA_LIB=/home/zhaosiying/miniforge3/envs/media7/lib
export LD_LIBRARY_PATH="$PTQAD_MEDIA_LIB:/usr/local/cuda/lib64:/usr/lib/wsl/lib"
export PATH="${PTQAD_MEDIA_LIB%/lib}/bin:/usr/local/cuda/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export QAD_DATASET="$GR00T_REPO/demo_data/libero_demo"
unset QAD_BSZ QAD_ACC QAD_ACCUM_STEPS QAD_INIT_ADAPTER QAD_OPD_KL_W OPD_CACHE_PATH
unset OPD_CAPTURE_DIR FP4VLA_LOG_DIR FP4VLA_SCOPE
export QAD_LORA_R=32 QAD_LORA_ALPHA=64 QAD_LORA_SCOPE=head+lang_all
export QAD_MICRO_BATCH=1 QAD_GLOBAL_BATCH=2 QAD_ACTIVATION_CHECKPOINTING=1 QAD_LR=1e-4
export QAD_OPD_MSE_W=0 OPD_EVERY=4 FP4VLA_QUANT=0
export PYTHONPATH="$PROJECT:$GR00T_REPO"
test -x "$PTQAD_PYTHON"
test -f "$BF16_TEACHER/config.json"
test -d "$QAD_DATASET"
test -e "$GR00T_BACKBONE_MODEL"
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$SCRATCH/stages"
mkdir "$SCRATCH/stages/$FIGURE"
date -u +%FT%TZ > "$SCRATCH/stages/$FIGURE/started_utc.txt"
cd "$PROJECT"
"$PTQAD_PYTHON" -m unittest discover -s tests -p test_activation_checkpoint.py -v

date -u +%FT%TZ > "$SCRATCH/stages/$FIGURE/completed_utc.txt"
