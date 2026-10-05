#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/common_v11.sh"
OUT="$(prepare_scratch shot_qad)"; assert_no_compute_apps
cd "$GR00T_REPO"
export GR00T_BASE_CKPT="$PTQ_BASE" QAD_OUT="$OUT/qad" QAD_DATASET="$DATASET" QAD_CAPTURE_DATASET="$CAPTURE_DATASET" QAD_CAPTURE_DATASET_SHA256="$CAPTURE_DATASET_SHA256"
export QAD_STEPS=2 QAD_SAVE_STEPS=2 QAD_SAVE_TOTAL_LIMIT=2 QAD_MICRO_BATCH=1 QAD_GLOBAL_BATCH=1 QAD_ACTIVATION_CHECKPOINTING=1 QAD_OPD_MSE_W=0 TRAIN_SEED=20261003 PROTOCOL_FILE="$PROTOCOL_FILE"
export QAD_LORA_R=32 QAD_LORA_ALPHA=64.0 QAD_LORA_SCOPE=all_ordinary_linear QAD_LR=0.0001 QAD_W4A4=1 FP4VLA_QUANT=0 FP4VLA_W4A4=1 FP4VLA_SATURATE_F16_ACTIVATIONS=0
unset QAD_INIT_ADAPTER QAD_MAX_GRAD_NORM OPD_CACHE_PATH || true
"$PTQAD_PYTHON" -u "$PROJECT/rl/lora_qad.py" 2>&1 | tee "$OUT/qad_2step.raw.log"
test -f "$OUT/qad/checkpoint-2/recovery_manifest.json"
printf '%s
' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
