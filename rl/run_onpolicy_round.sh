#!/usr/bin/env bash
# One audited collect -> label -> paired continuation -> heldout round.
# Caller must own the GPU exclusively. This script never runs GPU jobs in parallel.
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GR00T=${GR00T_REPO:-$HOME/codebase/groot-fsdp2/Isaac-GR00T}
PY=${PTQAD_PYTHON:-${TRAIN_PYTHON:-$GR00T/.venv/bin/python}}
: "${PTQ_BASE:?Set the corrected PTQ checkpoint used to train QAD}"
: "${BF16_TEACHER:?Set the original full BF16 LIBERO checkpoint}"
: "${QAD_ADAPTER:?Set the initial QAD checkpoint containing lora_A/B}"
: "${QAD_MERGED:?Set the initial QAD merged BF16 checkpoint}"
: "${ROUND_OUT:?Set a new output directory}"
COLLECTION_SEED=${COLLECTION_SEED:-110000}
HELDOUT_SEED=${HELDOUT_SEED:-220000}
EXTRA_STEPS=${EXTRA_STEPS:-100}
COLLECTION_EPISODES=${COLLECTION_EPISODES:-2}
HELDOUT_EPISODES=${HELDOUT_EPISODES:-10}
PORT=${PORT:-5595}
export QAD_MICRO_BATCH=${QAD_MICRO_BATCH:-1}
export QAD_GLOBAL_BATCH=${QAD_GLOBAL_BATCH:-16}
export QAD_ACTIVATION_CHECKPOINTING=${QAD_ACTIVATION_CHECKPOINTING:-1}
export QAD_LR=${QAD_LR:-1e-4}
export LD_LIBRARY_PATH=${PTQAD_MEDIA_LIB:-${LD_LIBRARY_PATH:-$HOME/miniforge3/envs/media7/lib}}
export HF_HUB_OFFLINE=1
test ! -e "$ROUND_OUT" || { echo "ROUND_OUT already exists; use a new evidence directory" >&2; exit 1; }
mkdir -p "$ROUND_OUT"
ROUND_OUT=$(realpath "$ROUND_OUT")
cd "$GR00T"

python3 "$PROJECT/eval/run_recovery_eval.py" \
  --checkpoint "$QAD_MERGED" --out "$ROUND_OUT/collection" --purpose collection \
  --seed "$COLLECTION_SEED" --episodes "$COLLECTION_EPISODES" --port "$PORT" \
  > "$ROUND_OUT/collection.log" 2>&1

"$PY" "$PROJECT/rl/opd_probe_cache.py" --teacher "$BF16_TEACHER" \
  --input-dir "$ROUND_OUT/collection/observations" --count 160 \
  --out "$ROUND_OUT/teacher_probes.pt" > "$ROUND_OUT/teacher_labeling.log" 2>&1

# Shared frozen base and exact same initial LoRA tensors. Both arms start fresh
# Adam states and LR schedules, and consume the same additional demo step budget.
# OPD adds probe compute; this is not an equal-compute comparison.
for arm in continued_qad qad_opd; do
  if [[ "$arm" == qad_opd ]]; then mse_weight=${OPD_WEIGHT:-1.0}; else mse_weight=0; fi
  GR00T_BASE_CKPT="$PTQ_BASE" QAD_INIT_ADAPTER="$QAD_ADAPTER" \
    QAD_OUT="$ROUND_OUT/$arm" QAD_STEPS="$EXTRA_STEPS" QAD_SAVE_STEPS="$EXTRA_STEPS" \
    QAD_OPD_MSE_W="$mse_weight" OPD_CACHE_PATH="$ROUND_OUT/teacher_probes.pt" \
    OPD_EVERY=${OPD_EVERY:-4} "$PY" "$PROJECT/rl/lora_qad.py" \
    > "$ROUND_OUT/$arm.train.log" 2>&1
  "$PY" "$PROJECT/rl/lora_merge_bake.py" \
    --base "$PTQ_BASE" --ckpt "$ROUND_OUT/$arm/checkpoint-$EXTRA_STEPS" \
    --out "$ROUND_OUT/${arm}_merged" --rank "${QAD_LORA_R:-32}" --alpha "${QAD_LORA_ALPHA:-64}" \
    > "$ROUND_OUT/$arm.merge.log" 2>&1
done

for arm in bf16 ptq qad continued_qad qad_opd; do
  case "$arm" in
    bf16) checkpoint="$BF16_TEACHER" ;;
    ptq) checkpoint="$PTQ_BASE" ;;
    qad) checkpoint="$QAD_MERGED" ;;
    *) checkpoint="$ROUND_OUT/${arm}_merged" ;;
  esac
  python3 "$PROJECT/eval/run_recovery_eval.py" \
    --checkpoint "$checkpoint" --out "$ROUND_OUT/heldout_$arm" --purpose heldout \
    --seed "$HELDOUT_SEED" --episodes "$HELDOUT_EPISODES" --port "$PORT" \
    --collection-manifest "$ROUND_OUT/collection/eval_manifest.json" \
    > "$ROUND_OUT/$arm.eval.log" 2>&1
done
python3 "$PROJECT/eval/compare_recovery.py" --round "$ROUND_OUT" > "$ROUND_OUT/comparison.log" 2>&1
echo "Completed paired round: $ROUND_OUT"
