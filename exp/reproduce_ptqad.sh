#!/usr/bin/env bash
# Serial, fail-closed reproduction. Does not run unless explicitly invoked.
# Set PTQAD_RUN_DIR to a new experiment root. Existing stage outputs/logs fail.
# Stages can be supplied individually to continue an already-created run root.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${PTQAD_RUN_DIR:?Set PTQAD_RUN_DIR to the experiment directory}"
RUN_DIR=$(realpath -m -- "$PTQAD_RUN_DIR")
RUN_ID=${PTQAD_RUN_ID:-$(basename -- "$RUN_DIR")}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { printf '%s\n' 'Invalid PTQAD_RUN_ID'; exit 2; }
GROOT=${GR00T_REPO:-$ROOT/third_party/Isaac-GR00T}
PY=${PTQAD_PYTHON:-$GROOT/.venv/bin/python}
SIM_PY=${LIBERO_PYTHON:-$GROOT/.venv-libero/bin/python}
BASE=${PTQAD_BASE:-$ROOT/weights/GR00T-N1.7-LIBERO/libero_10}
DATASET=${QAD_DATASET:-$GROOT/demo_data/libero_demo}
EVAL_RUNS_ROOT=${PTQAD_EVAL_RUNS_ROOT:-$RUN_DIR/evaluations}
RECOVERY_RECIPE=${PTQAD_RECOVERY_RECIPE:-fp8}
PROTOCOL_FILE=${PTQAD_PROTOCOL_FILE:-$ROOT/exp/recovery_protocol.json}
case "$RECOVERY_RECIPE" in
  rtn|fp8|mixed|aggr|calib|head_ffn|head_lang|head_lang_vision) ;;
  *) printf 'Invalid recovery recipe: %s\n' "$RECOVERY_RECIPE"; exit 2 ;;
esac
CAL_SCOPE=${PTQAD_CAL_SCOPE:-calib}
CAL_WINDOWS=${PTQ_CAL_WINDOWS:-128}
CAL_BATCH=${PTQ_CAL_BATCH:-1}
QAD_STEPS=${QAD_STEPS:-500}
CONT_STEPS=${PTQAD_CONT_STEPS:-100}
BSZ=${QAD_GLOBAL_BATCH:-16}
MICRO_BATCH=${QAD_MICRO_BATCH:-1}
export QAD_ACTIVATION_CHECKPOINTING=${QAD_ACTIVATION_CHECKPOINTING:-1}
RANK=${QAD_LORA_R:-32}
ALPHA=${QAD_LORA_ALPHA:-64}
LR=${QAD_LR:-0.0001}
EPISODES=${PTQAD_EVAL_EPISODES:-}
DEV_EPISODES=${PTQAD_DEV_EPISODES:-}
PORT_BASE=${PTQAD_PORT_BASE:-5610}
export HF_HUB_OFFLINE=1
export PYTHONHASHSEED=${PYTHONHASHSEED:-20260929}
export LD_LIBRARY_PATH="${PTQAD_MEDIA_LIB:-$HOME/miniforge3/envs/media7/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

[[ -x "$PY" && -d "$BASE" && -d "$DATASET" ]] || { printf '%s\n' 'Python, base or dataset unavailable'; exit 2; }
[[ -f "$PROTOCOL_FILE" ]] || { printf 'Protocol file missing: %s\n' "$PROTOCOL_FILE"; exit 2; }
# A versioned protocol owns the seed and episode count for every formal
# partition.  Environment overrides remain available for an explicitly
# documented protocol fork, but the default follows the selected JSON file.
mapfile -t _PROTOCOL_VALUES < <("$PY" - "$PROTOCOL_FILE" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
parts = data.get("partitions", data)
for name in ("development", "collection", "heldout"):
    item = parts.get(name, {})
    print(int(item["seed"]))
    print(int(item["episodes_per_task"]))
PY
)
[[ ${#_PROTOCOL_VALUES[@]} -eq 6 ]] || { printf 'Protocol lacks seed/episode declarations for development, collection and heldout\n'; exit 2; }
DEV_SEED=${PTQAD_DEV_SEED:-${_PROTOCOL_VALUES[0]}}
DEV_EPISODES=${DEV_EPISODES:-${_PROTOCOL_VALUES[1]}}
COLLECTION_SEED=${PTQAD_COLLECTION_SEED:-${_PROTOCOL_VALUES[2]}}
COLLECTION_EPISODES=${PTQAD_COLLECTION_EPISODES:-${_PROTOCOL_VALUES[3]}}
HELDOUT_SEED=${PTQAD_HELDOUT_SEED:-${_PROTOCOL_VALUES[4]}}
EPISODES=${EPISODES:-${_PROTOCOL_VALUES[5]}}
mkdir -p -- "$RUN_DIR/logs"
if [[ ! -f "$RUN_DIR/run_parameters.json" ]]; then
  "$PY" - "$RUN_DIR/run_parameters.json" "$ROOT" "$BASE" "$DATASET" "$GROOT" "$CAL_SCOPE" \
    "$CAL_WINDOWS" "$CAL_BATCH" "$QAD_STEPS" "$CONT_STEPS" "$BSZ" "$MICRO_BATCH" "$RANK" "$ALPHA" "$LR" "$EPISODES" "$RECOVERY_RECIPE" <<'PY'
import json,sys,hashlib,subprocess,os
from pathlib import Path
out,repo,base,data,groot,scope,windows,batch,steps,cont,bsz,micro,rank,alpha,lr,eps,recovery=sys.argv[1:]
paths=['quant/torch_fp4.py','quant/ptq/quantizers.py','quant/ptq/bake.py','quant/ptq/collector.py',
       'rl/lora_qad.py','rl/probe_distill.py','rl/lora_merge_bake.py','rl/activation_checkpoint.py',
       'rl/runtime_metrics.py','rl/gr00t_runtime.py','exp/reproduce_ptqad.sh']
record={'recovery_recipe':recovery,'repo':repo,'base':base,'dataset':data,'gr00t_repo':groot,'calibration_scope':scope,
        'calibration_windows':int(windows),'calibration_batch':int(batch),'qad_steps':int(steps),
        'continuation_steps_each_arm':int(cont),'global_batch':int(bsz),'micro_batch':int(micro), 'gradient_accumulation':int(bsz)//int(micro),
        'lora_rank':int(rank),'lora_alpha':float(alpha),'learning_rate':float(lr),
        'activation_checkpointing':os.environ.get('QAD_ACTIVATION_CHECKPOINTING') == '1',
        'requested_eval_episodes_per_task':int(eps),
        'source_sha256':{p:hashlib.sha256((Path(repo)/p).read_bytes()).hexdigest() for p in paths},
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        'note':'Source hashes include uncommitted working files; stages execute serially. Raw logs determine actual episode counts.'}
Path(out).write_text(json.dumps(record,indent=2)+'\n')
PY
else
  "$PY" - "$RUN_DIR/run_parameters.json" "$RECOVERY_RECIPE" "$BASE" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); previous=json.loads(path.read_text())
if previous['base'] != sys.argv[3]:
    raise ValueError('Existing run uses another base; use a new run directory')
if previous['recovery_recipe'] != sys.argv[2]:
    if list((path.parent/'logs').glob('train-*.log')):
        raise ValueError('Cannot change recovery recipe after training started')
    previous['recovery_recipe']=sys.argv[2]
    path.write_text(json.dumps(previous,indent=2)+'\n')
PY
fi

run_logged() {
  local stage=$1
  shift
  local log="$RUN_DIR/logs/$stage.log"
  [[ ! -e "$log" ]] || { printf 'Refusing existing stage log: %s\n' "$log"; return 2; }
  { printf 'UTC '; date -u +%FT%TZ; printf 'COMMAND '; printf '%q ' "$@"; printf '\n'; } > "$log"
  "$@" 2>&1 | tee -a "$log"
  printf 'COMPLETED UTC %s\n' "$(date -u +%FT%TZ)" >> "$log"
}

bake_recipe() {
  local recipe=$1
  if [[ "$recipe" == rtn || "$recipe" == fp8 ]]; then
    run_logged "bake-$recipe" "$PY" "$ROOT/quant/ptq/bake.py" --base "$BASE" \
      --out "$RUN_DIR/$recipe" --recipe "$recipe" --calibration-mode none
  else
    run_logged "bake-$recipe" "$PY" "$ROOT/quant/ptq/bake.py" --base "$BASE" \
      --out "$RUN_DIR/$recipe" --recipe "$recipe" --calibration-mode required \
      --calib "$RUN_DIR/calibration/calib.pt"
  fi
}

train_arm() {
  local arm=$1 steps=$2 init=${3:-} weight=${4:-0}
  [[ ! -e "$RUN_DIR/train_$arm" ]] || { printf 'Refusing existing training output: %s\n' "$RUN_DIR/train_$arm"; return 2; }
  local -a command=(env "GR00T_BASE_CKPT=$RUN_DIR/$RECOVERY_RECIPE" "QAD_DATASET=$DATASET"
    "QAD_OUT=$RUN_DIR/train_$arm" "QAD_STEPS=$steps" "QAD_SAVE_STEPS=$steps" "QAD_GLOBAL_BATCH=$BSZ" "QAD_MICRO_BATCH=$MICRO_BATCH"
    "QAD_LORA_R=$RANK" "QAD_LORA_ALPHA=$ALPHA" "QAD_LORA_SCOPE=${QAD_LORA_SCOPE:-head+lang_all}" "QAD_LR=$LR"
    "QAD_OPD_MSE_W=$weight" "OPD_EVERY=${OPD_EVERY:-4}")
  if [[ -n "$init" ]]; then command+=("QAD_INIT_ADAPTER=$init"); fi
  if [[ "$weight" != 0 ]]; then
    : "${OPD_CACHE_PATH:?Set OPD_CACHE_PATH to a new student-rollout teacher cache}"
    command+=("OPD_CACHE_PATH=$OPD_CACHE_PATH")
  fi
  command+=("$PY" "$ROOT/rl/lora_qad.py")
  (cd -- "$GROOT"; run_logged "train-$arm" "${command[@]}")
}

merge_arm() {
  local arm=$1 steps=$2
  run_logged "merge-$arm" "$PY" "$ROOT/rl/lora_merge_bake.py" --base "$RUN_DIR/$RECOVERY_RECIPE" \
    --ckpt "$RUN_DIR/train_$arm/checkpoint-$steps" --out "$RUN_DIR/$arm" --rank "$RANK" --alpha "$ALPHA"
}

evaluate() {
  local arm=$1 offset=$2 purpose=${3:-heldout}
  local tag="${RUN_ID}_${purpose}_${arm}" checkpoint="$RUN_DIR/$arm"
  [[ "$arm" == bf16 ]] && checkpoint="$BASE"
  local -a protocol
  case "$purpose" in
    development) protocol=(--seed "$DEV_SEED" --episodes "$DEV_EPISODES") ;;
    collection) protocol=(--seed "$COLLECTION_SEED" --episodes "$COLLECTION_EPISODES") ;;
    heldout)
      protocol=(--seed "$HELDOUT_SEED" --episodes "$EPISODES" --collection-manifest
        "${PTQAD_COLLECTION_MANIFEST:-$EVAL_RUNS_ROOT/${RUN_ID}_collection_qad/eval_manifest.json}") ;;
    *) return 2 ;;
  esac
  [[ -x "$SIM_PY" ]] || { printf 'Simulation Python missing: %s\n' "$SIM_PY"; return 2; }
    run_logged "eval-$purpose-$arm" "$PY" "$ROOT/eval/run_recovery_eval.py" \
    --checkpoint "$checkpoint" --out "$EVAL_RUNS_ROOT/$tag" --purpose "$purpose" \
    --gr00t "$GROOT" --server-python "$PY" --rollout-python "$SIM_PY" \
    --port "$((PORT_BASE + offset))" --protocol-file "$PROTOCOL_FILE" "${protocol[@]}"
  local actual_request=$EPISODES
  [[ "$purpose" == development ]] && actual_request=$DEV_EPISODES
  [[ "$purpose" == collection ]] && actual_request=2
  "$PY" "$ROOT/paper/collect_reevaluation.py" "$tag" --source-dir "$EVAL_RUNS_ROOT/$tag" \
    --requested-episodes "$actual_request" --output-root "$RUN_DIR/evidence"
}

if (( $# == 0 )); then
  printf '%s\n' 'Supply explicit stages. See docs/reproduce-ptqad.md; recipe selection must use development outcomes.'
  exit 2
fi
for stage in "$@"; do
  case "$stage" in
    rtn|fp8|mixed|aggr|calib|head_ffn|head_lang|head_lang_vision) bake_recipe "$stage" ;;
    calibrate)
      (cd -- "$GROOT"; run_logged calibrate "$PY" "$ROOT/quant/ptq/collector.py" \
        --base "$BASE" --dataset "$DATASET" --out "$RUN_DIR/calibration" \
        --recipe "$CAL_SCOPE" --windows "$CAL_WINDOWS" --batch "$CAL_BATCH") ;;
    train-qad) train_arm qad "$QAD_STEPS" ;;
    merge-qad) merge_arm qad "$QAD_STEPS" ;;
    train-continued) train_arm continued "$CONT_STEPS" "$RUN_DIR/train_qad/checkpoint-$QAD_STEPS" 0 ;;
    train-opd) train_arm opd "$CONT_STEPS" "$RUN_DIR/train_qad/checkpoint-$QAD_STEPS" "${QAD_OPD_MSE_W:-1}" ;;
    merge-continued) merge_arm continued "$CONT_STEPS" ;;
    merge-opd) merge_arm opd "$CONT_STEPS" ;;
    dev-*) evaluate "${stage#dev-}" 0 development ;;
    collect-qad) evaluate qad 1 collection ;;
    eval-bf16) evaluate bf16 0 ;;
    eval-head_ffn) evaluate head_ffn 6 ;;
    eval-head_lang) evaluate head_lang 7 ;;
    eval-head_lang_vision) evaluate head_lang_vision 8 ;;
    eval-calib) evaluate calib 9 ;;
    eval-rtn) evaluate rtn 0 ;;
    eval-fp8) evaluate fp8 1 ;;
    eval-mixed) evaluate mixed 2 ;;
    eval-qad) evaluate qad 3 ;;
    eval-continued) evaluate continued 4 ;;
    eval-opd) evaluate opd 5 ;;
    *) printf 'Unknown stage: %s\n' "$stage"; exit 2 ;;
  esac
done
