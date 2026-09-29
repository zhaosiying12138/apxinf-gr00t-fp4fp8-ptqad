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
COLLECTION_SEED=${COLLECTION_SEED:-}
HELDOUT_SEED=${HELDOUT_SEED:-}
EXTRA_STEPS=${EXTRA_STEPS:-100}
COLLECTION_EPISODES=${COLLECTION_EPISODES:-}
HELDOUT_EPISODES=${HELDOUT_EPISODES:-}
PORT=${PORT:-5595}
PROTOCOL_FILE=${PROTOCOL_FILE:-$PROJECT/exp/recovery_protocol.json}
test -f "$PROTOCOL_FILE" || { echo "PROTOCOL_FILE does not exist: $PROTOCOL_FILE" >&2; exit 1; }
mapfile -t _PROTOCOL_VALUES < <(python3 - "$PROTOCOL_FILE" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
parts = data.get("partitions", data)
for name in ("collection", "heldout"):
    item = parts.get(name, {})
    print(int(item["seed"]))
    print(int(item["episodes_per_task"]))
print(int(data.get("selection", {}).get("train_seed", 42)))
PY
)
[[ ${#_PROTOCOL_VALUES[@]} -eq 5 ]] || { echo "Protocol lacks collection/heldout seed and episode declarations" >&2; exit 1; }
COLLECTION_SEED=${COLLECTION_SEED:-${_PROTOCOL_VALUES[0]}}
COLLECTION_EPISODES=${COLLECTION_EPISODES:-${_PROTOCOL_VALUES[1]}}
HELDOUT_SEED=${HELDOUT_SEED:-${_PROTOCOL_VALUES[2]}}
HELDOUT_EPISODES=${HELDOUT_EPISODES:-${_PROTOCOL_VALUES[3]}}
TRAIN_SEED=${TRAIN_SEED:-${_PROTOCOL_VALUES[4]}}
export QAD_MICRO_BATCH=${QAD_MICRO_BATCH:-1}
export QAD_GLOBAL_BATCH=${QAD_GLOBAL_BATCH:-16}
export QAD_LORA_SCOPE=${QAD_LORA_SCOPE:-head+lang_all}
export QAD_LORA_R=${QAD_LORA_R:-32}
export QAD_LORA_ALPHA=${QAD_LORA_ALPHA:-64}
export QAD_ACTIVATION_CHECKPOINTING=${QAD_ACTIVATION_CHECKPOINTING:-1}
export QAD_LR=${QAD_LR:-1e-4}
export TRAIN_SEED
export PROTOCOL_FILE
export LD_LIBRARY_PATH=${PTQAD_MEDIA_LIB:-${LD_LIBRARY_PATH:-$HOME/miniforge3/envs/media7/lib}}
export HF_HUB_OFFLINE=1
test ! -e "$ROUND_OUT" || { echo "ROUND_OUT already exists; use a new evidence directory" >&2; exit 1; }
# Fail before collecting trajectories if continuation settings do not match QAD.
python3 - "$QAD_ADAPTER" "$PTQ_BASE" "$QAD_MERGED" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
adapter, base, merged = map(lambda p: Path(p).resolve(), sys.argv[1:])
manifest_path = adapter / "recovery_manifest.json"
manifest = json.loads(manifest_path.read_text())
expected = {"base": str(base), "rank": int(os.environ["QAD_LORA_R"]),
            "alpha": float(os.environ["QAD_LORA_ALPHA"]),
            "scope": os.environ["QAD_LORA_SCOPE"]}
if any(manifest.get(k) != v for k, v in expected.items()):
    raise ValueError("Continuation base/rank/alpha/scope differs from initial QAD")
protocol = Path(os.environ["PROTOCOL_FILE"])
protocol_data = json.loads(protocol.read_text())
if int(protocol_data.get("version", 1)) >= 3:
    if (manifest.get("train_seed") != int(os.environ["TRAIN_SEED"]) or
            manifest.get("protocol_sha256") != hashlib.sha256(protocol.read_bytes()).hexdigest()):
        raise ValueError("Initial QAD training seed/protocol differs from this on-policy round")
for field, filename in (("base_config_sha256", "config.json"),
                        ("base_statistics_sha256", "statistics.json"),
                        ("base_recipe_sha256", "ptq_recipe.json")):
    if manifest.get(field) != hashlib.sha256((base / filename).read_bytes()).hexdigest():
        raise ValueError(f"QAD base metadata changed: {filename}")
export = json.loads((merged / "merge_manifest.json").read_text())
if (Path(export["training_checkpoint"]).resolve() != adapter or
        Path(export["base"]).resolve() != base or export.get("status") != "complete"):
    raise ValueError("Collection policy is not the completed export of the initial QAD adapter")
if export["recovery_manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
    raise ValueError("Collection policy and QAD adapter manifest differ")
print("[round] initial QAD adapter, collection export and continuation settings match", flush=True)
PY
mkdir -p "$ROUND_OUT"
ROUND_OUT=$(realpath "$ROUND_OUT")
cd "$GR00T"

python3 "$PROJECT/eval/run_recovery_eval.py" \
  --checkpoint "$QAD_MERGED" --out "$ROUND_OUT/collection" --purpose collection \
  --seed "$COLLECTION_SEED" --episodes "$COLLECTION_EPISODES" --port "$PORT" \
  --protocol-file "$PROTOCOL_FILE" \
  > "$ROUND_OUT/collection.log" 2>&1

"$PY" "$PROJECT/rl/opd_probe_cache.py" --teacher "$BF16_TEACHER" \
  --input-dir "$ROUND_OUT/collection/observations" --count 160 \
  --out "$ROUND_OUT/teacher_probes.pt" > "$ROUND_OUT/teacher_labeling.log" 2>&1

# Run one complete OPD schedule period, so the final optimizer update actually
# executes teacher backward, and save at the production micro/global batch. This
# adapter is discarded from all comparisons; both arms below still start from
# the original QAD_ADAPTER, with the preregistered optimizer/update budgets.
OPD_SMOKE_STEPS=${OPD_EVERY:-4}
GR00T_BASE_CKPT="$PTQ_BASE" QAD_INIT_ADAPTER="$QAD_ADAPTER" \
  QAD_OUT="$ROUND_OUT/opd_smoke" QAD_STEPS="$OPD_SMOKE_STEPS" QAD_SAVE_STEPS="$OPD_SMOKE_STEPS" \
  QAD_OPD_MSE_W="${OPD_WEIGHT:-1.0}" OPD_CACHE_PATH="$ROUND_OUT/teacher_probes.pt" \
  OPD_EVERY=${OPD_EVERY:-4} "$PY" "$PROJECT/rl/lora_qad.py" \
  > "$ROUND_OUT/opd_smoke.train.log" 2>&1
python3 - "$ROUND_OUT/opd_smoke.train.log" "$OPD_SMOKE_STEPS" "$QAD_GLOBAL_BATCH" "$QAD_MICRO_BATCH" <<'PY'
import math, re, sys
from pathlib import Path
path, period, batch, micro = sys.argv[1:]
records = re.findall(r"\[opd\] step=(\d+) probe=\d+ mse=(\S+) weight=\S+ microbatch=1", Path(path).read_text())
expected = int(batch) // int(micro)
if len(records) != expected or any(int(step) != int(period) or not math.isfinite(float(mse)) or float(mse) < 0 for step, mse in records):
    raise RuntimeError(f"OPD smoke did not execute {expected} finite teacher microbatch backward passes at update {period}")
print(f"[round] OPD smoke verified {len(records)} teacher microbatch backward passes at update {period}", flush=True)
PY

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
    --protocol-file "$PROTOCOL_FILE" \
    --collection-manifest "$ROUND_OUT/collection/eval_manifest.json" \
    > "$ROUND_OUT/$arm.eval.log" 2>&1
done
python3 "$PROJECT/eval/compare_recovery.py" --round "$ROUND_OUT" > "$ROUND_OUT/comparison.log" 2>&1
echo "Completed paired round: $ROUND_OUT"
