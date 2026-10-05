#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/zhaosiying/codebase/fp4vla
GR00T_REPO=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
PTQAD_PYTHON=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python
LIBERO_PYTHON=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
BF16_TEACHER=/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10
PTQ_BASE=/home/zhaosiying/codebase/fp4vla/results/ptqad_20261003/w4a4_full_category
QAD_MODEL=/home/zhaosiying/codebase/fp4vla/results/ptqad_20261003/v11_recovery_r4/artifacts/merge_qad_lr_0.0001
QAD_ADAPTER=/home/zhaosiying/codebase/fp4vla/results/ptqad_20261003/v11_recovery_r4/artifacts/train_qad_lr_0.0001/checkpoint-2000
OPD_MODEL=/home/zhaosiying/codebase/fp4vla/results/ptqad_20261003/v11_recovery_r4/artifacts/merge_opd_025
DATASET=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/demo_data/libero_demo
CAPTURE_DATASET=/home/zhaosiying/codebase/fp4vla/results/ptqad_20261003/w4a4_teacher_supervision
CAPTURE_DATASET_SHA256=67fb84ea77bedd2197cf666dada80aa47eb556539029125268885f79c019bb85
PROTOCOL_FILE=/home/zhaosiying/codebase/fp4vla/exp/recovery_protocol_v11_w4a4_category.json
BACKBONE_MODEL=nvidia/Cosmos-Reason2-2B
CAPTURE_ROOT=/home/zhaosiying/codebase/fp4vla/paper/_build/capture_final_20261006
SCRATCH_ROOT="$CAPTURE_ROOT/scratch"
export PROJECT GR00T_REPO PTQAD_PYTHON LIBERO_PYTHON BF16_TEACHER PTQ_BASE QAD_MODEL QAD_ADAPTER OPD_MODEL DATASET CAPTURE_DATASET CAPTURE_DATASET_SHA256 PROTOCOL_FILE BACKBONE_MODEL CAPTURE_ROOT SCRATCH_ROOT
export GR00T_BACKBONE_MODEL="$BACKBONE_MODEL"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 NO_ALBUMENTATIONS_UPDATE=1
export PTQAD_MEDIA_LIB="${PTQAD_MEDIA_LIB:-$HOME/miniforge3/envs/media7/lib}"
export LD_LIBRARY_PATH="$PTQAD_MEDIA_LIB:/usr/local/cuda/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="${PTQAD_MEDIA_LIB%/lib}/bin:/usr/local/cuda/bin:$PATH" PYTHONPATH="$PROJECT:$GR00T_REPO"
require_sources() {
  for executable in "$PTQAD_PYTHON" "$LIBERO_PYTHON"; do test -x "$executable" || { echo "missing executable: $executable" >&2; exit 2; }; done
  for directory in "$BF16_TEACHER" "$PTQ_BASE" "$QAD_MODEL" "$QAD_ADAPTER" "$OPD_MODEL" "$DATASET" "$CAPTURE_DATASET"; do test -d "$directory" || { echo "missing source: $directory" >&2; exit 2; }; done
  test -f "$PROTOCOL_FILE" && test -f "$PTQ_BASE/category_ptq_recipe.json" && test -f "$QAD_MODEL/merge_manifest.json"
}
prepare_scratch() {
  local figure="$1"; require_sources; local out="$SCRATCH_ROOT/$figure"
  if [[ -e "$out" ]]; then echo "refusing to overwrite existing scratch: $out" >&2; exit 2; fi
  mkdir -p "$out"; printf '%s\n' "$(date -u +%FT%TZ)" > "$out/started_utc.txt"; printf '%s\n' "$out"
}
assert_no_compute_apps() {
  local rows
  if ! rows="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)"; then
    echo "nvidia-smi query failed; refusing to start capture" >&2; exit 2
  fi
  if [[ -n "${rows//[[:space:]]/}" ]]; then echo "CUDA process exists; do not start capture: $rows" >&2; exit 2; fi
}
wait_for_port() {
  local pid="$1" port="$2"; python3 - "$pid" "$port" <<'PY'
import os, socket, sys, time
pid, port = map(int, sys.argv[1:]); deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1): break
    except OSError: time.sleep(1)
else: raise TimeoutError("W4A4 recovery server did not bind")
PY
}
