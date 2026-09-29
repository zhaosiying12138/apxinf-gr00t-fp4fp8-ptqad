#!/usr/bin/env bash
export SHOT_GPU_RELEASED=1
#!/usr/bin/env bash
# Prepared smoke evidence only; formal experiments and their outputs are read-only.
set -euo pipefail
PROJECT=/home/zhaosiying/codebase/fp4vla
SCRATCH=/home/zhaosiying/codebase/fp4vla/paper/_build/capture_batch/scratch/20260929_r4
FIGURE=shot_evalserver
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
# Parent coordinator must release the GPU first. The capture launcher supplies this flag.
test "${SHOT_GPU_RELEASED:-0}" = 1 || { printf '%s\n' 'GPU handoff has not been acknowledged.' >&2; exit 2; }
python3 - <<'PY'
import subprocess
result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                        check=True, capture_output=True, text=True)
rows = [line for line in result.stdout.splitlines() if line.strip()]
if rows:
    raise RuntimeError('Existing CUDA compute process; do not start a capture: ' + repr(rows))
PY
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$SCRATCH/stages"
mkdir "$SCRATCH/stages/$FIGURE"
date -u +%FT%TZ > "$SCRATCH/stages/$FIGURE/started_utc.txt"
cd "$PROJECT"
test -f "$SCRATCH/stages/shot_qad/completed_utc.txt"
"$PTQAD_PYTHON" "$PROJECT/rl/lora_merge_bake.py" --base "$SCRATCH/head_lang" \
  --ckpt "$SCRATCH/qad/checkpoint-2" --out "$SCRATCH/qad_merged" --rank 32 --alpha 64
export GR00T_EVAL_SEED=10910000

# Start only our child in a new process group. Cleanup never uses broad pkill.
server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid"
    wait "$server_pid" || true
  fi
}
trap cleanup_server EXIT
PORT=5598
python3 - "$PORT" <<'PY'
import socket, sys
with socket.socket() as listener:
    listener.bind(('127.0.0.1', int(sys.argv[1])))
PY
start_server() {
  cd "$GR00T_REPO"
  setsid "$PTQAD_PYTHON" -u "$PROJECT/eval/serve_recovery.py" \
    --model-path "$SCRATCH/qad_merged" --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper --host 127.0.0.1 --port "$PORT" &
  server_pid=$!
  printf '%s\n' "$server_pid" > "$SCRATCH/stages/$FIGURE/server.pid"
  python3 - "$server_pid" "$PORT" <<'PY'
import os, socket, sys, time
pid, port = map(int, sys.argv[1:])
deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise TimeoutError('Our new server did not bind before the deadline')
PY
}
stop_server_cleanly() {
  "$PTQAD_PYTHON" - "$PORT" <<'PY'
import json, sys
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host='127.0.0.1', port=int(sys.argv[1]), timeout_ms=10000)
try:
    response = client.call_endpoint('ping', requires_input=False)
    print(json.dumps({'endpoint': 'ping', 'response': response}), flush=True)
    if response.get('status') != 'ok':
        raise RuntimeError('Our server did not return a healthy RPC response')
    client.kill_server()
finally:
    client.close()
PY
  wait "$server_pid"
  server_pid=""
  trap - EXIT
}
start_server
stop_server_cleanly

date -u +%FT%TZ > "$SCRATCH/stages/$FIGURE/completed_utc.txt"
