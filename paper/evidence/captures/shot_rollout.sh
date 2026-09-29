#!/usr/bin/env bash
export SHOT_GPU_RELEASED=1
#!/usr/bin/env bash
# Prepared smoke evidence only; formal experiments and their outputs are read-only.
set -euo pipefail
PROJECT=/home/zhaosiying/codebase/fp4vla
SCRATCH=/home/zhaosiying/codebase/fp4vla/paper/_build/capture_batch/scratch/20260929_r4
FIGURE=shot_rollout
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
test -f "$SCRATCH/stages/shot_evalserver/completed_utc.txt"
export GR00T_EVAL_SEED=10910000 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPD_CAPTURE_DIR="$SCRATCH/observations/task0"
export OPD_CAPTURE_EVERY=1 OPD_CAPTURE_PER_TASK=2 OPD_CAPTURE_LIMIT=2

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
"$LIBERO_PYTHON" -u "$PROJECT/eval/rollout_seeded.py" \
  --env-name libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket \
  --n-episodes 1 --n-envs 1 --seed 910000 --init-state-indices 4 \
  --max-episode-steps 720 --n-action-steps 8 --video-dir "$SCRATCH/videos/task0" \
  --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
  2>&1 | tee "$SCRATCH/rollout.raw.log"
stop_server_cleanly
cd "$PROJECT"
python3 - "$SCRATCH" <<'PY'
import json, sys
from pathlib import Path
from eval.run_recovery_eval import parse_log, validate_resets
root = Path(sys.argv[1])
result = parse_log(root / 'rollout.raw.log')
validate_resets(result, 910000, [4])
if result['episodes'] != 1:
    raise RuntimeError('Smoke did not complete exactly one episode')
files = sorted((root / 'observations').rglob('sample_*.pt'))
if len(files) != 2:
    raise RuntimeError(f'Expected two real student observations, received {len(files)}')
manifest = {'purpose': 'screenshot_smoke', 'task_count': 1, 'episodes': 1,
            'seed': 910000, 'init_state_indices': [4], 'settle_steps': 10,
            'not_formal_collection': True, 'captured_observations': len(files), 'result': result}
with (root / 'rollout_smoke_manifest.json').open('x') as stream:
    json.dump(manifest, stream, indent=2)
PY

date -u +%FT%TZ > "$SCRATCH/stages/$FIGURE/completed_utc.txt"
