#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/common_v11.sh"
OUT="$(prepare_scratch shot_rollout)"; assert_no_compute_apps
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPD_CAPTURE_DIR="$OUT/observations" OPD_CAPTURE_EVERY=1 OPD_CAPTURE_PER_TASK=2 OPD_CAPTURE_LIMIT=2 OPD_CAPTURE_PER_EPISODE=2
export FP4VLA_CAPTURE_PURPOSE=screenshot_smoke FP4VLA_CAPTURE_TASK_NAME="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" FP4VLA_CAPTURE_SEED=940000 FP4VLA_CAPTURE_INIT_STATE_INDICES=4 FP4VLA_CAPTURE_PROTOCOL_SHA256="$(sha256sum "$PROTOCOL_FILE" | cut -d' ' -f1)" FP4VLA_CAPTURE_EVENT_FILE="$OUT/reset_events.jsonl"
export GR00T_EVAL_SEED=20261003 FP4VLA_QUANT=0 FP4VLA_W4A4=1 FP4VLA_W4A4_ADAPTER=1 FP4VLA_SATURATE_F16_ACTIVATIONS=1 GR00T_BACKBONE_MODEL="$BACKBONE_MODEL"
PORT=5618
server_pid=""
cleanup() { if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" || true; fi; }
trap cleanup EXIT
cd "$GR00T_REPO"
"$PTQAD_PYTHON" -u "$PROJECT/eval/serve_recovery.py" --model-path "$QAD_MODEL" --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --host 127.0.0.1 --port "$PORT" > >(tee "$OUT/server.raw.log") 2>&1 &
server_pid=$!; wait_for_port "$server_pid" "$PORT"

"$LIBERO_PYTHON" -u "$PROJECT/eval/rollout_seeded.py" --env-name libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket --n-episodes 1 --n-envs 1 --seed 940000 --init-state-indices 4 --max-episode-steps 720 --n-action-steps 8 --video-dir "$OUT/videos" --policy-client-host 127.0.0.1 --policy-client-port "$PORT" 2>&1 | tee "$OUT/rollout_1task_1episode.raw.log"
test "$(find "$OUT/observations" -name 'sample_*.pt' | wc -l)" -eq 2
python3 - "$OUT" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1]); json.dump({"purpose":"screenshot_smoke","tasks":1,"episodes":1,"captured_observations":2,"not_formal_evaluation":True}, (root / "rollout_smoke_manifest.json").open("w"), indent=2)
PY
"$PTQAD_PYTHON" - <<'PY'
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host="127.0.0.1", port=5618, timeout_ms=10000)
try: client.kill_server()
finally: client.close()
PY
wait "$server_pid" || true; server_pid=""
printf '%s
' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
