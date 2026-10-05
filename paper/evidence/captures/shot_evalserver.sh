#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/common_v11.sh"
OUT="$(prepare_scratch shot_evalserver)"; assert_no_compute_apps
export GR00T_EVAL_SEED=20261003 FP4VLA_QUANT=0 FP4VLA_W4A4=1 FP4VLA_W4A4_ADAPTER=1 FP4VLA_SATURATE_F16_ACTIVATIONS=1 GR00T_BACKBONE_MODEL="$BACKBONE_MODEL"
PORT=5617
server_pid=""
cleanup() { if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" || true; fi; }
trap cleanup EXIT
cd "$GR00T_REPO"
"$PTQAD_PYTHON" -u "$PROJECT/eval/serve_recovery.py" --model-path "$OPD_MODEL" --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --host 127.0.0.1 --port "$PORT" > >(tee "$OUT/server.raw.log") 2>&1 &
server_pid=$!; wait_for_port "$server_pid" "$PORT"

"$PTQAD_PYTHON" - "$PORT" <<'PY' 2>&1 | tee "$OUT/ping.raw.log"
import json, sys
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host="127.0.0.1", port=int(sys.argv[1]), timeout_ms=10000)
try:
    response = client.call_endpoint("ping", requires_input=False)
    print(json.dumps({"endpoint":"ping", "scope":"health RPC only", "response":response}, ensure_ascii=False), flush=True)
    if response.get("status") != "ok": raise RuntimeError(response)
finally: client.close()
PY
kill -TERM "$server_pid"; wait "$server_pid" || true; server_pid=""
printf '%s
' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
