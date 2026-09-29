#!/usr/bin/env bash
# Root-coordinated GPU acceptance of the rebuilt APX native extension.
# This is separate from GR00T/QAD/OPD training and does not modify checkpoints.
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ${PTQAD_GPU_EXCLUSIVE:-0} != 1 ]]; then
  printf '%s\n' 'Run only after obtaining exclusive GPU ownership: PTQAD_GPU_EXCLUSIVE=1.' >&2
  exit 2
fi
if [[ $# != 1 ]]; then
  printf '%s\n' 'usage: exp/run_native_graph_gates.sh <new-output-directory>' >&2
  exit 2
fi
OUT="$(realpath -m "$1")"
[[ ! -e "$OUT" ]] || { printf 'Refusing existing output: %s\n' "$OUT" >&2; exit 2; }
APX="$PROJECT/third_party/apxinf-robo/apxinf"
NATIVE_PY="${NATIVE_PY:-$PROJECT/third_party/apxinf-robo/.venv/bin/python}"
MODEL="${NATIVE_PI05_MODEL:-$PROJECT/weights/native_pi05_20260929/model}"
WHEEL="$APX/target/wheel/wheels/apxinf_py-0.1.0-cp312-cp312-linux_x86_64.whl"
[[ -f "$WHEEL" && -x "$NATIVE_PY" && -f "$MODEL/fp4/manifest.json" ]]
[[ -x "$APX/target/wheel/release/examples/pi05_fp4_graph_smoke" ]]
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="$HOME/.cuda-stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export APXINF_CUDA_ARCH=sm_120 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=target/wheel
mkdir -p "$OUT"
trap 'status=$?; printf "%s\n" "$status" > "$OUT/exit_code.txt"' EXIT
date -u +%FT%TZ > "$OUT/start_utc.txt"
sha256sum "$WHEEL" "$PROJECT/patches/apxinf-fp4vla-engine.patch" \
  "$PROJECT/spike/fp4_opbench" \
  "$APX/target/wheel/release/examples/pi05_fp4_graph_smoke" > "$OUT/input_sha256.txt"
nvidia-smi > "$OUT/device_before.txt"
cd "$APX"
for test in fp4_contract_rowmajor_and_tensor_scale \
            fp4_graph_replay_bf16_and_distinct_scales \
            fp4_activation_padding_zero_after_capture; do
  /usr/bin/time -v cargo test --release -p apxinf-cuda "$test" \
    -- --ignored --nocapture > "$OUT/$test.log" 2>&1
  grep -Fq 'test result: ok. 1 passed; 0 failed;' "$OUT/$test.log"
done
/usr/bin/time -v ./target/wheel/release/examples/pi05_fp4_graph_smoke "$MODEL" 10 \
  > "$OUT/pi05_require_graph.log" 2>&1
grep -Fq 'NVFP4 full-model RequireGraph PASS:' "$OUT/pi05_require_graph.log"
# Install only after numerical/padding/replay/full-model acceptance passed.
cd /tmp
uv pip install --python "$NATIVE_PY" --reinstall --no-deps "$WHEEL" \
  > "$OUT/wheel_install.log" 2>&1
"$NATIVE_PY" - "$OUT/installed_extension.json" <<'PY'
import hashlib, json, pathlib, sys
import apxinf_py
folder=pathlib.Path(apxinf_py.__file__).resolve().parent
extensions={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.so')}
if not extensions:
    raise RuntimeError('No imported native extension found')
cls=getattr(apxinf_py,'ModelRunner',getattr(apxinf_py,'Model',None))
if cls is None or not hasattr(cls,'execution_mode'):
    raise RuntimeError('New read-only execution_mode getter missing')
pathlib.Path(sys.argv[1]).write_text(json.dumps({'native_extensions':extensions,
    'execution_mode_getter_present':True,'instances_created':0},indent=2)+'\n')
PY
nvidia-smi > "$OUT/device_after.txt"
date -u +%FT%TZ > "$OUT/end_utc.txt"
printf 'Graph gates passed and accepted wheel installed; remeasure native benchmarks. Evidence: %s\n' "$OUT"
