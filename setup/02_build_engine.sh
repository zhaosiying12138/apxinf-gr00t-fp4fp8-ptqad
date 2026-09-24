#!/usr/bin/env bash
# Build the ApxInf engine for the local GPU (sm_120 on RTX 5090 Laptop) + robo layer.
# Heavy: first run compiles all CUDA kernels (~10-40 min). Idempotent-ish (cargo caches).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/third_party/apxinf-robo"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
# WSL: linker cannot find -lcuda (only libcuda.so.1 in /usr/lib/wsl/lib); provide a stub
mkdir -p "$HOME/.cuda-stubs"
ln -sf /usr/lib/wsl/lib/libcuda.so.1 "$HOME/.cuda-stubs/libcuda.so"
export LIBRARY_PATH="$HOME/.cuda-stubs:${LIBRARY_PATH:-}"

[ -d "$REPO/apxinf" ] || { echo "missing $REPO (run git clone --recursive first)"; exit 1; }

# python 3.12 venv (system python 3.14 too new for some wheels)
cd "$REPO"
if [ ! -d .venv ]; then uv venv .venv --python 3.12; fi
source .venv/bin/activate
uv pip install -U maturin pip

# engine wheel (auto-detects local arch; override if it mis-detects)
cd "$REPO/apxinf"
CARGO_TARGET_DIR=target/wheel ${APXINF_CUDA_ARCH:+APXINF_CUDA_ARCH=$APXINF_CUDA_ARCH} \
  maturin build --release --features cuda --auditwheel skip -m crates/apxinf-py/Cargo.toml
uv pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
uv pip install -e "python/apxinf[serving]"

# robo layer
cd "$REPO"
uv pip install -e ".[libero,serve,dev]"

# NOTE: run import checks from outside the repo: the apxinf/ submodule directory
# shadows the installed `apxinf` python package when CWD is the repo root.
cd "$HOME"
python -c 'import apxinf_py; print("apxinf_py", apxinf_py.__version__)'
python -c 'import apxinf, apxinf_robo; print("apxinf_robo ok")'
echo "ENGINE BUILD DONE"

# apply fp4vla engine patches (fp4 path, fp8 sm_120 fix, instrumentation)
cd "$REPO/apxinf"
git apply --check "$ROOT/patches/apxinf-fp4vla-engine.patch" 2>/dev/null \
  && git apply "$ROOT/patches/apxinf-fp4vla-engine.patch" && echo "engine patches applied" \
  || echo "engine patches: already applied or failed (check manually)"
