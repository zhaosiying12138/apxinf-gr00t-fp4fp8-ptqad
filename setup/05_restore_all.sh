#!/usr/bin/env bash
# Complete environment restoration — reproduces EVERYTHING in this repo from
# a fresh WSL2 Ubuntu + RTX 5090 (sm_120) machine. Idempotent per step.
# Rough timeline on a 500Mbps line: ~1.5h (weights dominate).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
step(){ printf "\n=== %s ===\n" "$1"; }

step "0. environment snapshot (record what we build on)"
bash setup/00_env_report.sh || true

step "1. dev tools (rustup, uv; CUDA toolkit assumed at /usr/local/cuda-13.x + sudo GL libs)"
bash setup/01_install_dev_tools.sh || true
echo "MANUAL PREREQS (cannot be scripted): CUDA Toolkit >=12.8, NVIDIA driver (WSL host),
sudo apt install libegl1 libgl1 libgles2 libopengl0 libglvnd0 libosmesa6 libgl1-mesa-dri"

step "2. engine build (clone APXinf-robo + apply fp4vla patches + maturin sm_120)"
bash setup/02_build_engine.sh

step "3. weights (~22GB total, resumable; GR00T lean policy excludes optimizer states)"
bash setup/03_download_weights.sh

step "4. LIBERO eval environment (robosuite/mujoco/EGL, verified pin set)"
bash setup/04_install_libero.sh

step "5. PyTorch baselines envs (lerobot for pi05, gr00t pkg for N1.7)"
[ -d baselines/.venv ] || {
  export PATH="$HOME/.local/bin:$PATH"; cd baselines
  uv venv .venv-pi05 --python 3.12 -q && (source .venv-pi05/bin/activate &&
    uv pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128 &&
    uv pip install "lerobot[pi]" sentencepiece safetensors)
  uv venv .venv --python 3.12 -q && (source .venv/bin/activate &&
    uv pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128 &&
    uv pip install -e ./Isaac-GR00T) || echo "GR00T source missing: rsync from upstream"
  cd "$ROOT"; }

step "6. NVFP4 artifacts (pure CPU, ~15min total; all regenerate from checkpoints)"
source baselines/.venv-pi05/bin/activate 2>/dev/null || true
( cd quant && [ -d ../weights/pi05_libero_base ] && {
  python nvfp4_convert_packed.py --ckpt ../weights/pi05_libero_base/model.safetensors --out ../weights/pi05.nvfp4.packed || true
  python nvfp4_convert_packed.py --ckpt ../weights/pi05_libero_base/model.safetensors --out ../weights/pi05.nvfp4.lang --scope lang || true
  python nvfp4_convert_packed.py --ckpt ../weights/pi05_libero_base/model.safetensors --out ../weights/pi05.nvfp4.act --scope act || true
  python nvfp4_convert.py --ckpt ../weights/pi05_libero_base/model.safetensors --out ../weights/pi05.nvfp4 || true ; } )

step "7. self-tests (bit-exact quantizer, adapter GEMM, gold composition)"
( cd spike && make >/dev/null 2>&1 || true
  ./fp4_quant_test 2>/dev/null | tail -1
  ./fp4_adapter_test 2>/dev/null | tail -2
  ./gold_check4 2>/dev/null | tail -2 ) || true

echo
echo "RESTORE COMPLETE — sanity gates:"
echo "  bench:  cd ~ && <robo-venv>/python ~/codebase/fp4vla/exp/bench_engine.py --model-dir .../pi05_libero_base --variant nvfp4_static ..."
echo "  screen: <robo-venv>/python ~/codebase/fp4vla/exp/run_sensitivity.py --stage screen"
