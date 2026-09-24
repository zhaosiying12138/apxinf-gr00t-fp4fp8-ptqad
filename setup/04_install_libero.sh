#!/usr/bin/env bash
# LIBERO eval env (verified working 2026-09-25 on WSL2 Ubuntu 26.04, py3.12).
# Run AFTER setup/02_build_engine.sh. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/third_party/apxinf-robo/.venv"
source "$VENV/bin/activate"
export PATH="$HOME/.local/bin:$PATH"

# 1. system GL libs (EGL/OSMesa via mesa; needs sudo once):
#    sudo apt-get install -y libegl1 libgl1 libgles2 libopengl0 libglvnd0 \
#       libosmesa6 libgl1-mesa-dri
# 2. python deps, version-critical set (do NOT install LIBERO requirements.txt wholesale):
uv pip install "robosuite==1.4.1" "mujoco==3.1.6" "bddl==1.0.1" easydict "protobuf<4" \
   future matplotlib cloudpickle "gym==0.25.2" PyOpenGL pillow
# torch (CPU is enough; LIBERO benchmark imports it at module level)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. LIBERO package: clone + editable + direct path (editable finder is broken)
[ -d "$ROOT/third_party/LIBERO" ] || \
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$ROOT/third_party/LIBERO"
(cd "$ROOT/third_party/LIBERO" && uv pip install -e . ) || true
SITE="$VENV/lib/python3.12/site-packages"
echo "$ROOT/third_party/LIBERO" > "$SITE/libero-src.pth"

# NOTE: an existing ~/codebase/_external/LIBERO also works and may win via its own path;
# both ship the same bddl assets.

# 4. verify full chain (old-gym API: reset->obs; step->4-tuple)
cd "$HOME"
python - <<'EOF' 2>&1 | grep -E "RESET-OK|STEP-OK|FULL CHAIN|Error"
import os, pathlib
os.environ["MUJOCO_GL"] = "egl"
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark
b = get_benchmark("libero_10")()
task = b.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=224, camera_widths=224)
obs = env.reset(); print("RESET-OK", obs["agentview_image"].shape)
o, r, d, info = env.step(env.action_space.sample()); print("STEP-OK", r)
env.close(); print("=== LIBERO FULL CHAIN OK ===")
EOF
echo "LIBERO ENV DONE"
