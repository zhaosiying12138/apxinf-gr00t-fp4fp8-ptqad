#!/usr/bin/env bash
# LIBERO eval env (robosuite/MuJoCo, EGL headless) into the robo venv.
# Run AFTER setup/02_build_engine.sh. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/third_party/apxinf-robo/.venv"
source "$VENV/bin/activate"

# libero: PyPI first, git fallback (Lifelong-Robot-Learning/LIBERO)
uv pip install libero 2>/dev/null || {
  [ -d "$ROOT/third_party/LIBERO" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$ROOT/third_party/LIBERO"
  uv pip install -e "$ROOT/third_party/LIBERO"
}
uv pip install robosuite mujoco "protobuf<4" pillow

# headless rendering probe (EGL under WSL2)
export MUJOCO_GL=egl
python - <<'EOF'
import os
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
try:
    import mujoco
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><geom type='sphere' size='0.1'/></body></worldbody></mujoco>")
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 64, 64)
    r.update_scene(d); r.render(); print("EGL render OK")
except Exception as e:
    print("EGL probe failed:", e); print("fallback: try MUJOCO_GL=osmesa (apt install libosmesa6-dev)")
EOF
python -c "from libero.libero import get_libero_path; print('libero OK', get_libero_path('bddl_files'))"
echo "LIBERO ENV DONE"
