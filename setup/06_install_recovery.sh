#!/usr/bin/env bash
# Reconstruct captured training and separate simulation environments.
# A clean installation was not executed during the paper run.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="$HOME/.local/bin:$PATH"
GROOT=${GR00T_REPO:-$ROOT/third_party/Isaac-GR00T}
MEDIA=${PTQAD_MEDIA_PREFIX:-$ROOT/third_party/media7}
LIBERO_CONFIG=${LIBERO_CONFIG_PATH:-$ROOT/third_party/libero-config}
UV=${UV:-uv}
UPSTREAM=51d4c89f72fda44cbf77285c6a8114b52676b8a1
LIBERO_REV=8f1084e3132a39270c3a13ebe37270a43ece2a01
PATCH="$ROOT/patches/gr00t-recovery-runtime.patch"
command -v "$UV" >/dev/null
command -v git >/dev/null
command -v python3 >/dev/null
python3 - "$ROOT" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); lock=root/'setup/locks'; m=json.loads((lock/'manifest.json').read_text())
for name,sha in m['lock_sha256'].items():
    if hashlib.sha256((lock/name).read_bytes()).hexdigest()!=sha:
        raise ValueError(f'Lock checksum mismatch: {name}')
if hashlib.sha256((root/m['runtime_patch']).read_bytes()).hexdigest()!=m['runtime_patch_sha256']:
    raise ValueError('Runtime patch checksum mismatch')
PY
if [[ ! -e "$GROOT" ]]; then
  mkdir -p -- "$(dirname -- "$GROOT")"
  git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT"
  git -C "$GROOT" checkout --detach "$UPSTREAM"
fi
[[ "$(git -C "$GROOT" rev-parse HEAD)" == "$UPSTREAM" ]] || {
  printf 'Refusing to alter another checkout: %s; choose a new GR00T_REPO.\n' "$GROOT" >&2; exit 2;
}
if git -C "$GROOT" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  printf '%s\n' 'Recorded runtime patch is already present.'
else
  git -C "$GROOT" diff --quiet
  git -C "$GROOT" diff --cached --quiet
  git -C "$GROOT" apply --check "$PATCH"
  git -C "$GROOT" apply "$PATCH"
fi
git -C "$GROOT" submodule update --init external_dependencies/LIBERO
[[ "$(git -C "$GROOT/external_dependencies/LIBERO" rev-parse HEAD)" == "$LIBERO_REV" ]] || {
  printf '%s\n' 'LIBERO revision mismatch'; exit 2;
}

# torchcodec 0.8 used FFmpeg 7; the system FFmpeg 8 is not interchangeable.
if [[ ! -f "$MEDIA/conda-meta/history" ]]; then
  CONDA=${CONDA_EXE:-$HOME/miniforge3/bin/conda}
  [[ -x "$CONDA" ]] || { printf 'Set CONDA_EXE to an installed conda executable.\n' >&2; exit 2; }
  "$CONDA" create --yes --prefix "$MEDIA" --file "$ROOT/setup/locks/media7-linux-64.explicit.txt"
fi
python3 - "$ROOT/setup/locks/media7-environment.json" "$MEDIA" <<'PY'
import json,sys
from pathlib import Path
expected=json.loads(Path(sys.argv[1]).read_text())['packages']
installed={p['name']:p for path in (Path(sys.argv[2])/'conda-meta').glob('*.json')
           for p in [json.loads(path.read_text())]}
for package in expected:
    actual=installed.get(package['name'],{})
    if any(actual.get(k)!=package[k] for k in ('version','build')):
        raise ValueError(f'Media dependency differs: {package["name"]}')
PY
export LD_LIBRARY_PATH="$MEDIA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$MEDIA/bin:$PATH"
[[ -x "$GROOT/.venv/bin/python" ]] || "$UV" venv "$GROOT/.venv" --python 3.12.14
[[ -x "$GROOT/.venv-libero/bin/python" ]] || "$UV" venv "$GROOT/.venv-libero" --python 3.12.14
TRAIN_PY="$GROOT/.venv/bin/python"
SIM_PY="$GROOT/.venv-libero/bin/python"
export DS_BUILD_OPS=0
"$UV" pip install --python "$TRAIN_PY" --no-deps torch==2.9.0+cu128 torchvision==0.24.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
TMP_REQUIREMENTS=$(mktemp)
trap 'rm -f -- "$TMP_REQUIREMENTS"' EXIT
sed -E '/^(torch|torchvision|deepspeed)==/d' "$ROOT/setup/locks/recovery-py312.txt" > "$TMP_REQUIREMENTS"
"$UV" pip install --python "$TRAIN_PY" --no-deps -r "$TMP_REQUIREMENTS"
"$UV" pip install --python "$TRAIN_PY" --no-deps --no-build-isolation deepspeed==0.17.6
"$UV" pip install --python "$TRAIN_PY" --no-deps -e "$GROOT"
"$UV" pip install --python "$SIM_PY" --no-deps -r "$ROOT/setup/locks/libero-py312.txt"
"$UV" pip install --python "$SIM_PY" --no-deps -e "$GROOT" -e "$GROOT/external_dependencies/LIBERO"

# Keep the configuration within this checkout instead of overwriting ~/.libero.
mkdir -p -- "$LIBERO_CONFIG"
"$TRAIN_PY" - "$GROOT" "$LIBERO_CONFIG/config.yaml" <<'PY'
from pathlib import Path
import sys,yaml
root=Path(sys.argv[1])/'external_dependencies/LIBERO/libero/libero'
data={'benchmark_root':str(root),'bddl_files':str(root/'bddl_files'),
      'init_states':str(root/'init_files'),'datasets':str(root.parent/'datasets'),
      'assets':str(root/'assets')}
out=Path(sys.argv[2]); new=yaml.safe_dump(data)
if out.exists() and yaml.safe_load(out.read_text())!=data:
    raise ValueError(f'Existing LIBERO config differs: {out}')
out.write_text(new)
PY
for pair in "recovery:$TRAIN_PY" "libero:$SIM_PY"; do
  kind=${pair%%:*}; interpreter=${pair#*:}
  "$interpreter" - "$ROOT/setup/locks/${kind}-py312.txt" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
if sys.version_info[:3] != (3,12,14): raise RuntimeError(f'Python differs: {sys.version}')
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line or line.startswith('#'): continue
    name,want=line.split('==',1)
    if version(name)!=want: raise RuntimeError(f'{name}: {version(name)} != {want}')
print(f'All recorded package versions match: {sys.argv[1]}')
PY
done
# The backbone factory checks for the literal nvidia/Cosmos-Reason2 substring.
# Preserve that name in local paths; resolving a HF snapshot hash loses it.
BACKBONE="$ROOT/weights/nvidia/Cosmos-Reason2-2B"
if [[ ! -e "$BACKBONE" && -d "$ROOT/weights/Cosmos-Reason2-2B" ]]; then
  mkdir -p -- "$ROOT/weights/nvidia"
  ln -s ../Cosmos-Reason2-2B "$BACKBONE"
fi
[[ -f "$BACKBONE/config.json" ]] || {
  printf 'Missing local Cosmos config: %s; run setup/03_download_weights.sh.\n' "$BACKBONE" >&2; exit 2;
}
ENV_FILE="$ROOT/setup/recovery-env.sh"
{
  printf '# Generated by setup/06_install_recovery.sh; source this file.\n'
  printf 'export GR00T_REPO=%q\n' "$GROOT"
  printf 'export PTQAD_PYTHON=%q\n' "$TRAIN_PY"
  printf 'export LIBERO_PYTHON=%q\n' "$SIM_PY"
  printf 'export PTQAD_MEDIA_LIB=%q\n' "$MEDIA/lib"
  printf 'export LIBERO_CONFIG_PATH=%q\n' "$LIBERO_CONFIG"
  printf 'export GR00T_BACKBONE_MODEL=%q\n' "$BACKBONE"
  printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$MEDIA/lib"
} > "$ENV_FILE"
printf 'Versions and patch verified. Source %s before separate model/eval smoke runs.\n' "$ENV_FILE"
