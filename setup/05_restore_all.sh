#!/usr/bin/env bash
# Restore selected prerequisites; model/closed-loop validation is separate.
# --with-engine adds the independently benchmarked APXInf implementation.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$ROOT"
WITH_ENGINE=0
DOWNLOAD=1
for arg in "$@"; do
  case "$arg" in
    --with-engine) WITH_ENGINE=1 ;;
    --skip-download) DOWNLOAD=0 ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
bash setup/01_install_dev_tools.sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if (( DOWNLOAD )); then bash setup/03_download_weights.sh; fi
bash setup/06_install_recovery.sh
if (( WITH_ENGINE )); then
  ENGINE="$ROOT/third_party/apxinf-robo"
  ENGINE_REV=63540bb02b926a99cf9700a7e59a9253f054806f
  if [[ ! -e "$ENGINE" ]]; then
    git clone --no-checkout https://github.com/RLinf/APXinf-robo.git "$ENGINE"
    git -C "$ENGINE" checkout --detach "$ENGINE_REV"
    git -C "$ENGINE" submodule update --init --recursive
  fi
  [[ "$(git -C "$ENGINE" rev-parse HEAD)" == "$ENGINE_REV" ]] || {
    printf 'APXInf-robo checkout differs from pinned revision: %s\n' "$ENGINE"; exit 2;
  }
  [[ "$(git -C "$ENGINE/apxinf" rev-parse HEAD)" == e07dbe98da914bc0a3df8c4bb4b7251c2a726f69 ]] || {
    printf '%s\n' 'APXInf submodule revision mismatch'; exit 2;
  }
  bash setup/02_build_engine.sh
  bash setup/00_env_report.sh
fi
printf '%s\n' 'Selected setup stages completed. Model loading, training and LIBERO evaluation still require documented smoke runs.'
