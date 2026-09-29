#!/usr/bin/env bash
# Restore the paper's exact local model bytes. No model is executed here.
# Cosmos/pi05 revisions are proved by retained local HF metadata + content ETags.
# GR00T uses a verified byte-equivalent revision; original download is unresolved.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
W=${PTQAD_WEIGHTS_DIR:-$ROOT/weights}
MANIFEST=${PTQAD_MODEL_SOURCE_MANIFEST:-$ROOT/setup/locks/model-sources.json}
HF_HUB_VERSION=${PTQAD_HF_HUB_VERSION:-0.36.2}
COSMOS_HF_REVISION=${COSMOS_HF_REVISION:-9ce19a195e423419c349abfc86fd07178b230561}
PI05_HF_REVISION=${PI05_HF_REVISION:-a217bfd3b14673cf2ce597e69997ab21866438dd}
GR00T_HF_REVISION=${GR00T_HF_REVISION:-2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21}
MODELS=(gr00t cosmos pi05)
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --core-only) MODELS=(gr00t cosmos) ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) printf 'Usage: %s [--core-only] [--verify-only]\n' "$0" >&2; exit 2 ;;
  esac
done
if (( VERIFY_ONLY )); then
  exec python3 "$ROOT/setup/verify_weights.py" --manifest "$MANIFEST" --weights-root "$W" --models "${MODELS[@]}"
fi
mkdir -p -- "$W"
# An existing mismatched file must not be silently overwritten by a downloader.
python3 "$ROOT/setup/verify_weights.py" --manifest "$MANIFEST" --weights-root "$W" --models "${MODELS[@]}" --existing-only
if [[ "$GR00T_HF_REVISION" == main ]]; then
  printf '%s\n' 'GR00T original HF revision is unknown: main is only a download selector; known content hashes remain mandatory.' >&2
fi
HF() {
  local tries=0
  until (( tries >= 5 )); do
    if uvx --from "huggingface_hub==$HF_HUB_VERSION" hf download "$@"; then return 0; fi
    tries=$((tries+1)); printf 'Download failed; retry %s/5.\n' "$tries" >&2
    sleep 3
    export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  done
  return 1
}
for model in "${MODELS[@]}"; do
  mapfile -t spec < <(python3 - "$MANIFEST" "$model" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))['models'][sys.argv[2]]
print(m['repo_id']); print(m['local_dir'])
for row in m['files']: print(row['path'])
PY
  )
  (( ${#spec[@]} >= 3 )) || { printf 'Invalid model manifest entry: %s\n' "$model" >&2; exit 2; }
  case "$model" in
    gr00t) revision=$GR00T_HF_REVISION ;;
    cosmos) revision=$COSMOS_HF_REVISION ;;
    pi05) revision=$PI05_HF_REVISION ;;
  esac
  HF "${spec[0]}" --revision "$revision" --local-dir "$W/${spec[1]}" --include "${spec[@]:2}"
  python3 "$ROOT/setup/verify_weights.py" --manifest "$MANIFEST" --weights-root "$W" --models "$model" --skip-assets
done
if [[ " ${MODELS[*]} " == *' pi05 '* ]]; then
  # Existing accepted assets are preserved; a changed remote object fails closed.
  while IFS=$'\t' read -r relative url expected; do
    destination="$W/$relative"
    if [[ -e "$destination" ]]; then
      printf '%s  %s\n' "$expected" "$destination" | sha256sum --check -
      continue
    fi
    mkdir -p -- "$(dirname -- "$destination")"
    temporary=$(mktemp "${destination}.download.XXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    curl --fail --location --retry 3 "$url" --output "$temporary"
    printf '%s  %s\n' "$expected" "$temporary" | sha256sum --check -
    # No replacement if another process created the destination while downloading.
    ln -- "$temporary" "$destination"
    rm -f -- "$temporary"
    trap - EXIT
  done < <(python3 - "$MANIFEST" <<'PY'
import json,sys
for a in json.load(open(sys.argv[1]))['external_assets']:
    print(a['path'],a['source_url_recorded_in_existing_setup_script'],a['sha256'],sep='\t')
PY
  )
fi
python3 "$ROOT/setup/verify_weights.py" --manifest "$MANIFEST" --weights-root "$W" --models "${MODELS[@]}"
printf '%s\n' 'Selected model bytes verified. Original unresolved revisions remain explicitly recorded in model-sources.json.'
