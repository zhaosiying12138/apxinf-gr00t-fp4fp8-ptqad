#!/usr/bin/env bash
# Pass --run-dir, --wait-pid and --media-lib explicitly. The helper preserves
# the existing run's protocol and verifies completed stages before adoption.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$script_dir/continue_w4a4_run.py" "$@"
