#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/common_v12.sh"
python3 -B "$CAPTURE_ROOT/verify_completed_capture.py" --plan "$CAPTURE_ROOT/plan_v12_completed.json" --figure shot_rollout
