#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/common_v11.sh"
OUT="$(prepare_scratch shot_opdcache)"; assert_no_compute_apps
ROLLOUT="$SCRATCH_ROOT/shot_rollout"; test -d "$ROLLOUT/observations"
test "$(find "$ROLLOUT/observations" -name 'sample_*.pt' | wc -l)" -eq 2
cd "$GR00T_REPO"
export FP4VLA_QUANT=0 FP4VLA_W4A4=0 FP4VLA_W4A4_ADAPTER=0 FP4VLA_SATURATE_F16_ACTIVATIONS=0
"$PTQAD_PYTHON" -u "$PROJECT/rl/opd_probe_cache.py" --teacher "$BF16_TEACHER" --input-dir "$ROLLOUT/observations" --dataset "$DATASET" --count 2 --seed 20261003 --model-dtype float32 --autocast-dtype bfloat16 --device cuda --out "$OUT/teacher_probes.pt" 2>&1 | tee "$OUT/opdcache_2sample.raw.log"
test -f "$OUT/teacher_probes.pt"
printf '%s
' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
