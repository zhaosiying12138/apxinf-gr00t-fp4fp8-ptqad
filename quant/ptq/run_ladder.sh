#!/bin/bash
# PTQ ladder v2: bake + offline probe eval (recipes: rtn/calib/fp8/mixed)
set -x
cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib
PY=.venv/bin/python
PTQ=/home/zhaosiying/codebase/fp4vla/quant/ptq
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
mkdir -p "$OUT"
CAL=/mnt/c/fq_ptq_calib/calib.pt
LOG=/mnt/c/fq_ptq_calib/ladder2.log
BASE=/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10

for recipe in rtn calib fp8 mixed; do
  echo "=== BAKE $recipe $(date) ===" >> "$LOG"
  $PY "$PTQ/bake.py" --base "$BASE" --out "$OUT/gr00t_ptq_$recipe" \
      --recipe "$recipe" --calib "$CAL" >> "$LOG" 2>&1
  echo "=== PROBE $recipe $(date) ===" >> "$LOG"
  $PY "$PTQ/probe_eval.py" --ckpt "$OUT/gr00t_ptq_$recipe" >> "$LOG" 2>&1
done
echo "=== LADDER2 DONE $(date) ===" >> "$LOG"
