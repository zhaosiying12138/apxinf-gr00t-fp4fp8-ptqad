#!/bin/bash
set -x
cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib
PY=.venv/bin/python
PTQ=/home/zhaosiying/codebase/fp4vla/quant/ptq
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
BASE=/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10
LOG=/mnt/c/fq_ptq_calib/probe2.log

$PY "$PTQ/probe_eval.py" --ckpt "$BASE" --make-ref >> "$LOG" 2>&1
for r in rtn calib fp8 mixed; do
  $PY "$PTQ/probe_eval.py" --ckpt "$OUT/gr00t_ptq_$r" >> "$LOG" 2>&1
done
echo PROBE2_DONE >> "$LOG"
