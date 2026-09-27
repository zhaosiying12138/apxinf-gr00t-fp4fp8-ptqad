#!/bin/bash
# Sequential mini closed-loop (3 tasks x 5 eps) for PTQ candidates.
set -x
cd /home/zhaosiying/codebase/groot-fsdp2
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
LOG=/mnt/c/fq_ptq_calib/mini.log

PORT=5571 bash run_libero_eval_mini.sh "$OUT/gr00t_ptq_mixed"  mini_mixed  5 >> "$LOG" 2>&1
sleep 10
PORT=5572 bash run_libero_eval_mini.sh "$OUT/gr00t_ptq_fp8"    mini_fp8    5 >> "$LOG" 2>&1
sleep 10
PORT=5573 bash run_libero_eval_mini.sh "$OUT/gr00t_ptq_calib"  mini_calib  5 >> "$LOG" 2>&1
echo MINI_CHAIN_DONE >> "$LOG"
