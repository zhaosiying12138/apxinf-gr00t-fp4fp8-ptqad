#!/bin/bash
set -x
cd /home/zhaosiying/codebase/groot-fsdp2
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
LOG=/mnt/c/fq_ptq_calib/evalchain.log

PORT=5574 bash run_libero_eval_mini.sh "$OUT/gr00t_ptq_aggr" mini_aggr 5 >> "$LOG" 2>&1
sleep 10
PORT=5575 bash run_libero_eval_fp4vla.sh "$OUT/gr00t_ptq_mixed" full_mixed 10 >> "$LOG" 2>&1
echo EVAL_CHAIN_DONE >> "$LOG"
