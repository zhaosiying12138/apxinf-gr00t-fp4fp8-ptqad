#!/bin/bash
set -x
LOG=/mnt/c/fq_ptq_calib/rwr_final.log
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib

.venv/bin/python /home/zhaosiying/codebase/fp4vla/rl/lora_rwr.py \
  --rollout-dir /mnt/c/fq_rwr_rollout \
  --results-glob "/home/zhaosiying/codebase/groot-fsdp2/runs/eval_mini_rwr_collect/*.log" \
  --base "$OUT/gr00t_ptq_fp8" --out "$OUT/gr00t_rwr_fp8base" \
  --steps 200 --batch 8 >> "$LOG" 2>&1
echo "RWR final train exit $? $(date)" >> "$LOG"

cd /home/zhaosiying/codebase/groot-fsdp2
PORT=5581 bash run_libero_eval_mini.sh "$OUT/gr00t_rwr_fp8base" mini_rwr 5 >> "$LOG" 2>&1
echo RWR_FINAL_DONE >> "$LOG"
