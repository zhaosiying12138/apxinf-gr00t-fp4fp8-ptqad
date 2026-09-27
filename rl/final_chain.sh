#!/bin/bash
# Final overnight chain: 10x10 full evals for QAD-LoRA & OPD-LoRA recovery arms,
# then RWR arm (collect rollouts with logging -> train -> mini).
set -x
LOG=/mnt/c/fq_ptq_calib/final.log
CAL=/mnt/c/fq_ptq_calib
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
RL=/home/zhaosiying/codebase/fp4vla/rl
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib

cd /home/zhaosiying/codebase/groot-fsdp2
PORT=5578 bash run_libero_eval_fp4vla.sh "$OUT/gr00t_qadlora_fp8base" full_qadlora 10 >> "$LOG" 2>&1
sleep 10
PORT=5579 bash run_libero_eval_fp4vla.sh "$OUT/gr00t_opdlora_fp8base" full_opdlora 10 >> "$LOG" 2>&1
echo "FULL-10x10 BOTH DONE $(date)" >> "$LOG"
sleep 10

# RWR arm: collect on-policy rollouts from the fp8 base with server logging
rm -rf /mnt/c/fq_rwr_rollout
FP4VLA_LOG_DIR=/mnt/c/fq_rwr_rollout PORT=5580 \
  bash run_libero_eval_mini.sh "$OUT/gr00t_ptq_fp8" mini_rwr_collect 5 >> "$LOG" 2>&1
sleep 10

cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
.venv/bin/python $RL/lora_rwr.py \
  --rollout-dir /mnt/c/fq_rwr_rollout \
  --results-glob "/home/zhaosiying/codebase/groot-fsdp2/runs/eval_mini_rwr_collect/*.log" \
  --base "$OUT/gr00t_ptq_fp8" --out "$OUT/gr00t_rwr_fp8base" \
  --steps 200 --batch 8 >> "$LOG" 2>&1
echo "RWR train exit $? $(date)" >> "$LOG"

cd /home/zhaosiying/codebase/groot-fsdp2
PORT=5581 bash run_libero_eval_mini.sh "$OUT/gr00t_rwr_fp8base" mini_rwr 5 >> "$LOG" 2>&1
echo FINAL_CHAIN_DONE >> "$LOG"
