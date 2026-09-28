#!/bin/bash
# Data-volume ablation: QAD-500 trained on the FULL LIBERO-10 demo set (379 eps)
# vs the original 5-episode sample. Same recipe, same budget, same eval protocol.
set -x
LOG=/mnt/c/fq_ptq_calib/fulldata.log
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
RL=/home/zhaosiying/codebase/fp4vla/rl
DATA=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$HOME/miniforge3/envs/media7/lib:${LD_LIBRARY_PATH:-}"

cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T

# 1) train QAD-500 on full data (kl_w=0: pure QAD arm, matching the original)
QAD_DATASET="$DATA" QAD_OUT=/mnt/c/fq_lora_qad_full QAD_STEPS=500 \
  .venv/bin/python $RL/lora_qad.py >> "$LOG" 2>&1
RC=$?
echo "[chain] train rc=$RC $(date)" >> "$LOG"
[ $RC -ne 0 ] && { echo "[chain] TRAIN FAILED" >> "$LOG"; exit 1; }
[ -f /mnt/c/fq_lora_qad_full/checkpoint-500/model.safetensors ] || {
  echo "[chain] no checkpoint-500" >> "$LOG"; exit 1; }

# 2) merge into deployable bake
rm -rf "$OUT/gr00t_qadlora_fulldata"
.venv/bin/python $RL/lora_merge_bake.py --ckpt /mnt/c/fq_lora_qad_full/checkpoint-500 \
  --out "$OUT/gr00t_qadlora_fulldata" >> "$LOG" 2>&1
echo "[chain] merge rc=$? $(date)" >> "$LOG"

# 3) full 10x10 closed loop
cd /home/zhaosiying/codebase/groot-fsdp2
PORT=5585 bash run_libero_eval_fp4vla.sh "$OUT/gr00t_qadlora_fulldata" full_qadlora_fulldata 10 >> "$LOG" 2>&1
echo "[chain] FULLDATA_CHAIN_DONE $(date)" >> "$LOG"
