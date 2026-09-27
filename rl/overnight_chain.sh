#!/bin/bash
# Overnight chain: wait for mixed 10x10 -> QAD-LoRA (fp8 base) -> merge -> mini
# -> OPD-LoRA (probe-cached teacher-KL) -> merge -> mini.
set -x
LOG=/mnt/c/fq_ptq_calib/overnight.log
CAL=/mnt/c/fq_ptq_calib
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
RL=/home/zhaosiying/codebase/fp4vla/rl
cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib
PY=.venv/bin/python

# 1. wait for the mixed full eval chain to release the GPU
while ! grep -q EVAL_CHAIN_DONE "$CAL/evalchain.log" 2>/dev/null; do sleep 60; done
echo "eval chain done $(date)" >> "$LOG"

# 2. QAD-LoRA on the fp8-arm base (70.8% @ 2.44x), 500 steps
GR00T_BASE_CKPT=$OUT/gr00t_ptq_fp8 QAD_STEPS=500 QAD_BSZ=16 QAD_ACC=1 \
QAD_LR=1e-4 QAD_OUT=/mnt/c/fq_lora_qad \
  $PY $RL/lora_qad.py > $CAL/qad_lora_run.log 2>&1
echo "QAD-LoRA exit $? $(date)" >> "$LOG"

# 3. merge + bake the final checkpoint
LAST=$(ls -d /mnt/c/fq_lora_qad/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LAST" ]; then
  $PY $RL/lora_merge_bake.py --ckpt "$LAST" --out "$OUT/gr00t_qadlora_fp8base" >> "$LOG" 2>&1
  cd /home/zhaosiying/codebase/groot-fsdp2
  PORT=5576 bash run_libero_eval_mini.sh "$OUT/gr00t_qadlora_fp8base" mini_qadlora 5 >> "$LOG" 2>&1
  cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
fi
echo "QAD-LoRA mini done $(date)" >> "$LOG"

# 4. OPD-LoRA (same base, + teacher-KL on probe cache)
GR00T_BASE_CKPT=$OUT/gr00t_ptq_fp8 QAD_STEPS=500 QAD_BSZ=16 QAD_ACC=1 \
QAD_LR=1e-4 QAD_OPD_KL_W=1.0 QAD_OUT=/mnt/c/fq_lora_opd \
  $PY $RL/lora_qad.py > $CAL/opd_lora_run.log 2>&1
echo "OPD-LoRA exit $? $(date)" >> "$LOG"

LAST=$(ls -d /mnt/c/fq_lora_opd/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LAST" ]; then
  $PY $RL/lora_merge_bake.py --ckpt "$LAST" --out "$OUT/gr00t_opdlora_fp8base" >> "$LOG" 2>&1
  cd /home/zhaosiying/codebase/groot-fsdp2
  PORT=5577 bash run_libero_eval_mini.sh "$OUT/gr00t_opdlora_fp8base" mini_opdlora 5 >> "$LOG" 2>&1
fi
echo "OVERNIGHT_CHAIN_DONE $(date)" >> "$LOG"
