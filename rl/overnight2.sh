#!/bin/bash
# Overnight chain v2 (additive LoRA semantics, native-speed forward):
# QAD-LoRA -> merge -> mini -> OPD-LoRA -> merge -> mini. Mixed 10x10 already done.
set -x
LOG=/mnt/c/fq_ptq_calib/overnight2.log
CAL=/mnt/c/fq_ptq_calib
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
RL=/home/zhaosiying/codebase/fp4vla/rl
cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib
PY=.venv/bin/python

GR00T_BASE_CKPT=$OUT/gr00t_ptq_fp8 QAD_STEPS=500 QAD_BSZ=16 QAD_ACC=1 \
QAD_LR=1e-4 QAD_OUT=/mnt/c/fq_lora_qad \
  $PY $RL/lora_qad.py > $CAL/qad_lora_run2.log 2>&1
echo "QAD-LoRA exit $? $(date)" >> "$LOG"

LAST=$(ls -d /mnt/c/fq_lora_qad/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LAST" ]; then
  $PY $RL/lora_merge_bake.py --ckpt "$LAST" --out "$OUT/gr00t_qadlora_fp8base" >> "$LOG" 2>&1
  cd /home/zhaosiying/codebase/groot-fsdp2
  PORT=5576 bash run_libero_eval_mini.sh "$OUT/gr00t_qadlora_fp8base" mini_qadlora 5 >> "$LOG" 2>&1
  cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
fi
echo "QAD-LoRA mini done $(date)" >> "$LOG"

GR00T_BASE_CKPT=$OUT/gr00t_ptq_fp8 QAD_STEPS=500 QAD_BSZ=16 QAD_ACC=1 \
QAD_LR=1e-4 QAD_OPD_KL_W=1.0 QAD_OUT=/mnt/c/fq_lora_opd \
  $PY $RL/lora_qad.py > $CAL/opd_lora_run2.log 2>&1
echo "OPD-LoRA exit $? $(date)" >> "$LOG"

LAST=$(ls -d /mnt/c/fq_lora_opd/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LAST" ]; then
  $PY $RL/lora_merge_bake.py --ckpt "$LAST" --out "$OUT/gr00t_opdlora_fp8base" >> "$LOG" 2>&1
  cd /home/zhaosiying/codebase/groot-fsdp2
  PORT=5577 bash run_libero_eval_mini.sh "$OUT/gr00t_opdlora_fp8base" mini_opdlora 5 >> "$LOG" 2>&1
fi
echo "OVERNIGHT2_DONE $(date)" >> "$LOG"
