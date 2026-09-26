#!/bin/bash
# QAD resume loop: save/100 + resume-from-checkpoint + save_total_limit 2.
# Proven playbook from sister E4 (9 restarts to 2000 steps): accept VM crashes
# at save gathers, resume from last good checkpoint each time. Each crash costs
# <=100 steps (~15-22min). Wrapper relaunches until COMPLETED.
set -u
LOG=/home/zhaosiying/fq_qad_loop.log
PY=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python
SCRIPT=/home/zhaosiying/codebase/fp4vla/rl/full_head_qad.py
export LD_LIBRARY_PATH="$HOME/miniforge3/envs/media7/lib:${LD_LIBRARY_PATH:-}"
cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T

attempt=1
while [ $attempt -le 12 ]; do
  echo "[loop] attempt $attempt $(date)" >> $LOG
  OUT=/home/zhaosiying/fq_qad_run_a${attempt}.log
  if [ -d /home/zhaosiying/fq_qad_out/checkpoint-1000 ] && \
     [ -f /home/zhaosiying/fq_qad_out/checkpoint-1000/model.safetensors ]; then
    echo "[loop] checkpoint-1000 weights exist - DONE" >> $LOG; break
  fi
  RESUME=""
  # newest good checkpoint with weights
  for ck in $(ls -d /home/zhaosiying/fq_qad_out/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); do
    [ -f "$ck/model.safetensors" ] && RESUME="--resume-from-checkpoint $(basename $ck)" && break
  done
  if [ -n "$RESUME" ]; then
    echo "[loop] resuming with $RESUME" >> $LOG
    RESUME_ARGS="$RESUME"
  else
    RESUME_ARGS=""
  fi
  RESUME="$RESUME_ARGS" $PY $SCRIPT >> $OUT 2>&1
  RC=$?
  echo "[loop] attempt $attempt rc=$RC $(date)" >> $LOG
  if grep -q "COMPLETED" $OUT; then echo "[loop] COMPLETED" >> $LOG; break; fi
  if [ -f /home/zhaosiying/fq_qad_out/checkpoint-1000/model.safetensors ]; then
    echo "[loop] ckpt1000 written post-crash - DONE" >> $LOG; break
  fi
  attempt=$((attempt+1))
  sleep 10
done
echo "[loop] loop exit $(date)" >> $LOG
