#!/bin/bash
# OPD resume loop (same battle-tested pattern as QAD loop)
set -u
LOG=/home/zhaosiying/fq_opd_loop.log
PY=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python
SCRIPT=/home/zhaosiying/codebase/fp4vla/rl/full_head_opd.py
export LD_LIBRARY_PATH="$HOME/miniforge3/envs/media7/lib:${LD_LIBRARY_PATH:-}"
export MALLOC_ARENA_MAX=2
cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
attempt=1
while [ $attempt -le 15 ]; do
  echo "[opdloop] attempt $attempt $(date)" >> $LOG
  if [ -f /mnt/c/fq_opd_out/checkpoint-1000/fp4vla_shard_index.json ]; then
    echo "[opdloop] DONE (ckpt-1000 exists)" >> $LOG; break
  fi
  RESUME=""
  for ck in $(ls -d /mnt/c/fq_opd_out/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); do
    [ -f "$ck/fp4vla_shard_index.json" ] && RESUME="--resume-from-checkpoint $(basename $ck)" && break
  done
  RESUME="$RESUME" $PY $SCRIPT >> /home/zhaosiying/fq_opd_run_a${attempt}.log 2>&1
  echo "[opdloop] attempt $attempt rc=$? $(date)" >> $LOG
  grep -q COMPLETED /home/zhaosiying/fq_opd_run_a${attempt}.log && break
  [ -f /mnt/c/fq_opd_out/checkpoint-1000/fp4vla_shard_index.json ] && break
  attempt=$((attempt+1)); sleep 10
done
echo "[opdloop] exit $(date)" >> $LOG
