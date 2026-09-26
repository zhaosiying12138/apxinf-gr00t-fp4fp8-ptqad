#!/bin/bash
# Standalone save-window cache dropper: watches newest QAD run log, drops page
# cache when within 15 steps of a save boundary (N%100>=85). Dies with VM;
# relaunched together with the resume loop each coordination cycle.
LOG=/home/zhaosiying/qad_wd2.log
while true; do
  L=$(ls -t /home/zhaosiying/fq_qad_run_a*.log 2>/dev/null | head -1)
  N=$(grep -oE "[0-9]+/1000" "$L" 2>/dev/null | tail -1 | cut -d/ -f1)
  if [ "${N:-0}" -ge 1 ] 2>/dev/null; then
    M=$(( N % 100 ))
    if [ "$M" -ge 85 ]; then
      echo 511213 | sudo -S sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null
      echo "[wd2] dropped at $N $(date)" >> $LOG
      sleep 150
    fi
  fi
  sleep 15
done
