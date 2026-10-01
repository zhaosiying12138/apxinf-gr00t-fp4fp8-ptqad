#!/usr/bin/env bash
set -u
parent_pid=3799584
while kill -0 "$parent_pid" 2>/dev/null; do
  sleep 60
done
cd /home/zhaosiying/codebase/fp4vla
PY=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python
exec "$PY" exp/run_w4a4_recovery.py --run-dir results/ptqad_20261003/v11_recovery_r2 --protocol-file exp/recovery_protocol_v11_w4a4_category.json --ptq-selection results/ptqad_20261003/v11_selection/selection.json --base weights/GR00T-N1.7-LIBERO/libero_10 --gr00t-repo /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T --python "$PY" --rollout-python /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python --dataset /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/demo_data/libero_demo --capture-dataset results/ptqad_20261003/w4a4_teacher_supervision --port-base 5780 --adopt-complete --allow-opd-nonimprovement --until all
