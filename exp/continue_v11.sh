#!/usr/bin/env bash
set -u
pid=3799584
while kill -0 "$pid" 2>/dev/null; do
  sleep 60
done
if [ ! -f results/ptqad_20261003/v11_recovery_r2/stages/select_qad_lr.json ]; then
  echo '[continuation] driver exited before QAD selection marker; refusing automatic continuation' >&2
  exit 2
fi
# The first driver was started before QAD_SAVE_TOTAL_LIMIT was wired into the
# trainer. Keep the verified final checkpoints and remove only superseded
# full-model save directories before the continuation allocates more shards.
for train_root in results/ptqad_20261003/v11_recovery_r2/artifacts/train_qad_lr_*; do
  [ -d "$train_root" ] || continue
  for checkpoint in "$train_root"/checkpoint-*; do
    [ -d "$checkpoint" ] || continue
    case "$checkpoint" in
      */checkpoint-2000) ;;
      *) rm -rf "$checkpoint" ;;
    esac
  done
done
exec /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python exp/run_w4a4_recovery.py \
  --run-dir results/ptqad_20261003/v11_recovery_r2 \
  --protocol-file exp/recovery_protocol_v11_w4a4_category.json \
  --ptq-selection results/ptqad_20261003/v11_selection/selection.json \
  --base weights/GR00T-N1.7-LIBERO/libero_10 \
  --gr00t-repo /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T \
  --python /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python \
  --rollout-python /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
  --dataset /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/demo_data/libero_demo \
  --capture-dataset results/ptqad_20261003/w4a4_teacher_supervision \
  --port-base 5780 --cleanup-duplicates --until all
