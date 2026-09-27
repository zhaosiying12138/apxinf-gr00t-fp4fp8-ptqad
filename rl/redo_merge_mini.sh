#!/bin/bash
# Re-merge (fixed) both LoRA checkpoints and run their mini evals + collect
# the exact mixed 10x10 aggregate.
set -x
LOG=/mnt/c/fq_ptq_calib/redo.log
CAL=/mnt/c/fq_ptq_calib
OUT=/home/zhaosiying/codebase/fp4vla/weights/ptq_bakes
RL=/home/zhaosiying/codebase/fp4vla/rl
cd /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib
PY=.venv/bin/python

rm -rf "$OUT/gr00t_qadlora_fp8base" "$OUT/gr00t_opdlora_fp8base"
$PY $RL/lora_merge_bake.py --ckpt /mnt/c/fq_lora_qad/checkpoint-500 \
    --out "$OUT/gr00t_qadlora_fp8base" >> "$LOG" 2>&1
LAST_OPD=$(ls -d /mnt/c/fq_lora_opd/checkpoint-* | sort -V | tail -1)
$PY $RL/lora_merge_bake.py --ckpt "$LAST_OPD" \
    --out "$OUT/gr00t_opdlora_fp8base" >> "$LOG" 2>&1

# exact mixed aggregate
python3 - << 'EOF' >> "$LOG" 2>&1
import re, glob
tot = succ = 0; ntask = 0
for f in glob.glob("/home/zhaosiying/codebase/groot-fsdp2/runs/eval_full_mixed/*.log"):
    if "server" in f: continue
    txt = open(f, errors="ignore").read()
    m = re.search(r"results:.*?(\[.*?\])", txt)
    sr = re.search(r"success rate:\s*([0-9.]+)", txt)
    if sr:
        ntask += 1
        # episodes count from results booleans
        bools = m.group(1) if m else "[]"
        n = bools.count("True") + bools.count("False")
        tot += n; succ += bools.count("True")
print(f"MIXED-10x10: tasks={ntask} episodes={tot} successes={succ} rate={succ/max(tot,1):.4f}")
EOF

cd /home/zhaosiying/codebase/groot-fsdp2
PORT=5576 bash run_libero_eval_mini.sh "$OUT/gr00t_qadlora_fp8base" mini_qadlora 5 >> "$LOG" 2>&1
sleep 10
PORT=5577 bash run_libero_eval_mini.sh "$OUT/gr00t_opdlora_fp8base" mini_opdlora 5 >> "$LOG" 2>&1
echo REDO_DONE >> "$LOG"
