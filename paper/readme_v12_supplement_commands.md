### 补充动作诊断与校准 GPTQ

以下步骤在主五臂全部完成、GPU 空闲后串行执行。它们使用独立输出，保留主实验的选模和评测协议。现有清单可复用，但采集、量化、评测及发布输出不得覆盖；遇到基础设施故障时先保留原日志，再按补充协议登记重试目录。

```bash
set -euo pipefail
cd /home/zhaosiying/codebase/fp4vla
PTQAD_ROOT="$PWD"
PTQAD_GR00T=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
PTQAD_PY="$PTQAD_GR00T/.venv/bin/python"
PTQAD_BASE="$PTQAD_ROOT/weights/GR00T-N1.7-LIBERO/libero_10"
PTQAD_RUN="$PTQAD_ROOT/results/reruns/rtn_w4a4_release_20261006_01"
PTQAD_DIAG="$PTQAD_ROOT/paper/_build/w4a4_action_diagnostics_v12"
PTQAD_CAL="$PTQAD_ROOT/paper/_build/captured_ptq_v12"
PTQAD_GPTQ="$PTQAD_ROOT/results/supplement_gptq_capture_v12"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -s "$PTQAD_RUN/recovery_v12/final_manifest.json"
python3 - "$PTQAD_RUN/recovery_v12/run_manifest.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1]))['status'] != 'complete':
    raise SystemExit('Main five-arm run has not completed')
PY
PTQAD_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$PTQAD_GPU_PIDS"
if [ ! -e "$PTQAD_DIAG/inputs.json" ]; then
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
    exp/action_chunk_diagnostics.py freeze \
    --capture-root "$PTQAD_RUN/teacher_supervision_v12_clean/observations" \
    --protocol-file exp/recovery_protocol_v12_rtn_w4a4.json \
    --per-task 8 --seed 2026100600 --out "$PTQAD_DIAG/inputs.json"
fi
for arm in bf16 ptq qad continued_qad qad_opd; do
  test ! -e "$PTQAD_DIAG/$arm"
  test ! -e "$PTQAD_DIAG/$arm.log"
  OMP_NUM_THREADS=1 "$PTQAD_PY" exp/action_chunk_diagnostics.py collect \
    --inputs "$PTQAD_DIAG/inputs.json" \
    --final-manifest "$PTQAD_RUN/recovery_v12/final_manifest.json" \
    --arm "$arm" --gr00t "$PTQAD_GR00T" --out "$PTQAD_DIAG/$arm" \
    > "$PTQAD_DIAG/$arm.log" 2>&1
done
if [ ! -e "$PTQAD_CAL/inputs.json" ]; then
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
    quant/ptq/captured_calibration.py \
    --root "$PTQAD_RUN/teacher_supervision_v12_clean" \
    --protocol-file exp/recovery_protocol_v12_rtn_w4a4.json \
    --teacher "$PTQAD_BASE" --out "$PTQAD_CAL/inputs.json"
fi
python3 eval/compare_gptq_reference.py --preflight-only
test ! -e "$PTQAD_GPTQ"
mkdir -p "$PTQAD_GPTQ"
(
  cd "$PTQAD_GR00T"
  OMP_NUM_THREADS=4 "$PTQAD_PY" "$PTQAD_ROOT/quant/ptq/collector.py" \
    --base "$PTQAD_BASE" --out "$PTQAD_GPTQ/ordinary_h" \
    --capture-manifest "$PTQAD_CAL/inputs.json" \
    --recipe calib --windows 148 --batch 1 --seed 2026100601 \
    --device cuda --cpu-threads 4
) 2>&1 | tee "$PTQAD_GPTQ/ordinary_collect.log"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  quant/ptq/verify_calibration.py \
  --calib "$PTQAD_GPTQ/ordinary_h" --expected-windows 148
OMP_NUM_THREADS=4 "$PTQAD_PY" quant/ptq/bake.py \
  --base "$PTQAD_BASE" --out "$PTQAD_GPTQ/ordinary_parent" \
  --recipe calib --calib "$PTQAD_GPTQ/ordinary_h/calib.pt" \
  --calibration-mode required --gptq-damp 0.01 --device cuda \
  2>&1 | tee "$PTQAD_GPTQ/ordinary_bake.log"
(
  cd "$PTQAD_GR00T"
  OMP_NUM_THREADS=4 "$PTQAD_PY" "$PTQAD_ROOT/quant/ptq/collector_category.py" \
    --parent "$PTQAD_GPTQ/ordinary_parent" --out "$PTQAD_GPTQ/category_h" \
    --capture-manifest "$PTQAD_CAL/inputs.json" \
    --windows 148 --batch 1 --seed 2026100601 --device cuda --cpu-threads 4
) 2>&1 | tee "$PTQAD_GPTQ/category_collect.log"
"$PTQAD_PY" exp/bake_gptq_reference_category.py
export LIBERO_PYTHON="$PTQAD_GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
export PTQAD_MEDIA_LIB=/home/zhaosiying/miniforge3/envs/media7/lib
export PTQAD_ZMQ_TIMEOUT_MS=120000
export FP4VLA_SCOPE=all
"$PTQAD_PY" eval/run_recovery_eval.py \
  --checkpoint "$PTQAD_GPTQ/w4a4_category" \
  --out "$PTQAD_GPTQ/heldout_gptq" \
  --protocol-file exp/recovery_protocol_v12_rtn_w4a4.json \
  --purpose heldout --seed 970000 --episodes 16 --port 6980 \
  --collection-manifest "$PTQAD_RUN/recovery_v12/artifacts/collection_qad/eval_manifest.json" \
  --gr00t "$PTQAD_GR00T" --server-python "$PTQAD_PY" \
  --rollout-python "$LIBERO_PYTHON"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  eval/compare_gptq_reference.py --out "$PTQAD_GPTQ/paired_supplement.json"
```

主证据安装到 `paper/evidence/` 后，将两项补充结果转换为可搬移证据。动作归档保存完整有限数值张量的 JSON，可重算动作误差；GPTQ 归档保存原始评测日志及统计源码，权重和 Hessian 以源端验收收据与哈希标识。两项均绑定同一个最终五臂清单。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  paper/collect_action_diagnostics.py collect \
  --source-root "$PTQAD_DIAG" \
  --final-manifest "$PTQAD_RUN/recovery_v12/final_manifest.json" \
  --out paper/evidence/action_diagnostics
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  paper/collect_gptq_reference.py \
  --report "$PTQAD_GPTQ/paired_supplement.json" \
  --main-evidence paper/evidence --out paper/evidence/gptq_reference
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt \
  python paper/collect_action_diagnostics.py verify \
  --folder paper/evidence/action_diagnostics \
  --final-manifest paper/evidence/final_manifest.json
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt \
  python paper/collect_gptq_reference.py \
  --verify paper/evidence/gptq_reference --main-evidence paper/evidence
```

`requirements-evidence.txt` 固定 Ubuntu x86_64、Python 3.12 的官方 CPU PyTorch 2.9.0 wheel 及其 SHA，并使用 NumPy 1.26.4 重放指标；不需要 GPU，也不安装进正在训练的恢复环境。量化与推理仍使用安装章节中锁定的 GR00T 环境。
