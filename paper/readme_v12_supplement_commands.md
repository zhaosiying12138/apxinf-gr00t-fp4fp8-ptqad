### 补充证据与冻结范围

本轮训练、新量化和搜索已于 2026 年 10 月 10 日冻结，未完成的 v12 开发集、held-out 正式评测与既定固定观测诊断可收尾。校准 GPTQ 补充未执行；不再收集 Hessian、不量化新权重、不启动 GPTQ 评测，不报告其成功率、配对统计或校准成本。范围决定保存在 [publication_scope_v12.json](paper/publication_scope_v12.json)。

主五臂证据安装到 [paper/evidence/](paper/evidence/) 后，用下面的 CPU 命令将该决定绑定到实际最终清单。它要求完整的当前五臂结果、准确的协议 SHA 和本次发布运行路径；已有目录、缺少范围决定或旧清单都会失败。

~~~bash
python3 paper/gptq_reference_publication.py --record-not-performed
python3 paper/gptq_reference_publication.py --verify
~~~

动作诊断保留已授权的首次评测复现入口：主五臂全部结束、GPU 空闲后，对冻结的检查点按既定观测和噪声分别评测一次，不更新参数、不新建量化权重或搜索候选。已经冻结的 80 个观测清单可复用，不重新 rollout；已有输出禁止覆盖。执行顺序与来源核验保留如下。当前运行中的开发集与 held-out 评测享有优先权。

~~~bash
set -euo pipefail
cd /home/zhaosiying/codebase/fp4vla
PTQAD_ROOT="$PWD"
PTQAD_GR00T=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
PTQAD_PY="$PTQAD_GR00T/.venv/bin/python"
PTQAD_RUN="$PTQAD_ROOT/results/reruns/rtn_w4a4_release_20261006_01"
PTQAD_DIAG="$PTQAD_ROOT/paper/_build/w4a4_action_diagnostics_v12"
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
~~~

诊断完成后，以下 CPU 命令只归档已有输出。发布动作误差结论仍要求五臂的完整有限数值张量、实际噪声身份和指标重算全部通过。如果源输出不存在，发布门禁继续失败，不能以 GPTQ 的未执行决定跳过动作证据检查。

~~~bash
PTQAD_ROOT="$PWD"
PTQAD_RUN="$PTQAD_ROOT/results/reruns/rtn_w4a4_release_20261006_01"
PTQAD_DIAG="$PTQAD_ROOT/paper/_build/w4a4_action_diagnostics_v12"
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt \
  python paper/collect_action_diagnostics.py collect \
  --source-root "$PTQAD_DIAG" \
  --final-manifest "$PTQAD_RUN/recovery_v12/final_manifest.json" \
  --out paper/evidence/action_diagnostics
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt \
  python paper/collect_action_diagnostics.py verify \
  --folder paper/evidence/action_diagnostics \
  --final-manifest paper/evidence/final_manifest.json
python3 paper/install_final_evidence.py --register-supplements paper/evidence --supplement-root paper/evidence
python3 paper/install_final_evidence.py --verify paper/evidence
~~~

补充登记仍要求 runtime、search_costs、action_diagnostics、gptq_reference 与 frontier_comparison 全部齐备；GPTQ 目录只含明确的未执行收据，禁止与测量结果混用。其余已声明的证据标准保持完整。requirements-evidence.txt 固定 Ubuntu x86_64、Python 3.12 的官方 CPU PyTorch 2.9.0 wheel 及 SHA，并使用 NumPy 1.26.4 重放指标；不需要 GPU，也不安装进恢复环境。
