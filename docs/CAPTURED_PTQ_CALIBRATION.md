# 使用可追溯的教师观测构建 GPTQ 参考

此入口补充 [QVLA 对照审阅](QVLA_GAP_REVIEW_20261006.md) 指出的强 PTQ 基线缺口。它复用已有普通层、category 层 Hessian collector 和 GPTQ 算法，只增加捕获输入适配与来源校验，不改变当前 v12 的 RTN 恢复实验。

当前完成的是 CPU 输入审计、程序测试、普通权重分配预检及[补充协议冻结](../exp/gptq_reference_protocol_v12.json)；以下模型校准、重新量化与补充评测尚未执行，不能据此填写 GPTQ 成功率。GPU 步骤必须等待当前五臂流程结束。

补充协议在 v12 主实验 held-out 尚未开始时冻结，当前 SHA-256 为 `ba1f3972893fcc2bf2ab10ec9b0dd75569f6f5a10b79e594cd4d273837035a5b`。执行前审查补齐了 14 份数值与运行时依赖，初版原文及其 SHA 保存在协议 `amendment` 指向的快照中；模型、数据、配方与统计规则未变。它不参与原 v12 开发集选择，主实验协议的路径和字节保持原样。

## 输入与计算口径

- 输入为 v12 教师训练分区初态 20–23 的成功轨迹捕获，十任务共 148 个观测窗口。它们与当前 held-out 初态分离，但已用于恢复训练；148 个窗口不是 148 条独立轨迹。
- 清单复用 `verify_teacher_replay.audit_replay`，核对原始 rollout、成功回合、reset 身份、任务覆盖、协议和教师权重，并冻结每个样本的 SHA。collector 使用前重新执行审计，读取样本时再次校验 SHA。
- 每个捕获已经完成单样本 collate，按任务及文件名遍历一次，要求 `--batch 1 --windows 148`。不重新堆叠视觉 patch；不循环补足窗口数。
- 普通层在原 BF16 上收集 H；category 层在新 GPTQ parent 上收集 H。两者使用同一冻结观测与种子规则，但前向模型不同，不能互换或复用绑定旧 parent 的 category H。
- 前向使用捕获的教师动作端点及固定 seed，执行训练速度预测路径。当前收集器不在激活上安装 QDQ，也不优化最终动作误差；这是既有 GPTQ Hessian 参考，不能称为 QVLA 动作敏感度方法或专门优化激活误差的 PTQ。
- 后续 W4A4 评测需要与 v12 相同的激活 QDQ。磁盘检查点仍保存反量化值，未变成原生 packed NVFP4 部署文件。

## 冻结输入（仅 CPU）

下面在 WSL Ubuntu 执行。路径变量可调整，输出目录必须是新目录；已冻结的清单不覆盖。

```bash
cd /home/zhaosiying/codebase/fp4vla
PTQAD_ROOT="$PWD"
PTQAD_GR00T=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
PTQAD_PY="$PTQAD_GR00T/.venv/bin/python"
PTQAD_BASE="$PTQAD_ROOT/weights/GR00T-N1.7-LIBERO/libero_10"
PTQAD_RUN="$PTQAD_ROOT/results/reruns/rtn_w4a4_release_20261006_01"
PTQAD_CAL="$PTQAD_ROOT/paper/_build/captured_ptq_v12"

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  quant/ptq/captured_calibration.py \
  --root "$PTQAD_RUN/teacher_supervision_v12_clean" \
  --protocol-file exp/recovery_protocol_v12_rtn_w4a4.json \
  --teacher "$PTQAD_BASE" --out "$PTQAD_CAL/inputs.json"
```

## 执行前身份检查

在运行任何新增 GPU 步骤前，先核对已冻结的协议、输入和源码身份：

```bash
cd "$PTQAD_ROOT"
python3 - <<'PY'
import hashlib, json
from pathlib import Path
path = Path('exp/gptq_reference_protocol_v12.json')
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'ba1f3972893fcc2bf2ab10ec9b0dd75569f6f5a10b79e594cd4d273837035a5b'
plan = json.loads(path.read_text())
records = [plan['main_protocol'], plan['capture_manifest'], *plan['preparation_source_files'].values()]
for record in records:
    source = Path(record['path'])
    assert source.stat().st_size == record['bytes'], source
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256'], source
print('Frozen supplement identities verified; this does not run calibration or evaluation.')
PY
```

## 串行校准与量化（待执行）

只有在 GPU 空闲后才执行本节。`set -e` 使任一步失败时停止后续步骤；旧校准产物不得混入新输出。普通层用 `calib` 配方请求所有 eligible 普通权重的 NVFP4 GPTQ，别名和非 Linear 项按实际记录回退。category 的活动 LIBERO bank 比较 GPTQ 与 RTN 的校准误差，误差相同保留 RTN，其余 bank 使用 RTN；最终报告实际选择，不能预先写成七个活动 bank 都采用 GPTQ。

```bash
set -e
PTQAD_GPTQ="$PTQAD_ROOT/results/supplement_gptq_capture_v12"
mkdir -p "$PTQAD_GPTQ"
(
  cd "$PTQAD_GR00T"
  OMP_NUM_THREADS=4 "$PTQAD_PY" "$PTQAD_ROOT/quant/ptq/collector.py" \
    --base "$PTQAD_BASE" --out "$PTQAD_GPTQ/ordinary_h" \
    --capture-manifest "$PTQAD_CAL/inputs.json" \
    --recipe calib --windows 148 --batch 1 --seed 2026100601 \
    --device cuda --cpu-threads 4
)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  "$PTQAD_ROOT/quant/ptq/verify_calibration.py" \
  --calib "$PTQAD_GPTQ/ordinary_h" --expected-windows 148
OMP_NUM_THREADS=4 "$PTQAD_PY" "$PTQAD_ROOT/quant/ptq/bake.py" \
  --base "$PTQAD_BASE" --out "$PTQAD_GPTQ/ordinary_parent" \
  --recipe calib --calib "$PTQAD_GPTQ/ordinary_h/calib.pt" \
  --calibration-mode required --gptq-damp 0.01 --device cuda
(
  cd "$PTQAD_GR00T"
  OMP_NUM_THREADS=4 "$PTQAD_PY" "$PTQAD_ROOT/quant/ptq/collector_category.py" \
    --parent "$PTQAD_GPTQ/ordinary_parent" --out "$PTQAD_GPTQ/category_h" \
    --capture-manifest "$PTQAD_CAL/inputs.json" \
    --windows 148 --batch 1 --seed 2026100601 --device cuda --cpu-threads 4
)
"$PTQAD_PY" "$PTQAD_ROOT/exp/bake_gptq_reference_category.py"
```

最后一步通过薄包装调用原 `bake_category.py`，参数固定为 `--expected-windows 148 --method gptq_active --gptq-damp 0.01`，路径来自已冻结补充协议。可先加 `--print-command` 查看实际命令而不执行。原 category 产物不记录 damp 参数，因此包装另存完整 argv、退出码及日志 SHA；比较入口必须验收这份 `category_bake_invocation.json`，不能仅凭 producer 默认值推断实际参数。

保留两份 `calib_meta.json`、缓存、量化 recipe、bake manifest、调用记录、源码版本及完整日志。新增 `captured_input_provenance` 位于校准元数据顶层，不改变既有 category parent 身份合同。原 `verify_calibration.py` 检查 Hessian 数值与源权重；它的 PASS 本身不等于重新审计捕获分区。来源审计由适配器执行，并随元数据保存。

## 补充评测及统计规则（已冻结，待执行）

复用原 v12 的十任务、初态 9–19 与 24–28、seed 970000，每任务 16 回合。固定报告 GPTQ−RTN、QAD−GPTQ、OPD−GPTQ 三项差值，采用任务内配对 bootstrap（20,000 次、seed 20261007、95% 逐项区间），三项 exact McNemar 检验做 Holm 校正。GPTQ 与 BF16、continued-QAD 的比较另作描述，不增加显著性主张。补充 family 与主实验的四项 family 分别定义，不能宣称覆盖整篇论文的统一错误率控制。

QAD、continued-QAD、OPD 只能采用原 v12 开发集选定的检查点；无论结果方向如何，六臂和三项差值均完整报告。推断范围限于固定十任务与一个训练 seed；不能根据这些 held-out 分数再挑 GPTQ 参数。新增基线回答同覆盖校准 PTQ 的贡献，不能自动升级为“超过 PTQ SOTA”。

评测入口仍使用原 v12 JSON，而不是把补充 JSON 传入 `--protocol-file`。只有原五臂完成并验收、新 GPTQ 量化产物完成身份与覆盖检查、GPU 空闲后，才运行：

```bash
set -e
cd "$PTQAD_ROOT"
export LIBERO_PYTHON="$PTQAD_GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
export PTQAD_MEDIA_LIB=/home/zhaosiying/miniforge3/envs/media7/lib
export PTQAD_ZMQ_TIMEOUT_MS=120000
export FP4VLA_SCOPE=all
test -s "$PTQAD_RUN/recovery_v12/final_manifest.json"
test -s "$PTQAD_GPTQ/w4a4_category/category_bake_manifest.json"
test -s "$PTQAD_RUN/recovery_v12/artifacts/collection_qad/eval_manifest.json"
"$PTQAD_PY" eval/run_recovery_eval.py \
  --checkpoint "$PTQAD_GPTQ/w4a4_category" --out "$PTQAD_GPTQ/heldout_gptq" \
  --purpose heldout --protocol-file exp/recovery_protocol_v12_rtn_w4a4.json \
  --seed 970000 --episodes 16 --port 6980 \
  --collection-manifest "$PTQAD_RUN/recovery_v12/artifacts/collection_qad/eval_manifest.json" \
  --gr00t "$PTQAD_GR00T" --server-python "$PTQAD_PY" \
  --rollout-python "$LIBERO_PYTHON"
```

以上媒体环境路径对应本机；新安装按 `setup/06_install_recovery.sh` 生成的环境配置填写。`--collection-manifest` 必须指向已完成的学生 collection，不能替换成教师监督清单。基础设施失败保留失败记录和成本，用相同模型、协议及种子在新目录重试；成功率本身不是重跑理由。

比较入口 `eval/compare_gptq_reference.py` 复用 `run_high_fp4_v3.eval_audit` 核对原始日志、实际激活安装及逐回合 reset，再用 `require_pairing` 对齐六臂。统计复用 `paired_uncertainty.paired_effect` 与 `holm_adjust`；现有 `compare_round` 和 `analyze` 保持主实验五臂与四项比较。只有全部模型、校准、配对和结果检查通过才生成独立补充报告，不覆盖主实验结果。

```bash
cd "$PTQAD_ROOT"
# 此项只检查已冻结的协议、捕获清单和源码，不要求实验完成。
python3 eval/compare_gptq_reference.py --preflight-only
# 以下仅在主五臂与新 GPTQ 评测全部完成后执行，CPU 读取实际产物。
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  eval/compare_gptq_reference.py \
  --out "$PTQAD_GPTQ/paired_supplement.json"
```

当前尚无真实补充评测结果。CPU 测试与预检通过不代表六臂比较已验收，也不代表原生部署性能。

## 可搬移发布证据

补充评测及比较完成后，保留源权重和 Hessian 直到归档通过。归档器重新运行源端比较验收，再保存原始 JSON、日志、配方、校准元数据及源码。主五臂证据必须先安装到 `paper/evidence/`；补充目录不覆盖五臂核心结果。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  paper/collect_gptq_reference.py \
  --report "$PTQAD_GPTQ/paired_supplement.json" \
  --main-evidence paper/evidence --out paper/evidence/gptq_reference
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt \
  python paper/collect_gptq_reference.py \
  --verify paper/evidence/gptq_reference --main-evidence paper/evidence
```

公开复验从已归档的原始日志重建六臂逐回合结果和预先固定的配对统计，不访问原电脑的绝对路径。权重、相机输入与 Hessian 不进入发布包；其张量验收来自源端收据，公开复验不会把记录哈希说成已重新读取这些张量。论文和 README 只有在补充证据通过复验后才生成相应表格。

## CPU 测试

```bash
cd "$PTQAD_ROOT"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  -m unittest discover -s tests -p test_captured_calibration.py -v
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  -m unittest discover -s tests -p test_gptq_reference_evidence.py -v
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  -m unittest discover -s tests -p test_compare_gptq_reference.py -v
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  -m unittest discover -s tests -p test_collect_gptq_reference.py -v
```

测试分别验证捕获输入与原有 H 累积、校准到最终权重的证据链，以及六臂配对统计。证据链测试使用小型真实 safetensors 文件，覆盖合法 RTN 回退、参数及文件篡改拒绝；统计测试覆盖负向结果保留、固定比较方向与多重检验。它们不替代真实 GR00T 模型加载和校准验收。
