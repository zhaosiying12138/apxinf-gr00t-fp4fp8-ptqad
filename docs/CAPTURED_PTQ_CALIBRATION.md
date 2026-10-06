# 使用可追溯的教师观测构建 GPTQ 参考

此入口补充 [QVLA 对照审阅](QVLA_GAP_REVIEW_20261006.md) 指出的强 PTQ 基线缺口。它复用已有普通层、category 层 Hessian collector 和 GPTQ 算法，只增加捕获输入适配与来源校验，不改变当前 v12 的 RTN 恢复实验。

当前完成的是 CPU 输入审计和程序测试；以下模型校准、重新量化与补充评测尚未执行，不能据此填写 GPTQ 成功率。GPU 步骤必须等待当前五臂流程结束，补充评测协议也必须在读取结果前另行冻结。

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

## 串行校准与量化（待执行）

只有在 GPU 空闲后才执行本节。`set -e` 使任一步失败时停止后续步骤；旧校准产物不得混入新输出。普通层用 `calib` 配方请求所有 eligible 普通权重的 NVFP4 GPTQ，别名和非 Linear 项按实际记录回退。category 的活动 LIBERO bank 使用 GPTQ，其余 bank 使用 RTN。

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
OMP_NUM_THREADS=4 "$PTQAD_PY" "$PTQAD_ROOT/quant/ptq/bake_category.py" \
  --parent "$PTQAD_GPTQ/ordinary_parent" --calib "$PTQAD_GPTQ/category_h" \
  --out "$PTQAD_GPTQ/w4a4_category" --expected-windows 148 \
  --method gptq_active --gptq-damp 0.01
```

保留两份 `calib_meta.json`、缓存、量化 recipe、bake manifest、源码版本及完整日志。新增 `captured_input_provenance` 位于校准元数据顶层，不改变既有 category parent 身份合同。原 `verify_calibration.py` 检查 Hessian 数值与源权重；它的 PASS 本身不等于重新审计捕获分区。来源审计由适配器执行，并随元数据保存。

补充闭环评测仍需独立协议：锁定同一 GR00T 根权重、实际激活覆盖、配对初态与 seed、主要比较、置信区间和多重检验规则；不能根据主实验 held-out 分数挑选 GPTQ 超参数。新增基线首先回答 GPTQ 相对 RTN 的贡献，以及恢复模型相对这一校准参考的差异，不能自动升级为“超过 PTQ SOTA”。

## CPU 测试

```bash
cd "$PTQAD_ROOT"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  -m unittest discover -s tests -p test_captured_calibration.py -v
```

测试验证输入轴和单次遍历、JSON 冻结往返、文件变更、元数据与权重身份拒绝，以及原有 H 累积。它不替代真实 GR00T 模型加载和校准验收。
