# GR00T 固定观测动作诊断

该工具用于测量 BF16、PTQ、QAD、continued-QAD 和 OPD 在相同观测及相同初始噪声下生成的完整动作块差异，补充成功率表背后的行为分析。它独立于训练、开发集选择和正式闭环评测，不修改 v12 协议。

截至 2026-10-06：CPU 测试通过，已从真实教师训练数据冻结十任务各八个观测，共 80 个；GPU 推理尚未运行，尚无动作误差结果。待最终五臂清单形成且 GPU 空闲后再执行采集。这里的命令是后续诊断入口，不是已完成实验的证明。

## 测量定义

- 使用正式 `serve_recovery.py` 和 `run_gr00t_server_fp4vla.py` 加载器；用进程内回调替代网络服务。每个臂单独启动进程，保留正式 W4A4 与 BF16 LoRA 的执行方式。
- 从已有捕获读取经过处理的相机、语言和状态输入；删除训练用 `action`、`action_mask`，避免误触发实时动作块拼接（RTC）。
- 调用模型的 `get_action`，执行完整积分，保留全部模型输出；不把训练前向的速度预测当成动作。
- 记录动作编码器第一次收到的积分状态。比较前要求初始噪声的 dtype、形状和数值全部相同，仅 seed 相同不足以通过。
- 有效动作范围由检查点的 processor/statistics 推导。当前是 16 步 × 7 维，指标覆盖完整动作块；线上一次执行前八步不改变诊断的范围。排除其他填充位置。
- 同时报归一化空间的整体及逐维 MSE。仅在配置允许不依赖原始状态解码时，调用正式 processor 输出 x/y/z、roll/pitch/yaw、gripper 各组的反归一化 MSE。它们是控制器输入坐标的误差，不能当成实际末端位置或姿态偏移。
- 夹爪指令沿用 LIBERO 的 `-sign(2*x-1)`，`x=0.5` 保留中性指令；报告指令不一致比例，不将它称为抓取失败率。
- 输出逐观测、逐任务及任务宏平均；观测窗口不等于独立闭环回合，本工具不生成成功率或显著性结论。

## 数据与模型身份

当前清单来自 `teacher_supervision_v12_clean/observations`，这些观测已用于 QAD 训练。结果必须标为 **训练分布上的拟合诊断**，不能解释为未见状态泛化，也不能用于反向选择当前冻结的五臂。若要增加独立诊断，需要另行核验数据来源与分区。

`freeze` 按各任务文件名顺序选取固定数量的观测，记录源文件 SHA、任务、回合、初态和推理 seed。它拒绝 held-out 分区；`collect` 再次核对完整清单及其来源。不会根据动作误差挑选观测。

`collect` 必须提供本轮完成后的 `final_manifest.json`，工具从中解析五臂检查点，无需手填 checkpoint 路径。采集前核对所选模型、量化基座和实际加载的适配器权重，核对实际 processor 的归一化与 embodiment，并检查激活量化覆盖。比较结果要求来自同一个最终清单和相同版本的测量、推理及解码源码。

## 执行命令

以下在 WSL Ubuntu 执行。`freeze` 和 `compare` 只用 CPU；五个 `collect` 必须等待训练和正式评测全部完成后串行运行。输出已存在时工具拒绝覆盖。

```bash
cd /home/zhaosiying/codebase/fp4vla
PTQAD_PY=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python
PTQAD_GR00T=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
PTQAD_RUN=results/reruns/rtn_w4a4_release_20261006_01
PTQAD_DIAG=paper/_build/w4a4_action_diagnostics_v12

# 已执行过的清单不要覆盖；首次复现时运行一次。
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  exp/action_chunk_diagnostics.py freeze \
  --capture-root "$PTQAD_RUN/teacher_supervision_v12_clean/observations" \
  --protocol-file exp/recovery_protocol_v12_rtn_w4a4.json \
  --per-task 8 --seed 2026100600 --out "$PTQAD_DIAG/inputs.json"
```

下面命令仅在最终清单生成、GPU 空闲后运行；当前未执行：

```bash
set -e
test -s "$PTQAD_RUN/recovery_v12/final_manifest.json"
for arm in bf16 ptq qad continued_qad qad_opd; do
  OMP_NUM_THREADS=1 "$PTQAD_PY" exp/action_chunk_diagnostics.py collect \
    --inputs "$PTQAD_DIAG/inputs.json" \
    --final-manifest "$PTQAD_RUN/recovery_v12/final_manifest.json" \
    --arm "$arm" --gr00t "$PTQAD_GR00T" --out "$PTQAD_DIAG/$arm" \
    > "$PTQAD_DIAG/$arm.log" 2>&1
done
for arm in ptq qad continued_qad qad_opd; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
    exp/action_chunk_diagnostics.py compare \
    --reference "$PTQAD_DIAG/bf16" --candidate "$PTQAD_DIAG/$arm" \
    --out "$PTQAD_DIAG/${arm}_vs_bf16.json"
done
```

每臂保留 `actions.pt`、`manifest.json` 和原始日志。完成真实运行后，需要检查实际噪声配对、覆盖记录、反归一化范围及数值结果，再决定哪些表格纳入论文。这里只测固定观测上的完整动作块，尚未测量独立状态上的速度场误差或短程闭环末端偏移；这两项不能由本工具的 MSE 替代。

## CPU 验证

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  -m unittest discover -s tests -p test_action_chunk_diagnostics.py -v
```

测试覆盖完整积分入口与 RTC 排除、随机状态和模型模式恢复、实际噪声配对、padding 排除、完整 16 步计量、夹爪中性点、分区隔离、最终模型身份与适配器字节变化，以及不同源码版本之间的比较拒绝。CPU 测试不证明真实 GPU 推理或部署性能。
