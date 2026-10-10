# GR00T 固定观测动作诊断

该工具用于测量 BF16、PTQ、QAD、continued-QAD 和 OPD 在相同观测及相同初始噪声下生成的完整动作块差异，补充成功率表背后的行为分析。它独立于训练、开发集选择和正式闭环评测，不修改 v12 协议。

截至 2026-10-11：首次五臂 GPU 诊断及四项 BF16 参照比较已完成。输入为真实教师训练数据中的十任务各八个固定观测，共 80 个；五臂清单均记录完成，四项比较均通过实际初始噪声一致性检查，并绑定最终清单 SHA-256：`b33445dd687a68e58a7cb829f4cdbb150ad53ee25d6489288dc19e8b034b5fc8`。这些结果属于训练分布上的拟合诊断；公开归档的独立 CPU 导出和复验仍须完成后才能注册为发布证据。

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

以下是已完成首次采集与比较的复现命令。现有结果保存在上述目录中，不要覆盖或重复运行；首次复现到新目录时，仍须等最终清单生成、训练和正式评测全部结束且 GPU 空闲后串行执行：

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

每臂已保留 `actions.pt`、`manifest.json` 和原始日志。五臂各完成 80 个观测，四项比较各含 80 个逐观测结果和十个逐任务结果；实际噪声配对已通过，任务宏平均已核对为十任务指标的等权平均。这里只测固定观测上的完整动作块，尚未测量独立状态上的速度场误差或短程闭环末端偏移；这两项不能由本工具的 MSE 替代。

## 首次诊断结果

以下均为相对于 BF16 的任务宏平均；归一化 MSE 覆盖有效的 16 步 × 7 维动作块。MSE 降幅以 PTQ 的动作差异为基准，不是成功率增幅。

| 候选臂 | 归一化动作 MSE | 相对 PTQ 的 MSE 降幅 | LIBERO 夹爪指令不一致比例 |
| --- | ---: | ---: | ---: |
| PTQ | 0.0111099510 | — | 0.78125% |
| QAD | 0.0088363732 | 20.46% | 0.78125% |
| continued-QAD | 0.0091572452 | 17.58% | 0.859375% |
| QAD+OPD | 0.0089182975 | 19.73% | 0.78125% |

三个恢复臂各在七个任务上降低归一化 MSE、在三个任务上升高，不能描述为所有任务均改善。QAD+OPD 的整体归一化 MSE 比 QAD 高约 0.93%，比 continued-QAD 低约 2.61%；因此本次诊断不支持“OPD 在 QAD 基础上进一步降低整体动作误差”的结论。

这组结果支持有限的描述：在已用于训练的固定观测上，恢复后的动作块通常更接近 BF16。它不能证明未见状态泛化、成功率提升、统计显著性或因果机制。尤其是 continued-QAD 在这里的 MSE 接近 QAD 与 QAD+OPD，但其正式闭环成功率为 104/160（65%），低于后两者的 145/160（90.625%）和 144/160（90%）。固定输入误差没有覆盖动作执行后状态分布和后续决策的变化，不能据此解释或替代闭环表现。

正式五臂评测的四项预声明配对比较中，只有 QAD+OPD 相对 continued-QAD 的差异获得统计支持；QAD 相对 PTQ、QAD+OPD 相对 QAD 的成功率优势均未得到确立。诊断 MSE 的下降不能补足这些统计结论，也不能证明臂间等效。

## 可搬移发布证据

真实五臂诊断完成后，在原检查点和捕获数据仍可读取时导出。工具重新验证选中模型字节、输入分区、源码及五臂完成状态，再把完整噪声、动作和反归一化输出无损转换为 JSON。输出已存在时拒绝覆盖；这一环节只用 CPU，不执行策略。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  paper/collect_action_diagnostics.py collect \
  --source-root "$PTQAD_DIAG" \
  --final-manifest "$PTQAD_RUN/recovery_v12/final_manifest.json" \
  --out paper/evidence/action_diagnostics
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt \
  python paper/collect_action_diagnostics.py verify \
  --folder paper/evidence/action_diagnostics \
  --final-manifest paper/evidence/final_manifest.json
```

最后一个命令要求主五臂证据已安装。独立搬移归档时可省略 `--final-manifest`，用包内最终清单核验。公开归档保存原始清单、日志、源码以及有限数值张量的 dtype、shape、数值和字节摘要，可以离开原机器重算四项动作比较。模型和相机输入仅保存身份，不声称归档本身足以重新运行推理。正式论文及 README 从复验后的结果生成表格，缺失该证据时拒绝生成终稿。

首次真实 GPU 诊断和四项比较已完成，原始结果位于 `paper/_build/w4a4_action_diagnostics_v12`。当前 CPU 导出与哈希复验由发布主流程执行；在其成功且完成补充证据注册之前，不应把原始工作目录视为已通过可搬移验证的发布归档。

## CPU 验证

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  -m unittest discover -s tests -p test_action_chunk_diagnostics.py -v
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
  -m unittest discover -s tests -p test_collect_action_diagnostics.py -v
```

测试覆盖完整积分入口与 RTC 排除、随机状态和模型模式恢复、实际噪声配对、padding 排除、完整 16 步计量、夹爪中性点、分区隔离、最终模型身份与适配器字节变化，以及不同源码版本之间的比较拒绝。CPU 测试不证明真实 GPU 推理或部署性能。
