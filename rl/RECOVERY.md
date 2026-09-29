# 单轮 on-policy 蒸馏复现

本目录采用冻结的 PTQ 数值基座加 LoRA 旁路。`QAD` 训练演示数据的流匹配损失；`QAD+OPD` 从同一个 QAD 适配器继续训练，并加入 BF16 教师在学生实际访问观测上的速度场 MSE。一次采集对应一次策略改进轮次，更新后的策略不会自动刷新旧缓存。

## 批量、显存和梯度

本地 GR00T 的 `--global-batch-size` 表示**累积前**各 GPU 的总批量。为避免歧义，本项目入口只使用单 GPU，并显式定义：

```text
QAD_MICRO_BATCH=1   QAD_GLOBAL_BATCH=16  -> accumulation=16
QAD_MICRO_BATCH=2   QAD_GLOBAL_BATCH=16  -> accumulation=8
```

`QAD_ACCUM_STEPS` 可以省略；若显式设置，必须与上述比值一致。入口验证实际 Trainer 参数并把三者写入 `recovery_manifest.json`。`QAD_BSZ/QAD_ACC` 是兼容别名，其含义仍是微批量与累积次数。

主损失先完成反向传播，释放主计算图；每 `OPD_EVERY=4` 个优化器更新，再对每个累积微步执行一个独立探针前向和反向传播。因此，第 4、8、12……次更新的目标是“微批平均演示损失 + λ × 微批平均教师 MSE”，其余更新只有演示损失。长期平均教师系数为 λ/4。该实现复用 HF Trainer/Accelerator 的梯度累积和优化器，不把多个探针的计算图叠在主任务图上。主任务与探针仍分别消耗计算；两臂相同演示量、优化器步数不意味着相同算力预算。

教师和学生均在探针前向中保持 eval 模式，禁用 dropout；eval 本身不会禁用 autograd。逐样本 `fork_rng` 固定并恢复 CPU 与模型 CUDA 设备的随机状态，以共享前向内采样的噪声和 Beta 时间，不改变下一批主训练随机数。缓存记录并验证参数 dtype、autocast dtype、流匹配配置及 16 层语言栈、32 层 DiT、4 层 VL 模块。默认教师参数为 FP32、矩阵计算使用 BF16 autocast，匹配当前训练实际路径。

版本 3 缓存还记录处理器定义的有效动作范围。当前基座是 16 步 × 7 维，来自 `processor_config.json` 的 LIBERO action 时间索引及 `statistics.json` 的各动作分组维度。师生仍使用完整 40×132 的同一学生动作端点构造插值，损失仅在 112 个有效输出元素上计算平方误差均值。无效维度、尾部填充不贡献损失；全零掩码会报错。

`head`、`head+lang`、`head+lang_all` 分别训练动作头、动作头加语言注意力、动作头加语言注意力及 MLP。视觉参数始终冻结。rank=32 时的静态清单分别为 38,658,048、45,998,080、58,580,992 个参数；完整运行会再次报告实际注入数。首次主损失反向后，入口断言 head 和被选择的 language LoRA B 梯度非零；零初始化 B 导致首步 A 梯度为零是正常情况。

## 先做两步 smoke

先准备完整 GR00T 环境、原始 BF16 模型和新 PTQ checkpoint，并确保 GPU 没有其他实验。

```bash
export PROJECT=$(pwd)
source "$PROJECT/setup/recovery-env.sh"
export PROTOCOL_FILE="$PROJECT/exp/recovery_protocol_v3_high_fp4.json"
export TRAIN_SEED=20260929
cd "$GR00T_REPO"
GR00T_BASE_CKPT=/absolute/path/to/selected_ptq \
QAD_OUT=/absolute/path/to/new_smoke \
QAD_STEPS=2 QAD_SAVE_STEPS=2 \
QAD_LORA_SCOPE=head+lang_all \
QAD_MICRO_BATCH=1 QAD_GLOBAL_BATCH=2 QAD_OPD_MSE_W=0 QAD_ACTIVATION_CHECKPOINTING=1 \
  .venv/bin/python "$PROJECT/rl/lora_qad.py"
```

不要把 CPU 测试通过当作 24 GB 显存验证。应先核对 smoke 的最大显存、实际加载层数、LoRA B 梯度和保存产物，再决定生产微批量。

正式驱动脚本默认启用 `QAD_ACTIVATION_CHECKPOINTING=1`；直接调用 Python 训练入口时也应显式设置。该选项对有可训练参数的语言层、DiT 层和 VL 层使用 PyTorch 非重入梯度检查点，保留 RNG；也覆盖保持 eval 模式的可微学生探针。它不改参数精度、动作掩码或梯度累积预算，但增加反向重算时间，并关闭训练中不用的 KV 缓存。完整 `head+lang_all` 已通过两步 GPU smoke，覆盖 16/32/4 个块，实际数量写入训练 manifest。当前上游 DiT forward 没有调用其检查点标志，冻结模式的语言层又保持 eval，因此不能仅用上游同名开关代替本实现。三条正式训练臂使用相同的检查点设置。

训练完成后输出 `runtime_metrics.json`：包括从脚本初始化到训练及保存结束的 wall time、Trainer 实际 `global_step`、参数 dtype/字节数、各 CUDA 设备的 allocated/reserved 峰值及结束时占用。峰值来自该进程的 PyTorch allocator，不包含桌面或其他进程。教师缓存已有 `elapsed_seconds`，另记录各设备峰值；其计时范围是 checkpoint 加载与标注，缓存序列化另行注明排除。

## 独立采集、标注和最终评测

维护的入口为 `eval/run_recovery_eval.py`。它逐任务启动一个模型服务器，单环境评测，每个 episode 独立设置重置种子，再通过官方 `get_task_init_states` / `set_init_state` 恢复指定的初态，并执行 10 步原始仿真器七维零动作以稳定物理状态。稳定步骤不计入策略的 720 步上限。`development` 用于选择量化覆盖率；`collection` 收集学生轨迹；`heldout` 必须提供采集 manifest，拒绝初态索引或重置种子重叠。所有正式结果覆盖完整 10 任务，报告逐任务布尔结果、真实分母和任务宏平均；`smoke` 可缩减任务数，但不给完整宏平均。

本流程显式传入 `exp/recovery_protocol_v3_high_fp4.json`：

| 分区 | 每任务回合 | 官方初态索引 | 起始 seed |
|---|---:|---|---:|
| development | 5 | 4–8 | 440000 |
| collection | 4 | 20–23 | 550000 |
| heldout | 10 | 30–39 | 660000 |
| smoke | 1 | 0 | 770000 |

每任务 seed 加 `1000 × task_index`，环境每集再加 episode 索引。正式模式核对分区、集数与十任务覆盖。最终五臂共享相同 heldout 条件，开发结果不充当最终成功率。

日志逐次记录初态库文件 SHA-256、库索引、恢复后及稳定后物理状态 SHA-256。结果汇总严格核对前 N 次 reset 与种子、索引、数量；允许最后一次自动 reset，但不把它计为新 episode。`eval/compare_recovery.py` 在生成 `paired_comparison.json` 前逐 episode 比较五臂的状态散列；不一致会报错。这里配对的是环境初态，策略扩散噪声只按任务设置种子，不能声称每个 episode 的完整随机轨迹相同。

评测保留视频到 `<out>/videos/<task>`。`PTQAD_MEDIA_LIB` 指向 FFmpeg 环境的 `lib` 目录，入口同时把同级 `bin` 加入 `PATH`；特殊布局可用 `PTQAD_MEDIA_BIN` 指定可执行目录。启动前必须找到 `ffmpeg`，动态库仍从 `LD_LIBRARY_PATH` 加载。

采集钩子记录实际送给 QAD 策略的观测，在 CPU 上为每个样本重新 collate，并按策略实际输入路径将浮点观测转换为 BF16；不能直接沿第 0 维切分 Qwen 的扁平图像块。学生生成的归一化动作块作为插值路径的动作端点。教师离线对这些相同观测、相同插值条件预测速度，而不是把学生动作当作专家标签。归一化统计必须与原教师一致；入口保持并验证冻结基座统计。

单独收集和标注的命令如下：

```bash
python3 "$PROJECT/eval/run_recovery_eval.py" \
  --checkpoint /absolute/path/to/qad_merged --purpose collection \
  --out /absolute/path/to/new_collection --seed 550000 --episodes 4 \
  --protocol-file "$PROTOCOL_FILE" --gr00t "$GR00T_REPO" \
  --server-python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON"
"$PTQAD_PYTHON" "$PROJECT/rl/opd_probe_cache.py" \
  --teacher /absolute/path/to/original_bf16 \
  --input-dir /absolute/path/to/new_collection/observations --count 160 \
  --out /absolute/path/to/new_teacher_probes.pt --seed 20260929 \
  --dataset "$GR00T_REPO/demo_data/libero_demo"
```

不传 `--input-dir` 的缓存工具只从演示数据生成离线探针，其元数据明确标记 `demo`，不能称为 on-policy。

## 从共同 QAD 起点运行两组续训

`QAD_INIT_ADAPTER` 读取原 QAD checkpoint 的精确 A/B 张量，冻结基座仍是最初 PTQ checkpoint。两组都重置 Adam 和学习率计划，不能一组恢复优化器而另一组重新初始化。入口核对 base、rank、alpha、scope 及基座配置、统计、配方散列；不从已合并 BF16 模型重新分解 LoRA。

v3 由 `exp/run_high_fp4_v3.py` 统一调度。先完成两个 QAD 学习率的 500 步训练与开发选择，再从选中模型的精确 A/B 创建 continued-QAD 和两个 OPD 权重候选，各训练 100 步；最后按开发结果选择 OPD 权重并评测五臂。不要用固定教师权重的一轮脚本代替这项开发搜索。

```bash
# 从仓库根目录调用；路径由安装及开发流程设置。
python3 "$PROJECT/exp/run_high_fp4_v3.py" \
  --run-dir "$RECOVERY_ROOT" --ptq-selection "$DEV_ROOT/selection.json" \
  --protocol-file "$PROJECT/exp/recovery_protocol_v3_high_fp4.json" \
  --base "$PTQAD_BASE" --gr00t-repo "$GR00T_REPO" \
  --python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON" \
  --dataset "$QAD_DATASET" --until all
```

可把 `--until all` 改为 `qad_selection` 或 `opd_selection` 分段执行，再用相同命令续跑。协议固定训练 seed=20260929、rank32、alpha64、`head+lang_all`、microbatch1、有效 batch16 及激活重算。QAD 学习率候选 5e-5 和 1e-4；OPD 权重候选 0.25 和 1.0，均按开发宏平均选择，平分取较小值。完整环境和目录初始化见 `docs/reproduce-ptqad.md`。

OPD 启动前可用独立输出执行 4 步短测，`QAD_INIT_ADAPTER`指向选中的 QAD 适配器、`OPD_CACHE_PATH`指向学生观测缓存，`QAD_OPD_MSE_W`使用待验证候选权重，`OPD_EVERY=4`、microbatch1、有效 batch16。检查第 4 次更新中的 16 次教师微批反向均有限，且学生梯度可达；短测参数不进入正式分支。

最终检查 `artifacts/heldout_round/heldout_*/summary.json` 与逐任务 `task_results.json`。`final_manifest.json`记录实际选中模型和协议身份。恢复效果、教师额外计算与编码覆盖均以这些完整证据为准。

`paired_comparison.json` 包含五臂逐任务成功数/分母、OPD 相对 QAD 及相对 continued-QAD 的配对 2×2 表，以及来源 JSON 文件和汇总代码的 SHA-256。精确双侧 McNemar p 值按不一致配对的条件二项分布计算，与 [statsmodels 的 exact 定义](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)一致，并用 [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)核对。它仅作探索性描述；同一组任务内的 episode 不能被无条件看作独立任务样本，两个总体比较与逐任务比较没有做多重检验校正。
