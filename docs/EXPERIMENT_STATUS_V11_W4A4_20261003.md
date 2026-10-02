# v11 全覆盖 W4A4 实验状态与交接说明

本文档给后续 agent 做只读交接，不是论文正文，也不产生新的实验结果。后续分析必须先阅读 `exp/recovery_protocol_v11_w4a4_category.json`、`exp/run_w4a4_recovery.py`（及其委托的 `exp/run_high_fp4_v3.py`）、量化与恢复训练源码，以及本目录下的最终日志。除已在运行的 v11 进程外，不要另启 GPU 实验，不要覆盖现有 checkpoint。

## 冻结的问题与协议

当前主线固定为 **GR00T N1.7 + 全覆盖 NVFP4 W4A4 + QAD + OPD**。479 个可量化权重张量全部使用 NVFP4；469 个普通 Linear 和 7 个 CategorySpecificLinear 使用 W4A4 activation QDQ；3 个 embedding/position 张量只有权重量化。恢复训练冻结 W4A4 base，只更新 468 个普通 Linear 的 LoRA A/B；残差分支读取原始 BF16 输入。QAD 使用 BF16 成功 rollout 形成的 148 个窗口，OPD 在 QAD 学生访问状态上重放固定的 BF16 教师 cache。

冻结协议文件为 `exp/recovery_protocol_v11_w4a4_category.json`，SHA-256 为：

```text
158acbd49fbabbd49e5af18079ab1d0c5e9b067e4e2f72f2c2b44060b6fb26cb
```

development 使用 bank 4–8（5 回合/任务）；教师监督和学生 collection 使用 bank 20–23（训练专用）；最终 held-out 使用 bank 9–19 与 24–28（16 回合/任务，共 160 回合/臂）。选择不得读取 held-out。

## 已完成的压力门控

在 development 上，BF16 与全 NVFP4 W4A4 PTQ 的差距已满足协议的压力门槛。该门控只决定是否进入恢复链，不证明差距已经可以恢复；正文必须等待五臂 held-out ledger。

教师采集已完成 10 个任务、148 个有效 snapshot，CPU replay audit 通过。当前运行目录为：

```text
results/ptqad_20261003/v11_recovery_r4
```

## 当前 GPU 进程与续跑纪律

截至 2026-10-02 12:30 左右（Asia/Shanghai），QAD 的 `5e-5` 臂已超过 1,260/2,000 optimizer steps，driver PID 为 4118527，训练 PID 为 4118579；这些是当时的快照，接手时必须重新检查。续跑服务仍为 active，尚无 `final_manifest.json`；WSL 可用约 266 GiB，宿主 C 盘可用约 79 GB。随后按冻结顺序完成另一学习率、QAD 选择、学生状态采集、continued-QAD、两个 OPD 权重及 held-out 五臂比较。

续跑由 systemd 用户服务 `fp4vla-v11-r4-continuation.service` 托管，入口为 `exp/continue_w4a4_run.py`。服务先等待绑定的 driver 退出，然后校验并复用已完成阶段。当前 driver 没有 `PTQAD_MEDIA_LIB`，第一条 development 评测预计会在 FFmpeg 预检时退出；续跑器仅在日志精确匹配该错误、评测目录完全为空、没有评测 manifest 或完成标记且 GPU 空闲时，归档原日志并记录 SHA，再使用已验证的 media7 路径续跑。它不删除已有结果，不重跑部分训练，不修改冻结的训练或评测实现。

```bash
python3 exp/high_fp4_status.py --recovery-root results/ptqad_20261003/v11_recovery_r4
systemctl --user status fp4vla-v11-r4-continuation.service
journalctl --user -u fp4vla-v11-r4-continuation.service -n 20 --no-pager
```

`paper/_build/formal_status.py` 是同一只读状态入口的兼容调用；默认目录现为 r4。状态输出中的 `incomplete` 仅描述文件尚未完成，不能据此认定进程已停止；是否仍在训练须结合现场 PID 和服务状态判断。

续跑调用和后续 driver 日志位于运行目录的 `operations/continuation-*/`。`exp/continue_v11.sh` 现在是通用入口，调用时必须显式提供 `--run-dir`、活着的原 driver 的 `--wait-pid` 和 `--media-lib`，没有硬编码旧 PID。不要重复启动已有服务。此服务不保证跨 Windows/WSL 重启恢复；重启后先检查日志和完整阶段证据。

开发集 OPD 若没有严格超过 QAD 和 continued-QAD，脚本可以显式使用 `--allow-opd-nonimprovement` 完成 paired held-out。这个开关只解除“提前停止”，不会把失败写成增益：`selection.json` 会记录 `opd_gate_override` 和实际的 `opd_beats_qad`、`opd_beats_continued_qad` 布尔值。正文报告 held-out 相对两条基线的差值与配对不确定性，点估计高于二者不是独立增益的充分证据。

训练阶段的日志、manifest 和 checkpoint 不得手工移动。若 driver 在阶段标记前退出，先运行对应的 `train_verify`/`merge_verify`，确认 `runtime_metrics.json`、`recovery_manifest.json`、recipe SHA、category manifest SHA 和协议 SHA 全部一致，再用 `--adopt-complete` 续跑；不能删除唯一 checkpoint 后重训。

当前保存的是 **model-only checkpoint**，不含 Adam、scheduler 和 RNG 状态。它可以用于模型诊断或显式设计的新训练阶段，不能完整恢复被中断的 optimizer 轨迹，也不能根据它手工补写 `runtime_metrics.json`。训练中仅保留最近两份模型 checkpoint；完成后的重复根目录权重可由 driver 的 `--cleanup-duplicates` 校验后删除并留存 receipt。

2026-10-02 已按用户授权清理被当前运行替代的 r2 五份未完成 checkpoint 分片及旧 OPD smoke 导出，共 17 个 `.safetensors`、71,252,118,056 字节。删除前核对当前运行、协议、selection、教师采集和基座没有引用，活进程没有使用；日志、manifest、索引与 Trainer 状态保留。两个旧目录的 `PAYLOAD_REMOVED.json` 标明不可直接加载。完整回执为当前运行的 `operations/cleanup-superseded-20261002/receipt.json`。未删除 r4、当前 PTQ 基座或教师数据；后续仍须同时观察 WSL 文件系统和宿主 C 盘空间。

## 发布门槛

### 本次 CPU 审计与修复

`rl/probe_distill.py` 原来只接受 1 或配置的 16 个累积微批，但 148 个窗口每轮实际形成 `9×16+4`。OPD 每四更新触发，第 20 更新会遇到四样本尾批并异常退出。现已允许合法尾批，按实际组长归一化，且在演示反传前检查非法计数。真实 Transformers/Accelerator CPU 回归覆盖了第 20 更新及两种累积缩放方式。当前 QAD 未安装 OPD hook，无须因此丢弃或重跑；后续 OPD 自动使用修复版本。

每个完整 2,000 更新阶段的真实演示读取预算为 29,600；OPD 额外有 6,800 次学生探针反传。训练顺序与 sampler 未改，最后四个窗口的单次平均损失权重大于完整组中的单个窗口，须保留这一限制。

原始 Trainer `checkpoint-*` 含 A/B，但不能当作普通模型交给评测。两个评测入口现已在加载前拒绝无完整部署清单的恢复 checkpoint；正式 driver 的部署打包路径不受影响。训练与服务均使用 `base(QDQ(x)) + LoRA(raw x)`，这仍是数值 W4A4，不是 GR00T 原生四位内核。

修复记录位于当前运行的 `operations/opd-tail-fix/receipt.json`，变更前后字节保存在 `source_snapshots/<sha>/<相对路径>`。首条 QAD manifest 保留原始源码 SHA，后续阶段记录新 SHA；成本收集器按阶段验证并归档对应版本，不要求未使用的 OPD 源文件哈希跨阶段相同，不补写历史启动证明。协议、数据、模型来源和完整阶段产物均未改写。

发布收集器已支持 v11 的 160 回合和九个训练依赖源文件；样本预算依据绑定的 capture 数量与 Trainer epoch 推导，标注为源码推导计数。`paper/analysis_plan_w4a4.json` 在首条 QAD 期间补充四项比较、任务内配对 bootstrap 与条件 McNemar/Holm 规则，没有修改被冻结的实验协议，也没有保证统计功效。`extract_final_evidence.py` 只接受 v11 W4A4 的完整五臂结果，并逐任务交叉核验 episode 数组。

### 最终回填

最终公开结果只从 `final_manifest.json`、每臂 `eval_manifest.json`、十个任务日志、逐回合 reset identity 和 `paired_comparison.json` 生成。发布前必须从最终选中的 checkpoint 原子重建：

```text
paper/evidence/selected_recipe/category_ptq_recipe.json
paper/evidence/selected_recipe/category_bake_manifest.json
paper/evidence/selected_recipe/category_memory.json
paper/evidence/recipe_inventory.json
```

不得沿用旧 `.incomplete-*` 路径、旧 recipe SHA 或其他协议 bank 的结果。最终正文、图表、摘要和 README 只接受当前 v11 的同一协议、分母和 reset ledger。

正文目前保留红色 `xxx` 占位符是有意的。必须等五臂 160 回合证据和压缩账本通过 `extract_final_evidence.py`、`materialize_final_evidence.py`、`audit_recovery_release.py` 与 publication validation 后，才回填成功率、恢复幅度和压缩比例；训练 loss、development 分数和 smoke 回合都不能代替闭环成功率。

审阅稿的 17 个截图环节中，11 张已核对，6 个以文字占位：`shot_bake`、`shot_qad`、`shot_opdcache`、`shot_opd`、`shot_evalserver`、`shot_rollout`。内部保留原始资产，但这些旧画面不再出现在 HTML、知乎稿或审阅 ZIP。新 `shot_probe` 已实际重拍并核对，13 项 CPU 检查覆盖教师探针、有效掩码及真实 Trainer 的尾批。最终须实拍替换六图、记录 capture 来源、解除 `figures.json` 中对应的 `refresh_pending`，再重新构建和验收。不能只解除标记而沿用旧图片。

源码导航已更新为 37 文件、105 符号。完整 CPU 检查包含 tests/ discovery 与四套论文检查，报告保存在 `paper/validation/cpu-tests.json`；测试夹具的合成数值只用于校验程序，不进入真实结果。初次迁移时的失败报告原样保存在 `paper/_build/final_cpu_20261002_failed_initial/`。
