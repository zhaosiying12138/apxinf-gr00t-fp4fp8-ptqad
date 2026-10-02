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

截至 2026-10-02 11:00（Asia/Shanghai），QAD 的 `5e-5` 臂约为 978/2000 optimizer steps，driver PID 为 4118527，训练 PID 为 4118579；这些是当时的快照，接手时必须重新检查。随后按冻结顺序完成另一学习率、QAD 选择、学生状态采集、continued-QAD、两个 OPD 权重及 held-out 五臂比较。

续跑由 systemd 用户服务 `fp4vla-v11-r4-continuation.service` 托管，入口为 `exp/continue_w4a4_run.py`。服务先等待绑定的 driver 退出，然后校验并复用已完成阶段。当前 driver 没有 `PTQAD_MEDIA_LIB`，第一条 development 评测预计会在 FFmpeg 预检时退出；续跑器仅在日志精确匹配该错误、评测目录完全为空、没有评测 manifest 或完成标记且 GPU 空闲时，归档原日志并记录 SHA，再使用已验证的 media7 路径续跑。它不删除已有结果，不重跑部分训练，不修改冻结的训练或评测实现。

```bash
systemctl --user status fp4vla-v11-r4-continuation.service
journalctl --user -u fp4vla-v11-r4-continuation.service -n 20 --no-pager
```

续跑调用和后续 driver 日志位于运行目录的 `operations/continuation-*/`。`exp/continue_v11.sh` 现在是通用入口，调用时必须显式提供 `--run-dir`、活着的原 driver 的 `--wait-pid` 和 `--media-lib`，没有硬编码旧 PID。不要重复启动已有服务。此服务不保证跨 Windows/WSL 重启恢复；重启后先检查日志和完整阶段证据。

开发集 OPD 若没有严格超过 QAD 和 continued-QAD，脚本可以显式使用 `--allow-opd-nonimprovement` 完成 paired held-out。这个开关只解除“提前停止”，不会把失败写成增益：`selection.json` 会记录 `opd_gate_override` 和实际的 `opd_beats_qad`、`opd_beats_continued_qad` 布尔值；只有 held-out 同时超过两条基线时，论文才允许宣称 OPD 有独立提升。

训练阶段的日志、manifest 和 checkpoint 不得手工移动。若 driver 在阶段标记前退出，先运行对应的 `train_verify`/`merge_verify`，确认 `runtime_metrics.json`、`recovery_manifest.json`、recipe SHA、category manifest SHA 和协议 SHA 全部一致，再用 `--adopt-complete` 续跑；不能删除唯一 checkpoint 后重训。

当前保存的是 **model-only checkpoint**，不含 Adam、scheduler 和 RNG 状态。它可以用于模型诊断或显式设计的新训练阶段，不能完整恢复被中断的 optimizer 轨迹，也不能根据它手工补写 `runtime_metrics.json`。训练中仅保留最近两份模型 checkpoint；完成后的重复根目录权重可由 driver 的 `--cleanup-duplicates` 校验后删除并留存 receipt。

## 发布门槛

最终公开结果只从 `final_manifest.json`、每臂 `eval_manifest.json`、十个任务日志、逐回合 reset identity 和 `paired_comparison.json` 生成。发布前必须从最终选中的 checkpoint 原子重建：

```text
paper/evidence/selected_recipe/category_ptq_recipe.json
paper/evidence/selected_recipe/category_bake_manifest.json
paper/evidence/selected_recipe/category_memory.json
paper/evidence/recipe_inventory.json
```

不得沿用旧 `.incomplete-*` 路径、旧 recipe SHA 或其他协议 bank 的结果。最终正文、图表、摘要和 README 只接受当前 v11 的同一协议、分母和 reset ledger。

正文目前保留红色 `xxx` 占位符是有意的。必须等五臂 160 回合证据和压缩账本通过 `extract_final_evidence.py`、`materialize_final_evidence.py`、`audit_recovery_release.py` 与 publication validation 后，才回填成功率、恢复幅度和压缩比例；训练 loss、development 分数和 smoke 回合都不能代替闭环成功率。
