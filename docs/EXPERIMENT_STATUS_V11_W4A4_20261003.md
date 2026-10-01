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

在 development 上，BF16 为 46/50，全 NVFP4 W4A4 PTQ 为 41/50，已经形成约 10 个百分点的恢复窗口。该数字只证明压力候选满足进入恢复链的门槛，不是最终主表结果；正文必须等待五臂 held-out ledger。

教师采集已完成 10 个任务、148 个有效 snapshot，CPU replay audit 通过。当前运行目录为：

```text
results/ptqad_20261003/v11_recovery_r2
```

## 当前 GPU 进程与续跑纪律

QAD 的 `5e-5` 臂正在训练 2000 个 optimizer steps，随后由同一个 v11 driver 继续训练另一学习率、选择 QAD、收集学生状态、训练 continued-QAD 与两个 OPD 权重，并运行 held-out 五臂比较。续跑脚本为 `exp/continue_v11.sh`，使用 `--adopt-complete` 只审计并采用已经完成的稳定路径。

开发集 OPD 若没有严格超过 QAD 和 continued-QAD，脚本可以显式使用 `--allow-opd-nonimprovement` 完成 paired held-out。这个开关只解除“提前停止”，不会把失败写成增益：`selection.json` 会记录 `opd_gate_override` 和实际的 `opd_beats_qad`、`opd_beats_continued_qad` 布尔值；只有 held-out 同时超过两条基线时，论文才允许宣称 OPD 有独立提升。

训练阶段的日志、manifest 和 checkpoint 不得手工移动。若 driver 在阶段标记前退出，先运行对应的 `train_verify`/`merge_verify`，确认 `runtime_metrics.json`、`recovery_manifest.json`、recipe SHA、category manifest SHA 和协议 SHA 全部一致，再用 `--adopt-complete` 续跑；不能删除唯一 checkpoint 后重训。

## 发布门槛

最终公开结果只从 `final_manifest.json`、每臂 `eval_manifest.json`、十个任务日志、逐回合 reset identity 和 `paired_comparison.json` 生成。发布前必须从最终选中的 checkpoint 原子重建：

```text
paper/evidence/selected_recipe/category_ptq_recipe.json
paper/evidence/selected_recipe/category_bake_manifest.json
paper/evidence/selected_recipe/category_memory.json
paper/evidence/recipe_inventory.json
```

不得沿用旧 `.incomplete-*` 路径、旧 recipe SHA 或 v5/v7 bank 的结果。旧的“43.4%→99.0%”、v5 的 86/89/92/96 等数字没有当前 v11 的同一协议、分母和 reset ledger，不能进入正文、图表、摘要或 README。

正文目前保留红色 `xxx` 占位符是有意的。必须等五臂 160 回合证据和压缩账本通过 `extract_final_evidence.py`、`materialize_final_evidence.py`、`audit_recovery_release.py` 与 publication validation 后，才回填成功率、恢复幅度和压缩比例；训练 loss、development 分数和 smoke 回合都不能代替闭环成功率。
