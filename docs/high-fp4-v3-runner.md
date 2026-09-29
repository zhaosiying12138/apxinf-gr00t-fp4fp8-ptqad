# 高 FP4 v3 恢复实验编排

`exp/run_high_fp4_v3.py` 接收开发评测生成的 `selection.json`，重新核对原始日志、初态配对与选择规则后，串行执行恢复实验。没有满足预注册压力条件的 PTQ 配方时，入口拒绝训练。

```bash
python3 exp/run_high_fp4_v3.py \
  --protocol-file exp/recovery_protocol_v3_high_fp4.json \
  --ptq-selection results/ptqad_20260929/high_fp4_v3/evaluations/development/selection.json \
  --run-dir results/ptqad_20260929/high_fp4_v3_recovery \
  --cleanup-duplicates
```

环境沿用 `GR00T_REPO`、`PTQAD_PYTHON`、`LIBERO_PYTHON`、`PTQAD_BASE`、`QAD_DATASET`、`GR00T_BACKBONE_MODEL`、`PTQAD_MEDIA_LIB` 等设置。仅核对输入可加 `--validate-only`；它会生成运行清单，但不启动训练或评测。必须使用独立的新恢复目录。

执行顺序：

1. 同一 PTQ 基座分别进行两组 500 步 QAD，以开发集成功率选择学习率，同分选较低值。
2. 胜出 QAD 在 collection 分区收集观测，每任务 16 条、共 160 条；BF16 教师对这批观测生成缓存。
3. continued-QAD、OPD 权重 0.25、OPD 权重 1.0 均加载同一 QAD 检查点的 A/B，分别以新的优化器训练 100 步。
4. 开发集评估三个续训分支；两个 OPD 候选按成功率选择，同分选较低权重。
5. 配置冻结后，heldout 分区评估 BF16、PTQ、QAD、continued-QAD、胜出 OPD 五臂并验证逐回合配对。

`--until qad_dev|qad_selection|recovery_dev|opd_selection` 可在相应阶段停止。再次使用相同命令，会重新验证并跳过已完成阶段。更改协议、输入选择或已完成证据会导致退出。

## 输出路径与中断处理

模型、检查点、评测和捕获观测从开始即写入稳定的 `artifacts/` 路径，结束后不搬运。manifest 内的 `out`、`video_root`、模型、adapter、collection 和缓存源文件绝对路径因此始终可用。阶段验证完成后才原子发布 `stages/<name>.json`。

未标记目录按未完成处理，默认拒绝覆盖。只有检查后明确使用 `--adopt-complete`，且全部阶段证据校验通过，才能补记完成。它不会恢复半截训练的优化器，也不会重跑或覆盖半截评测。旧版 `work → artifacts` 布局会被拒绝，不能直接接续。

`--cleanup-duplicates` 只删除与保留检查点 SHA256 完全相同的训练根目录权重副本。检查点、merged 模型、原始日志和观测均保留；清理收据不会在接续时被空收据覆盖。

## CPU 验证

```bash
python3 -m unittest exp.test_high_fp4_v3_cpu -v
CUDA_VISIBLE_DEVICES= "$PTQAD_PYTHON" -m unittest exp.test_teacher_cache_cpu -v
```

第一组覆盖日志与 JSON 不一致、计数/Boolean 错误、初态配对错误、选择结果被改写、稳定路径和中断恢复。第二组使用极小 CPU 张量覆盖缓存缺样本、sidecar 不一致、错误回放 seed、学生来源、非有限教师输出和动作掩码；不会运行模型或初始化 CUDA。
