# v12 W4A4 实验状态与后续分析说明

本文档是给分析 agent 的只读上下文。**不要启动训练、量化、评测或截图，不要改动冻结协议和已发布证据。** 只阅读源码、运行清单与日志，指出风险和下一步；GPU 运行由主流程统一管理。

## 研究问题

目标是验证一条完整的 W4A4 恢复链：全覆盖 NVFP4 训练后量化（PTQ）制造真实行为压力，QAD 在成功演示上恢复量化策略，OPD 再在 QAD 学生真实访问状态上加入 BF16 教师速度监督。必须同时保留 BF16、PTQ、QAD、continued-QAD、QAD+OPD 五个臂，才能分别回答：量化损失是否存在、演示恢复是否有效、OPD 是否超过同预算续训。

## 冻结协议

- 协议：`exp/recovery_protocol_v12_rtn_w4a4.json`
- SHA-256：`31addae911f92db65dda3c1c3145e8b798dca46c49a5ae12bbe10d232b5a4521`
- 479 个 eligible 权重张量全部 NVFP4；469 个 ordinary Linear 和 7 个 CategorySpecificLinear 使用 W4A4 激活 QDQ；3 个 embedding/position 张量只量化权重。
- QAD/continued-QAD/OPD 均为 2,000 steps；468 个 ordinary Linear 训练 rank=32、alpha=64 的 BF16 LoRA，类别层冻结。
- 开发集为官方初态 4–8（每任务 5 回合）；教师监督和学生采集使用 20–23；held-out 为 9–19、24–28（每任务 16 回合，共 160 回合/臂）。选择禁止读取 held-out。
- 预注册效应目标：PTQ 相对 BF16 至少下降 5 pp；QAD 相对 PTQ 至少恢复 5 pp；OPD 相对 QAD 至少提升 5 pp。160 配对回合不保证分辨 5 pp 差异；结论还须结合配对区间及同预算 continued-QAD 对照。统计解释以 `paper/analysis_plan_w4a4.json` 为准。若数据未达到目标，必须如实报告，不得调整协议或用旧结果替换。

## 当前运行状态（2026-10-08，动态核验）

v12 开发压力已完成：BF16 46/50，W4A4 RTN PTQ 41/50，说明当前全覆盖 W4A4 基座产生了约 10 pp 的开发集压力。正式恢复目录为：

```text
results/reruns/rtn_w4a4_release_20261006_01/recovery_v12/
```

本次核查时 QAD 第一学习率阶段仍在运行，已保存 checkpoint-1500；五臂 held-out 尚未开始。步数与进程状态会继续变化，接手时运行以下只读命令获取当前记录，并用 `ps` 核对日志所对应的训练进程。状态 JSON 中的 `incomplete` 只表示缺少完成收据，不等于训练已经停止。

本次动态核验（2026-10-08 03:55 CST）显示主驱动 PID `2467476` 仍在运行，当前阶段为 `train_opd_025`，日志已到约 `117/2000` optimizer steps；`stages/train_opd_025.json` 尚未生成，说明该阶段尚未完成。不要重启该进程、不要并行占用 GPU，也不要把旧 v11 checkpoint 当作 v12 候选。完成 `opd_025` 后，驱动仍需训练 `opd_100`、在开发集选择 OPD 权重，再执行五臂 held-out；在此之前不得生成最终论文结果或截图。

```bash
python3 exp/high_fp4_status.py \
  --development-root results/reruns/rtn_w4a4_release_20261006_01/selection_final \
  --recovery-root results/reruns/rtn_w4a4_release_20261006_01/recovery_v12
```

主命令通过 `PTQAD_ZMQ_TIMEOUT_MS=120000` 延长 GR00T PolicyClient 的请求超时。驱动依次完成两个 QAD 学习率及开发评测、选择 QAD、采集学生状态、缓存教师、训练 continued-QAD 与两种 OPD 权重、选择 OPD，最后运行五臂 held-out。当前接续进程会等原驱动退出后，用已提交的验证器核验完整产物并继续；它不会覆盖真正不完整的阶段。若发生失败，先审计日志再恢复，不因观察超时而重启，不并发启动第二个 GPU 作业。

QVLA 对照审阅见 [`QVLA_GAP_REVIEW_20261006.md`](QVLA_GAP_REVIEW_20261006.md)。新增的[固定观测动作诊断](ACTION_CHUNK_DIAGNOSTICS.md)已通过 CPU 测试，并从教师训练分区冻结 80 个观测；尚未运行 GPU 测量。该入口用于训练分布上的拟合分析，不是独立泛化证据；它要求本轮最终五臂清单，不能在当前训练中途冒用检查点标签。

## 为什么不能使用旧 v11 数字

旧 `paper/evidence/final_results.json` 来自 v11 协议和不同运行目录；它没有绑定 v12 的协议 SHA 和 RTN 基座，不能证明本轮重新量化与训练的效果。旧数字可以用于审计差异，不能进入 v12 论文、README 或截图说明。`paper/extract_final_evidence.py` 校验 final manifest 格式与来源身份；发布生成器另以冻结 v12 SHA 拒绝旧协议。

## 读源码时重点核对

1. `exp/run_high_fp4_v3.py` 的 stage receipt 是否与输出目录和 protocol SHA 一致；不完整目录不能通过 `--adopt-complete`。
2. `eval/run_recovery_eval.py` 是否保存每个 episode 的 reset hash、seed、初态索引和 `PTQAD_ZMQ_TIMEOUT_MS`；ZeroMQ 超时重试不得改变 episode 顺序。
3. `eval/compare_recovery.py` 是否从同一 held-out round 计算五臂配对差异；只看总成功率不能替代配对分析。
4. `paper/extract_final_evidence.py`、`paper/v12_publication_contract.py` 是否拒绝旧协议、缺臂、少于 160 回合或 task 分子分母不一致。
5. QAD/OPD 的 merge manifest 是否同时加载冻结 W4A4 基座和独立 LoRA；把 LoRA 合入后再量化会改变训练函数，属于无效复现。

## 快速分析问题

- 若评测出现 `zmq.error.Again`，先检查任务对应 `.server.log`、`.log` 与 `eval_manifest.json` 的 timeout 字段，确认是否是服务启动/首次 warm-up 超时；不要删除部分日志，也不要手工补成功标记。
- 若 PTQ 压力消失，检查运行时是否同时安装了 W4A4 activation QDQ、479 张量 recipe 和 category bank；仅把权重存成 FP4 但服务走 BF16 激活不能算 W4A4。
- 若 QAD 无恢复，检查 teacher capture 是否只含 BF16 成功轨迹、监督覆盖十个任务，以及恢复 scope 是否覆盖所有 active ordinary Linear。
- 若 OPD 不超过 continued-QAD，应报告“本配置下未观察到独立增益”，不能把 OPD−QAD 的追加训练差异冒充教师监督效果；只有完整五臂完成后才可决定论文结论。

## 交付门槛

最终发布必须同时具备：v12 `final_manifest.json`、五臂各 160 回合逐回合日志和配对统计、recipe/LoRA 字节账本、训练成本、17 张 3840×2280（去除任务栏）Ubuntu 截图、`paper/paper.html`、`paper/zhihu/article.md` 和根目录中文 `README.md`。README 必须列出安装、依赖、编译、PTQ 制备/选择、教师采集、QAD、OPD、评测、证据材料化、HTML/知乎构建和验证命令。
