# W4A4 实验状态

更新时间：2026-10-02。本文档记录冻结 v11 W4A4 路径的工程验证和当前阶段，不是论文最终结果表。

## 已冻结的执行契约

W4A4 的主分支按以下顺序计算：输入先转为 F16；沿最后一维每 16 个元素求块最大值；用固定二级 scale `1.0` 和 E4M3 表示块 scale；再用 E2M1 对归一化输入舍入。QAD/OPD 的前向保持两条分支：

```text
base = Linear(QA(x), Wq)
residual = (B(A(x))) * alpha / rank
output = base + residual
```

`x` 是原始 BF16 输入。只有 `base` 使用 `QA(x)`；把 `QA(x)` 送入残差会改变 QAD/OPD 的定义，因此服务端和训练端都用同一条显式分支实现。正式部署不再把 `Wq + BA` 当成 W4A4 权重使用：`lora_merge_bake.py` 生成的 dense 导出只作为诊断文件，W4A4 服务从 `merge_manifest.json` 回到冻结 base，再加载 A/B 张量。

当前路径是 **GPU Torch 数值 QDQ 仿真**。APXInf 的原生 FP4 GEMM 已在 π0.5 路径中存在，但 GR00T 的原生 executor 尚未提供同一套 W4A4 loader；因此本阶段不把 GR00T 结果写成“原生 APXInf W4A4”。

## 已完成的验证

- `tests.test_native_activation`、`tests.test_w4a4_lora`、`tests.test_w4a4_deploy`：共 17 项通过。
- W4A4 PTQ smoke：真实 GR00T 服务日志报告 469 个普通 Linear 和 5 个 CategorySpecificLinear 安装激活 QDQ，1/1 回合成功；这是接口 smoke，正式 v11 full-category 配方覆盖 7 个活动 category bank。
- QAD smoke：5 步训练，364 个 LoRA Linear 使用 `base(QA(x)) + residual(x)`；梯度审计通过。独立 adapter 服务 1/1 回合成功。
- OPD smoke：从同一 QAD A/B 检查点继续训练 5 步，5/5 次更新执行 teacher velocity MSE；独立 adapter 服务 1/1 回合成功。

## v11 开发集与教师数据

v11 development 已完成：BF16 为 46/50，W4A4 全覆盖 PTQ 为 41/50，下降 10 个百分点，因此满足恢复实验的压力门槛。BF16 教师监督分区生成 148 个有效快照（十任务、每任务至少两条成功轨迹），CPU replay audit 已通过。正式 held-out 仍未读取。

## 快速压力探针

同一 `v9` smoke bank、同一 10 个任务、每任务 1 回合：

| 臂 | 成功 | 回合 | 成功率 | 用途 |
|---|---:|---:|---:|---|
| BF16 | 10 | 10 | 100% | 配对压力参照 |
| 全 NVFP4-GPTQ + W4A4 | 8 | 10 | 80% | 恢复候选筛选 |

这组数字只说明存在约 20 个百分点的压力窗口；它使用 smoke 分区，不能替代开发集或 160 回合 held-out 结果。正式门控仍要求：PTQ 低于 BF16、QAD 高于 PTQ、OPD 高于 QAD 且高于 continued-QAD，并在同一新协议上完成配对评测。

## 为什么旧的高分不能复用

此前出现过的 47/50 或 43/50 不能作为本轮 W4A4 结果。旧的 rollout 入口只把
`FP4VLA_QUANT` 设为关闭，没有显式传入 `FP4VLA_W4A4=1`；服务端因此加载了
已经 bake 的权重，却没有安装本轮的激活 QDQ。那条路径没有验证
`Linear(QA(x), Wq)`。本轮 `eval/run_recovery_eval.py` 会把
`FP4VLA_W4A4` 和 `FP4VLA_W4A4_ADAPTER` 写入每个子进程环境，服务端还必须在
`*.server.log` 输出 `activation-only` 报告；恢复入口会核对这份报告后才接受分数。

旧 category 日志还因 `K % 16 != 0` 直接跳过了 132 维输入方向的 padding 路径，
所以只安装了部分 category 层。修复后的 installer 先把 K 补到 16 的倍数，完成
QDQ 后再切回原始维度；新的 full-category 配方必须重新绑定当前 parent 的
`ptq_recipe.json` 和 `bake_manifest.json`，旧 calibration cache 不能直接复用。

这也解释了早期“PTQ 很低、QAD 接近满分”的记录为何不进入 v11。那批记录的
任务覆盖、episode 分母、LoRA scope、batch/累积设置和 base→adapter 链都与本协议
不同；其中 OPD 是从独立基座启动，并使用固定离线 probe，不是本协议定义的
“QAD 后学生访问状态蒸馏”。此外，早期格式化代码的 E4M3/E2M1 舍入边界与当前
实现不等价。它们只能用于定位实现差异，不能和当前 W4A4 五臂结果拼接或互相
证明。公开表只接受带协议 SHA、activation installer 报告、reset identity 和
完整 episode ledger 的 v11 记录。

## 当前主线（v11）

为了让“必须 W4A4”可审计，主线使用 `exp/recovery_protocol_v11_w4a4_category.json`。
它要求 479 个 weight tensor 全部有 NVFP4；其中 476 个可执行 Linear（469 个普通、7 个 category）走 W4A4 activation QDQ；3 个 embedding/位置参数没有 activation matmul。
selection 和最终 manifest 还必须保存 activation 安装报告。执行顺序是：

1. 对当前 `all_nvfp4_gptq` parent 重新采集 category H（128 windows），再 bake 出新的 full-category checkpoint；
2. 用同一 v11 bank 重跑 BF16 与 W4A4 PTQ development，生成 development-only selection；
3. 采集 BF16 成功教师轨迹，训练 QAD；从同一 QAD A/B checkpoint 分出 continued-QAD 与 OPD；
4. 在 160 个 held-out 回合上评测五臂，并以配对 ledger 检查 `PTQ < QAD < OPD` 以及 OPD 对 continued-QAD 的独立增益。

任何一臂缺少 activation installer 报告、category 覆盖不全或协议 SHA 不一致，都会在
selection 阶段停止，不把该分数写入论文。

## 当前进行中的主实验

1. QAD 学习率 `5e-5` 正在进行 2000 个优化步；完成后按同一 run-dir 运行 `1e-4` 并只在 development 选择。
2. 选择 QAD 后采集 W4A4 学生访问状态并生成 160 条教师探针缓存；随后串行训练 continued-QAD 与两个 OPD 权重。
3. OPD 选择必须同时高于 QAD 和 continued-QAD；否则 driver fail-closed，不把不支持的增益写入正文。
4. 通过后在 160 个 held-out 回合评测五个臂，再把唯一一套 v11 数字回填到论文、HTML、知乎 Markdown、图表和受影响的 Ubuntu 运行截图。
