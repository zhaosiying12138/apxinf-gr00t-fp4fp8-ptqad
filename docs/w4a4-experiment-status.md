# W4A4 实验与证据状态

本文档对应 `exp/recovery_protocol_v11_w4a4_category.json`，协议 SHA-256 为 `158acbd49fbabbd49e5af18079ab1d0c5e9b067e4e2f72f2c2b44060b6fb26cb`。完整复现命令见[项目 README](../README.md)，方法、结果与原生执行基准见[中文论文](../paper/paper.html)。

## 量化与恢复的范围

479 个可量化权重张量全部采用 NVFP4：469 个普通 Linear、7 个 CategorySpecificLinear，以及 3 个 embedding/position 张量。前两类共 476 个线性算子同时量化输入激活；最后 3 个张量仅量化权重。激活沿最后一维每 16 个元素分块，使用 E4M3 块尺度与固定二级尺度 `1.0`；类别层的 132 维输入补零到 144 维，QDQ 后切回原始维度。

QAD/OPD 在 468 个参与动作前向的普通 Linear 上训练低秩分支，`rank=32`、`alpha=64`，共 72,998,912 个参数。`lm_head` 不参与动作前向，类别层不含低秩分支。部署保持两条计算路径：

```text
base = Linear(QA(x), Wq)
residual = B(A(x)) * alpha / rank
output = base + residual
```

`x` 是原始 BF16 输入。量化主分支读取 `QA(x)`，低秩分支读取 `x`。评测从导出清单定位冻结基座和独立 A/B 参数；稠密 `Wq + BA` 文件只用于诊断，不能代替这个前向。

当前 GR00T 行为结果来自 **Torch W4A4 QDQ 数值路径**。APXInf 的原生 FP4 GEMM、单层管线和 π0.5 模型执行单独验收、单独计时；尚未测量恢复后 GR00T 的原生 packed 执行速度。

## 数据分区与比较

| 分区 | 每任务官方初态索引 | 用途 |
|---|---|---|
| 开发集 | 4–8 | PTQ 筛选及 QAD/OPD 超参数选择 |
| 教师演示、学生采集 | 20–23 | 演示与教师探针数据 |
| 独立评测集 | 9–19、24–28 | 五臂各 160 回合的冻结比较 |

教师成功轨迹提供十任务共 148 个演示窗口。学生采集与教师标注构成 160 个探针样本。QAD 训练 2,000 次更新；continued-QAD 与 OPD 从同一 QAD 检查点分别追加 2,000 次更新。每阶段读取 29,600 次演示窗口，OPD 额外执行 6,800 次探针前向与反传。

初始 QAD 训练对 F16 溢出报错。continued-QAD 与 OPD 的追加训练采用有限值饱和与 `max_grad_norm=0.25`。所有量化臂的正式推理统一启用有限值饱和。训练收据和评测环境分别保存这两类设置，不能用其中一个替代另一个。

因此，OPD−QAD 包含追加预算及训练数值设置变化；OPD−continued-QAD 才隔离相同演示与更新预算下教师监督的作用。两者不等计算量，OPD 的探针与标注成本单列。

## 当前完成度

<!-- BEGIN VERIFIED STATUS -->
截至 2026-10-06，五个正式评测臂均完成 160 回合。BF16/PTQ/QAD/continued-QAD/QAD+OPD 分别为 145/160、145/160、144/160、104/160、134/160；完整逐回合日志、配对统计和环境清单已通过发布校验。
<!-- END VERIFIED STATUS -->

## 证据入口

| 内容 | 公开工件 |
|---|---|
| 冻结实验与分析规则 | `exp/recovery_protocol_v11_w4a4_category.json`、`paper/analysis_plan_w4a4.json` |
| 最终五臂结果与配对统计 | `paper/evidence/final_results.json`、`paper/evidence/paired_comparison.json` |
| 逐回合原始记录 | `paper/evidence/heldout_raw_logs.json` 及其中引用的 rollout/server 日志 |
| 量化覆盖、编码预算 | `paper/evidence/recipe_inventory.json` |
| 训练、教师与采集成本 | `paper/evidence/training/costs.json` |
| 运行环境与源文件身份 | `paper/evidence/runtime/` |
| 实拍与来源 | `paper/evidence/captures.json`、`paper/evidence/retained_captures.json` |

发布校验重新核算成功率、配对初态、原始日志散列和图像来源；截图用于展示执行过程，完整日志与结果数组用于核验实验结论。复现者应创建自己的输出目录和协议副本，不覆盖仓库内的发布证据。
