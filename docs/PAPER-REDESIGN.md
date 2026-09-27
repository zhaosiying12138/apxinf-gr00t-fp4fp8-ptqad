# 论文重组设计（2026-09-27 11:1x，用户指令：tutorial 式论文 + 恢复阶梯消融）

## 用户明确要求

1. 只微调动作头权重（full-weight action head）——OPD 已是此形态
2. **backbone 必须量化**——OPD 全局 fake-quant（含 backbone）满足；最终交付 = 全模型 NVFP4
3. **消融实验展示 PTQ→QAD→OPD 逐步闭环提升**——主消融表
4. 或许跟 PPO 对比——设计为讨论 + 可选对照臂
5. **tutorial 式论文**：PTQ/QAD/OPD 详细原理一五一十，所有量化方法讲清楚，篇幅不限

## 主消融表设计（恢复阶梯，GR00T N1.7 LIBERO-10 × 10 eps）

**2026-09-28 更新：PTQ 攻坚后阶梯重构**——纯校准（GPTQ/裁剪）救不了全 NVFP4（calib 0%），
NVIDIA 式混合分配一举回到 93.3%（mini）；恢复训练（QAD/OPD-LoRA）以 aggr（2.88×，仅护
o/down+解码器）为基座，对标 mixed（2.02×，部署级）与 BF16 上界。新增 §12 PTQ 教程章。

| 阶梯 | 训练配方 | 量化部署 | 压缩 | 闭环 | 状态 |
|---|---|---|---|---|---|
| BF16 上界 | — | 无 | 1× | 96.7% | ✅ |
| SFT 无量化（姊妹 E4） | 官方配方 1000 步 | 无 | 1× | 87.4% | ✅（姊妹数据） |
| PTQ rtn | — | 全模型 NVFP4 RTN | 3.56× | 0% | ✅ |
| PTQ calib | — | 全 NVFP4 + GPTQ + MSE 裁剪 | 3.56× | 0%（mini） | ✅ 09-28 |
| PTQ fp8 | — | backbone FP8 + head NVFP4 | 2.44× | 70.8%（mini） | ✅ 09-28 |
| **PTQ mixed（部署级）** | — | NVIDIA 分配复刻 | 2.02× | 93.3%（mini）/ [10×10 TBD] | ✅ 09-28 |
| PTQ aggr（恢复基座） | — | 全 NVFP4 仅护 o/down+解码器 | 2.88× | [TBD-aggr] | 🔄 |
| QAD-LoRA | demo loss + merged-quant STE | aggr | 2.88×+LoRA | ? | 🔄 就绪待跑 |
| OPD-LoRA | + probe-cached teacher-KL | aggr | 2.88×+LoRA | ? | 设计就绪 |
| QAD（旧全量头） | +fake-quant SFT | 全模型 NVFP4 | 3.56× | 0.63% | ✅（历史） |
| PPO 对照（同预算） | 任务奖励微调（LoRA） | aggr | 2.88×+LoRA | ? | 设计中 |

辅证消融：混合精度二分（head-FP4 71.2 / backbone-FP4 0）、双模型验证（π0.5
BF16 90% / nvfp4 0%）、离线-闭环脱节（信号扫描 + 09-28 probe 混沌教训：连 FP8/mixed
都 relMSE≈2 corr≈0 而闭环 93%——闭环是唯一裁判）、AWQ α=0 全站点负结果
（weight-only NVFP4 块缩放已吸收通道异常值）。

## 新论文结构（两篇合一的教程型长文）

### 第一部分：教程（Tutorial，读者不需要先修量化知识）

**§1 引言与问题设定**
VLA 部署约束；闭环成功率 vs 离线指标的鸿沟（用我们的数据做引子）。

**§2 VLA 模型解剖**（以 GR00T N1.7 / π0.5 为例）
VLM backbone（Qwen3-VL / Gemma）+ DiT flow-matching action head；
KV-cache 复用与误差复合的结构性原因（连向 §7 混合精度结论）。

**§3 数值格式与量化基础**
3.1 FP32/BF16/FP16 回顾
3.2 E2M1（4-bit）：位级布局、值域 {0,±0.5,...,±6}、中点舍入、饱和
3.3 E4M3（8-bit scale）、UE8M0（MX scale）
3.4 缩放粒度：per-tensor / per-channel / per-block(16) / 双重量化
3.5 舍入模式：RNE、ties-to-even、为什么它改变了我们的调试

**§4 NVFP4 深度解析**
4.1 格式定义：E2M1 + per-16 E4M3 + per-tensor FP32
4.2 硬件执行路径：cuBLASLt block-scaled GEMM
4.3 scale swizzle 布局（128×4 tile 行交错）——含我们逆向的故事与公式
4.4 fake-quant 模拟：STE、逐层数值验证方法（我们的 gold-check 方法论）
4.5 与 MXFP4 的对比（sm_120 不支持的实证）

**§5 量化恢复方法论**（论文核心教程）
5.1 PTQ：流程、失效模式（权重均匀但复合放大）、何时够用
5.2 QAD（Quantization-Aware Distillation）
    - QAT 原理：fake-quant 前向 + STE 反向
    - NVIDIA 配方：冻结 BF16 teacher、KL 损失替换任务损失、scale 冻结
    - 我们的实现与结果（demo loss 版的失效——重要负结果）
5.3 OPD（On-Policy Distillation）
    - DAgger 协变量漂移理论：O(T²ε)→O(Tε)
    - λ 混合目标、reverse-KL、向量场匹配
    - 我们的 twin-forward 实现（同模型交换量化/原始 forward）
    - 成功率损失分解（Δ_floor/Δ_shift/Δ_opt）
5.4 与 PPO 的关系：教师塑形奖励下的 PPO ≡ OPD 加权形式；
    稠密监督 vs 稀疏奖励的样本效率；超越教师上限的条件

**§6 闭环评测方法论**
LIBERO 协议、Wilson CI、为什么离线 corr/L2 不能预测闭环（我们的实证）。

### 第二部分：实验研究

**§7 系统平台**：5090/sm_120、ApxInf 引擎 NVFP4 路径、算子数据（506 TFLOPS）
**§8 主消融：恢复阶梯**（上表）+ PPO 讨论/对照
**§9 混合精度二分**：backbone vs head 定位 + 机制解释
**§10 双模型验证**：π0.5 引擎路径
**§11 讨论、局限、结论**

## PPO 对照臂设计（可选，视时间）

同预算（1000 步、同 batch）任务奖励微调 fake-quant 模型；
工程路径：RLinf 或极简 PPO 循环 + LIBERO 环境奖励。
若不跑：§5.4 给理论对比 + §8 引用 E5 Muon 数据作 RL-相邻证据。
决策点：OPD 结果出来后定（若 OPD 已显著恢复，PPO 升级为"锦上添花"）。

## 与姊妹论文的边界（不变）

FSDP2/offload = 系统使能技术一节 + 交叉引用；Muon 不进本文主表。
