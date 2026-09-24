# 顾问建议审计：PTQ + QAD + OPD（2026-09-24）

**建议原文**："ptq+qad+opd 去提高量化后评测的成功率，qad 的时候可以 freeze moe 之外的 weight。"

## 术语溯源（已核实）

| 术语 | 出处 | 核心做法 |
|---|---|---|
| PTQ | GPTQ/AWQ/SmoothQuant 一脉 | 权重直接转换+校准，无训练 |
| **QAD** | NVIDIA, *Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery*, [arXiv:2601.20088](https://arxiv.org/abs/2601.20088)（2026-01，Nemotron-3-Nano-30B MoE） | QAT 模拟 NVFP4 + 蒸馏：BF16 teacher 冻结，KL 损失替换任务损失；**训练时冻结 scale、更新权重**；官方开源 toolkit |
| **OPD** | GKD: Agarwal et al., *On-Policy Distillation of Language Models*, [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)（ICLR 2024）；及 2025 年 Gemini Nano 上的 OPD 实践 | **学生自采样** + teacher 逐 token 纠偏，reverse-KL（mode-seeking、贴合学生支撑集）优于 supervised-KD/SeqKD，并可衔接 RLHF |

## 审计结论：**方向采纳，两处必须纠正，一处按我们预算降档**

### ✅ 采纳（且优于我们原 L1/L2 设计）
1. **三段阶梯 PTQ→QAD→OPD 命名与结构**：这是 NVIDIA 在 LLM NVFP4 上已验证的 SOTA 恢复配方，
   术语有出处、审稿人可直接对号入座。我们原来的"蒸馏+小规模 PPO"重命名为 QAD + OPD，
   PPO 保留为对照臂（理由见下）。
2. **QAD 冻结 scale、更新权重**（NVIDIA 实践）：推翻了我原 L1"只学 scale"的设想，
   按报告做法权重走 LoRA、scale 冻结。把"仅学 scale"降级为一个消融臂（R5b-α）。
3. **OPD 作为主恢复机制**：与我们的闭环动机（协变量漂移）严丝合缝，GKD 的 reverse-KL
   论证可直接迁移到 flow-matching action 分布上（用向量场匹配损失实现）。

### ⚠️ 纠正一："freeze MoE 之外的 weight" 对我们不适用
GR00T N1.7（Cosmos-Reason2-2B backbone + DiT action head）和 π0.5 都是**稠密模型，没有 MoE**。
该建议是 Nemotron MoE 语境的（冻结 dense/attention/shared 层、只训 expert 以省显存）。
**正确的等价移植**：冻结 VLM backbone，只训 action head 的 LoRA（这正是敏感性分析预期
量化损伤集中处，训练预算也最小）。冻结策略本身保留为一个**实验臂**（R5b）：
{仅学 scale / 仅 action-head LoRA / +backbone LoRA}，而不是照搬 MoE 处方。

### ⚠️ 纠正二：LLM 配方恢复的是"开环指标"，我们的靶是"闭环成功率"——这正是论文的科学增量
NVIDIA QAD/OPD 恢复的是 perplexity/静态 benchmark（teacher-forced、开环）。
VLA 的成功率是闭环指标：量化误差导致的状态漂移使策略离开训练分布、误差平方级复合（DAgger 理论）。
**离线 QAD 不解决漂移，OPD 才解决**——这个差距本身就是我们的可检验假设（H1，实验 R2 vs R3）。
即：朋友的配方必要但不充分，"不充分"恰好是我们论文的贡献点。

### 📉 降档（预算约束）
NVIDIA QAD 是全参数、集群规模；我们是单卡笔记本 + 一天预算。
降档为：LoRA(r8, action head) + 冻结 scale + 少量 LIBERO demo 数据；OPD 用
ApxInf BF16 teacher（引擎高速服务 teacher 动作）+ fake-quant 学生 rollout，保持
"推理引擎在训练闭环里"的生态卖点。PPO 臂经 RLinf 保留，用于 (a) 对比稠密监督 vs 稀疏奖励
的样本效率，(b) 检验"RL 能否超越 teacher 上限"。

## 修订后的恢复流水线（替换原 L1/L2）

```
L0 PTQ       权重转换 + 激活校准（无训练）
L1 QAD       teacher-forced 蒸馏：D_demo 上向量场匹配损失，
             冻结 backbone 与 scale，只训 action-head LoRA        [预算 ~2h]
L2 OPD       学生 rollout（fake-quant）+ ApxInf BF16 teacher 逐步纠偏，
             reverse-KL/向量场匹配，λ 混合 on/off-policy 数据      [预算 ~8h]
L3 PPO(对照) RLinf 任务奖励微调，同预算                             [预算 ~8h]
```

详细数学原理与实验矩阵见 `paper/sections/02-恢复理论与实验设计.md` 与 `docs/PLAN.md`（已更新）。
