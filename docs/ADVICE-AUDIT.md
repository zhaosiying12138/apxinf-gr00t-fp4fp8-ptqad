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

---

## 附：2026-09-24 晚间补充决策——两阶段容量策略（跨会话协调，用户已确认）

来源：docs/COORDINATION-directive-2026-09-24.md（协调会话决策记录）。要点与审计意见：

1. **Phase 1（立即，不阻塞）**：head-LoRA(r8) 主线跑通 R0–R8（PTQ→QAD→OPD），
   保持论文最小可发表单元。
2. **Phase 2（等待门控）**：新增 full-weight action head 训练臂。**参数量实测修正
   （姊妹项目 Phase 0 param_audit，2026-09-24）：可训练 action head = 1.62B**
   （DiT 1083M + VL-SA 201M + action_encoder 227M + state_encoder 55M + action_decoder 38M），
   非早期估计的 300–400M；AdamW fp32 m+v ~13GB（bf16-state 路径 ~6.5GB），
   offload 后 CPU RAM 需求 ~20–26GB（本机可用 47GB）。需 FSDP2+CPUOffload，
   基建来自姊妹项目 groot-fsdp2。
   门控：姊妹项目 E2 完成 → 10-min 冒烟（fake_quant fwd + 1 step bwd）；
   E4 完成 → 正式实验。**R0–R8 的 LoRA 路线不等待。**
   另：姊妹项目实测本机 PCIe pinned H2D 38.2 / D2H 23.7 GB/s，unpinned 仅 3.5–4.5 GB/s
   （10× 惩罚）——teacher serving 数据通路若涉及 host staging，pinned memory 为硬前提。
3. **R5b 扩展为容量阶梯**：仅 scale 修正 → head-LoRA(r8) → full head。
   full-head 显著胜出则主路径切换，R5b 升级为主结果之一。
4. **优化器锁定 AdamW**：Muon 禁入本论文全部主表/消融（优化器变更污染
   "恢复增益来自流水线"的归因）。Muon 只属于姊妹论文。
5. **交付物（full-head 情形）**：直接训 W + STE，导出即纯 NVFP4 权重，无 LoRA 旁路
   张量，引擎直接部署——比 LoRA 的 Q(W+BA) 折叠重量化方案（保留作对照）干净。
6. **新增假设 H6**：补偿容量（scale-only / LoRA-r8 / full-head）与闭环恢复上限的关系。
7. **审计意见（本会话补充）**：同意该决策；附加理由——NVFP4 误差全局散布、修正未必
   低秩，r8 低秩约束可能系统性低估可恢复量，full-head 臂正好检验这一点（H6 的机制
   解释）；单卡 FSDP2 实质是 CPUOffloadPolicy+prefetch（sharding 维度用不上），够用。
   风险：wall-clock +20–50%、时序依赖姊妹项目 Phase 进度、π0.5 侧无现成 patches。
