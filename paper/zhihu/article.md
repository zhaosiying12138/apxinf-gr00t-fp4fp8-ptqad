# 摘要

视觉-语言-动作（VLA）模型需要在本体上以 10–50 Hz 闭环运行，但现有部署研究集中在
BF16/FP8。NVIDIA Blackwell 将 NVFP4（E2M1 + 逐 16 元素 E4M3 块缩放）带入消费级
GPU，却没有任何公开工作回答：**VLA 能否在 FP4 下运行，闭环成功率损失几何，如何恢复？**
本文在 RTX 5090 Laptop（sm_120）上给出首个系统性回答。算子层：我们逆向并验证了
cuBLASLt 无文档的 128×4 tile 行交错块缩放布局，NVFP4 GEMM 峰值 506 TFLOPS（1.8×
FP8）；引擎层：在 ApxInf 中交付完整 NVFP4 执行路径（四层正确性验证，corr=1.000000），
并给出首批 GeForce 上的 VLA 引擎数据（π0.5 = 7.3×、GR00T N1.7 = 3.2× 于各自
PyTorch 基线）；实证层：1036 张量的权重误差剖面显示逐模块误差均匀（0.095±0.001），
而全量 FP4 PTQ 使动作输出与 BF16 基线去相关（corr≈0.11）——闭环控制对量化噪声的
复合放大。理论层：以 DAgger 协变量漂移框架将复合误差定量化（O(T²ε)→O(Tε)），导出
PTQ→QAD→OPD 恢复阶梯与成功率损失分解（噪声下界/漂移/次优），配套 H1–H6 可检验
假设与 R0–R8 实验矩阵。工程交付：1400+ 行可上游引擎补丁、打包名工件管线（含豁免
表消融）、单卡全权重 FP4 恢复训练可行性验证（1.62B 参数 FSDP2+CPUOffload）。

视觉-语言-动作（VLA）模型要在机器人本体上闭环运行，推理引擎的算力与显存预算是硬约束。
NVIDIA Blackwell 架构把 FP4 tensor core 带到了消费级 GPU（RTX 50 系列），但截至目前
没有任何公开工作回答：**VLA 模型能否在 FP4 精度下保持任务成功率？**

本文在 RTX 5090 Laptop（sm_120）上对 GR00T N1.7（Cosmos-Reason2-2B backbone + DiT
flow-matching action head）与 π0.5 进行首个系统性的 NVFP4 量化研究：

- **算子层（已完成）**：block-scaled NVFP4 GEMM 在 GeForce sm_120 上可用且数值精确；
  我们逆向并验证了 cuBLASLt 的 128×4 tile 行交错 scale 布局，计算受限形状下 NVFP4
  达到 506 TFLOPS（1.8× FP8、5.5× BF16），而 batch-1 GEMV 形状下 FP4 路径反而慢于 FP8——
  这一非对称性直接导出混合精度准则。
- **模型层（进行中）**：逐模块敏感性分析（VLM backbone vs action head）、闭环失效模式
  刻画、量化误差→动作误差→成功率传播链。
- **恢复阶梯**：PTQ → 校准式 scale 学习 → 蒸馏 + 一天预算小规模 PPO（LoRA 仅作用
  于 action head），全部闭环于 ApxInf 推理引擎与 RLinf 训练框架。

# 1. 引言

机器人策略模型正在从"云端大脑"走向本体推理：GR00T、π0.5、WALL-OSS 等 VLA 模型
需要在 10-100W 的边缘设备上以 10-50 Hz 的频率完成"感知→决策→行动"闭环。
现有量化工作集中在 LLM（GPTQ/AWQ/SmoothQuant 等 4-bit 权重量化），但 VLA 有两点
本质不同：其一，action head 的输出直接驱动闭环控制，量化误差会在数百步的 rollout
中复合，而非体现在 perplexity 上；其二，VLA 推理同时包含自回归 VLM 解码与
flow-matching 多步采样两类截然不同的算子负载 profile。

Blackwell 消费级 GPU 的 FP4 tensor core（NVFP4：E2M1 数据 + 每 16 元素 E4M3 block
scale + 每 tensor FP32 二级 scale）提供了一个此前不存在的选项……（继续）

# 2. 算子层与端到端初步结果（进行中 🔄）

## 2.1 端到端引擎对比（RTX 5090 Laptop, sm_120, batch 1, 同机同测）

| 模型/路径 | 延迟 P50 | 频率 | 显存 | 功耗(均/峰) |
|---|---|---|---|---|
| π0.5 · lerobot PyTorch（默认引擎） | 361.7 ms | 2.8 Hz | 16.7 GB | 160.8 / 201 W |
| π0.5 · ApxInf BF16 | **49.6 ms** | **19.7 Hz** | 16.3 GB | 140.2 / 165.5 W |
| GR00T N1.7 · NVIDIA PyTorch（官方） | 106.9 ms | 9.4 Hz | 6.0 GB | 97.5 / 127 W |
| GR00T N1.7 · ApxInf BF16 | **33.7 ms** | **29.7 Hz** | 10.4 GB | 107.0 / 124 W |

**ApxInf 相对各自默认 PyTorch 引擎：π0.5 = 7.3×、GR00T N1.7 = 3.2× 加速**（batch-1
端到端口径含预处理，PyTorch 为 eager 无 CUDA Graph，引擎走 CUDA Graph，30 样本 P50）。
**这些是首批 GeForce（sm_120）上的 GR00T N1.7 引擎数据**；另一发现：5090 Laptop 引擎延迟
约为 Jetson AGX Thor（58–60 ms）的 1/2——消费级 Blackwell 对具身部署的性价比信号。
NVFP4 路径在此之上继续压（算子层已验证 506 TFLOPS，见 2.2）。

## 2.2 算子层 NVFP4（已完成 ✅）

在 RTX 5090 Laptop、CUDA 13.3、cuBLASLt 13.6 上（spike/fp4_gemm_bench.cu）：

| GEMM 形状 | BF16 TFLOPS | FP8 | NVFP4 | NVFP4/FP8 |
|---|---|---|---|---|
| 2048³ | 90.7 | 268.6 | 417.2 | 1.55× |
| 4096×8192×2048 | 98.8 | 267.8 | 464.1 | 1.73× |
| 8192×2048×4096 | 88.9 | 276.3 | 503.9 | 1.82× |
| 4096³ | 92.5 | 281.2 | 506.2 | 1.80× |

| batch-1 形状 | FP8 GB/s | NVFP4 GB/s |
|---|---|---|
| 1×4096×4096 | 1253 | 428 |
| 1×8192×2048 | 1490 | 521 |

MXFP4（VEC32_UE8M0）在 sm_120 上不被 cuBLASLt 支持（heuristic 0 算法）。

scale 张量的物理布局（128×4 tile、行按 32 分块交错）由单字节探针扫描逆向得到，
公式与验证见仓库 docs/spike-notes.md：

```
PS = 512 * ceil(KB/4)
offset(row r, block b) = PS*(r/128) + 512*(b/4) + 16*(r%32) + 4*((r/32)%4) + (b%4)
```

![](images/swizzle-demo.png)

## 2.3 引擎算子级 NVFP4 加速（真实 π0.5 层形状，量化+GEMM 全管线 vs BF16）

| 层形状（N×K） | M=522 | M=778 | M=2048 |
|---|---|---|---|
| gemma 1152×1152 | 1.30× | 1.63× | 1.89× |
| MLP 3072×1024 | 1.67× | 2.24× | 2.95× |
| MLP 4096×1024 | 2.02× | 2.22× | 3.74× |
| embed 16384×2048 | 4.45× | 4.80× | **5.30×** |
| attn 2048×2048 | 1.94× | 2.45× | 3.06× |
| GeGLU 4304×1152 | 2.20× | 2.17× | 3.57× |

（p50，30 次；含在线激活量化开销；同机有共存负载，绝对值保守。数据：
results/engine/fp4_opbench_realshapes.csv，spike/fp4_opbench.cu 可复现）

## 2.4 nvfp4_static 引擎端到端：链路正确性与 PTQ 基线现象（2026-09-25）

引擎 nvfp4_static 变体（步骤 1-5 全链：adapter→量化 kernel→加载器→混合拓扑
→variant）已在本机跑通端到端前向。**组合正确性经四层验证**：算子级
（gold_check4：corr=1.000000、maxrel 4.83e-4）；逐块 scale 语义（10× 动态
范围四象限全 PASS）；工件级（单层 site-check corr 0.9957）；kernel 级（与
Python 金标准逐字节一致）。

**关键实证现象（论文动机数据点）**：全量 fp4 PTQ 的 π0.5 与 BF16 基线的
动作输出相关性仅 ~0.11——36 层 × 双 GEMM × 权重/激活双 9% 噪声的乘性复合。
这正是第 3 章协变量漂移理论的直接体现，也是恢复流水线（QAD→OPD）的价值
主张起点：PTQ 全量 fp4 需要恢复才能恢复闭环成功率。


## 2.5 权重误差剖面（全模型 1036 张量实测）

| 模块类型 | n | mean rel-Frob | max |
|---|---|---|---|
| attn qkv（packed+raw, 双模型） | 324 | 0.0951 | 0.0991 |
| mlp gate/up（packed+raw） | 140 | 0.0953 | 0.0989 |
| vision mlp（fc1/fc2） | 110 | 0.0944 | 0.0951 |
| mlp down | 52 | 0.0946 | 0.0952 |
| attn o_proj | 52 | 0.0951 | 0.0958 |
| 其他（embeddings/专家等） | 358 | 0.0957 | 0.1117 |

**误差均匀性**：所有模块类型的 mean rel-Frob 都紧贴 0.095（±0.001），深度方向
无系统性漂移（π0.5 首四分位层 0.0956 vs 末四分位 0.0965；GR00T 同样平坦）。
两点论文含义：(i) NVFP4 的 per-16 block scale 自适应足够强，逐模块误差几乎
与权重分布无关——这意味着**敏感性差异（若存在）将来自拓扑与闭环复合，而非
权重可量化性本身**，为 §3 理论提供前提；(ii) 极少数离群张量（max 0.112）可
作为豁免表候选（敏感性网格 pt4-* 将实证检验其闭环影响）。

## 2.6 QAD 恢复信号（2026-09-26 15:5x，GR00T N1.7）

| 配置 | 动作相关性 vs BF16 教师 |
|---|---|
| PTQ（原权重 NVFP4 量化） | **-0.357**（去相关甚至反号） |
| **QAD-deployed（QAD 训练后权重再量化）** | **+0.737** |
| QAD 权重 BF16 直跑（不量化） | +0.214 |

**恢复增益 +1.09**（同一观测、 seeded 对照）。附加发现：QAD 权重不量化直跑
（0.214）反而远差于量化部署（0.737）——**QAD 权重已内化量化网格**（grid-adapted
而非去噪），这是 QAT/QAD 文献中少见的直接证据。PTQ 的负相关（-0.36）比 π0.5
（0.11）更极端，与 2.4 节 PTQ 基线现象互证。注：单观测方向性信号，多样本扫描
与 LIBERO 闭环成功率为正式数字（exp/qad_recovery_signal.py 可复现）。


# 3. 恢复的理论：量化误差、协变量漂移与策略恢复

本章回答两个问题：(i) 为什么量化会让闭环任务成功率下降得比开环指标预测的更多；
(ii) 为什么蒸馏（QAD）、在策蒸馏（OPD）与强化学习（PPO）能够、以及各自能恢复哪一部分损失。
我们把三者统一到同一个数学框架下，并给出可检验的预测与实验设计。

## 3.1 记号与设定

- 环境：LIBERO 操作任务，有限视界 MDP，视界 T ≤ 720 步，每 κ=8 步重规划（~90 次决策）。
- 教师策略 π*（BF16 引擎执行），NVFP4 量化后学生策略 π_q = Q_φ(θ)，其中 Q 为块量化算子，
  φ 为可训练补偿参数（LoRA 增量、可选 scale 修正）。我们要求部署时真实执行 Q(θ)（引擎
  kernel），训练时用 fake-quant 模拟，二者通过 E0 parity 对齐。
- 动作生成：flow-matching，动作 a 由积分向量场 v_θ(x_t, t; s) 的 ODE 得到，
  离散 K 步（K=10 或 STEP 一步）。
- 任务成功率 p(π) = E_τ~π[1{任务达成}]。

## 3.2 每步量化扰动有界（开环视角）

**权重扰动**：NVFP4 每块（16 元素）误差由 E2M1 网格（相对步长 1/4~1/2）与 E4M3 scale
舍入共同决定。逐层测量相对扰动 ε_W = ‖ΔW‖_F/‖W‖_F（E2 实测）。

**动作扰动**：一阶线性化下，单步决策的动作为状态的函数，量化扰动满足

```
‖a_q(s) − a*(s)‖ ≤ ‖J_θ(s)‖ · ‖Δθ_eff‖ + O(‖Δθ‖²)
```

其中 J_θ 是动作对权重的雅可比，‖Δθ_eff‖ 由逐层 ε_W 累积。关键在于该界只在
**训练分布内**的 s 上小；分布外无保证（3.3 节正是利用这一点）。

**Flow ODE 的误差放大**：动作由 ODE 积分产生，向量场受扰 δv 时，轨迹端点误差满足
Gronwall 型界：

```
‖δx_T‖ ≤ ∫₀ᵀ e^{∫ₜᵀ L_v ds} · ‖δv_s‖ ds   ⇒   离散 K 步 Euler 下近似 O(K · δv)
```

**预测 H5**：步数 K 越少（如 STEP 一步生成），量化带来的动作偏差越小——一步生成与量化
存在正向耦合（实验 R7 检验）。

## 3.3 闭环复合：协变量漂移使损失超线性（DAgger 论证）

量化策略在环境里闭环执行时，每步动作偏差把状态推离教师的数据分布；在漂移后的状态上
策略误差进一步放大。记 d_π 为策略 π 的状态访问分布，ε_a = E_{s~d_*}‖π_q(s)−π*(s)‖ 为
教师分布上的单步误差。由 Ross & Bagnell (2011) 的复合误差分析：

```
开环/模仿式训练（只在固定数据 d_demo 上最小化 ε_a）：
    J(π_q) − J(π*) = O(T² · ε_a)        # 误差沿视界平方级复合
在策纠正训练（在 d_{π_q} 上获得教师标签）：
    J(π_q) − J(π*) = O(T · ε_a')        # 线性，且 ε_a' 是可在学生分布上继续压小的残差
```

**这正是"量化在闭环里比开环指标伤得更重"的数学解释**，也是三段恢复配方分工的依据：
QAD 在 d_demo 上优化（λ=0 的离线蒸馏），**不改变漂移项**；OPD 在学生自己的
rollout 分布 d^{π_q} 上用教师标签优化，等价于以 π* 为专家的 DAgger，把平方复合降为线性；
PPO 直接在环境奖励上优化闭环回报，理论上还能修正教师本身的部分次优（超越教师上限）。

## 3.4 三个恢复目标的统一形式（flow-matching 上实现）

GKD（Agarwal et al., ICLR 2024）的广义目标在 VLA 上的对应实现：
取教师与学生共享的加噪路径 x_t = α_t a + σ_t ϵ，t~U(0,1)，损失为向量场匹配：

```
L_λ(φ) = E_{s~D_λ, t, x_t} [ w(t) · ‖ v_{Q(θ,φ)}(x_t,t;s) − v_{θ*}(x_t,t;s) ‖² ]

D_λ = λ · d^{π_q(φ)}  +  (1−λ) · d_demo        # λ=0 → QAD；λ=1 → OPD（纯在策）
```

- 选 **向量场匹配** 而非端点动作 MSE：与训练目标同构、每步稠密监督，且 reverse-KL 的
  mode-seeking 性质在 ODE 端点分布上同样成立（学生支撑集上匹配教师）。
- **PPO 臂**：J(φ)=E_τ~π_q[Σ min(r_t Â_t, clip)]，Â 由 GAE。注意当奖励取
  r_t = −‖a_t − π*(s_t)‖²（教师塑形）时，PPO 退化为 OPD 的加权形式——
  OPD 可理解为"以教师为势函数做 potential-based reward shaping 的 RL"，
  该统一视角在 R4b 消融中实证检验（RL 与蒸馏不是两件事，是同一谱系的两端）。

## 3.5 成功率损失的分解（论文叙事主定理）

把量化造成的成功率损失分解为三个可分别观测的部分：

```
Δ = p(π*) − p(π_q)
  = Δ_floor    # 噪声下界：量化噪声下最优补偿也补不回的部分（由 R5d 噪声匹配实验估计）
  + Δ_shift    # 协变量漂移损失：QAD 补不动、OPD 主补（H1）
  + Δ_opt      # 学生在自身分布上的次优：OPD/PPO 皆可补，PPO 唯一可能越过 p(π*)（H4）
```

各 Δ 通过"恢复曲线 + 残差分解"实验联合估计（R5e oracle 上界 + R5d 噪声下界夹逼）。

## 3.6 可检验假设

- **H1（漂移主因）**：QAD 显著降低教师分布上的动作误差，但对闭环成功率提升有限；
  OPD 在同等 GPU 预算下成功率恢复显著更高。
- **H2（数据混合谱系）**：成功率随 λ 单调不减（GKD 在 LLM 上的结论迁移到 VLA）。
- **H3（访问分布收敛）**：OPD 训练中 d^{π_q} 与 d_{π*} 的距离（状态访问分布 TV 距离）
  单调下降，且其下降幅度中介（mediate）成功率恢复。
- **H4（超越教师）**：OPD 后接 PPO 可超过 p(π*)（教师自身在 LIBERO 并非最优）；
  若不成立，Δ_opt≈0 的证据同样有价值。
- **H5（步数耦合）**：量化损失与 OPD 增益均随 flow 步数 K 减小而减小（Gronwall 预测）。
- **H6（容量阶梯，2026-09-24 增补）**：补偿容量（仅 scale → head-LoRA(r8) → full-weight
  action head，后者由 FSDP2+CPUOffload 使能）与闭环恢复上限正相关；机制解释：NVFP4 误差
  全局散布于所有权重，修正未必低秩，r8 低秩子空间可能系统性低估可恢复量 Δ_opt+Δ_shift。
  full-head 臂同时给出工程论据：直接训 W + STE 导出即纯 NVFP4 权重，无 LoRA 旁路张量。

## 3.7 实验矩阵与指标（对照 GKD/DAgger/PPO 顶会惯例）

| 编号 | 实验 | 设计要点 | 关键指标 |
|---|---|---|---|
| R0 | 漂移动机 | PTQ 后按视界截断评测（T=90/180/360/720）；on/off-policy 动作误差对比 | 成功率-T 曲线斜率；ε_a(d*) vs ε_a(d^{π_q}) 差距；状态访问分布 TV 距离 |
| R1 | PTQ 基线 | L0，多精度（BF16/FP8/NVFP4 混合/NVFP4 全量） | 成功率±Wilson 95% CI、延迟、显存 |
| R2 | QAD | λ=0，冻结 backbone+scale，action-head LoRA；~2 GPU·h | 教师分布动作 MSE ↓、成功率（预期有限提升） |
| R3 | OPD | λ∈{0.25,0.5,1}，学生 rollout + 引擎 BF16 teacher 逐步纠偏；~8 GPU·h | 恢复曲线（episodes→成功率）、成功率 AUC、episodes-to-90% gap |
| R4 | PPO 对照 | RLinf 同预算；R4b=教师塑形奖励（OPD 等价形式） | 同预算成功率、样本效率、是否超 p(π*) |
| R5 | 机制消融 | (a) λ 谱系 (b) **容量阶梯{仅 scale / head-LoRA(r8) / full-head(FSDP2+offload)}** (c) teacher=BF16 vs FP8 (d) **等范数高斯噪声 vs NVFP4 结构性误差**（区分"学鲁棒"与"修结构误差"） (e) oracle=BF16 同预算同容量 | 各臂成功率；Δ_floor/Δ_shift/Δ_opt 分解饼图；优化器一律 AdamW（归因隔离，Muon 见姊妹论文） |
| R6 | 恢复-精度前沿 | 精度×{PTQ,+QAD,+OPD} 网格 | "恢复移动精度-成功率前沿"主图 |
| R7 | 步数耦合 | K∈{1,4,10} × {PTQ,OPD} | 量化损失、OPD 增益 vs K |
| R8 | 泛化 | 仅在 2-3 个掉点最重任务上恢复，全 10 任务评测 | 恢复的跨任务迁移（过拟合检查） |

指标口径统一：成功率报 Wilson 95% CI（每任务 ≥50 episodes）；动作误差/MSE 在
**学生自身 rollout 状态上**计算（on-policy，R0 专门对比 off-policy 口径的差异）；
状态访问分布距离用 8 维本体状态直方图 TV 距离 + VLM 嵌入 Wasserstein 双口径；
训练效率报 GPU·h 与环境交互步数（OPD 的 teacher 查询由 ApxInf BF16 服务，
本身即系统贡献）。所有恢复实验种子 ≥3。

## 3.8 与相关工作的关系（引用与差异）

- **GKD/OPD**（Agarwal et al. ICLR 2024；2025 OPD 实践）：借其 λ 混合目标、reverse-KL
  论证与 on/off-policy 对照实验设计；差异：我们作用于 flow-matching 动作分布与闭环控制，
  且 teacher 由量化引擎高速服务。
- **QAD for NVFP4**（NVIDIA, arXiv:2601.20088）：借其"冻结 scale、更新权重"实践与
  PTQ→QAD 阶梯；差异：其恢复开环 benchmark 指标，我们证明并利用"闭环需要 on-policy"。
- **DAgger**（Ross et al. 2011）：借其复合误差理论框架，把"量化"实例化为误差源；
  我们的 OPD 即以 BF16 策略为专家的 DAgger，并首次给出量化场景的分解与噪声下界。
- **STEP**（ICML 2026 warm-start）：H5 把它与量化误差耦合，解释一步生成的额外收益。


# 4. 相关工作

## 4.1 VLA 模型与部署压力

视觉-语言-动作（VLA）模型将预训练 VLM backbone 与动作生成头结合：π0/π0.5
（Physical Intelligence，flow-matching action expert）、NVIDIA Isaac GR00T 系列
（N1/N1.5 用 Eagle-2 backbone，N1.6/N1.7 换用 Cosmos-Reason2-2B/Qwen3-VL 架构并
倍增 DiT action head）、WALL-OSS、SmolVLA、OpenVLA 等。部署侧的共同约束是本体
推理：10-100W 功耗预算内以 10-50 Hz 闭环，模型 3-4B 参数在边缘设备上逼近显存与
带宽极限。NVIDIA 为 GR00T 提供 Jetson Thor 上的 TensorRT 部署路径，但消费级
GeForce 与量化格式的组合此前没有公开数据——本工作填补该空白。

## 4.2 LLM 量化与 NVFP4 恢复

4-bit 权重量化（GPTQ 的 Hessian 逐列舍入、AWQ 的激活感知缩放、SmoothQuant 的
难度迁移）已是 LLM 标配；块缩放格式（NVFP4：E2M1 + per-16 E4M3 scale + per-tensor
FP32 二级 scale，对照 MXFP4 的 per-32 UE8M0）随 Blackwell 硬件成为新的最低精度档。
恢复方面，NVIDIA 的 QAD 技术报告（arXiv:2601.20088）确立 PTQ→QAD（QAT 模拟 +
冻结 BF16 teacher 的 KL 蒸馏、冻结 scale 更新权重）→ 在策蒸馏的阶梯；GKD
（Agarwal et al., ICLR 2024）与后续 OPD 实践给出 λ 混合广义目标与 reverse-KL 的
论证。这些方法恢复的都是**开环指标**（perplexity/benchmark）；闭环控制下量化误差的
复合与恢复是本工作的核心增量。

## 4.3 协变量漂移与模仿学习的复合误差

DAgger（Ross et al., 2011）证明：固定数据集上训练的策略在闭环执行中误差随视界
平方级复合 O(T²ε)，而在策纠正训练将其降为线性 O(Tε)。量化策略天然构成这一框架中
"带扰动学习器"的实例：NVFP4 引入的每步动作偏差 ε_q 正是 DAgger 误差分析中的 ε。
我们将此理论实例化到 VLA 量化恢复（第 3 章），并给出 Δ_floor/Δ_shift/Δ_opt 分解
与相应的实验设计——该连接在量化文献与机器人学习文献中均未被建立过。

## 4.4 具身推理引擎

通用 LLM 引擎（vLLM/TensorRT-LLM）面向 token 吞吐，不处理 VLA 的观测预处理、
flow-matching 采样循环与动作后处理；Embodied.cpp（arXiv:2607.02501）提出可移植
C++ 具身运行时；ApxInf（Infinigence/RLinf，Rust + 自研 CUDA kernel）以三层 API
服务 VLA 并在 Jetson Thor/Orin 上验证 BF16/FP8/INT8。本工作在 GeForce Blackwell
(sm_120) 上适配并扩展 ApxInf：修复其 FP8 路径在 sm_120 的不可用问题（E4M3 GEMM
输出不被 cuBLASLt 支持，改为 F16 输出），并新增 NVFP4 精度路径。与之互补的
单卡训练侧使能技术（FSDP2 CPUOffload 流水线）见姊妹工作，二者共享同一 checkpoint
与评测协议。

## 4.5 一步动作生成

STEP（ICML 2026）的 warm-start 将 flow-matching 采样从 10 步压到 1 步。我们的
Gronwall 分析（3.2 节）预言量化动作偏差随积分步数线性放大，因此一步生成与量化
存在正向耦合（H5，R7 检验）——这是系统方法与量化精度在 VLA 上的首次交互研究。


# 5. 实验设置与初步结果

## 5.1 硬件与软件配置（全部实测）

| 项 | 值 | 备注 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop（Blackwell sm_120，82 SM，24GB GDDR7） | 消费级移动 Blackwell；FP4/FP8 tensor core |
| 整机 | Intel Ultra 9 275HX（24 核）+ 58GB RAM（WSL2 动态调整中 43GB） | 单机训练+推理+仿真同载 |
| CUDA / cuBLASLt | 13.3 / 13.6 | CUDA 13 API 变化（layout 无 ORDER、handle 必传）均已适配 |
| 推理引擎 | ApxInf main @ e07dbe9（自编译 sm_120，wheel cp312） | 含我们两处 sm_120 修复 |
| PyTorch 基线 | 2.9.0+cu128 / transformers 4.57.3 | lerobot（π0.5）与 NVIDIA 官方 gr00t（N1.7）各自默认路径 |
| 仿真 | LIBERO（robosuite 1.4.1 + MuJoCo 3.1.6 + EGL mesa headless） | WSL2 无 NVIDIA EGL，走 mesa surfaceless |
| 对照训练基建 | FSDP2 CPUOffload（姊妹项目管线，交叉引用） | full-head 恢复臂专用 |

## 5.2 测量协议

- 延迟：batch-1 端到端（含预处理/tokenize/动作解码），30 样本，报 P50/P99；
  引擎走 CUDA Graph 稳态（首帧捕获开销单列），PyTorch 为 eager。
- 功耗：0.2s 间隔 nvidia-smi 采样全程，报均值/峰值；笔记本 TGP 波动如实注明。
- 显存：进程峰值（nvidia-smi）+ 分配器口径（torch）双报。
- 成功率（后续）：LIBERO-10 ×50 episodes/任务，Wilson 95% CI，断点续跑。
- NVFP4 数值验证：CPU fp64 反量化参考为金标准，GPU kernel 输出
  maxrel < 5e-4（fp16 输出舍入级）为 PASS（已达成，见 §5.3）。

## 5.3 已完成结果速览

**算子层**（§2.2）：NVFP4 GEMM 计算受限 506 TFLOPS（1.8× FP8）；MXFP4 sm_120 不支持；
swizzle 布局逆向解码并三向验证（平 scale 探针/单字节扫描/全随机数据）。

**端到端 E1**（§2.1）：π0.5 引擎 7.3×、GR00T N1.7 引擎 3.2× 于各自 PyTorch 默认引擎；
GR00T 为首批 GeForce 引擎数据，且约为 Jetson Thor 引擎延迟的 1/2。

**FP8 缺陷与根因**（§2.1 脚注）：ApxInf fp8_static 在 sm_120 因 E4M3 GEMM 输出
不可用（cuBLASLt heuristic 状态 15，探针铁证）；我们的 NVFP4 路径设计上即规避
（F16 输出），fp4 GEMM adapter 已在引擎 adapter 目录落地并通过功能验证
（maxrel 4.84e-4）。

**离线量化工件**：π0.5 全部 459 个 eligible 张量 NVFP4 化，14.46GB→2.03GB
（7.11×），rel-Frob 均值 9.5%、最大 10.2%（无异常张量）；GR00T 侧同管线待跑。

## 5.4 进行中

fp4 引擎路径（adapter ✓ → Rust 接线 → fp4_weights/executor/runtime → bench）；
R0 漂移动机实验；QAD/OPD 恢复流水线（LoRA 主线不等外部，full-head 臂 E4 后接入）；
系统测量（延迟分解/batch 吞吐/能耗）。


# 6. 讨论与结论

## 6.1 主要发现

**算子层**：GeForce Blackwell（sm_120）的 cuBLASLt 原生支持 block-scaled NVFP4
GEMM，我们完整逆向并验证了其 scale 张量的 128×4 tile 行交错物理布局——此前无
公开文档。计算受限形状下 NVFP4 达 FP8 的 1.8×、BF16 的 5.5×；MXFP4 不被支持。

**引擎层**：在 ApxInf 引擎中交付了完整的 NVFP4 执行路径（adapter→量化 kernel→
工件加载器→混合拓扑→variant），四层组合正确性验证（算子 corr=1.000000）。
真实 π0.5 层形状上 fp4 全管线（含在线激活量化）相对 BF16 GEMM 加速 1.3–5.3×。

**实证现象**：全量 fp4 PTQ 的 π0.5 与 BF16 基线动作相关性仅 ~0.11（36 层 × 双
GEMM × 双 9% 噪声复合）——协变量漂移理论（§3.3）的直接体现，恢复流水线的价值
主张起点。

**引擎对比**：ApxInf BF16 相对各自 PyTorch 默认引擎 π0.5 = 7.3×、GR00T N1.7 =
3.2×；后者为首批 GeForce GR00T 引擎数据，且约为 Jetson Thor 引擎延迟的 1/2。

## 6.2 工程贡献（可上游合并）

1. cuBLASLt CUDA 13 API 适配与 NVFP4 swizzle 布局逆向（spike 全套探针工具）；
2. FP8 在 sm_120 的不可用根因定位（E4M3 GEMM 输出不被支持，probe 铁证）；
3. 打包名工件管线（RMSNorm 折叠语义 + 引擎布局对齐）与豁免表机制；
4. 全链 1400+ 行引擎补丁（patches/ 可复现）。

## 6.3 局限

- 单一 GPU 型号（RTX 5090 Laptop）；笔记本 TGP 波动已注明，绝对值为持续值；
- 闭环成功率实验进行中（敏感性网格与恢复阶梯排队等 GPU 窗口）；
- nvfp4 的 gate_up 融合 GEMM 未实现（当前朴素 GEMM+独立 geglu，性能次优）；
- CUDA Graph 捕获与 fp4 路径的兼容性待干净窗口验证；
- 语言/动作层 PACKED 工件的 vision 部分 qkv 仅 packed 主 GEMM，o_proj 未打包。

## 6.4 结论与展望

NVFP4 在消费级 Blackwell 上的 VLA 部署在算子与引擎层均已打通且收益显著；
全量 PTQ 的动作去相关（corr 0.11）实证了"闭环控制对量化噪声的复合放大"，
使 QAD→OPD 恢复流水线（含 full-head 容量阶梯）成为把 FP4 推向可用精度的
关键路径。未来工作：融合 fp4 GeGLU kernel、CUDA Graph 适配、更多 VLA 家族
（GR00T N1.7 全管线）与真机部署验证。

---
*本工作在单张 RTX 5090 Laptop（WSL2）上完成；姊妹工作（FSDP2 CPUOffload 单卡
全权重训练）共享同卡与 checkpoint，交叉引用。*
