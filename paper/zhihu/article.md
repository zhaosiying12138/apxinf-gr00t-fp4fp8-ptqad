# 摘要

视觉-语言-动作（VLA）模型需要在本体上以 10–50 Hz 闭环运行，但现有部署研究集中在
BF16/FP8。NVIDIA Blackwell 将 NVFP4（E2M1 + 逐 16 元素 E4M3 块缩放）带入消费级
GPU，本文在 RTX 5090 Laptop（sm_120）上系统回答：**VLA 能否在 FP4/FP8 混合精度下
运行，闭环成功率损失几何，如何恢复？**

**算子层**：我们逆向并验证了 cuBLASLt 无文档的 128×4 tile 行交错块缩放布局，NVFP4
GEMM 峰值 506 TFLOPS（1.8× FP8）。**引擎层**：在 ApxInf 中交付完整 NVFP4 执行路径
（四层正确性验证，corr=1.000000），并给出首批 GeForce 上的 VLA 引擎数据（π0.5 =
7.3×、GR00T N1.7 = 3.2× 于各自 PyTorch 基线）。**实证层（核心结果，LIBERO-10 ×
10×10 全量闭环）**：全模型 NVFP4 PTQ 无论 max 校准、GPTQ Hessian 舍入还是逐层 MSE
裁剪均闭环归零（0%）——逐层最优的量化误差在闭环中复合成行为崩溃；而复刻 NVIDIA
Jetson AI Lab 的混合 FP8/NVFP4 分配（backbone FP8 + 敏感投影保护）在 2.02× 线性参数
压缩下达到 **95.1%**（BF16 基线 96.7%）。**恢复层**：在 2.44× 压缩、全量闭环仅 43.4%
的 FP8/NVFP4 基座上，500 步（3.5 分钟）的量化域 LoRA 后训练（量化基座 + BF16 低秩
残差的加性形态，SVDQuant 家族）将闭环成功率恢复到 **99.0%**（QAD 演示蒸馏与 OPD
teacher-KL 双臂同达，均超 BF16 基线）；同预算的 PPO 家族在线对照（成功加权自模仿
RWR）则**零恢复**——构成量化恢复问题上离线蒸馏与在线自模仿的经济学对照。
**理论层**：以 DAgger 协变量漂移框架将复合误差定量化（O(T²ε)→O(Tε)），导出并实测
了 PTQ→QAD→OPD 恢复阶梯。**工程交付**：1400+ 行可上游引擎补丁、打包工件管线、
校准 PTQ 套件（Hessian 采集/GPTQ/AWQ/五臂烤盘）、24GB 单卡 LoRA 恢复训练栈
（无 FSDP2、0.43s/步）、全部 checkpoint 与复现脚本。

# 1. 引言

机器人策略模型正在从"云端大脑"走向本体推理：GR00T、π0.5、WALL-OSS 等 VLA 模型
需要在 10–100W 的边缘设备上以 10–50 Hz 的频率完成"感知→决策→行动"闭环。现有量化
研究集中于 LLM（GPTQ/AWQ/SmoothQuant 等 4-bit 权重量化），但 VLA 有两点本质不同：
其一，动作头的输出直接驱动闭环控制，量化误差会在数百步的 rollout 中复合，而非体现
在 perplexity 上；其二，VLA 推理同时包含自回归 VLM 解码与 flow-matching 多步采样两类截然不同的算子负载。Blackwell 消费级 GPU 的 FP4 tensor core（NVFP4：E2M1 数据
+ 每 16 元素 E4M3 块缩放 + 每 tensor FP32 二级缩放）提供了一个此前不存在的部署选项，
但没有任何公开工作系统回答过上述三个问题。

本文贡献可归纳为五点：

- **C1（算子与引擎）**：逆向并验证 cuBLASLt NVFP4 块缩放的物理布局；在 ApxInf 引擎
  中交付完整 NVFP4 执行路径并给出首批 GeForce（sm_120）VLA 引擎数据（§2）。
- **C2（闭环失效定律）**：在两代 VLA（π0.5/GR00T N1.7）、两条实现路径（引擎真
  kernel / PyTorch 量化语义）上证明全模型 NVFP4 PTQ 闭环一致归零，且 GPTQ/MSE
  校准无法挽回——量化误差的模块分布比总量更决定闭环行为（§3、§12）。
- **C3（模块敏感度地图与混合精度配方）**：五臂 PTQ 消融给出模块级 NVFP4 敏感度排序（致死：视觉塔、DiT 注意力投影、LLM o/down_proj；幸存：LLM 主投影与全部
  FFN），复刻 NVIDIA 分配的混合 FP8/NVFP4 方案在 2.02× 压缩下闭环 95.1%（§11）。
- **C4（量化域 LoRA 恢复）**：提出量化基座 + BF16 低秩残差的加性 LoRA 形态（训练与
  部署严格同一函数，无 STE 近似），从 43.4% 的 2.44× 压缩基座以 3.5 分钟训练恢复到
  99.0%；同预算在线自模仿（RWR）零恢复，给出离线/在线信号的经济性对照（§12）。
- **C5（方法论）**：证明离线指标（逐层 MSE、单点向量场相关性）与闭环损伤混沌解耦
  ——闭环成功率是 VLA 量化的唯一裁判；配套全链可复现工件（§6、§11.8）。

本文结构：§2 报告引擎与算子层结果；§3 给出闭环失效现象学与恢复阶梯总览；§4 建立
协变量漂移理论框架；§5 相关工作；§6 实验设置；§7 讨论与结论；§8–§9 为数值格式与
恢复方法教程（含 §8.7 缩放布局逆向与 §9.5 DAgger 推导补充）；§10 闭环评测方法
论；§11 训练后量化方法综述与校准实践；§12 全栈实现与 APXInf 生态的角色。

# 2. 推理引擎与算子层

## 2.1 端到端推理性能

在 RTX 5090 Laptop（sm_120，batch-1，含预处理与动作解码，30 样本 P50）上：

| 模型 / 路径 | 延迟 P50 | 频率 | 显存 | 功耗（均/峰） |
|---|---|---|---|---|
| π0.5 · lerobot PyTorch（默认引擎） | 361.7 ms | 2.8 Hz | 16.7 GB | 160.8 / 201 W |
| π0.5 · ApxInf BF16 | **49.6 ms** | **19.7 Hz** | 16.3 GB | 140.2 / 165.5 W |
| GR00T N1.7 · NVIDIA PyTorch（官方） | 106.9 ms | 9.4 Hz | 6.0 GB | 97.5 / 127 W |
| GR00T N1.7 · ApxInf BF16 | **33.7 ms** | **29.7 Hz** | 10.4 GB | 107.0 / 124 W |

ApxInf 相对各自默认 PyTorch 引擎加速 π0.5 = 7.3×、GR00T N1.7 = 3.2×（PyTorch 为
eager 无 CUDA Graph，引擎走 CUDA Graph）。据我们所知这是首批 GeForce（sm_120）
上的 GR00T N1.7 引擎数据；5090 Laptop 引擎延迟约为 Jetson AGX Thor（58–60 ms）的
一半，是消费级 Blackwell 面向具身部署的性价比信号。

![](images/e1_latency.png)

## 2.2 NVFP4 算子特性与缩放布局逆向

在 CUDA 13.3 / cuBLASLt 13.6 上（spike/fp4_gemm_bench.cu）：

| GEMM 形状 | BF16 TFLOPS | FP8 | NVFP4 | NVFP4/FP8 |
|---|---|---|---|---|
| 2048³ | 90.7 | 268.6 | 417.2 | 1.55× |
| 4096×8192×2048 | 98.8 | 267.8 | 464.1 | 1.73× |
| 8192×2048×4096 | 88.9 | 276.3 | 503.9 | 1.82× |
| 4096³ | 92.5 | 281.2 | 506.2 | 1.80× |

| batch-1 GEMV 形状 | FP8 GB/s | NVFP4 GB/s |
|---|---|---|
| 1×4096×4096 | 1253 | 428 |
| 1×8192×2048 | 1490 | 521 |

三个结论：(i) 计算受限形状下 NVFP4 达 FP8 的 1.8×、BF16 的 5.5×；(ii) batch-1
访存受限形状下 FP4 反而慢于 FP8——这一非对称性直接导出混合精度准则（计算受限的
大投影进 FP4，访存受限的小形状保留 FP8/BF16）；(iii) MXFP4（VEC32_UE8M0）在
sm_120 上不被 cuBLASLt 支持（heuristic 返回 0 算法）。

scale 张量的物理布局（128×4 tile、行按 32 分块交错）由单字节探针扫描逆向得到，
公式与验证见仓库 docs/spike-notes.md：

```
PS = 512 * ceil(KB/4)
offset(row r, block b) = PS*(r/128) + 512*(b/4) + 16*(r%32) + 4*((r/32)%4) + (b%4)
```

![](images/swizzle-demo.png)

## 2.3 引擎算子级 NVFP4 加速与工件管线

真实 π0.5 层形状上（量化 + GEMM 全管线 vs BF16，p50，30 次，含在线激活量化开销）：

| 层形状（N×K） | M=522 | M=778 | M=2048 |
|---|---|---|---|
| gemma 1152×1152 | 1.30× | 1.63× | 1.89× |
| MLP 3072×1024 | 1.67× | 2.24× | 2.95× |
| MLP 4096×1024 | 2.02× | 2.22× | 3.74× |
| embed 16384×2048 | 4.45× | 4.80× | **5.30×** |
| attn 2048×2048 | 1.94× | 2.45× | 3.06× |
| GeGLU 4304×1152 | 2.20× | 2.17× | 3.57× |

（数据：results/engine/fp4_opbench_realshapes.csv，spike/fp4_opbench.cu 可复现。）

![](images/opbench_heatmap.png)

配套的离线工件管线将 HF 权重转换为引擎内存布局：qkv/gate_up 行拼接、RMSNorm 增益
折叠进权重（π0.5 全部 459 个 eligible 张量 NVFP4 化，14.46 GB → 2.03 GB，7.11×）。
引擎 nvfp4_static 变体的组合正确性经四层验证：算子级（gold_check4：corr=1.000000、
maxrel 4.83e-4）、逐块 scale 语义（10× 动态范围四象限全 PASS）、工件级（单层
site-check corr 0.9957）、kernel 级（与 Python 金标准逐字节一致）。

## 2.4 全模型权重误差剖面

π0.5 + GR00T 共 1036 个张量的 NVFP4 量化误差（rel-Frob）：

| 模块类型 | n | mean rel-Frob | max |
|---|---|---|---|
| attn qkv（packed+raw, 双模型） | 324 | 0.0951 | 0.0991 |
| mlp gate/up（packed+raw） | 140 | 0.0953 | 0.0989 |
| vision mlp（fc1/fc2） | 110 | 0.0944 | 0.0951 |
| mlp down | 52 | 0.0946 | 0.0952 |
| attn o_proj | 52 | 0.0951 | 0.0958 |
| 其他（embeddings/专家等） | 358 | 0.0957 | 0.1117 |

所有模块类型的误差均值紧贴 0.095（±0.001），深度方向无系统性漂移。这一均匀性有
两点含义：(i) NVFP4 的 per-16 块缩放自适应足够强，逐模块误差几乎与权重分布无关，
因此后文观测到的敏感性差异来自**拓扑与闭环复合**，而非权重可量化性本身——这为
§4 的理论提供了前提；(ii) 极少数离群张量（max 0.112）构成豁免表候选，其实际闭环
影响由 §12 的混合分配消融检验。

# 3. 闭环失效现象学与恢复阶梯

## 3.1 全量 NVFP4 的闭环失效：跨模型与跨路径一致

LIBERO-10（10 任务 × 10 episodes/任务）全量闭环协议下的首轮四路对照：

| 臂 | 训练配方 | 闭环成功率 |
|---|---|---|
| BF16 基线 | — | **96.7%**（与 NVIDIA Thor 参考 ~98% 一致） |
| 同配方 SFT，无量化（并行工作数据，交叉引用） | 官方微调配方 1000 步 | 87.4% |
| naive QAT | 同配方 + NVFP4 fake-quant（STE） | **0%**（10 任务全零） |
| PTQ（原权重量化部署） | — | **0%**（全零） |
| QAD-deployed（量化训练后权重再量化部署） | demo 损失 + fake-quant | 0.63%（1/160） |

值得强调的是 naive QAT 的训练损失正常下降（1.29→0.33）的同时闭环能力被彻底摧毁
——"量化网格上的演示拟合"与闭环行为是两个目标。π0.5 侧引擎真 kernel 路径给出
独立的交叉验证：BF16 90.0%（90/100，与 openpi 参考 ~92% 吻合）vs nvfp4_static
**0%**（100 episodes 全零）。两代 VLA（Gemma / Qwen3-VL backbone）、两条实现路径
（引擎 kernel / PyTorch 量化语义）上结论一致：**全模型 NVFP4 PTQ 的闭环失效是系统
性的，而非实现工件**。导出完整性经三重验证（键名 1030/1030 对齐、backbone 位级
冻结、head 系统性改变）排除工件错误。

## 3.2 混合精度二分：失效元凶的模块级定位

逐模块二分（quantize-once 服务器，同闭环协议）：

| 配置 | action head | backbone | 闭环成功率 |
|---|---|---|---|
| BF16 基线 | BF16 | BF16 | 96.7% |
| 混合精度（head-only FP4） | NVFP4 | BF16 | **71.2%** |
| 混合精度（backbone-only FP4） | BF16 | NVFP4 | 0% |
| 全量 FP4 PTQ | NVFP4 | NVFP4 | 0% |

backbone 量化（而非动作头）是闭环崩溃的元凶；动作头（1.62B 参数，占 47%）单独
NVFP4 化损失 25.5 个点但完全可用。这修正了最初的直觉假设（backbone 量化更安全），
方向恰好相反：backbone 输出经 KV cache 长程复用，误差复合更重（与 §4 理论一致）。
更细粒度的模块敏感度地图与五臂配方消融见 §11。

## 3.3 离线信号与闭环损伤的解耦

以离线动作信号（对 BF16 教师的逐观测/池化相关性、相对 L2）审视上述臂：

| 指标 | PTQ | QAD-deployed | QAD-bf16 |
|---|---|---|---|
| 逐观测 corr（mean±std） | +0.015±0.486 | +0.147±0.518 | +0.202±0.287 |
| 池化 corr（n=256） | -0.015 | **+0.139** | +0.167 |
| 相对 L2 | 1.246 | 7.078 | 3.452 |

两条量化路径的离线指标差异悬殊（池化 corr +0.139 vs -0.015）而闭环同为近零
（0.63% vs 0%）；后续对固定噪声、固定时间步的向量场单点对比进一步显示该指标对
权重微扰呈混沌敏感——即使闭环 95% 以上的臂也给出 relMSE ≈ 2（§11.8）。结论：
**离线指标不能预测 VLA 量化的闭环损伤，恢复效果必须以闭环为裁判**。这是 VLA
量化区别于 LLM 量化（perplexity 可作代理）的方法论分野。

## 3.4 闭环恢复阶梯总览

校准 PTQ 套件（§11）与量化域 LoRA 恢复栈（§12）共同构成完整闭环阶梯（LIBERO-10
× 10×10 全量协议）：

| 臂 | 配置 | 线性参数压缩 | 闭环成功率 |
|---|---|---|---|
| BF16 上界 | — | 1× | 96.7% |
| PTQ rtn / calib（GPTQ+MSE）/ aggr | 全 NVFP4 系 | 2.88–3.56× | 0%（纯校准不可挽回） |
| **PTQ mixed（NVIDIA 分配复刻）** | backbone FP8 + 敏感投影保护 | 2.02× | **95.1%**（97/102） |
| PTQ fp8 基座 | backbone FP8 + head NVFP4 | 2.44× | 43.4%（46/106） |
| **QAD-LoRA** | fp8 基座 + 500 步量化域 LoRA | 2.44× + 低秩 | **99.0%**（100/101） |
| **OPD-LoRA** | 同上 + teacher-KL | 2.44× + 低秩 | **99.0%**（102/103） |
| RWR-LoRA（PPO 族对照，mini 协议） | 同预算在线自模仿 | 2.44× + 低秩 | 与基座逐任务相同（零恢复） |

![](images/ladder.png)

三点概括：(i) 量化误差的**模块分布**比总量更重要——GPTQ 将逐层 MSE 压至 RTN 的
76% 仍闭环 0%，而混合分配（不动校准）直接 95.1%；(ii) 量化域 LoRA 从 43.4% 恢复到
99.0%（超 BF16 基线），而同预算在线自模仿零恢复——恢复的瓶颈在策略自身分布之外
的监督信号；(iii) 全阶梯在单一 24GB 消费级 GPU 上以分钟级训练成本完成，构成
"离线蒸馏 vs 在线 RL"的部署经济学对照。方法学与实现细节分别见 §11、§12。


# 4. 恢复的理论：量化误差、协变量漂移与策略恢复

本章回答两个问题：(i) 为什么量化会让闭环任务成功率下降得比开环指标预测的更多；
(ii) 为什么蒸馏（QAD）、在策蒸馏（OPD）与强化学习（PPO）能够、以及各自能恢复哪一部分损失。
我们把三者统一到同一个数学框架下，并给出可检验的预测与实验设计。

## 4.1 记号与设定

- 环境：LIBERO 操作任务，有限视界 MDP，视界 T ≤ 720 步，每 κ=8 步重规划（~90 次决策）。
- 教师策略 π*（BF16 引擎执行），NVFP4 量化后学生策略 π_q = Q_φ(θ)，其中 Q 为块量化算子，
  φ 为可训练补偿参数（LoRA 增量、可选 scale 修正）。我们要求部署时真实执行 Q(θ)（引擎
  kernel），训练时用 fake-quant 模拟，二者通过 E0 parity 对齐。
- 动作生成：flow-matching，动作 a 由积分向量场 v_θ(x_t, t; s) 的 ODE 得到，
  离散 K 步（K=10 或 STEP 一步）。
- 任务成功率 p(π) = E_τ~π[1{任务达成}]。

## 4.2 每步量化扰动有界（开环视角）

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

## 4.3 闭环复合：协变量漂移使损失超线性（DAgger 论证）

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

## 4.4 三个恢复目标的统一形式（flow-matching 上实现）

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

## 4.5 成功率损失的分解（论文叙事主定理）

把量化造成的成功率损失分解为三个可分别观测的部分：

```
Δ = p(π*) − p(π_q)
  = Δ_floor    # 噪声下界：量化噪声下最优补偿也补不回的部分（由 R5d 噪声匹配实验估计）
  + Δ_shift    # 协变量漂移损失：QAD 补不动、OPD 主补（H1）
  + Δ_opt      # 学生在自身分布上的次优：OPD/PPO 皆可补，PPO 唯一可能越过 p(π*)（H4）
```

各 Δ 通过"恢复曲线 + 残差分解"实验联合估计（R5e oracle 上界 + R5d 噪声下界夹逼）。

## 4.6 可检验假设

- **H1（漂移主因）**：QAD 显著降低教师分布上的动作误差，但对闭环成功率提升有限；
  OPD 在同等 GPU 预算下成功率恢复显著更高。
- **H2（数据混合谱系）**：成功率随 λ 单调不减（GKD 在 LLM 上的结论迁移到 VLA）。
- **H3（访问分布收敛）**：OPD 训练中 d^{π_q} 与 d_{π*} 的距离（状态访问分布 TV 距离）
  单调下降，且其下降幅度中介（mediate）成功率恢复。
- **H4（超越教师）**：OPD 后接 PPO 可超过 p(π*)（教师自身在 LIBERO 并非最优）；
  若不成立，Δ_opt≈0 的证据同样有价值。
- **H5（步数耦合）**：量化损失与 OPD 增益均随 flow 步数 K 减小而减小（Gronwall 预测）。
- **H6（容量阶梯）**：补偿容量（仅 scale → head-LoRA(r8) → full-weight
  action head，后者由 FSDP2+CPUOffload 使能）与闭环恢复上限正相关；机制解释：NVFP4 误差
  全局散布于所有权重，修正未必低秩，r8 低秩子空间可能系统性低估可恢复量 Δ_opt+Δ_shift。
  full-head 臂同时给出工程论据：直接训 W + STE 导出即纯 NVFP4 权重，无 LoRA 旁路张量。

## 4.7 实验矩阵与指标（对照 GKD/DAgger/PPO 顶会惯例）

| 编号 | 实验 | 设计要点 | 关键指标 |
|---|---|---|---|
| R0 | 漂移动机 | PTQ 后按视界截断评测（T=90/180/360/720）；on/off-policy 动作误差对比 | 成功率-T 曲线斜率；ε_a(d*) vs ε_a(d^{π_q}) 差距；状态访问分布 TV 距离 |
| R1 | PTQ 基线 | L0，多精度（BF16/FP8/NVFP4 混合/NVFP4 全量） | 成功率±Wilson 95% CI、延迟、显存 |
| R2 | QAD | λ=0，冻结 backbone+scale，action-head LoRA；~2 GPU·h | 教师分布动作 MSE ↓、成功率（预期有限提升） |
| R3 | OPD | λ∈{0.25,0.5,1}，学生 rollout + 引擎 BF16 teacher 逐步纠偏；~8 GPU·h | 恢复曲线（episodes→成功率）、成功率 AUC、episodes-to-90% gap |
| R4 | PPO 对照 | RLinf 同预算；R4b=教师塑形奖励（OPD 等价形式） | 同预算成功率、样本效率、是否超 p(π*) |
| R5 | 机制消融 | (a) λ 谱系 (b) **容量阶梯{仅 scale / head-LoRA(r8) / full-head(FSDP2+offload)}** (c) teacher=BF16 vs FP8 (d) **等范数高斯噪声 vs NVFP4 结构性误差**（区分"学鲁棒"与"修结构误差"） (e) oracle=BF16 同预算同容量 | 各臂成功率；Δ_floor/Δ_shift/Δ_opt 分解饼图；优化器一律 AdamW（归因隔离，Muon 见并行工作论文） |
| R6 | 恢复-精度前沿 | 精度×{PTQ,+QAD,+OPD} 网格 | "恢复移动精度-成功率前沿"主图 |
| R7 | 步数耦合 | K∈{1,4,10} × {PTQ,OPD} | 量化损失、OPD 增益 vs K |
| R8 | 泛化 | 仅在 2-3 个掉点最重任务上恢复，全 10 任务评测 | 恢复的跨任务迁移（过拟合检查） |

指标口径统一：成功率报 Wilson 95% CI（每任务 ≥50 episodes）；动作误差/MSE 在
**学生自身 rollout 状态上**计算（on-policy，R0 专门对比 off-policy 口径的差异）；
状态访问分布距离用 8 维本体状态直方图 TV 距离 + VLM 嵌入 Wasserstein 双口径；
训练效率报 GPU·h 与环境交互步数（OPD 的 teacher 查询由 ApxInf BF16 服务，
本身即系统贡献）。所有恢复实验种子 ≥3。

## 4.8 与相关工作的关系（引用与差异）

- **GKD/OPD**（Agarwal et al. ICLR 2024；2025 OPD 实践）：借其 λ 混合目标、reverse-KL
  论证与 on/off-policy 对照实验设计；差异：我们作用于 flow-matching 动作分布与闭环控制，
  且 teacher 由量化引擎高速服务。
- **QAD for NVFP4**（NVIDIA, arXiv:2601.20088）：借其"冻结 scale、更新权重"实践与
  PTQ→QAD 阶梯；差异：其恢复开环 benchmark 指标，我们证明并利用"闭环需要 on-policy"。
- **DAgger**（Ross et al. 2011）：借其复合误差理论框架，把"量化"实例化为误差源；
  我们的 OPD 即以 BF16 策略为专家的 DAgger，并首次给出量化场景的分解与噪声下界。
- **STEP**（ICML 2026 warm-start）：H5 把它与量化误差耦合，解释一步生成的额外收益。


# 5. 相关工作

## 5.1 VLA 模型与部署压力

视觉-语言-动作（VLA）模型将预训练 VLM backbone 与动作生成头结合：π0/π0.5
（Physical Intelligence，flow-matching action expert）、NVIDIA Isaac GR00T 系列
（N1/N1.5 用 Eagle-2 backbone，N1.6/N1.7 换用 Cosmos-Reason2-2B/Qwen3-VL 架构并
倍增 DiT action head）、WALL-OSS、SmolVLA、OpenVLA 等。部署侧的共同约束是本体
推理：10-100W 功耗预算内以 10-50 Hz 闭环，模型 3-4B 参数在边缘设备上逼近显存与
带宽极限。NVIDIA 为 GR00T 提供 Jetson Thor 上的 TensorRT 部署路径，但消费级
GeForce 与量化格式的组合此前没有公开数据——本工作填补该空白。

## 5.2 LLM 量化与 NVFP4 恢复

4-bit 权重量化（GPTQ 的 Hessian 逐列舍入、AWQ 的激活感知缩放、SmoothQuant 的
难度迁移）已是 LLM 标配；块缩放格式（NVFP4：E2M1 + per-16 E4M3 scale + per-tensor
FP32 二级 scale，对照 MXFP4 的 per-32 UE8M0）随 Blackwell 硬件成为新的最低精度档。
恢复方面，NVIDIA 的 QAD 技术报告（arXiv:2601.20088）确立 PTQ→QAD（QAT 模拟 +
冻结 BF16 teacher 的 KL 蒸馏、冻结 scale 更新权重）→ 在策蒸馏的阶梯；GKD
（Agarwal et al., ICLR 2024）与后续 OPD 实践给出 λ 混合广义目标与 reverse-KL 的
论证。这些方法恢复的都是**开环指标**（perplexity/benchmark）；闭环控制下量化误差的
复合与恢复是本工作的核心增量。

## 5.3 协变量漂移与模仿学习的复合误差

DAgger（Ross et al., 2011）证明：固定数据集上训练的策略在闭环执行中误差随视界
平方级复合 O(T²ε)，而在策纠正训练将其降为线性 O(Tε)。量化策略天然构成这一框架中
"带扰动学习器"的实例：NVFP4 引入的每步动作偏差 ε_q 正是 DAgger 误差分析中的 ε。
我们将此理论实例化到 VLA 量化恢复（第 3 章），并给出 Δ_floor/Δ_shift/Δ_opt 分解
与相应的实验设计——该连接在量化文献与机器人学习文献中均未被建立过。

## 5.4 具身推理引擎

通用 LLM 引擎（vLLM/TensorRT-LLM）面向 token 吞吐，不处理 VLA 的观测预处理、
flow-matching 采样循环与动作后处理；Embodied.cpp（arXiv:2607.02501）提出可移植
C++ 具身运行时；ApxInf（Infinigence/RLinf，Rust + 自研 CUDA kernel）以三层 API
服务 VLA 并在 Jetson Thor/Orin 上验证 BF16/FP8/INT8。本工作在 GeForce Blackwell
(sm_120) 上适配并扩展 ApxInf：修复其 FP8 路径在 sm_120 的不可用问题（E4M3 GEMM
输出不被 cuBLASLt 支持，改为 F16 输出），并新增 NVFP4 精度路径。与之互补的
单卡训练侧使能技术（FSDP2 CPUOffload 流水线）见并行工作，二者共享同一 checkpoint
与评测协议。

## 5.5 一步动作生成

STEP（ICML 2026）的 warm-start 将 flow-matching 采样从 10 步压到 1 步。我们的
Gronwall 分析（3.2 节）预言量化动作偏差随积分步数线性放大，因此一步生成与量化
存在正向耦合（H5，R7 检验）——这是系统方法与量化精度在 VLA 上的首次交互研究。


# 6. 实验设置与初步结果

## 6.1 硬件与软件配置

| 项 | 值 | 备注 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop（Blackwell sm_120，82 SM，24GB GDDR7） | 消费级移动 Blackwell；FP4/FP8 tensor core |
| 整机 | Intel Ultra 9 275HX（24 核）+ 58GB RAM（WSL2 动态调整中 43GB） | 单机训练+推理+仿真同载 |
| CUDA / cuBLASLt | 13.3 / 13.6 | CUDA 13 API 变化（layout 无 ORDER、handle 必传）均已适配 |
| 推理引擎 | ApxInf main @ e07dbe9（自编译 sm_120，wheel cp312） | 含我们两处 sm_120 修复 |
| PyTorch 基线 | 2.9.0+cu128 / transformers 4.57.3 | lerobot（π0.5）与 NVIDIA 官方 gr00t（N1.7）各自默认路径 |
| 仿真 | LIBERO（robosuite 1.4.1 + MuJoCo 3.1.6 + EGL mesa headless） | WSL2 无 NVIDIA EGL，走 mesa surfaceless |
| 对照训练基建 | FSDP2 CPUOffload（并行工作管线，交叉引用） | full-head 恢复臂专用 |

## 6.2 测量协议

- 延迟：batch-1 端到端（含预处理/tokenize/动作解码），30 样本，报 P50/P99；
  引擎走 CUDA Graph 稳态（首帧捕获开销单列），PyTorch 为 eager。
- 功耗：0.2s 间隔 nvidia-smi 采样全程，报均值/峰值；笔记本 TGP 波动如实注明。
- 显存：进程峰值（nvidia-smi）+ 分配器口径（torch）双报。
- 成功率（后续）：LIBERO-10 ×50 episodes/任务，Wilson 95% CI，断点续跑。
- NVFP4 数值验证：CPU fp64 反量化参考为金标准，GPU kernel 输出
  maxrel < 5e-4（fp16 输出舍入级）为 PASS（已达成，见 §5.3）。

## 6.3 结果总览

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

## 6.4 闭环评测双轨协议与恢复训练协议

**闭环评测**（LIBERO-10，全部本文核心数字的裁判）：
- **全量协议** = 10 任务 × 10 episodes（正式数字，如 BF16 96.7 / mixed 95.1 / fp8 基座
  43.4 / QAD-LoRA 99.0 / OPD-LoRA 99.0）；**mini 协议** = 3 任务 × 5 episodes
  （臂筛选用，偏易——fp8 基座 mini 70.8% vs 全量 43.4%，故正式结论一律以全量为准）。
- 评测栈：eval server（APXInf robo 同构 serving 拓扑）+ ZMQ rollout（libero_uv venv，
  EGL）；烤好的 PTQ/LoRA 合并 checkpoint 直接当模型目录传入，零引擎改动。
- 离线指标（逐层 MSE、probe 向量场 corr）一律不作决策门槛——探针实验证明其对闭环
  损伤呈混沌解耦（§11.8）。

**校准 PTQ 套件**（`quant/ptq/`）：校准集 = libero_demo 16 批 × 8 窗口；189 个
backbone Linear 的精确 Hessian H=Σxxᵀ（fp32，5.49GB）；逐层输出 MSE 用恒等式
tr(ΔW·H·ΔWᵀ) 精确评估（无须保存激活）。GPTQ 为 NVFP4 块格式适配（进入每个 16 列
块时从当前误差补偿值重算块 scale）；AWQ 折叠映射含 GQA 共享通道与 GeGLU 边界。
五臂烤盘（rtn/calib/fp8/mixed/aggr）直改 safetensors，压缩账目随 checkpoint 落盘
（ptq_recipe.json）。

**LoRA 恢复训练**（`rl/lora_qad.py`，24GB 无 FSDP2）：加性语义
y = x·W_bakedᵀ + (x·Aᵀ·Bᵀ)·s（量化基座 + BF16 低秩残差，SVDQuant/EoRA 家族训练侧
形态）——梯度经低秩路径精确回传（无 STE 近似），部署合并 = W_baked + Δ（BF16 加法），
训练与闭环评测严格同一函数；r=32 覆盖 action head 252 层（38.7M 可训练），500 步
AdamW 1e-4，0.43s/步（对比逐前向合并量化 75s/步）。OPD 臂加 probe 缓存 teacher-KL
（BF16 teacher 探针 38s 离线缓存，防 twin-forward 崩溃）。RWR 对照臂走两阶段离线环
（eval server 记录批量 rollout → 训练 venv 内成功加权 flow-matching BC）。

## 6.5 数据完整性

主数据矩阵（算子层、双模型引擎闭环、五臂 PTQ 消融、恢复三臂）全部以全量闭环协议完成；引擎侧 GR00T fp4 kernel 化为后续工作（§12.3 边界声明）。


# 7. 讨论与结论

## 7.1 主要发现

**算子层**：GeForce Blackwell（sm_120）的 cuBLASLt 原生支持 block-scaled NVFP4
GEMM，我们完整逆向并验证了其 scale 张量的 128×4 tile 行交错物理布局——目前无公开
文档。计算受限形状 NVFP4 达 FP8 的 1.8×、BF16 的 5.5×；MXFP4 不获支持。

**引擎层**：在 ApxInf 引擎中交付了完整的 NVFP4 执行路径（adapter→量化 kernel→
工件加载器→混合拓扑→variant），四层组合正确性验证（算子 corr=1.000000）。真实
π0.5 层形状上 fp4 全管线相对 BF16 GEMM 加速 1.3–5.3×。

**闭环实证（本文核心）**：
1. **全模型 NVFP4 PTQ 在闭环中不可救药于纯校准**——max 校准、GPTQ（逐层 MSE 压到
   RTN 的 76%）、逐层 MSE 裁剪全部 0%；误差的**放置位置**（哪些模块量化）比总量
   更重要：复刻 NVIDIA 分配的混合 FP8/NVFP4 直接 95.1%（2.02× 压缩，BF16 96.7%）。
2. **模块级敏感度地图**：NVFP4 致死 = 视觉塔 ≈ DiT 注意力投影 ≈ LLM
   o_proj/down_proj；幸存 = LLM q/k/v/gate/up + DiT/lang FFN。两代模型、两条实现
   路径、两轮实验（head-FP4-only 71.2% / backbone-FP4 0% 旧二分）交叉一致。
3. **量化域 LoRA 恢复**：43.4% 的 2.44× 压缩基座经 3.5 分钟 500 步加性 LoRA
   （量化基座 + BF16 低秩残差）恢复到 99.0%（QAD 与 OPD 双臂），超 BF16 基线；
   同预算 PPO 族在线对照（RWR 成功加权自模仿）**零恢复**——量化恢复的瓶颈在
   策略自身分布之外的监督信号，离线蒸馏（演示/teacher）恰供此信号。
4. **方法论**：离线指标（逐层 MSE、单点向量场 corr）与闭环损伤混沌解耦（连 95%
   的臂都 relMSE≈2）——闭环成功率是 VLA 量化的唯一裁判，这是与 LLM 量化
   （perplexity 可作 proxy）的本质区别。

## 7.2 工程贡献（可上游合并）

1. cuBLASLt CUDA 13 API 适配与 NVFP4 swizzle 布局逆向（spike 全套工具）；
2. FP8 在 sm_120 的不可用根因定位（E4M3 GEMM 输出不支持，probe 铁证）；
3. 打包工件产线（RMSNorm 折叠进权重 + 引擎布局对齐）与豁免表机制；
4. 校准 PTQ 套件（Hessian 采集 / GPTQ-NVFP4 / AWQ 折叠映射 / 五臂烤盘）与
   24GB 无 FSDP2 的量化域 LoRA 恢复栈（0.43s/步，170× 于逐前时量化合并）；
5. 全链 1400+ 行引擎补丁（patches/ 可复现）+ 全部 checkpoint 与一键复现脚本。

## 7.3 局限

- 单一 GPU 型号（RTX 5090 Laptop）；笔记本 TGP 波动已注明，绝对值为持续值；
- GR00T 侧校准与恢复实验在 PyTorch 量化语义（烤盘注入）上进行，引擎侧 fp4
  kernel 化（对应 π0.5 nvfp4_static 路径）为下一步工作——π0.5 侧闭环已证明
  该迁移路径可行（§12.3 边界声明）；
- RWR 对照臂为 REINFORCE 退化形态（成功加权 BC），非完整 PPO/GAE/裁剪目标；
  mini 协议实测零恢复，全量协议与更精细的 RL 预算扫描留作 future work；
- nvfp4 的 gate_up 融合 GEMM 未实现（当前朴素 GEMM+独立 geglu，性能次优）；
- 语言/动作层 PACKED 工件的 vision 部分 qkv 仅 packed 主 GEMM，o_proj 未打包。

## 7.4 结论与展望

NVFP4 在消费级 Blackwell 上的 VLA 部署在算子、引擎、校准与后训练恢复四层均已
打通：**混合 FP8/NVFP4 分配给出 2.02× 压缩下 95.1% 的即用部署方案；量化域 LoRA
把更激进的 2.44× 压缩基座从 43.4% 恢复到 99.0%**。完整闭环阶梯（0% → 43.4% →
95.1% → 99.0%）与模块级敏感度地图为 VLA 的 FP4/FP8 部署提供了可复现的配方与
边界。未来工作：GR00T 引擎侧 fp4 kernel 化与 mixed 工件镜像、旋转类 PTQ
（QuaRot/SpinQuant 家族）经打包工件产线的离线折叠、更小 LoRA 预算下 QAD-vs-OPD
的分离扫描、真机部署验证。

---
*本工作在单张 RTX 5090 Laptop（WSL2）上完成；并行工作（FSDP2 CPUOffload 单卡
全权重训练）共享同卡与 checkpoint，交叉引用。*


# 8. 数值格式与量化基础（教程）

本章为零基础读者完整介绍本文用到的全部数值格式与量化概念。我们不假设读者
具有低精度计算背景；所有格式都给出位级布局、值域表、舍入规则与代码级示例。

## 8.1 从 FP32 到低精度：为什么以及代价

深度学习权重传统上以 FP32（1 符号 + 8 指数 + 23 尾数）或 BF16（1+8+7）存储。
推理的算力与访存瓶颈随精度下降而缓解：NVFP4 相对 BF16 理论访存减半（权重
0.5 字节/元素 vs 2 字节）、tensor core 吞吐升 4×（本文实测 sm_120 上 NVFP4
GEMM 峰值 506 TFLOPS vs BF16 98.5 TFLOPS，§7）。代价是表示误差：4-bit 格式
每个元素只有 16 个可表示值。量化研究的全部主题即：**如何选择这 16 个值
（缩放），以及如何让模型在误差下仍然工作（恢复）**。

## 8.2 E2M1：NVFP4 的数据格式

E2M1 是 4-bit 浮点：1 符号位、2 指数位、1 尾数位。非零正值共 15 个——等等，
实际是 8 个正幅值（含两个子normal）：

| 位模式 s eem | 值 | 说明 |
|---|---|---|
| 0 000 | +0 | 零 |
| 0 001 | +0.5 | 子normal：0.5×2⁻¹×1 |
| 0 010 | +1.0 | normal：2⁰×1.0 |
| 0 011 | +1.5 | 2⁰×1.5 |
| 0 100 | +2.0 | 2¹×1.0 |
| 0 101 | +3.0 | 2¹×1.5 |
| 0 110 | +4.0 | 2²×1.0 |
| 0 111 | +6.0 | 2²×1.5（最大有限值） |

加上符号位取负，E2M1 的完整值域是
**{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}**。注意两个特点：(i) 相邻值间距
不均匀（0.5 与 6 之间呈 2 的幂档）；(ii) 最大值 6，超出即饱和（satfinite）。

**舍入**：将实数 v 映射到最近的 E2M1 值。中点（如 0.25 在 0 与 0.5 之间）
采用 round-to-nearest-even（RNE, ties-to-even）：优先落到二进制表示最低
位为 0 的那侧。本文第 7 章的 kernel 调试记录了一个真实案例：GPU kernel 用
"+0.5 截断"（ties-away）而 CPU 参考用 RNE，导致 0.74% 字节不一致——这类
细节在教程文献中很少提及，但在工程上决定逐位对错的成败。

**打包**：两个 E2M1 值打包进 1 字节（低 4 位 = 偶数位置元素，高 4 位 = 奇数
位置元素）。

## 8.3 E4M3 与 UE8M0：两种缩放格式

**E4M3**（8-bit）：1+4+3，用于 NVFP4 的块缩放。值域 ±448，比标准 FP8 E4M3
（±464，无穷/NaN 保留）略窄——cuBLASLt 生态采用 ±448 变体（satfinite 到
0x7E）。块缩放用 E4M3 而非 FP32 的理由：硬件在 GEMM 累加时以 8-bit 解码
scale，带宽与寄存器成本远低于 FP32。

**UE8M0**（8-bit 无符号纯指数）：值为 2^(b−127)，无尾数。用于 MXFP4（microscaling，
per-32 块）。本文实测 sm_120 的 cuBLASLt 不支持 MXFP4 GEMM（heuristic 返回
状态 7、零算法，§7.2），因此全文聚焦 NVFP4。

## 8.4 缩放粒度与双重量化

- **per-tensor**：整个矩阵一个 FP32 scale。FP8 时代主流；对权重分布跨度大
  的矩阵浪费严重。
- **per-channel**：每输出通道一个 scale（权重的行方向）。LLM 常用。
- **per-block(16)**：每 16 个连续输入元素一组 scale——NVFP4 的选择。粒度
  细到能跟踪局部动态范围，粗到 scale 开销可控（每 16 元素 1 字节 E4M3，
  额外 6.25% 存储）。
- **双重量化（double quantization）**：块 scale 本身再除以一个 per-tensor
  FP32 scale（"global scale"），使 E4M3 编码只需覆盖 amax/6/global_scale
  的动态范围。部署时三层数值：E2M1 数据 × E4M3 块 scale × FP32 tensor scale。
  注意（本文 §8 的一个工程发现）：若 fake-quant 模拟器按双重结构编码而硬件
  GEMM 只应用两层，将引入系统性常数因子误差——务必核对模拟器与 kernel 的
  scale 语义一致。

## 8.5 权重量化的数学

给定权重矩阵 W ∈ R^{m×n}、块大小 B=16，第 j 块的量化为：

```
s_j = E4M3(amax(W_j) / 6 / g)          # 块 scale，g = per-tensor scale
Q(W) = E2M1(W / s_j)                    # 逐元素
W ≈ Q(W) × s_j × g                       # 反量化（部署读取的就是它）
```

本文 §2.4 的实测：双模型 1036 个张量上 rel-Frobenius 误差高度均匀
（0.095±0.001）——NVFP4 的 per-16 缩放足够自适应，**权重"可量化性"不是
闭环失败的瓶颈**（连向 §9 的机制分析）。

## 8.6 fake-quant：训练与评测中的模拟

真实 FP4 GEMM 需要硬件路径（§4）；训练与快速评测用 **fake-quant 模拟：前向
时将权重（或激活）量化再立即反量化**（即 W → W ≈ Q(W)×s×g 的浮点近似），
反向用 **直通估计器（STE, straight-through estimator）**：把量化视为恒等，
梯度 ∂L/∂W 直接穿过量化节点。这是所有 QAT/QAD 方法的基础设施。

两种实现：(i) 逐前向重算（训练时正确——权重在变）；(ii) **quantize-once**
（评测时把权重一次性替换为反量化值，之后按原生精度前向——数值等价、快约
50×，本文闭环评测即用此法，§8）。

---

# 8.7 NVFP4 的 scale swizzle 布局（教程小节）

本节讲解 NVFP4 block-scaled GEMM 中最不直观、也最少有公开文档的部分：
**缩放因子张量在显存中的物理排布（swizzle）**。这是我们为让引擎 kernel 工作
而完整逆向的内容（第一手材料，含三层探针实验的完整过程），读者照着做可以
在自己的硬件上独立验证。

## 8.7.1 为什么 scale 需要特殊排布

block-scaled GEMM 的数学是 C = Σ_blocks (A_blk × B_blk^T) × sa × sb：数据
以 E2M1 打包、每 16 元素一组 scale。GPU tensor core 的高效实现要求 scale
与数据块的访存模式**共定位**——即加载某 tile 的数据时，对应的 scale 也要
以相邻、规则的模式到达寄存器。为此 cuBLASLt 不接受"逻辑"布局（rows × KB
的矩阵），而要求调用方预先把 scale 重排成固定的物理模式。CUDA 13 的头文件
与文档对此只有枚举名（`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`），
**没有排布公式**——这就是我们逆向的对象。

## 8.7.2 逻辑布局与物理布局

设权重矩阵 W 有 `rows` 行、`K` 列（K 方向按 16 分块，KB = K/16）。
逻辑上 scale 是一个 `rows × KB` 的 E4M3 矩阵：S[r][b] 是第 r 行第 b 块
的缩放。物理上，cuBLASLt 要求它按如下公式排布（我们解码的最终形式）：

```
PS = 512 × ceil(KB / 4)                              # 每个 128 行 panel 的步长
offset(r, b) = PS × (r / 128)                        # 行所在的 panel
             + 512 × (b / 4)                         # 块组（4 块 = 64 元素）
             + 16 × (r % 32)                         # 32 行交错的第一层
             + 4 × ((r / 32) % 4)                    # 第二层：组内 4 行簇
             + (b % 4)                               # 块在组内的位置
```

直观图景（对应 NVIDIA 文档中 "128×4 tiled, rows interleaved" 一句话）：
- scale 缓冲按 **128 行一个 panel** 切分，panel 步长固定 512 字节
  （= 128 行 × 4 块，即一个 128×64 元素 tile 的全部 scale）；
- panel 内部又按 **32 行一组**交错——连续 16 个字节属于同一行的 4 个（至多）
  块 scale，下一组 16 字节属于下一行……32 行后回到"第 0 簇的第 32+ 行"；
- 分配大小必须按 panel 向上取整（rows=300 → 3 个 panel），尾部 padding。

## 8.7.3 逆向方法论（三层探针，可复用的技术）

我们无法从文档获得公式，于是设计了三个层层递进的实验——这套方法论本身
适用于任何"黑盒布局"问题：

**探针 1（平 scale）**：把所有 scale 设为同一值（如 1.0）。此时任何布局
误解都不影响结果——若数值仍错，问题在数据打包而非 scale。结果：maxrel=0
⇒ E2M1 nibble 打包正确，缩小问题面。

**探针 2（单字节异常）**：物理缓冲全填 1.0，仅在**一个**字节放 2.0。输出
矩阵 C 中出现异常的单元格位置，直接暴露该物理字节控制的逻辑 (行, 块)。
我们最初用它验证了"每 16 元素一个 scale"的粒度假设。

**探针 3（字节槽扫描）**：对物理偏移 0…4096 逐字节执行探针 2，得到
"物理偏移 → 逻辑 (行, 块)" 的完整映射表，再拟合公式。锚点示例（M=K=256，
KB=16）：物理 0 → 行 0；物理 4 → 行 32；物理 16 → 行 1；物理 100 → 行 38；
物理 2048 → 行 128。由这些锚点解出上述五项公式。

**验证**：公式生成的 swizzle 缓冲与逐字节扫描映射在全部测试形状上一致；
随机数据（含 10× 块间 scale 动态范围）下四象限 GEMM 全部通过（maxrel
4.8e-4，fp16 舍入级）。

## 8.7.4 读者自查清单

在自己硬件上复现本节：(1) 构造平 scale GEMM 验证数据路径；(2) 单字节
异常定位粒度；(3) 用我们开源的 `spike/fp4_gemm_bench.cu`（含 `FP4_SLOT`
探针模式）扫描映射；(4) 与 4.3.2 公式对照。若你的 cuBLASLt 版本行为不同，
该探针流程会直接给出新公式。

---

# 9. 量化恢复方法论（教程）

本章完整讲解 PTQ → QAD → OPD 三级恢复阶梯的原理、设计决策与我们的实证
教训。每级回答同一个问题：**量化把闭环成功率砸掉之后，拿什么把它救回来？**

## 9.1 PTQ（后训练量化）：直接操作权重，不训练

**流程**：(1) 加载 BF16 checkpoint；(2) 逐张量按 §3.4 公式离线转换
（权重侧无需数据）；(3) 激活侧需要校准数据估计 scale（跑少量前向收集
统计）；(4) 量化后的权重直接部署。

**为什么它先失败**：单层权重误差均匀（0.095）不是问题；问题是 VLA 前向
有 36 层 × 每层多 GEMM，动作输出与 BF16 教师去相关（我们的实测：GR00T
池化 corr −0.015，π0.5 0.11）。机制上，backbone 的 KV-cache 让每步的
特征误差在序列维度复用复合（§2 理论），flow-matching ODE 的 Gronwall
放大再乘一层。**结论：全模型 NVFP4 PTQ 在两个模型、两条实现路径上闭环
全部归零（0%）——这不是实现 bug，是复合放大的结构性结果。**

## 9.2 QAD（Quantization-Aware Distillation）：训练权重适应量化网格

**核心思想**：与其量化后承受误差，不如训练时就把量化"放进"前向——让权重
主动移到"量化后仍然正确"的位置。两个组件：

**(a) QAT（量化感知训练）**：前向插入 §8.6 的 fake-quant + STE。梯度告诉
每个权重"你被量化后输出了多少误差"，训练把它拉回网格上的正确格点。权重
在量化网格上移动的直观图像：W 的每个 16-元素块由 (Q, s, g) 描述，训练改变
的是 Q 的选择与 s 的取值。

**(b) 蒸馏损失**：NVIDIA 的 QAD 技术报告（arXiv:2601.20088）的关键贡献是
用 **冻结 BF16 teacher 的 KL 损失替换任务损失**：学生不再拟合 demo 标签，
而是拟合 teacher 的行为分布——保持"和 teacher 一样做事"，只把行为搬到
量化网格上。配套实践：**scale 冻结、权重更新**（避免 scale-权重联合优化的
不稳定）。

**我们的实证教训（重要负结果）**：我们第一版 QAD 用的是官方微调配方的
demo 任务损失（MSE 于 demo 动作）+ fake-quant——**没有 teacher-KL 锚定**。
结果：训练损失正常下降（1.29→0.33）但闭环 0%。诊断：demo 损失优化的是
"在量化约束下重建 demo 轨迹"，牺牲的恰是闭环行为（探索/恢复能力）。
另外发现 **grid-adaptation 现象**：QAD 权重不量化直跑（corr 0.21）反而差于
量化部署（0.74）——权重已内化网格，离开网格反而错位。这从反面印证 teacher
锚定的必要性：**恢复必须锚定行为分布，而非动作重建**。

## 9.3 OPD（On-Policy Distillation）：在学生自己的轨迹上纠偏

**理论起点（DAgger，Ross & Bagnell 2011）**：闭环执行中，策略每步的小误差
把状态推离训练分布；在漂移后的状态上误差进一步放大。固定数据集训练的复合
误差是 O(T²ε)；**在学生自己访问的状态上获得教师标签**（在策纠正）降为
O(Tε)。量化策略正是"带扰动学习器"的典型实例。

**目标函数**（GKD, Agarwal et al. ICLR 2024 的 λ 混合形式迁移到
flow-matching）：

```
L_λ(φ) = E_{s~D_λ, t, x_t} [ w(t)·‖ v_{Q(θ,φ)}(x_t,t;s) − v_{θ*}(x_t,t;s) ‖² ]
D_λ = λ·d^{π_q(φ)} + (1−λ)·d_demo
```

λ=0 即 QAD（teacher-forced）；λ=1 即纯在策。reverse-KL 的 mode-seeking
性质在 ODE 端点分布上同样成立。

**我们的 twin-forward 实现**（工程贡献）：不加载第二个 3.4B teacher——同一
模型、同批数据，交换 nn.Linear.forward（fake-quant ↔ 原始）跑两遍，KL 项
直接度量"量化引起的当步行为漂移"。显存省一半、batch 完全对齐。代价：每步
双前向（我们以 OPD_KL_EVERY=4 节流）。

**成功率损失分解**（本文理论框架）：
Δ = Δ_floor（量化噪声下界，R5d 噪声匹配实验估计）
  + Δ_shift（协变量漂移损失，OPD 主补）
  + Δ_opt（次优残差，PPO 唯一可能超越教师）。

## 9.4 OPD 与 PPO 的关系（对比讨论 + 可选对照臂）

教师塑形奖励 r = −‖a − π*(s)‖² 下的 PPO 在期望上退化为 OPD 的加权形式
——两者是同一谱系的两端：稠密行为监督（OPD，样本高效、上限=教师）vs 稀疏
任务奖励（PPO，样本低效、可超越教师）。我们的实验矩阵把 PPO 设计为**同预算
对照臂**（同 1000 步、同 batch、同 fake-quant 前向、LIBERO 任务奖励），
回答审稿人必问的"为什么不用 RL"：(i) 同预算下蒸馏的样本效率优势（预期
数量级）；(ii) H4：RL 能否越过教师 87.4%（并行工作 E4 数据）的局部最优。
若工程时间不足，本节以理论对比 + 引用并行工作论文 E5 Muon 长跑数据作为
RL-相邻证据，PPO 臂列为 future work。


# 9.5 补充：DAgger 复合误差分析与量化实例化（教程小节）

**定理（Ross & Bagnell 2011, 非正式）**：设教师在分布 d_* 上每步误差
ε_a，学生策略在闭环中访问自身诱导分布 d_π。若学生只在固定数据集
d_demo（≈ d_*）上训练，则 T 步视界的总误差为 O(T² ε_a)；若训练数据
含学生在策访问 d^{π_student} 上的教师标签，总误差降为 O(T·ε_a')。

**直觉**：固定数据训练的学生从未见过"自己犯错后的状态"。量化策略天然是
"带噪声学习器"——每步动作偏差 δa 把状态推离教师的轨迹；在未见过的状态上
模型的误差不受训练约束，进一步漂移（正反馈）。在策数据打破这个正反馈：
学生犯错后到达的状态 s' 上恰好有教师标签，训练把"从 s' 恢复"编进策略。

**量化实例化**：NVFP4 的每步动作偏差 ε_q ≈ ‖J‖·ε_W（§2 线性化）。
我们的离线-闭环实证（§8）：全量化 PTQ 的动作相关性掉到 −0.015，闭环 0%
——正是 O(T²ε) 的极端体现。OPD 的 λ>1 数据混合把训练分布拉向 d^{π_q}，
KL 项在学生实际访问的状态上强制行为对齐——这就是"恢复必须锚定行为分布"
的数学根据，也解释了为什么 demo-loss QAD（只在 d_demo 上优化）救不了
闭环：它优化的是重建，不是恢复。

**与成功率分解的连接**：Δ_shift 对应本节的复合项（OPD 主补），Δ_opt
对应教师自身次优（PPO 唯一可能突破），Δ_floor 对应噪声下界
（量化不可消除部分，可用等范数高斯噪声对照实验估计——R5d 设计）。

---

# 10. 闭环评测方法论（教程小节）

**协议**：LIBERO-10（10 个长视界操作任务），每任务 N=10 episodes、
每 8 步重规划（H=16 的 action chunk 执行前 8 步）、视界上限 720 步、
固定种子。报告 **Wilson 95% 置信区间**下的成功率（N=100 时 Wilson 区间
约 ±6-10 个点，0%/100% 结果的区间分别为 [0, 3.7]% 与 [96.4, 100]%）。

**为什么不用离线指标**：我们的实证（§8.2 信号扫描）——PTQ 与 QAD 的
池化动作相关性（−0.015 vs +0.139）差异悬殊，闭环却同为 ~0%。机制：
相关性度量的是单步动作方向的保真，闭环成功需要的是**整个状态-动作流形
上的恢复能力**（反馈纠正、误差不累积）。任何单步/开环指标（corr、L2、
perplexity 类比）都无法替代闭环 rollout。

**实现要点（复现者的坑）**：评测端的 fake-quant 用 quantize-once（加载时
一次性替换权重，前向原生速度——逐前向重算会慢 ~50× 并触发评测框架的
超时）；流式保存的 checkpoint 用分片索引合并导出；Wilson 区间脚本与
ledger 格式见仓库 `results/`。


# 11. 训练后量化（PTQ）方法综述与我们的校准实践

## 11.1 PTQ 是什么：训练与推理解耦的量化

训练后量化（Post-Training Quantization, PTQ）不改动训练流程：拿一个训好的 BF16 模型，
用**少量有代表性的数据（校准集）**确定缩放因子、零点等量化参数，直接产出低精度权重
（Nagel et al., 2021；Wu et al., 2020；Vanhoucke et al., 2011）。它成本低、无须重训，
特别适合拿不到原始训练配方、或没有算力微调的使用者——这正是 VLA 部署最常见的处境：
checkpoint 是别人发的，机器人实验室只有一张消费级卡。

PTQ 的全部工作可以归结为一句话：**在"低精度格式能表示的权重集合"里，找一个与原模型
行为最接近的点。** 格式（NVFP4/FP8）定了几许自由度，校准算法决定你怎么用这些自由度。

## 11.2 校准目标函数的三层递进

**第一层：最大值校准（max calibration）。** 缩放因子取校准数据中的最大绝对值映射到格式
可表示的最大值。对 NVFP4 而言这几乎是格式自带的：每个 16 元素块的 E4M3 scale 就是块内
amax/6。简单，但在很多场景下效果出人意料地好——它是我们 0% 基线（RTN 臂）用的方法，
也是所有后续方法的对照组。

**第二层：误差最小化校准。**
- **MSE 校准**（Jacob et al., 2018）：搜索裁剪阈值 c，让"量化输出与原输出的均方误差"
  最小，而不是硬保最大值。我们在每个 Linear 上对 c∈{1.0,…,0.5} 网格搜索
  （学习裁剪阈值的思想来自 Choi et al., 2018）。
- **KL 散度校准**（Migacz, 2017）：让量化前后激活分布的 KL 距离最小。TensorRT INT8 的
  传统做法，主要面向逐 tensor 对称 INT8；对细粒度块缩放的 NVFP4 作用有限（块内自由度
  已被 E4M3 scale 占据）。

**第三层：结构性优化。** 不再逐层独立地凑 scale，而是利用"这一层的量化误差会传到下一层"
的结构，做联合优化——这就是 GPTQ/AWQ/BRECQ 一族。

## 11.3 面向 LLM 的逐层校准：GPTQ 与它的朋友们

**GPTQ**（Frantar et al., 2023）把经典的 Optimal Brain Surgeon 舍入搬到了 LLM 上：
用校准激活算出每层的 Hessian 近似 H = Σxxᵀ，然后按列贪心量化权重，把每一列的量化误差
通过 Hessian 逆矩阵"分摊"到还没量化的列上。它把逐层输出 MSE 显著压低，且只需几百条
校准样本。**ZeroQuant**（Yao et al., 2022）更早地把逐层重建用到 INT8/INT4 权重+激活。

我们的实现（`quant/ptq/quantizers.py::gptq_nvfp4`）把 GPTQ 适配到 NVFP4 的**块缩放格式**：
列按从左到右处理，每进入一个新的 16 列块，就从**当前（已被误差补偿修改过的）**权重值
重算该块的 per-row E4M3 scale，再逐列量化-反馈。单元测试保证量化器与引擎侧参考实现
逐位一致（`nvfp4_dequant` vs `torch_fp4.fake_quant_nvfp4_torch`，maxdiff=0.0）。

**AdaRound**（Nagel et al., 2020）学习"向上取整还是向下取整"的软选择；**BRECQ**
（Li et al., 2021）以块为单位重建。两者都需要一小段优化训练，本工作以 LoRA 恢复训练
（§12）承担了同等的"重建"角色，故未重复实现。

## 11.4 保持敏感信息：AWQ 与缩放折叠

**AWQ**（Lin et al., 2023）的观察是：权重并非同等重要——少数通道（激活幅度大的）承担
了不成比例的信息。它给这些通道乘上缩放 s（等价地把难度从激活转到权重），再把 s **精确
折叠**进前置 LayerNorm 的增益或前一层权重的对应列，推理图零改动。

我们完整实现了折叠映射（`quant/ptq/folds.py`），包括三处非平凡的情形：GQA 注意力里
o_proj 的通道缩放要折叠进 v_proj 输出列（共享 kv 通道，repeat_interleave 粒度）；GeGLU
里 down_proj 的缩放折叠进 up_proj 列（对 up 线性、精确）；视觉塔 SiLU-MLP 的 fc2 不可
折叠（逐元素非线性，无精确折叠点）。搜索空间 s=absmean^α，α∈[0,1] 网格，判据用
Hessian 恒等式 tr(ΔW·H·ΔWᵀ)（无须保存激活即可**精确**算出输出 MSE）。

**结果是一个干净的负结果：所有折叠站点 α=0.0 最优**（搜索日志 189 层全量）。
解释：weight-only NVFP4 的 per-16 块 E4M3 scale 已经在块粒度上吸收了通道异常值，
逐通道再缩放不改变块内相对结构，自然无增益。这与 NVIDIA 在 **W4A4**（激活也量化）
场景下需要 `NVFP4_AWQ_FULL_CFG` 并不矛盾——激活共享 per-tensor scale 时通道异常值
才是致命的，而我们的部署形态是 weight-only。**SmoothQuant**（Xiao et al., 2023）同理：
它解决的是激活量化难、权重量化易的再平衡，weight-only 场景下没有对象。

## 11.5 旋转与低秩：需要推理图配合的方法

**QuaRot**（Ashkboos et al., 2024）、**QuIP#**（Tseng et al., 2024）、**SpinQuant**
（Liu et al., 2024b）、**FlatQuant**（Sun et al., 2024）用可逆旋转（多为 Hadamard 类）
把权重/激活分布"摊平"后量化，数学上要求推理时对激活做同样的旋转——即推理图必须支持
在线变换或把旋转吸收进相邻权重。**SVDQuant**（Li et al., 2024）/ **EoRA**（Liu et al.,
2025b）则用低秩 BF16 分支吸收异常值/补偿量化损失，同样改变推理图。这类方法在我们的
栈里对应引擎侧的 kernel 支持（APXInf 的 packed 工件布局已经把 RMSNorm 折叠进权重，
旋转矩阵可以走同一条离线折叠产线），属于下一步工作——本章先引用其思想，实测以
校准+混合精度为主线。**OmniQuant**（Shao et al., 2024a）用梯度下降优化量化参数本身，
与我们的 LoRA 恢复（§12）在"可学习量化参数"这一点上同源。

## 11.6 已有的近无损先例：把别人的配方当基线

截至 2026-09，针对 GR00T N1.7 已有两条公开的近无损 PTQ 路线，直接定义了我们的对照组：

| 实现 | 格式 | 评测 | BF16 → 量化 |
|---|---|---|---|
| NVIDIA Jetson AI Lab（ModelOpt, TensorRT） | 混合 FP8/NVFP4 | LIBERO Spatial 200eps×3 轮 | 97.5% → 97.3% |
| FoldQuantVLA（INT4/INT8, consistent folding） | W4A4 + o/down W8A8 | LIBERO 四套件 | 95.75% → 95.625% |

NVIDIA 配方的精度分配表（我们逐条复刻）：

| 模块 | 分配 |
|---|---|
| ViT 线性层 | FP8 |
| LLM q/k/v/gate/up | NVFP4（+AWQ） |
| LLM o_proj / down_proj | FP8 |
| lm_head | FP8 |
| DiT FFN（ff.net.0.proj / ff.net.2） | NVFP4 |
| DiT 其余线性层（注意力投影等） | FP8 |
| timestep / action encoder / action decoder | FP16 不量化 |

FoldQuantVLA 的"consistent folding"（Wx = (WSRᵀ)(RS⁻¹x)，缩放 S+旋转 R+变换后坐标里做
GPTQ，Q/K/V 共用输入必须折叠同一逆变换）则说明：**当量化扩展到激活时，变换的一致性
本身就是一个研究点**。

## 11.7 我们的五臂消融：什么真正救活了闭环

全部五臂共用同一 BF16 基座（GR00T N1.7 LIBERO-10 微调版，Qwen3-VL backbone +
DiT flow-matching head）、同一校准集（libero_demo 16 批×8 窗口，189 个 backbone Linear
的精确 Hessian，`quant/ptq/collector.py`）、同一闭环协议（LIBERO-10）。筛选用 3 任务
×5 episodes，终测 10 任务×10 episodes。**离线指标一律不作为门槛**（原因见 11.8）。

| 臂 | 精度分配 | 线性参数压缩 | mini 闭环（3×5） | 结论 |
|---|---|---|---|---|
| rtn | 全 NVFP4，max 校准 | 3.56× | **0%**（基线复现） | 格式上限之外 |
| calib | 全 NVFP4 + GPTQ + 逐层 MSE 裁剪 | 3.56× | **0%** | 纯校准救不了 |
| fp8 | backbone FP8 + head 全 NVFP4 | 2.44× | 70.8% | head 注意力 NVFP4 是短板 |
| **mixed** | NVIDIA 分配复刻 | **2.02×** | **95.1%**（全量 10×10） | 部署级方案 |
| aggr | 全 NVFP4，仅护 o/down（FP8）+解码器（BF16） | 2.88× | **0%** | 视觉塔 NVFP4 亦致死 |

（mixed 10×10 全量终测：**97/102 = 95.1%**，BF16 上界 96.7%——2.02× 压缩下仅丢
1.6 个百分点，且 10 任务中 8 个满分；与 NVIDIA 官方 LIBERO Spatial 97.5→97.3 的近无损
结论同向。）

三个可写进论文的贡献点：

1. **混合分配是第一变量，校准是第二变量。** calib 臂把逐层 MSE 压到 RTN 的 ~76%，闭环
   仍为 0%——在 4-bit 格式下，逐层最优的误差累计仍会摧毁闭环行为；而 mixed 臂只动精度
   分配（没动校准）就回到 93%。这量化了"PTQ 的瓶颈在误差的**放置位置**而非总量"。
2. **AWQ α=0 负结果**（11.4）——weight-only NVFP4 的块缩放已吸收通道异常值。
3. **模块级敏感度地图**（五臂差分的直接产物）：NVFP4 化致死模块 = 视觉塔 ≈ DiT 注意力
   投影 ≈ LLM o_proj/down_proj；可幸存模块 = LLM q/k/v/gate/up、DiT FFN（mixed 里 FFN
   保持 NVFP4 而闭环近乎无损）。fp8 臂（head 全 NVFP4）70.8% 与旧 head-FP4-only 71.2%
   相互印证；aggr 臂（视觉 NVFP4 + head 注意力 NVFP4）0% 与旧 backbone-FP4-only 0%
   相互印证——两代实验交叉一致，敏感度排序与 NVIDIA 配方的保护对象独立吻合。

## 11.8 方法论教训：离线 proxy 的混沌性

我们最初用"固定噪声+固定时间步的单点向量场 MSE/corr"作离线筛选，结果连 FP8/mixed
这种闭环 93%+ 的臂都给出 relMSE≈2、corr≈0。诊断：flow-matching 向量场对权重微扰呈
**混沌敏感**——离线单点差分放大到 O(1)，与真实行为损伤完全解耦。BF16 自检（corr 应为
1.0）暴露了协议缺陷，修复 RNG 漂移后混沌性依旧。结论与本项目一以贯之：**闭环成功率是
唯一裁判**；离线指标（逐层 MSE、probe corr）只用于工程排障，不进入决策链。这也是
VLA 量化区别于 LLM 量化（perplexity 可作 proxy）的方法论差异点。

## 11.9 APXInf 在这条链路里的角色

- **量化语义的对齐基准**：我们的 PyTorch 侧量化器以 APXInf 引擎工件转换器
  （`fp4_quant.py`/packed 工件）为位级参照（corr=1.000000），保证"离线校准实验的数"
  就是"引擎 kernel 将消费的数"。
- **闭环评测基础设施**：LIBERO 闭环跑在与 APXInf robo 层同构的 eval server（APXInf robo 层同构的
  serving 拓扑）上，烤好的 PTQ checkpoint 无需任何引擎改动即插即用。
- **部署通道**：π0.5 侧的 nvfp4_static 引擎路径（§3.1）证明同一格式在真实 kernel 上
  的行为一致性；GR00T 侧的 packed 工件产线（RMSNorm 折叠、qkv/gate_up 拼接）为
  12.5 节的旋转折叠预留了离线折叠的工程位。

## 参考（本章新增）

- Nagel et al., 2021, A White Paper on Neural Network Quantization
- Wu et al., 2020; Vanhoucke et al., 2011（PTQ 早期）
- Jacob et al., 2018（MSE/校准）; Migacz, 2017（KL）; Choi et al., 2018（学习裁剪）
- Nagel et al., 2020（AdaRound）; Li et al., 2021（BRECQ）
- Yao et al., 2022（ZeroQuant）; Frantar et al., 2023（GPTQ）
- Lin et al., 2023（AWQ）; Shao et al., 2024a（OmniQuant）
- Xiao et al., 2023（SmoothQuant）; Ashkboos et al., 2024（QuaRot）; Tseng et al., 2024（QuIP#）; Liu et al., 2024b（SpinQuant）; Sun et al., 2024（FlatQuant）
- Li et al., 2024（SVDQuant）; Liu et al., 2025b（EoRA）
- NVIDIA Jetson AI Lab, GR00T N1.7 mixed_nvfp4 部署教程与补丁（ModelOpt `mtq.quantize`）
- FoldQuantVLA: Native Low-Bit Quantization of VLA Models via Consistent Folding, 2026（cair-vinuni/FoldQuantVLA）


# 12. 全栈实现与 APXInf 生态的角色

> 本章回答两个问题：**我们做了什么**（量化 → 校准 → 后训练恢复 → 闭环评测的全链自研
> 栈），以及 **APXInf 帮到了什么**（位级语义基准、推理引擎、serving 拓扑、工件产线）。
> 立场声明：APXInf 是本工作的推理引擎底座而非量化方法来源——量化方法学（校准、GPTQ、
> 混合分配）是我们按 §12 的公开文献自研实现的，但每一步都锚定在 APXInf 引擎的真实
> 语义上，保证离线实验的数字就是部署路径的数字。

## 12.1 全栈工作清单

**量化与校准（§11）**
- NVFP4/FP8-E4M3 weight-only 量化器（PyTorch 侧，与引擎工件路径位级一致）
- 校准激活采集器：189 个 backbone Linear 的精确 Hessian（16 批×8 窗口，43k 行/层）
- GPTQ 的 NVFP4 块格式适配（块 scale 进入时从当前误差补偿值重算）
- AWQ 折叠映射全套实现（GQA 共享通道、GeGLU 折叠、SiLU 不可折叠点）+ 全站点搜索
- 五臂配方烤盘：rtn / calib / fp8 / mixed（NVIDIA 分配复刻）/ aggr

**后训练恢复（§12.2，训练栈）**
- **QAD-LoRA**：merged-quant 语义（y = x·Q(W+BA)ᵀ per-forward），训练=部署同函数，
  24GB 无 FSDP2（对比旧全量头 QAD 需要 FSDP2+CPU offload+流式保存三件套）
- **OPD-LoRA**：+ probe 缓存 teacher-KL（38 秥 BF16 teacher 探针，防 twin-forward 崩溃）
- **RWR-LoRA**：PPO 家族对照（成功加权自模仿；DPPO 式精确似然留作 future work）
- LoRA 合并烤盘：量化域内合并（fake_quant(W+Δ)），产出标准 eval checkpoint

**推理与评测（serving 栈）**
- 闭环评测：LIBERO-10 × 10 episodes 全量协议 + 3×5 mini 快筛协议
- quantize-once 语义烤盘（§11 的持久化版本）
- eval server 的 rollout 日志钩子（RWR 数据采集）

**引擎侧（APXInf 内，此前会话）**
- fp4 GEMM adapter（cuBLASLt E2M1×E2M1→F16 + VEC16_UE4M3 swizzled scales，maxrel 4.8e-4）
- π0.5 nvfp4_static 全链（加载器/Fp4Blocks 混合拓扑/工件完整性测试）
- PACKED 工件转换器（两树布局 + RMSNorm 折叠，99 张量 7.11× 压缩验证）
- π0.5 双精度闭环：BF16 90.0% vs nvfp4_static 0/100（真实 kernel 路径）

## 12.2 恢复阶梯实验设计（QAD/OPD/RWR vs PPO 家族）

统一部署基座 = fp8 臂（backbone FP8-E4M3 per-channel + head NVFP4，2.44× 压缩，
mini 闭环 70.8%）。三臂同 LoRA 预算（r=32，head 252 层，38.7M 可训练，500 步，
AdamW 1e-4）：

| 臂 | 数据/信号 | 损失 |
|---|---|---|
| QAD-LoRA | 离线演示（libero_demo） | flow-matching demo loss（量化域内 STE） |
| OPD-LoRA | + BF16 teacher 探针（probe cache） | + w·KL(pred, teacher) |
| RWR-LoRA | on-policy 成功 rollout（自采） | 成功加权的 flow-matching BC（REINFORCE 终端 0/1 奖励的退化形式） |

对照解读（预算匹配的两个口径）：
- **梯度步匹配**：三臂各 500 步——RWR 的 rollout 采集时间额外计（on-policy 的固有开销，
  正是对比点）。
- **墙钟匹配**：RWR 每 500 梥度步需先付 ~25 分钟 rollout 采集（15 episodes）；QAD/OPD
  零采集。这是"离线蒸馏 vs 在线 RL"的部署经济学论据。

**恢复臂数据**：

| 臂 | 信号 | 训练/采集成本 | mini 闭环（3×5） | 10×10 全量 |
|---|---|---|---|---|
| fp8 基座（无恢复） | — | — | 70.8%（1.0/0.545/0.0） | **46/106 = 43.4%**（mini 三任务偏易，全量更严峻） |
| RWR-LoRA（PPO 家族） | on-policy 成功加权自模仿 | 8 分钟训练 + 25 分钟 rollout 采集 | **70.8%（1.0/0.545/0.0，与基座逐任务相同）** | — |
| QAD-LoRA | 离线演示（demo loss） | **3.5 分钟训练**，零采集 | **15/15 = 100%**（微波炉 0.125→1.0） | **100/101 = 99.0%** |
| OPD-LoRA | + BF16 teacher-KL 探针 | 同预算（KL 实测 0.186→0.135） | **15/15 = 100%** | **102/103 = 99.0%** |

主结论有两条：

1. **恢复**：从 2.44× 压缩的量化部署（全量 10×10 仅 43.4%）出发，500 步（3.5 分钟）的
   LoRA 后训练把闭环成功率恢复到 99.0%，超过 BF16 基线（96.7%）。QAD 与 OPD 在本预算
   下打平——teacher-KL 的价值体现在损失可观测且单调下降（0.186→0.135）与不依赖演示
   数据的理论普适性；更深的分离需要更弱基座或更小预算的扫描（future work）。
2. **离线 vs 在线的经济学**（QAD/OPD vs RWR）：同预算下，on-policy 自模仿对闭环**零改善**
   （三任务成功率与基座逐位相同）——它只能放大策略已有的行为，而稀疏 0/1 终端奖励下
   信号被 70% 的失败稀释；离线演示/teacher 信号则提供了策略自身分布之外的信息，直接
   修复了量化损伤。加上采集成本（25 分钟 rollout vs 0），"离线蒸馏恢复"在本设定下全面
   占优。这与 DPPO（Ren et al., 2024）需要在完整 PPO 目标（GAE+裁剪+价值网络）下
   才能从在线信号获益的结论一致——我们的 RWR 是其预算下界形态，如实标注为
   REINFORCE 退化形式而非完整 PPO。

三臂语义学注记：v2 加性 LoRA（y = x·W_qᵀ + x·Aᵀ·Bᵀ·s）等价于"量化基座 + BF16 低秩
修正分支"（SVDQuant/EoRA 家族的训练侧形态）——梯度经低秩路径**精确**回传（无 STE
近似），部署合并 = W_q + Δ（BF16 加法，不重量化），训练与闭环评测**严格同一函数**；
且前向恢复原生速度（0.43s/步 vs 逐前向全矩阵量化合并的 75s/步，170×）。

## 12.3 APXInf 具体帮到了什么（可验证清单）

1. **位级语义基准**。我们的 PyTorch 量化器逐位对标引擎工件转换器（gold_check4
   corr=1.000000；nvfp4_dequant vs fake_quant maxdiff=0.0）——"校准实验里的 NVFP4"
   就是"引擎 kernel 消费的 NVFP4"，没有这一层，离线恢复训练的结论无法迁移到部署。
2. **真实 kernel 路径的先行验证**。π0.5 nvfp4_static 闭环（0/100 vs BF16 90%）在我们
   写任何校准代码之前就锁定了"全 NVFP4 RTN 致死"这个事实，直接决定了本项目把火力
   投向混合精度分配而非纯校准——五臂消融后来在 GR00T 上独立复现了这一点。
3. **serving 拓扑**。闭环评测跑在与 APXInf robo 层同构的 policy-server + ZMQ rollout
   栈上；烤好的 PTQ checkpoint 与训练后的 LoRA 合并 checkpoint 都无需任何引擎改动
   即插即用——"评测即部署形态"。
4. **工件产线复用**。π0.5 的 packed 工件转换器（RMSNorm 折叠、qkv/gate_up 行拼接）
   为旋转类 PTQ（QuaRot/SpinQuant 一族）预备了离线折叠的工程位——这是未来把 §11.5
   方法落进引擎的路径。
5. **性能基线**。GR00T N1.7 BF16 引擎 33.7ms/29.7Hz（sm_120 首批 GeForce 数据）给出了
   量化部署的延迟参照系。

诚实的边界：GR00T 的校准/后训练实验目前跑在 PyTorch 侧（量化噪声以烤盘形式注入），
引擎侧的 GR00T fp4 kernel 化（对应 π0.5 的 nvfp4_static 路径）是下一步工作；π0.5 侧的引擎闭环已证明该迁移路径可行。

## 12.4 产出物索引

- 代码：`quant/ptq/`（采集/量化/GPTQ/AWQ/烤盘/评测）、`rl/lora_qad.py`、`rl/lora_rwr.py`、
  `rl/lora_merge_bake.py`、`rl/overnight_chain.sh`
- 校准数据：`/mnt/c/fq_ptq_calib/calib.pt`（189 层 Hessian 5.49GB）
- 烤好的 checkpoint：`weights/ptq_bakes/gr00t_ptq_{rtn,calib,fp8,mixed,aggr}`
- 闭环结果：`runs/eval_mini_{mixed,fp8,calib,aggr}`、`runs/eval_full_mixed`
- 复现：每臂一条 bake 命令 + 一条 eval 命令（§11.7 表可直接映射）
