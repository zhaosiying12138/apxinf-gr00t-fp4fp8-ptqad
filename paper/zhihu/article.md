# 摘要

视觉-语言-动作（VLA）模型需要在本体上以 10–50 Hz 闭环运行，但现有部署研究集中在 BF16/FP8。NVIDIA Blackwell 将 NVFP4（E2M1 + 逐 16 元素 E4M3 块缩放）带入消费级 GPU，本文在 RTX 5090 Laptop（sm_120）上系统回答：**VLA 能否在 FP4/FP8 混合精度下运行，闭环成功率损失几何，如何恢复？**

**算子层**：我们通过受控探针实验测定并逐字节验证了 cuBLASLt 的 NVFP4 块缩放物理布局，据此实现了 sm_120 上可用的 block-scaled NVFP4 GEMM，峰值 506 TFLOPS（1.8× FP8）。**引擎层**：在 APXInf 中交付完整 NVFP4 执行路径（四层正确性验证，corr=1.000000），并给出首批 GeForce 上的 VLA 引擎数据（π0.5 = 7.3×、GR00T N1.7 = 3.2× 于各自 PyTorch 基线）。**实证层（核心结果，LIBERO-10 × 10×10 全量闭环）**：全模型 NVFP4 PTQ 无论 max 校准、GPTQ Hessian 舍入还是逐层 MSE 裁剪均闭环归零（0%）——逐层最优的量化误差在闭环中复合成行为崩溃；而复刻 NVIDIA Jetson AI Lab 的混合 FP8/NVFP4 分配（backbone FP8 + 敏感投影保护）在 2.02× 线性参数压缩下达到 **95.1%**（BF16 基线 96.7%）。**恢复层**：在 2.44× 压缩、全量闭环仅 43.4% 的 FP8/NVFP4 基座上，500 步（3.5 分钟）的量化域 LoRA 后训练（量化基座 + BF16 低秩残差的加性形态，SVDQuant 家族）将闭环成功率恢复到 **99.0%**（QAD 演示蒸馏与 OPD teacher-KL 双臂同达，均超 BF16 基线）；同预算的 PPO 家族在线对照（成功加权自模仿 RWR）则**零恢复**——构成量化恢复问题上离线蒸馏与在线自模仿的经济学对照。**理论层**：以 DAgger 协变量漂移框架将复合误差定量化（O(T²ε)→O(Tε)），导出并实测了 PTQ→QAD→OPD 恢复阶梯。**工程交付**：1400+ 行可上游引擎补丁、打包工件管线、校准 PTQ 套件（Hessian 采集/GPTQ/AWQ/五臂烤盘）、24GB 单卡 LoRA 恢复训练栈（无 FSDP2、0.43s/步）、全部 checkpoint 与复现脚本。

# 贡献亮点

**1. 在 NVIDIA 混合方案之上把压缩从 2.02× 推进到 2.44×——继续量化了什么？** NVIDIA 的 mixed_nvfp4 配方在动作头一侧留有大量余量：DiT 的注意力投影（to_q/to_k/to_v/to_out）与 adaLN 调制投影保持 FP8，timestep 编码器与动作解码器输出投影保持 FP16。我们把**动作头的全部线性层——包括 NVIDIA 保留的这四类——整体推进到 NVFP4**，backbone（ViT + LLM + lm_head）以 per-channel FP8-E4M3 权重量化承载，线性参数占用从 6.25 GB 压到 2.56 GB（2.44×；NVIDIA 配方复刻为 2.02×）。代价是全量闭环从 95.1% 跌至 43.4%——这恰好构成恢复层的严格测试床：**量化更狠、恢复更难，恢复方法本身的贡献才可分离**。

**2. 从 PTQ-only（NVIDIA 路线）到 PTQ + 量化域恢复（本文）：43.4% → 99.0%。** NVIDIA 止步于 PTQ（其报告 97.5%→97.3%）。我们在 43.4% 的 2.44× 基座上证明，仅 500 步（3.5 分钟）的 LoRA 后训练即可恢复到 **99.0%，反超 BF16 基线（96.7%）**，且 QAD（演示蒸馏）与 OPD（teacher-KL）双臂同达。做对了三件事：(a) **恢复训练发生在量化域内**——加性 LoRA 形态（量化基座 + BF16 低秩残差）使训练前向与部署函数严格同一，梯度经低秩路径精确回传、无 STE 近似，且以原生速度运行（0.43 秒/步，是逐前向全矩阵量化合并方案的 170 倍）；(b) **闭环成功率是唯一裁判**——我们实证离线指标与闭环损伤混沌解耦，所有臂的取舍均以 10×10 全量闭环裁定；(c) **用离线监督信号而非在线自模仿**——同预算的 PPO 家族对照（成功加权自模仿）对基座零改善，说明量化恢复的瓶颈在策略自身分布之外的信息，恰是演示/teacher 所提供的。

**3. 算子与引擎的完整实现（而非黑箱调用）。** 在受控探针实验测定并逐字节验证 cuBLASLt 的 NVFP4 块缩放布局之后，我们在 ApxInf 引擎内实现了 CUDA 适配器（E2M1×E2M1→F16 block-scaled GEMM + 在线激活量化核）、Rust 算子（fp4_linear）、工件加载器（fp4_weights）与混合拓扑执行器（Fp4Blocks），四层正确性验证收敛于 corr=1.000000；完整源码走读见附录 A。

# 1. 引言

机器人策略模型正在从"云端大脑"走向本体推理：GR00T、π0.5、WALL-OSS 等 VLA 模型需要在 10–100W 的边缘设备上以 10–50 Hz 的频率完成"感知→决策→行动"闭环。现有量化研究集中于 LLM（GPTQ/AWQ/SmoothQuant 等 4-bit 权重量化），但 VLA 有两点本质不同：其一，动作头的输出直接驱动闭环控制，量化误差会在数百步的 rollout 中复合，而非体现在 perplexity 上；其二，VLA 推理同时包含自回归 VLM 解码与 flow-matching 多步采样两类截然不同的算子负载。Blackwell 消费级 GPU 的 FP4 tensor core（NVFP4：E2M1 数据 + 每 16 元素 E4M3 块缩放 + 每 tensor FP32 二级缩放）提供了一个此前不存在的部署选项，但没有任何公开工作系统回答过上述三个问题。

本文贡献可归纳为五点：

- **C1（算子与引擎）**：以受控探针实验测定并验证 cuBLASLt NVFP4 块缩放的物理布局；在 APXInf 引擎中交付完整 NVFP4 执行路径并给出首批 GeForce（sm_120）VLA 引擎数据（§3.1、§4.2）。
- **C2（闭环失效定律）**：在两代 VLA（π0.5/GR00T N1.7）、两条实现路径（引擎真 kernel / PyTorch 量化语义）上证明全模型 NVFP4 PTQ 闭环一致归零，且 GPTQ/MSE 校准无法挽回——量化误差的模块分布比总量更决定闭环行为（§4.3、§4.4）。
- **C3（模块敏感度地图与混合精度配方）**：五臂 PTQ 消融给出模块级 NVFP4 敏感度排序（致死：视觉塔、DiT 注意力投影、LLM o/down_proj；幸存：LLM 主投影与全部 FFN），复刻 NVIDIA 分配的混合 FP8/NVFP4 方案在 2.02× 压缩下闭环 95.1%（§4.4）。
- **C4（量化域 LoRA 恢复）**：提出量化基座 + BF16 低秩残差的加性 LoRA 形态（训练与部署严格同一函数，无 STE 近似），从 43.4% 的 2.44× 压缩基座以 3.5 分钟训练恢复到 99.0%；同预算在线自模仿（RWR）零恢复，给出离线/在线信号的经济性对照（§4.5）。
- **C5（方法论）**：证明离线指标（逐层 MSE、单点向量场相关性）与闭环损伤混沌解耦——闭环成功率是 VLA 量化的唯一裁判；配套全链可复现工件（§4.6、附录 A）。

本文结构：§2 背景与相关工作（VLA 解剖、NVFP4 格式、LLM 量化谱系、VLA 量化先行工作、从蒸馏到在线自模仿的理论脉络）；§3 方法（执行层、校准 PTQ、复合误差理论、量化域恢复、闭环评测方法论）；§4 实验（按因果链组织的全部实证：执行层特性 → 全量失效解剖 → 混合精度与敏感度 → 三臂恢复 → 方法论消融）；§5 讨论与结论；附录 A 结合核心源码走读全部实现并解析 APXInf 框架。

# 2. 背景与相关工作

本章给出理解后续章节所需的全部背景：被量化对象的解剖（§2.1）、数值格式（§2.2）、LLM 训练后量化的方法谱系（§2.3）、VLA 量化的先行工作（§2.4），以及恢复训练所处的学习理论脉络——从 DAgger 式蒸馏到在线自模仿（§2.5）。第 4 章的每一个实验设计都能在本章找到动机。

## 2.1 VLA 模型解剖与部署压力

本文的实验对象是 GR00T N1.7（LIBERO 微调版），其 backbone 为 Qwen3-VL：语言侧 12 层、隐藏维 2048、GQA 注意力（16 个查询头 / 8 个 KV 头，头维 128，共享 KV 通道）；视觉侧 24 层、隐藏维 1022 的 SigLIP 型塔（qkv 融合投影 [3072,1024]），输出经 merger 与三层 deepstack 合并器投影到语言维；动作侧为 16 层 DiT（隐藏维 1536，交叉注意力从 backbone 取上下文），以 flow-matching 目标生成 64 步动作块。全模型 3.36B 参数中线性层占 3.13B（6.25GB BF16），是显存压缩的唯一有效靶点。对照模型 π0.5（PaliGemma/Gemma backbone）用于跨模型验证。两类模型的推理都混合了自回归 VLM 解码与多步去噪两类算子负载，且动作输出直接驱动闭环控制——这是 VLA 量化与 LLM 量化的本质区别：误差不在 perplexity 里显形，而在数百步 rollout 中复合（§3.3 给出定量分析）。

## 2.2 NVFP4 数值格式

NVFP4 由三层结构组成：4-bit E2M1 数据格（值域 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}，相对格长在 1/4 到 1/2 之间）、每 16 个连续元素的 E4M3 块缩放、以及每张量一个 FP32 二级缩放（实践中取 tscale = 全张量块内 amax 最大值 /448，用于把块缩放装进 E4M3 的表示域）。反量化值为 q·s_block·（隐含 tscale 已并入 s_block）。这个双重量化结构是后面多个结论的技术根源：其一，per-16 块缩放使**逐模块的权重量化误差几乎与分布无关**——我们在双模型 1036 个张量上实测 rel-Frob 全部落在 0.095±0.001（§4.2），因此任何模块间的闭环敏感性差异都来自拓扑与误差复合，而非"某些权重更难量化"；其二，块缩放已经在 16 元素粒度上吸收了通道间异常值，这使得 AWQ 式逐通道缩放在 weight-only 场景下失去作用对象（§4.6 的 α=0 负结果）。fake-quant 模拟（训练与评测中注入等价噪声）与真实 kernel 的语义对齐由逐位一致的参考实现保证（附录 A.3.2）。对照格式 FP8-E4M3（per-channel 权重缩放）作为混合精度的另一档；MXFP4（VEC32_UE8M0）在 sm_120 的 cuBLASLt 中不被支持（heuristic 返回零算法），构成本文格式选择的硬约束。

## 2.3 LLM 训练后量化：方法谱系

训练后量化（PTQ）不改动训练流程，用少量校准数据确定缩放因子等量化参数（Nagel et al., 2021 综述；Wu et al., 2020；Vanhoucke et al., 2011）。方法按其所利用的结构信息分三层递进。

**校准目标**：最简单的最大值校准把校准数据的最大绝对值映射到格式满量程；MSE 校准（Jacob et al., 2018）与 KL 校准（Migacz, 2017）转而最小化输出误差或分布距离；学习裁剪阈值（Choi et al., 2018）把"保最大值还是保典型值"变成每层一个小规模搜索——本文的逐层 clip 网格即属此类。

**逐层结构性优化**：GPTQ（Frantar et al., 2023）用校准激活的 Hessian 近似 H=Σxxᵀ 做 Optimal Brain Surgeon 式的逐列舍入与误差反馈；ZeroQuant（Yao et al., 2022）更早把逐层重建用于 INT8/INT4；AdaRound（Nagel et al., 2020）学习舍入方向；BRECQ（Li et al., 2021）以块为单位重建。AWQ（Lin et al., 2023）观察到少数激活幅度大的通道承载不成比例的信息，给这些通道乘缩放并把缩放精确折叠进前置归一化增益或前层权重列；OmniQuant（Shao et al., 2024a）以梯度下降优化量化参数本身。

**变换与低秩**：SmoothQuant（Xiao et al., 2023）把量化难度在激活与权重间再平衡（面向激活量化）；QuaRot（Ashkboos et al., 2024）、QuIP#（Tseng et al., 2024）、SpinQuant（Liu et al., 2024b）、FlatQuant（Sun et al., 2024）用 Hadamard 类正交旋转摊平分布，代价是推理图须配合在线变换或离线折叠；SVDQuant（Li et al., 2024）与 EoRA（Liu et al., 2025b）用低秩 BF16 分支吸收异常值/补偿损失——本文恢复层的加性 LoRA 形态与这一族在"量化基座 + 全精度低秩修正"的结构上同源（§3.4）。

## 2.4 VLA 量化的先行工作与本文位置

针对 GR00T N1.7 的公开近无损结果有两条路线。NVIDIA Jetson AI Lab 的 mixed_nvfp4 配方（ModelOpt + TensorRT）：ViT 线性层 FP8，LLM 的 q/k/v/gate/up 投影 NVFP4，LLM 的 o_proj 与 down_proj 回退 FP8，DiT 的 FFN 用 NVFP4 而其余线性层（注意力投影、adaLN 调制）FP8，timestep 编码器与动作解码器保持 FP16；在 LIBERO Spatial 上 200 episodes × 3 轮报告 97.5%→97.3%。FoldQuantVLA（2026，INT4/INT8）：以"一致折叠"（Wx = (WSRᵀ)(RS⁻¹x)，通道缩放 S + 分块 Hadamard 旋转 R + 变换后坐标里的 GPTQ + 按 token 动态激活量化，敏感投影 W8A8）在 LIBERO 四套件上报告 95.75%→95.625%。两者共同点是：**都止步于 PTQ**，且都在动作头一侧保留了可观的 FP8/FP16 余量。本文的位置由此确定：复刻并验证 mixed 分配（95.1%，§4.4），然后把 NVIDIA 保留的动作头四类投影整体推进到 NVFP4（2.44×），再用量化域恢复把随之而来的 43.4% 拉回 99.0%（§4.5）——压缩、敏感性与恢复三个变量第一次被分离测量。

## 2.5 恢复训练的学习理论脉络：从蒸馏到在线自模仿

量化后的策略恢复本质上是"带约束的策略改进"。三条技术路线在本文中同台对比，其理论谱系如下。

**策略蒸馏与协变量漂移**。行为克隆只在演示分布 d_demo 上最小化单步误差，Ross & Bagnell（2011；DAgger，Ross et al., 2011）证明其闭环性能损失按视界平方复合 O(T²ε)；专家标注的在线纠偏（DAgger）把损失压到线性 O(Tε)。广义知识蒸馏（GKD，Agarwal et al., 2023）进一步给出分布选择上的教师-学生散度视角。落到量化场景：QAD（量化感知蒸馏）在演示分布上把学生推向量化网格上对演示的最优解——它优化的是噪声项、不触碰漂移项；OPD（在线策略蒸馏）在学生自己的 rollout 分布上用教师标签纠偏，等价于以 BF16 教师为专家的 DAgger——这正是 §3.3 把平方复合降为线性的机制。

**强化学习与策略梯度**。REINFORCE（Williams, 1992）以 ∇J(θ)=E[R·∇log π_θ(a|s)] 直接在环境回报上优化；PPO（Schulman et al., 2017）以裁剪代理目标与优势估计成为当前默认。对扩散/流策略，DPPO（Ren et al., 2024）把去噪 ELBO 作为代理对数似然，使策略梯度可微穿过多步去噪。

**在线自模仿：RWR 与 SIL**。奖励加权回归（RWR；Peters & Schaal, 2007；Kober & Peters, 2009 综述）把策略改进写成加权最大似然：π_{k+1} ∝ π_k·exp(R/β)，其解是"以回报为权的轨迹加权 BC"，免除了显式值函数。自模仿学习（SIL；Oh et al., 2018）指出确定型策略的每次成功轨迹都携带"好于当前期望"的信息，其损失 ℓ_SIL = E[max(A,0)·(−log π)] 在优势为正时退化为对**自身成功轨迹**的模仿；SIL 的理论分析同时给出其边界——自模仿只能放大行为策略支撑集内已被访问的行为，无法恢复从未到达的状态-动作区域。这一族思想在离线侧的延续包括优势加权回归 AWR（Peng et al., 2019）、AWAC（Nair et al., 2020）与回报条件建模（Decision Transformer，Chen et al., 2021）。

**0/1 终端奖励下的退化形式（本文的 RWR 臂）**。LIBERO 的回报是 episode 末的二值成功信号，REINFORCE 的梯度变为 ∇J = E_{成功轨迹}[∇log π]——即只对成功 episode 做加权克隆，权重恰为任务成功率。当逐步优势不可得（无值网络）而以任务级成功率加权时，RWR 就是"在自己 43.4% 的成功轨迹上做 BC"。把它与 §3.3 并置即得本文的核心对照预言：量化损伤把策略推离教师分布（协变量漂移），自模仿只能在**策略自身支撑集内**重新加权，无法提供修复漂移所需的分布外信息；而 OPD 的教师标签恰恰作用在学生自己的（漂移后的）状态上。§4.5 的实验逐字兑现了这个预言——RWR 训练损失正常下降而闭环成功率与基座逐任务相同。我们把 RWR 如实标注为 PPO 家族的预算下界形态（无 GAE、无裁剪、无价值网络），DPPO 式完整实现留作扩展。

# 3. 方法

本章把全部技术要素组织成一条连贯的流水线：NVFP4 在 sm_120 上的执行层（§3.1）→ 校准 PTQ 与混合分配（§3.2）→ 量化误差为何在闭环中放大的理论（§3.3）→ 三种恢复方法（§3.4）→ 以闭环为唯一裁判的评测方法论（§3.5）。源码级细节统一放在附录 A。

## 3.1 sm_120 上的 NVFP4 执行层

**缩放布局的探针测定**。cuBLASLt 的 block-scaled 接口要求块缩放字节按一个未公开的物理公式排布。我们以三层递进的受控实验测定它（完整过程见 §A.2.1 的方法自述与仓库 spike/）：第一层用"平缩放"张量（所有块缩放相同）跑 GEMM 并二分替换 scale 缓冲的字节，定位"哪些字节位置真正被消费"；第二层用单字节非零探针（每次只置一个偏移为可分辨值）扫描出逻辑块 (row r, block b) 与物理偏移的映射；第三层以全随机数据在不同 K/行数下验证公式的普适性。最终得到 off(r,b) = PS·(r/128) + 512·(b/4) + 16·(r%32) + 4·((r/32)%4) + (b%4)，PS = 512·ceil(KB/4)——128 行为一个 panel、行内按 32 行四簇交错、块按 4 交错。

**为什么必须是这个布局**。它不是任意约定，而是内存系统友好的重排：逻辑上行主序的 scale 矩阵被组织成以"32 行簇 × 4 块组"为单位的 512 字节连续区间——某一行的 4 个块缩放在物理上紧挨着（4 字节），GEMM kernel 的每个线程用一次 16 字节向量读取即可取到该行 4 个块共 64 个权重元素的缩放；一个 warp 的 32 行恰好消费一段连续的 512 字节，访存完全合并。若按行主序直接存放，同一行相邻 4 块的缩放会相距 KB 字节，每次向量读取都跨缓存行。下图给出逻辑视图到物理缓冲的静态映射、公式与实例计算（同色为同一逻辑单元）：

![](images/swizzle_layout.png)

这套"平探针定位消费字节 → 单字节扫描建映射 → 随机数据验公式"的三层流程对任何黑箱布局问题可复用。

**引擎集成**。在测得布局后，NVFP4 GEMM 以 CUDA 适配器形式接入 APXInf（extern "C" 的 cuBLASLt 封装 + 在线激活量化核，CUDA 13 列主序适配；E2M1×E2M1→F16，scale 模式 VEC16_UE4M3），其上是 Rust 算子 fp4_linear、工件加载器（离线量化工件只上传不重量化）与组合式混合拓扑执行器 Fp4Blocks——每个线性层经 gemm_maybe_fp4 路由：有权重工件则"激活在线量化 + 块缩放 GEMM"，无则 BF16 直通，混合精度部署因此零代码改动（§A.2）。量化核手写位级 E4M3/E2M1 编码器而非调用硬件内建转换，以保证与离线参考逐位一致——四层验证（算子 corr=1.000000、逐块 scale 语义、工件级、kernel 级）收敛后才进入闭环实验。

**算子特性与混合准则**。计算受限形状下 NVFP4 达 506 TFLOPS（1.8× FP8、5.5× BF16）；但 batch-1 访存受限形状下 FP4 反而慢于 FP8（428 vs 1253 GB/s）——小投影进 FP4 无收益。这一非对称性与 §3.2 的精度分配共同决定"哪些层值得 4-bit"。

## 3.2 校准 PTQ：Hessian、GPTQ 块适配与混合分配

**校准统计**。在 libero_demo 上以 16 批 × 8 窗口跑 BF16 教师，对 189 个 backbone 线性层的前向输入钩子采集精确 Hessian H=Σxxᵀ（fp32 累积，5.49GB）与逐通道绝对均值。任意权重扰动 ΔW 的输出影响由恒等式 Σᵢ‖xᵢΔWᵀ‖² = tr(ΔW·H·ΔWᵀ) 精确给出——校准评估无须保存激活。

**GPTQ 的 NVFP4 块适配与逐层裁剪**。经典 GPTQ 假设逐列固定量化格，而 NVFP4 的块缩放随每 16 列的当前值变化。我们的适配：列从左到右处理，每进入一个 16 列块先从**当前（已被误差反馈修改过的）**权重重算逐行 E4M3 块缩放，再在该固定缩放下逐列量化并按 Hessian 逆做 OBS 误差反馈；每层的裁剪系数 c∈[0.5,1] 以输出 MSE 网格搜索（max 校准与 MSE 校准在此合流）。

**精度分配即策略**。五种配方是五个模块名到精度的纯函数（代码见 §A.3.3）：rtn（全 NVFP4 max 校准）与 calib（全 NVFP4 + GPTQ/clip）检验"纯校准能走多远"；mixed 复刻 NVIDIA 分配；aggr 保护仅 o/down_proj；恢复基座 = backbone per-channel FP8 + 动作头全部线性层 NVFP4。烤盘把量化值以 place-value 形式写回 safetensors（BF16 张量携带量化噪声），压缩账目随 checkpoint 落盘——现有评测栈零改动。部署格式账目：NVFP4 0.5625 字节/参数（含 E4M3 缩放），FP8 约 1 字节 + 行缩放。

## 3.3 量化误差的闭环复合：协变量漂移分析

设教师策略 π*（BF16）与量化学生 π_q = Q_φ(θ)。单步层面，一阶分析给出动作偏差界 ‖a_q(s)−a*(s)‖ ≤ ‖J_θ(s)‖·ε_W^eff + O(ε_W²)，其中权重雅可比的谱范数由逐层 ε_W 复合；动作由流 ODE 积分产生，向量场逐点偏差 δv 经 Gronwall 型界放大为轨迹偏差 O(K·δv)。关键的闭环效应在状态分布：每步动作偏差把状态推离教师的数据分布，在漂移后的状态上策略误差进一步放大。记 ε_a 为教师分布上的单步动作误差，由 Ross & Bagnell (2011) 的复合误差分析：行为克隆式训练（只在固定演示分布上最小化 ε_a，不触碰分布漂移）的闭环损失为 J(π*) − J(π_q) = O(T²·ε_a)；而在学生自身分布上获得专家标注的在线纠偏（DAgger 型）把损失降为线性 O(T·ε_a')。**这就是"量化在闭环里比在开环指标里伤得更重"的数学形式**，也是三种恢复配方分工的依据：QAD 在 d_demo 上优化，压低噪声项但不动漂移项；OPD 在学生的 rollout 分布上用教师标签纠偏，把平方复合降为线性，还能部分修正教师自身可改进的部分；直接环境奖励（PPO 家族）则优化闭环回报本身。成功率损失可相应分解为 ε_floor（量化噪声下界，由格式与分配决定）+ ε_shift（分布漂移，QAD 不可修）+ ε_opt（策略次优，PPO 理论上可越教师），第 4 章的实验逐一对应这三项。

## 3.4 量化域恢复：加性 LoRA 的 QAD、OPD 与在线自模仿对照

**加性 LoRA 形态**。恢复训练的第一决定是把"量化"从训练循环里请出去。每个受恢复的线性层计算 y = x·W_bakedᵀ + (x·Aᵀ·Bᵀ)·(α/r)：基座权重 W_baked 是烤盘产出的量化值（冻结、永不再量化），低秩残差是 BF16 全精度分支。三个由此而来的性质：(i) 梯度经低秩路径**精确**回传，无 STE 近似——量化基座不需要梯度；(ii) **训练前向与部署函数严格同一**——部署合并只是把加法算一次（W_baked + BA·α/r，BF16 加法，不重量化），闭环评测与训练所见函数一致；(iii) **原生速度**——0.43 秒/步，逐前时把 W+BA 整体量化的朴素写法为 75 秒/步（170×），在 24GB 单卡上把 500 步训练从 10 小时量级拉回 3.5 分钟。该形态与 SVDQuant/EoRA 的"量化主体 + 全精度低秩修正"结构同源（§2.3），但以 LoRA 训练动态实现。

**三个恢复臂**。QAD：演示分布上的 flow-matching 损失（量化域内的监督微调）。OPD：同预算外加探针缓存的 teacher-KL——BF16 教师在固定探针批上的预测离线缓存（38 秒），训练循环只跑学生前向对缓存做 MSE，避免 twin-forward 在 WSL 上的确定性崩溃；KL 项可观测且单调下降（§4.5）。RWR（在线自模仿对照，理论见 §2.5）：两阶段离线环——评测服务器记录批量 rollout（观测 + 动作块，任务成功率由评测日志给出），训练侧把成功 episode 的 64 步动作窗口（由已执行的 8 步前缀拼接、经处理器归一化，z-range 校验）以任务成功率为权做 flow-matching BC。预算与 QAD/OPD 对齐（500 梯度步当量），另计 25 分钟采集成本——on-policy 方法的固有开销，正是对照的一部分。它被如实标注为 REINFORCE 的退化形式（0/1 终端奖励、无优势估计）。

## 3.5 闭环评测方法论

**闭环是唯一裁判**。我们实证（§4.6）离线指标与闭环损伤混沌解耦，故一切臂间取舍以闭环成功率为准；离线量（逐层 MSE、向量场相关性）只用于工程排障。

**双轨协议**。全量协议 = LIBERO-10 × 每任务 10 episodes（全部正式数字）；mini 协议 = 3 任务 × 5 episodes（快速筛臂，已知偏易：恢复基座在 mini 上 70.8%、全量 43.4%——本文所有结论性数字均取全量）。评测栈与部署形态同构：策略常驻服务器 + ZMQ rollout 客户端，烤好的 checkpoint 换入模型路径即测。**quantize-once 语义**：加载时把受量化线性层的权重一次性替换为 NVFP4 反量化值，前向恢复原生 BF16 速度——数值与逐前向 fake-quant 完全等价，避免评测慢 50 倍导致的 rollout 超时（§A.5）。

# 4. 实验

本章按因果链组织：先把执行层特性交代清楚（§4.2），然后解剖"全模型 NVFP4 为何闭环归零"（§4.3），再测"量化误差该放在哪里"（§4.4），最后回答"放错之后能否救回、怎么救"（§4.5），并以两组方法论消融收尾（§4.6）。每个实验按"目的—操作—结果—解读"叙述；表格只承载数字，因果解释都在正文里。

## 4.1 实验设置

全部实验在单张 RTX 5090 Laptop（Blackwell sm_120，24GB，WSL2）上完成，训练、推理与仿真同载一机。闭环评测协议：LIBERO-10 的 10 个操作任务、每任务 10 episodes、每 episode 上限 720 步（每 8 步重新决策一次），以任务完成判定成功；正式数字一律取全量协议，3 任务 × 5 episodes 的 mini 协议仅用于臂间快筛并如实标注。基线：BF16 教师（官方 LIBERO 微调 checkpoint）全量闭环 96.7%，与 NVIDIA Thor 参考值 ~98% 相当。所有量化臂与恢复臂共用同一教师、同一评测栈、同一随机种子协议；每个臂的 checkpoint、烤盘配方（ptq_recipe.json）与逐任务日志全部入库可复现（§A.6）。

## 4.2 执行层：BF16 引擎加速与 NVFP4 算子特性

**目的**：确定 NVFP4 在目标硬件上的真实收益区间，为"哪些层值得 4-bit"提供算子层依据。**操作**：batch-1 端到端延迟（30 样本 P50，含预处理与动作解码）对比 APXInf 引擎与各自官方 PyTorch 路径；纯 GEMM 微基准覆盖计算受限（2048³–4096³）与访存受限（batch-1 GEMV）两类形状；真实 π0.5 层形状上对比"在线量化 + 块缩放 GEMM"全管线与 BF16 GEMM。**结果**：端到端上 APXInf BF16 相对 PyTorch 加速 π0.5 = 7.3×（361.7→49.6 ms）、GR00T N1.7 = 3.2×（106.9→33.7 ms，首批 GeForce 数据，约为 Jetson Thor 的 1/2）；计算受限 GEMM 上 NVFP4 达 506 TFLOPS（FP8 的 1.8×）；但 batch-1 GEMV 上 FP4 带宽利用率反而低于 FP8（428 vs 1253 GB/s）；真实层形状的 fp4 全管线相对 BF16 加速 1.3×（小形状）到 5.3×（嵌入大形状）。**解读**：NVFP4 的收益集中于大 K 的计算受限投影；小形状留 FP8/BF16 不损失性能——§4.4 的精度分配恰好遵循这一准则的互补面（性能不敏感处可以按精度敏感性自由分配）。

![](images/e1_latency.png)

![](images/opbench_heatmap.png)

## 4.3 全模型 NVFP4：闭环失效的解剖

**目的**：回答第一个核心问题——NVFP4 全量 PTQ 的闭环代价是多少，以及"训练时感知量化"能否避免它。**操作**：四路对照同时闭环。BF16 教师直接评测；PTQ 臂把原权重量化部署（max 校准）；naive QAT 臂以同配方微调 1000 步、训练中逐前向 NVFP4 fake-quant（STE 直通梯度）；QAD-deployed 臂在量化感知训练后再量化部署。另在 π0.5 上以 APXInf 引擎的真实 fp4 kernel 路径（nvfp4_static）独立闭环 100 episodes 作为跨模型、跨实现路径的验证。**结果**：BF16 96.7%；PTQ 0%（10 任务全零）；naive QAT 0%（训练损失从 1.29 正常降到 0.33，闭环却全零）；QAD-deployed 0.63%（160 episodes 中 1 次成功）；π0.5 BF16 90.0% vs 引擎 nvfp4_static 0/100。失败 episode 的形态本身是证据：策略几乎从不提前终止，全部跑满 720 步——动作流形上的微小但持续的偏差让物体永远到不了目标位。**解读**：三层结论。(i) 失效是系统性的：两代 VLA、两条实现路径（引擎 kernel 与 PyTorch 量化语义）一致归零，排除实现工件（导出经键名/冻结/改变三重验证）。(ii) 训练损失正常下降而闭环归零，说明"量化网格上的演示拟合"与"闭环行为"是两个目标——这正是 §3.3 的预言：QAT 只压低教师分布上的 ε_a，对分布漂移项无能为力，而 O(T²·ε_a) 的平方复合把每步百分之几的动作偏差放大成行为崩溃。(iii) QAD-deployed 的 0.63% 高于绝对零但无实质意义，说明温和的网格自适应不足以跨越这道鸿沟——恢复需要的是 §3.4 的低秩残差容量与分布外信号。

## 4.4 混合精度：误差该放在哪里

**目的**：把"全量归零"拆解到模块级——量化误差的**放置位置**比总量更重要吗？**操作**：第一轮二分：head-only FP4（backbone BF16）与 backbone-only FP4（head BF16）分别闭环。第二轮五臂：以 §3.2 的烤盘固定校准不变，只改精度分配——rtn（全 NVFP4，max 校准，即 4.3 的 PTQ 臂复现）、calib（全 NVFP4 + GPTQ + 逐层 MSE 裁剪）、aggr（视觉塔与动作头全 NVFP4，仅 o_proj/down_proj 保 FP8、解码器保 BF16）、fp8（backbone per-channel FP8 + 动作头全部 NVFP4）、mixed（复刻 NVIDIA 分配）。全部经 mini 协议筛除零臂后，mixed 与 fp8 基座进入全量协议。**结果**：二分给出方向性事实——head-only FP4 = 71.2%（可用），backbone-only FP4 = 0%。五臂（mini）：rtn 0%、calib 0%、aggr 0%、fp8 70.8%、mixed 93.3%；全量：mixed 97/102 = 95.1%、fp8 基座 46/106 = 43.4%。逐任务剖面（fp8 基座，全量）呈现清晰双峰：两个粗视觉运动任务保持满分（字母汤放入篮 10/10、书本入架 10/10），而精细操作坍塌（微波炉任务 1/16 = 6%，场景 5 双杯摆放 2/10，场景 8 双摩卡壶 3/10）。mixed 的唯一失分同样在微波炉任务（7/11）。**解读**：(i) **校准救不了放置**——calib 把逐层输出 MSE 压到 rtn 的 76%，闭环仍为 0%；aggr 只保护两个投影仍为 0%（视觉塔 NVFP4 致死）；而 mixed 不动校准、只动分配即达 95.1%。在 4-bit 格式下，误差放在哪些模块比误差总量更具决定性，这是 §3.3"复合效应取决于拓扑位置"的直接证据。(ii) 五臂差分联合两轮二分给出**模块敏感度地图**：NVFP4 致死模块 = 视觉塔 ≈ DiT 注意力投影 ≈ LLM 的 o_proj/down_proj；幸存模块 = LLM 主投影（q/k/v/gate/up）与全部 FFN——与 NVIDIA 配方的保护对象独立吻合，也与旧二分（head-FP4 71.2%、backbone-FP4 0%）交叉一致。机制上可理解为：视觉特征与注意力输出经 KV cache / 交叉注意力长程复用，误差按 §3.3 的链条平方复合；FFN 逐 token 局部作用，误差不跨步传播。(iii) fp8 基座的双峰剖面是后续恢复实验的完美测试床：损伤是**任务依赖**的（不是全盘退化），恢复必须修复的是精细操作的闭环几何，而非全局幅值。

## 4.5 量化域恢复：三臂对照

**目的**：在 43.4% 的 2.44× 压缩基座上检验三种恢复信号——离线演示（QAD）、教师标签（OPD）、自身成功（RWR）——并把预算固定使三者可严格比较。**操作**：三臂共用加性 LoRA（r=32，动作头 252 个线性层，38.7M 可训练参数，其余冻结），AdamW 1e-4，500 梯度步、batch 16。QAD 与 OPD 用演示数据流；OPD 额外以探针缓存 teacher-KL（权重 1.0，每 4 步计一次）；RWR 先以基座策略采集 15 episodes 的带日志 rollout（25 分钟），再对成功窗口做任务成功率加权的 flow-matching BC（其 0/1 终端奖励下的 REINFORCE 退化形式，§2.5）。训练后合并低秩残差（不重量化）、以全量协议闭环。**结果**：训练动力学上三臂都"看起来在学"：QAD 演示损失两段式下降（0.233→0.055 于前 100 步，此后 0.015–0.025 平台）；OPD 的 teacher-KL 单调衰减 **0.186→0.0074（25×）**，前 200 步降一个数量级后在第 400 步附近进入 0.0075 平台——学生在量化域内向 BF16 教师收敛的直接证据；RWR 的自模仿损失同样从 0.024 降到 0.010。闭环结果则尖锐分化：**QAD-LoRA 100/101 = 99.0%，OPD-LoRA 102/103 = 99.0%（双双超过 BF16 基线 96.7%）**；**RWR 与基座逐任务完全相同**（字母汤 10/10、灶台 4/10、微波炉 1/16——无一任务变化）。恢复的解剖集中在基座最深的伤口：微波炉任务从 6% 恢复到 100%（两臂同达），场景 5 与场景 8 从 20%/30% 回到满分或 90–91%；QAD/OPD 各自仅剩一次失败（分别 101 与 103 分之一，恰都在场景 8 双摩卡壶任务）。**解读**：(i) 三件事做对了（对应贡献亮点 2）：恢复发生在量化域内（训练=部署同一函数）；闭环为唯一裁判（若以训练损失判断，RWR 也是"成功"的恢复）；信号必须来自策略自身分布之外（演示或教师）。(ii) RWR 的零恢复不是实现失败而是理论预言的兑现：43.4% 的成功率意味着它的加权 BC 目标里 57% 的信息来自失败轨迹的缺失，且自模仿只能在行为策略支撑集内重新加权（§2.5 的 SIL 边界），而量化损伤恰恰把策略推到了支撑集外——它损失降到 0.010 学到的是"更好地复现自己已有的（不足）行为"。对照之下 OPD 的教师标签作用在学生自己的状态上（DAgger 式纠偏），QAD 的演示则提供分布外的动作几何参照，二者都能修复漂移项。(iii) 经济学：QAD/OPD 的恢复成本为 3.5 分钟训练 + 零采集；RWR 为同预算训练 + 25 分钟 rollout 采集——离线蒸馏在成本与效果两个维度同时占优。恢复的逐任务解剖（全量协议）：

| 任务（代表） | fp8 基座 | QAD-LoRA | OPD-LoRA |
|---|---|---|---|
| 微波炉放杯（最精细） | 1/16 = 6% | **10/10 = 100%** | **10/10 = 100%** |
| 双杯摆放（场景 5） | 2/10 | 10/10 | 10/10 |
| 双摩卡壶（场景 8） | 3/10 | 9/10 | 10/11 |
| 字母汤入篮（粗任务） | 10/10 | 10/10 | 10/10 |
| **合计** | **46/106 = 43.4%** | **100/101 = 99.0%** | **102/103 = 99.0%** |

![](images/ladder.png)

这一对照划定了在线方法的位置：当环境奖励可大量采样且无教师/演示时（本文设定之外），PPO 家族的完整形式（GAE + 裁剪 + 价值网络，DPPO 的 ELBO 代理）才是必要的。

## 4.6 方法论消融：离线指标的失效与 AWQ 的空转

**离线 proxy 为何不能用**。我们最初以"固定噪声、固定时间步的向量场单点对比"作臂间快筛，结果连闭环 95% 以上的臂都给出 relMSE ≈ 2、相关 ≈ 0；连 BF16 教师对自身缓存预测的自检也只有 0.10——诊断出 DiT 向量场对初始噪声的混沌敏感主导了单点对比。修复协议（固定生成器 + 固定 t=0.5）消除了 RNG 漂移，混沌性依旧：该指标对权重微扰的放大使任何臂间比较失去分辨率。结合 4.3 中"训练损失正常而闭环归零"与"离线 corr +0.139 vs −0.015 的两臂闭环同为近零"，结论是三重独立的：**VLA 量化的闭环损伤不可由离线指标预测**——这与 LLM 量化（perplexity 是有效 proxy）形成方法论分野，也是本文全部实验以全量闭环为唯一裁判的原因。

**AWQ 的空转**。AWQ 的逐通道缩放需要可折叠点，我们实现了完整折叠映射（含 GQA 共享 KV 通道、GeGLU 的 up 列折叠、SiLU-MLP 不可折叠点识别）并以 Hessian 恒等式精确搜索缩放指数 α——全部站点的最优 α 均为 0（即不缩放）。机制：NVFP4 的 per-16 块 E4M3 缩放已在块粒度吸收通道异常值，逐通道重排不改变块内相对结构。这与 NVIDIA 在 W4A4（激活共享粗粒度缩放）场景需要 AWQ 的事实互补而非矛盾：**AWQ 的作用对象是激活量化时的通道异常值，weight-only 场景下没有对象**。该负结果连同 GPTQ 的"MSE 降 24% 而闭环不动"，共同把本文的 PTQ 结论收窄为一句可操作的话：在 NVFP4 上，把工程预算花在精度分配与恢复训练上，而不是逐层校准上。

# 5. 讨论与结论

## 5.1 主要发现

**算子与引擎层**：以受控探针实验测定并逐字节验证了 cuBLASLt 的 NVFP4 块缩放物理布局，据此在 APXInf 引擎内交付了完整的 NVFP4 执行路径（CUDA 适配器、Rust 算子、工件加载器、混合拓扑执行器），四层组合正确性验证收敛于 corr=1.000000。算子特性上，NVFP4 在计算受限形状达 FP8 的 1.8×，而 batch-1 访存受限形状反而劣于 FP8——收益区间与精度分配互补。

**闭环实证（本文核心）**：(1) 全模型 NVFP4 PTQ 的闭环失效是系统性的——跨两代 VLA、跨引擎 kernel 与 PyTorch 两条实现路径一致归零，且 naive QAT 训练损失正常下降的同时闭环崩溃，证实"量化网格上的演示拟合"与闭环行为是两个目标（§4.3）。(2) 量化误差的**放置位置**比总量更具决定性：GPTQ 将逐层 MSE 压至 RTN 的 76% 仍闭环 0%，而复刻 NVIDIA 的混合分配（不动校准）即达 95.1%（2.02× 压缩，BF16 基线 96.7%）；五臂差分给出模块级敏感度地图（致死：视觉塔、DiT 注意力投影、LLM o/down_proj；幸存：LLM 主投影与全部 FFN，§4.4）。(3) 量化域恢复：43.4% 的 2.44× 基座经 3.5 分钟 500 步加性 LoRA 恢复到 99.0%（QAD 与 OPD 双臂，均超 BF16），teacher-KL 单调衰减 25× 给出收敛的直接证据；同预算在线自模仿（RWR）训练损失正常下降而闭环逐任务零变化——量化恢复的瓶颈在策略自身分布之外的监督信号（§4.5）。(4) 方法论：离线指标（逐层 MSE、向量场相关性）与闭环损伤混沌解耦，闭环成功率是 VLA 量化的唯一裁判（§4.6）。

## 5.2 APXInf 生态在本工作中的角色

本工作不是把 APXInf 当黑箱，也不是重造引擎——五点可验证的具体受益：(i) **位级语义基准**：我们的 PyTorch 量化器逐位对齐引擎工件转换路径（corr=1.000000、maxdiff=0.0），保证离线校准与恢复训练的数字就是部署 kernel 消费的数字；(ii) **真实 kernel 路径的先行验证**：π0.5 的 nvfp4_static 闭环（0/100 vs BF16 90.0%）在写任何校准代码之前就锁定了"全 NVFP4 RTN 致死"，直接决定了把火力投向混合分配与恢复而非纯校准；(iii) **serving 拓扑**：闭环评测跑在与 robo 层同构的策略服务器 + ZMQ rollout 栈上，全部实验臂零代码改动换入即测——评测即部署形态；(iv) **工件产线复用**：packed 工件转换器的 RMSNorm 折叠与行拼接为旋转类 PTQ 的离线折叠预留了工程位；(v) **性能参照系**：GR00T N1.7 引擎 33.7 ms / 29.7 Hz 给出量化部署的延迟基线。诚实边界：GR00T 侧的校准与恢复实验目前跑在 PyTorch 量化语义（烤盘注入）上，引擎侧 GR00T fp4 kernel 化是下一步工作；π0.5 侧的引擎闭环已经证明该迁移路径可行。

## 5.3 工程贡献

cuBLASLt CUDA 13 API 适配与布局探针测定方法（含全套 spike 工具）；FP8 在 sm_120 不可用性的根因定位（E4M3 GEMM 输出 dtype 不支持，探针铁证）；1387 行可上游引擎补丁与打包工件产线；校准 PTQ 套件（Hessian 采集 / GPTQ-NVFP4 / AWQ 折叠映射 / 五臂烤盘）；24GB 无 FSDP2 的量化域 LoRA 恢复栈（0.43 秒/步，170× 于逐前向量化合并）；全部 checkpoint、配方与一键复现脚本。

## 5.4 局限

单一 GPU 型号（RTX 5090 Laptop），笔记本 TGP 波动如实注明，绝对值为该机持续值；GR00T 引擎侧 fp4 kernel 化未完成（§5.2 边界）；RWR 对照臂是 REINFORCE 退化形式（0/1 终端奖励、无优势估计与价值网络），其全量协议与 DPPO 式完整实现留作扩展；nvfp4 的 gate_up 融合 GEMM 未实现（当前朴素 GEMM + 独立激活核，性能次优）；语言/动作层 packed 工件的 vision 部分仅 qkv 打包。

## 5.5 结论与展望

NVFP4 在消费级 Blackwell 上的 VLA 部署在算子、引擎、校准与后训练恢复四层均已打通：**混合 FP8/NVFP4 分配给出 2.02× 压缩下 95.1% 的即用方案；把动作头整体推进 NVFP4（2.44×）后，量化域 LoRA 以分钟级成本把 43.4% 恢复到 99.0%**。完整闭环阶梯（0 → 43.4 → 95.1 → 99.0）与模块敏感度地图为 VLA 的 FP4/FP8 部署提供了可复现的配方与边界，离线蒸馏对在线自模仿的经济学对照（恢复效果与采集成本双维度）为恢复方法选择给出了直接依据。未来工作：GR00T 引擎侧 fp4 kernel 化与 mixed 工件镜像、旋转类 PTQ（QuaRot/SpinQuant 族）经工件产线的离线折叠、更小 LoRA 预算下 QAD 与 OPD 的分离扫描、完整 PPO/DPPO 对照、真机部署验证。

---*本工作在单张 RTX 5090 Laptop（WSL2）上完成；并行工作（FSDP2 CPUOffload 单卡全权重训练）共享同卡与 checkpoint，交叉引用。*

# 附录 A. 核心源码走读与 APXInf 框架解析

本附录面向希望复现或扩展本工作的读者，分两部分：A.1 解释 APXInf 引擎的代码框架——我们的全部引擎侧工作都在它的既有模式内完成；A.2–A.5 逐段走读我们实现的核心源码（引擎侧 CUDA/Rust 与 PyTorch 侧校准/恢复/评测栈），每段代码都说明"为什么这样写"。全部代码位于 fp4vla 仓库（`quant/ptq/`、`rl/`）与引擎补丁（`patches/apxinf-fp4vla-engine.patch`，1387 行，可对引擎主干一键应用）。

# A.1 APXInf 引擎代码框架解析

## A.1.1 三层 crate 拓扑

APXInf 是 Rust 编写的推理引擎，其 crate 分层决定了"加一种新精度/新模型"的全部工作量边界：

- **apxinf-core**：与设备无关的类型层——`Tensor`（dtype+shape+storage）、`DType`（F16/BF16/…）、`Shape`、统一错误 `Error`。任何算子签名都以 core 类型书写，因此上层模型代码不感知 CUDA 细节。
- **apxinf-cuda**：设备层——`CudaContext`（stream 句柄）、`CudaBuffer`（显存分配，`copy_to_host`/`ptr()` 回读与借用）、`kernels/`（Rust 算子入口）与 `ffi/`（对 C 接口的声明绑定）。**关键约束：`ffi` 模块是私有的**（`lib.rs` 里 `mod ffi;`），上层 crate 不能直接 import——所以我们的跨 crate 小工具（如 bf16↔f16 cast）以 `pub fn xxx_ffi(...)` 薄包装的形式放在 `kernels/fp4.rs` 里转发，这是引擎内跨 crate 暴露能力的既定模式。
- **apxinf-model**：模型层——每个支持的组织（π0.5、GR00T N1.7、Qwen3-VL、WALL-OSS…）一个模块，内含 weights 加载、blocks（执行拓扑）、runtime/executor（推理循环）。

CUDA 调用不走 Rust FFI 直连，而是经过 **适配器层**：`adapters/*.cu` 中的 `extern "C"` C++ 函数由 nvcc 编译，经 `build.rs` 的 `kernel_files` 显式列表链接进 crate，Rust 侧在 `ffi/cublaslt.rs` 声明签名。新增一个 GEMM 算子的完整集成链条是：写 adapter `.cu` → `build.rs` 注册 → Rust `ffi` 声明 → `kernels/` 包装 → 模型层调用。我们的 fp4 路径严格按此链条落地。

## A.1.2 精度 monomorphization：一种精度 = 一套 Blocks

引擎对精度的组织方式是**编译期单态化**而非运行期分派：以 π0.5 为例，`model/blocks/` 下每种精度一个完整实现（`bf16.rs` 621 行 / `fp8_static.rs` 1014 行 / `int8_dynamic.rs` 575 行），`ModelVariant` 枚举在 `model.rs` 的 `infer()/preprocess_rgb()` 等 match 点静态分派：

```rust
pub(in crate::pi05) enum ModelVariant { Fp8Static(Pi05Model<Fp8StaticBlocks>), Bf16(...), ... }
```

这套模式直接决定了我们的 fp4 实现形态：新增 `Nvfp4Static` 变体需要覆盖 `ModelVariant` 的**全部** match 臂（漏一处是 E0004 非穷尽错误），并且要么克隆一份完整 Blocks（fp8 的做法，千行级），要么做**组合式**设计。我们选择了后者（见 A.2.4）——这既是对引擎模式的遵循，也是对它的一次改进实验。

## A.1.3 工件与校准约定

引擎的量化模型是**离线工件 + 在线执行**：权重以量化格式离线转换成工件目录，加载器只上传字节、从不重量化。π0.5 的 FP8 profile 约定放在 `<model-dir>/calibration.json`（`load.rs` 的 `options.calibration_path` 或模型根自动识别，fp8_weights.rs 甚至带校准 JSON 的身份校验）。我们的 NVFP4 工件沿用并显式化这一约定：

```
<dir>/<stem>.packed.u8   # rows x K/2 字节，E2M1 对（偶数 k 在低 nibble）<dir>/<stem>.scale.u8    # cuBLASLt VEC16_UE4M3 物理 swizzle 布局<dir>/manifest.json      # 每张量 {shape, block=16, tscale, rel_frob}
```

**部分工件是合法状态**：豁免表（哪些张量留在 BF16）由工件的存在性表达——加载器查不到就回退 BF16。这个"存在即量化"的约定让混合精度部署不需要改任何引擎代码。

## A.1.4 Python 侧：robo 层与 serving 拓扑

APXInf-robo（RLinf/APXinf-robo）是模型之上的机器人层：`Gr00tPolicy` 家族用 `precision=/calibration=/embodiment=` 构造（π0.5 走 `AutoPolicy`+`model_variant=`），LIBERO 评测与 OpenPI 兼容 server 都在这层。本文的闭环评测栈与该拓扑同构：eval server（策略常驻、批处理观测）+ ZMQ rollout 客户端（MuJoCo/EGL 在独立 venv）——**评测即部署形态**，烤好的 PTQ/LoRA 合并 checkpoint 换入模型路径即可，零代码改动。引擎侧需在同一 venv 安装 NVIDIA gr00t 包（`--no-deps`）+ transformers 4.57.3。

# A.2 引擎内的 NVFP4 执行路径：五步源码走读

## A.2.1 CUDA 适配器：block-scaled GEMM（cublaslt_fp4_adapter.cu）

适配器是整条 fp4 路径里唯一"跟硬件对话"的文件。核心是三件事：GEMM 描述符的 scale 模式设置、CUDA 13 列主序布局适配、以及自持句柄：

```c
static cublasLtHandle_t fp4_lt_handle() {static cublasLtHandle_t handle = nullptr;static std::once_flag once;std::call_once(once, [] { cublasLtCreate(&handle); });return handle;}

extern "C" cublasStatus_t apxinf_fp4_gemm_f16(int m, int n, int k,const void* a_packed, const void* b_packed,     // E2M1 对，各 M*K/2 与 N*K/2 字节const void* a_scale_swz, const void* b_scale_swz, // 物理 swizzle 后的 E4M3 块缩放__half* c, void* workspace, size_t workspace_bytes, cudaStream_t stream){cublasLtMatmulDesc_t op; ...;cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof(nn));

cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_swz, ...);cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_swz, ...);

// CUDA 13：只有列主序布局；A 存 KxM (ld=k, opT 得 MxK)，B 存 KxN，C 为 MxN F16cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, k, m, k);cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, k, n, k);cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F,     m, n, m);...cublasLtMatmulAlgoGetHeuristic(fp4_lt_handle(), op, la, lb, lc, lc, pref, 1, &heur, &nres);cublasLtMatmul(fp4_lt_handle(), op, &one, a_packed, la, b_packed, lb, &zero,c, lc, c, lc, &heur.algo, workspace, workspace_bytes, stream);}
```

写这段代码时踩过的坑值得记录：(i) CUDA 13 移除了行主序布局常量，全部按列主序重新推导存储方向——权重 `[N,K]` 行主序与 `[K,N]` 列主序 ld=K 是同一段字节，这是让 HF 转置习惯与 cuBLASLt 共存的关键；(ii) scale 指针与 scale 模式是描述符属性而非布局属性，设错位置会静默落到逐 tensor 缩放；(iii) `CUBLAS_COMPUTE_32F + CUDA_R_32F` 的组合是该 scale 模式下唯一可用的高精度路径。

**物理 swizzle 布局**——cuBLASLt 要求块缩放字节按固定物理公式排布（§3.1 的探针测定过程），适配器与离线转换器共用同一公式：

```
PS = 512 * ceil(KB/4)off(r, b) = PS*(r/128) + 512*(b/4) + 16*(r%32) + 4*((r/32)%4) + (b%4)
```

## A.2.2 在线激活量化核：与 CPU 参考逐位一致

适配器里的第二个导出函数把 F16 激活在线量化成 NVFP4（packed + swizzled scale）。这段代码的目标不是"快"而是**逐位确定**——它必须与离线转换器（`fp4_quant.py`）产出完全相同的字节，否则在线路径与离线工件的数值语义就分裂了：

```c
__device__ __forceinline__ uint8_t fp4_enc_e4m3(float v) {if (v >= 448.0f) return 0x7E;              // satfiniteif (v <= 0.0f)   return 0;float m = v, e = 0;while (m >= 2.0f) { m *= 0.5f; e += 1.0f; } // 规格化到 [1,2)while (m <  1.0f) { m *= 2.0f; e -= 1.0f; }int exp = (int)e + 7;if (exp <= 0) { /* 次正规格上的 RNE（ties-to-even） */ ... }float gm = (m - 1.0f) * 8.0f; int fl = (int)gm; float frac = gm - fl;int man = frac > 0.5f ? fl + 1 : frac < 0.5f ? fl : (fl % 2 == 0 ? fl : fl + 1);if (man > 7) { man = 0; exp += 1; }return (uint8_t)((exp << 3) | man);}

__global__ void nvfp4_quantize_activation_kernel(const __half* __restrict__ x, uint8_t* __restrict__ packed,uint8_t* __restrict__ scale_swz, int rows, int kb){int tid = blockIdx.x * blockDim.x + threadIdx.x;int r = tid / kb, b = tid - r * kb;        // 每线程负责一个 16 元素块float vals[16], amax = 0.0f;for (int j = 0; j < 16; ++j) { vals[j] = __half2float(blk[j]); amax = fmaxf(amax, fabsf(vals[j])); }uint8_t sb = fp4_enc_e4m3(amax * (1.0f / 6.0f));   // 块缩放：amax/6 编码为 E4M3float sd = fp4_dec_e4m3(sb);for (int j = 0; j < 16; j += 2) {float q0 = min(fp4_e2m1_mag(fabsf(vals[j]) / sd), 6.0f);  // 除法（非倒数乘）...  // E2M1 阶梯码 + 符号位，两值装一字节（偶数 k 低 nibble）}size_t off = PS*(r/128) + 512ull*(b/4) + 16ull*(r%32) + 4ull*((r/32)%4) + (b%4);scale_swz[off] = sb;                       // 直接写入物理 swizzle 位置}
```

两个刻意的设计决定：(i) **E4M3 编码用手写位级函数而非 `__nv_cvt_float_to_fp8`**——硬件内建函数的舍入模式与 CPU 参考不同，我们要的是 RNE-on-log-lattice 的精确复刻；(ii) **归一化除法而非乘倒数**——`x / sd` 与 `x * (1/sd)` 在 fp32 下给出不同舍入，注释里明确标注这是为了匹配 CPU 路径。E2M1 阶梯函数 `fp4_e2m1_mag` 用显式分支处理全部中点（0.25→0、0.75→1.0、1.75→2.0、3.5→4.0、5.0→4.0），保证 ties-to-even 的**码字**序而非数值序。这些细节共同保证了 kernel 与 Python 金标准逐字节一致（gold_check4 corr=1.000000）。

## A.2.3 Rust 算子与工件加载器

`kernels/fp4.rs` 的 `fp4_linear` 是模型层看到的唯一入口——借用工件缓冲的零拷贝视图、在线量化激活、调用 GEMM：

```rust
pub struct Fp4WeightView<'a> {pub packed: &'a CudaBuffer,   // E2M1 对pub scale:  &'a CudaBuffer,   // 物理 swizzle 字节pub rows: usize, pub k: usize,}

pub fn fp4_linear(ctx: &CudaContext, x: &Tensor, w: &Fp4WeightView<'_>) -> Result<Tensor> {let (m, k) = (dims[0], dims[1]);let packed_bytes = CudaBuffer::alloc_on(ctx, m * k / 2)?;let kb = k / 16;// 物理布局的缓冲尺寸公式：512*ceil(kb/4) * ceil(m/128)let scale_bytes = 512 * ((kb + 3) / 4) * ((m + 127) / 128);let scale_buf = CudaBuffer::alloc_on(ctx, scale_bytes)?;unsafe { ffi::apxinf_nvfp4_quantize_activation(x_buf.ptr(), packed_bytes.ptr(),scale_buf.ptr(), m as i32, k as i32, stream) };unsafe { ffi::apxinf_fp4_gemm_f16(m, w.rows, k, packed_bytes.ptr(), w.packed.ptr(),scale_buf.ptr(), w.scale.ptr(), out.ptr(), ws.ptr(), ws.len(), stream) };Ok(out.into_tensor(Shape::new(vec![m, w.rows]), DType::F16))}
```

`pi05/fp4_weights.rs` 加载器只做三件事：解析 manifest、按 stem 约定定位文件、`CudaBuffer` 上传——**注释里写明"工件不可变（QAD 冻结 scale），本加载器永不重量化"**，这是引擎"离线工件/在线执行"哲学的贯彻。它还带真实工件的设备往返回归测试（逐字节比对，环境变量门控），以及前述"部分工件合法"的豁免表语义。

## A.2.4 混合拓扑执行器：组合式 Fp4Blocks 与逐站路由

fp8_static.rs 的做法是克隆整个执行拓扑（千行级）。我们没有克隆，而是组合：`Fp4Blocks { inner: Bf16Blocks, fp4: Fp4TensorMap }`——视觉前处理、嵌入、KV cache 类型全部委托给既有 BF16 实现（`vision_layer_bf16` 等自由函数直接复用），只在 GEMM 站点插入路由。路由的心脏是这一个函数：

```rust
/// 混合拓扑的路由心脏：权重有工件张量则走 fp4 GEMM，否则 BF16 直通（豁免语义）。pub fn gemm_maybe_fp4(ctx: &Context, x_bf16: &Tensor, weight_bf16: &Tensor,fp4: Option<&Fp4DeviceWeight>,) -> Result<Tensor> {let Some(w4) = fp4 else {return kernels::gemm::bf16(ctx, x_bf16, weight_bf16);   // 豁免层：BF16 直通};let x_f16 = bf16_to_f16(ctx, x_bf16)?;   // fp4 站点付一次 dtype 转换let view = Fp4WeightView { packed: &w4.packed, scale: &w4.scale, rows: w4.rows, k: w4.k };kernels::fp4::fp4_linear(ctx, &x_f16, &view)  // 输出 F16（引擎 dtype 边界）}
```

语言层与动作层的每个 Linear 站点都换成 `gemm_maybe_fp4(...)`。两个非平凡的工程点：其一，引擎的 Gemma 执行器把 qkv 与 gate_up 存为 **PACKED 权重**（k/q/v 行拼接、dual-geglu 交错），而 HF checkpoint 是分开的张量——所以工件转换器（`quant/nvfp4_convert_packed.py`）必须**先按引擎内存布局拼接、再量化**，工件名用引擎内存名；RMSNorm 增益按输入通道折叠进 q/k/v（`w *= (1 + ln.weight)`）、gate/up 则乘 `(1 + post_ln.weight)`——这正是 §4.6 讨论的"变换折叠"在工件产线里的既有先例。其二，fp4 站点的 dtype 边界：引擎中间张量是 BF16，而量化核要求 F16 输入，故每个 fp4 站点付一次 bf16→f16 cast（输出侧同理），这在批注里如实标注为"豁免语义的代价"。

## A.2.5 变体接线与可见性

`load.rs` 与 `model.rs` 的接线是纯机械但坑密布的部分：`ModelVariantChoice::Nvfp4Static`（config.rs 的 as_str/from_str/resolve 三处）+ `ModelVariant::Nvfp4Static` 第四变体（model.rs 的全部 match 点）+ load.rs 分支（`fp4/` 目录约定或 `LoadOptions.fp4_artifact` 显式路径）+ builder `build_nvfp4_static_model`。引擎的封装边界要求解锁六处可见性（`CudaBuffer::as_ptr`、`into_tensor`、`Bf16Blocks::ctx` 等）——这些是"向引擎添加精度"的固有摩擦，patches 文件里逐处可见。

# A.3 PyTorch 侧：校准 PTQ 套件（quant/ptq/）

## A.3.1 校准采集器：精确 Hessian 的一钩子采集（collector.py）

GPTQ 与逐层 MSE 评估都需要每层的二阶统计。我们没有存激活本身（189 层 × 43k 行会占数 GB），而是利用恒等式 **Σᵢ‖xᵢΔWᵀ‖² = tr(ΔW·H·ΔWᵀ)**（H=Σxxᵀ），只存 K×K 的 Hessian 与 K 维绝对均值——评估任意权重扰动对的输出影响是精确的而非采样的：

```python
def mk_hook(name):def hook(mod, args):x = args[0]K = mod.weight.shape[1]if K % 16 != 0: returnxf = x.detach().reshape(-1, K).float()e = acc[name]e["H"] = xf.t() @ xf if e["H"] is None else e["H"] + xf.t() @ xf  # fp32 累积e["abs"] += xf.abs().sum(0)e["n"] += xf.shape[0]return hook
```

`forward_pre_hook` 保证拿到的是线性层的**输入**（对于 q/k/v 就是各自 norm 之后的张量）；Hessian 在 GPU 上按批累积、结束时落盘 CPU fp32（5.49GB）。AWQ 的通道显著度（absmean）在同一钩子里免费获得。

## A.3.2 量化器与 GPTQ 的 NVFP4 块格式适配（quantizers.py）

`nvfp4_dequant` 与引擎侧参考（`torch_fp4.fake_quant_nvfp4_torch`）逐位一致（单测 maxdiff=0.0）——这是"离线校准的数字就是引擎 kernel 消费的数字"的保证。GPTQ 的适配难点在于 NVFP4 的缩放是**每 16 列一块、随块内当前值变化**的，而经典 GPTQ 假设逐列固定的均匀格。我们的做法：列仍从左到右处理，但**每进入一个新 16 列块，就从当前（已被误差反馈修改过的）权重值重算该块的逐行 E4M3 缩放**，再在该固定缩放下逐列量化-反馈：

```python
for i in range(K):if i % block == 0:                      # 块入口：从当前值重算块缩放blk = W[:, i:i+block]amax = blk.abs().amax(1) * clip     # clip<1 是学习裁剪旋钮（MSE 校准）ts = (amax.max() / 448.0).clamp_min(1e-30)      # per-tensor 二级缩放s_row = _quant_e4m3(amax / 6.0 / ts).clamp_min(1e-9) * tsx = W[:, i] / s_rowidx = torch.clamp(torch.searchsorted(mids, x.abs()), 0, 6)q = grid[idx] * torch.sign(x) * s_row  # E2M1 中点阶梯d = Hi[i, i]                            # Hessian 逆（Cholesky 上三角）err = (W[:, i] - q) / dW[:, i+1:] -= err.unsqueeze(1) * Hi[i, i+1:].unsqueeze(0)   # OBS 误差反馈W[:, i] = q
```

这就是正文 §4.4 的结论的出处：即便逐层输出 MSE 被压到 RTN 的 76%，全 NVFP4 的闭环仍为 0%——误差的**放置**比总量重要。

## A.3.3 配方烤盘：分配表即策略（bake.py）

五种配方是五个纯函数，把模块名映射到精度——NVIDIA 分配的复刻与我们的激进版只差几行：

```python
def alloc_mixed(name, W):            # NVIDIA mixed_nvfp4 复刻（2.02×）if name.startswith("backbone.model.model.visual."): return "fp8"   # ViT 全 FP8if name == "backbone.lm_head":                     return "fp8"if ".language_model.layers." in name:if name.endswith(("self_attn.o_proj", "mlp.down_proj")): return "fp8"  # 敏感投影保护return "nvfp4_gptq"                                  # LLM 主投影 NVFP4+GPTQif name.startswith("action_head."):base = name.split("action_head.model.")[-1]if base.startswith("ff.net.0.proj") or base == "ff.net.2": return "nvfp4_gptq"if base.startswith("transformer_blocks.") or base == "proj_out_1": return "fp8"return "bf16"                       # timestep 编码器/动作解码器不量化return "fp8"

def alloc_aggr(name, W):             # 本文激进方案（2.88×）：动作头整体 NVFP4...if name.startswith("action_head."):...  # 全部 nvfp4；仅 o_proj/down_proj → fp8return "nvfp4_gptq"              # 视觉塔也 NVFP4（实测：闭环致死）
```

而**恢复基座（2.44×）**就是"backbone 全 FP8 + 动作头全 NVFP4"——与 mixed 的差别恰是把 NVIDIA 保留 FP8/FP16 的动作头四类投影推进到 NVFP4（贡献亮点 1）。烤盘把量化值以 **place-value 形式写回 safetensors**（BF16 张量携带量化噪声），因此现有评测栈零改动即可闭环；压缩账目（nvfp4=0.5625 字节/参数含 E4M3 缩放、fp8=1 字节+行缩放）随 checkpoint 落盘成 `ptq_recipe.json`。

## A.3.4 AWQ 折叠映射：三个非平凡边界（folds.py）

折叠映射表覆盖了 GQA 注意力（o_proj 的通道缩放须折叠进 v_proj 输出列，且同一 kv 通道被 2 个 q 头共享——映射函数处理 head_dim 分解）、GeGLU（down_proj 缩放精确折叠进 up_proj 列，因为 y=gelu(g)·u 对 u 线性）、以及**不可折叠点**（视觉塔 SiLU-MLP 的 fc2：逐元素非线性阻挡精确折叠）。搜索判据用 A.3.1 的 Hessian 恒等式精确评估输出 MSE——结果是全部站点 α=0（§4.6 的负结果）。

# A.4 量化域 LoRA 恢复栈（rl/）

## A.4.1 加性 LoRA：训练前向与部署函数严格同一（lora_qad.py）

恢复训练的核心决定是把"量化"从训练循环里**请出去**：基座权重是烤好的量化值（不变），LoRA 残差是精确梯度的 BF16 低秩分支：

```python
def make_fwd(m, s):   # s = alpha/r# v2 加性语义：量化基座（烤在 m.weight 里，值永不再变）+ 精确梯度低秩残差。# 部署 = lora_merge_bake（W_baked + (B@A)*s，BF16 加法，不重量化）。def fwd(x):y = torch.nn.functional.linear(x, m.weight, m.bias)z = torch.nn.functional.linear(x, m.lora_A)     # (…, r)z = torch.nn.functional.linear(z, m.lora_B)     # (…, N)return y + (z * s).to(y.dtype)return fwd
```

三个由此而来的性质：(i) **无 STE 近似**——量化基座不需要梯度（冻结），残差路径是普通线性函数，梯度精确；(ii) **训练=部署**——合并只是把这个加法算一次（W_baked + BA·s），闭环评测的函数与训练完全一致；(iii) **原生速度**——0.43 秒/步，而"逐前向把 W+BA 量化合并"的朴素写法是 75 秒/步（170×），后者曾在 24GB 卡上把 500 步推到 10 小时量级。训练器通过替换 `Gr00tTrainer.__init__` 注入（构造后遍历模块装 LoRA、冻结基座），`QAD_OPD_KL_W>0` 时再包一层 `compute_loss` 做 probe 缓存 teacher-KL（BF16 教师探针 38 秒离线缓存，训练循环内只跑学生前向——这是对早期 twin-forward 方案在 WSL 上确定性崩溃的修复）。

## A.4.2 合并烤盘的教训（lora_merge_bake.py）

合并逻辑本身三行（全局加载全部 shard → `W_baked + (B@A)·α/r` → 按原索引回写），但它踩中了两个值得记录的坑：HF 保存的 LoRA 键是 `X.lora_A` 而 base 是 `X.weight`（查表要补后缀）；lora_A 与其 base 权重可能落在**不同 shard**（必须全局加载后合并再回写，逐 shard 处理会 KeyError 崩在半路）。

## A.4.3 RWR 在线对照臂（lora_rwr.py）

PPO 家族对照的落地是两阶段离线环：eval server 以 `FP4VLA_LOG_DIR` 记录批量 rollout（观测 JPEG 压缩 + 动作块），训练侧按任务日志 mtime 分段、按 slot 拼接 64 步目标、经 `apply_action`（含相对坐标转换与归一化，z-range 校验=1.00）注入训练批。GR00T 的统一动作空间是 132 维，libero 占据前 7 槽——这个映射不是从文档查的，而是从演示数据的方差结构测出来的（var>0 恰在 0..6）：

```python
def inject_action(batch, target):a  = torch.zeros(1, 64, 132, dtype=torch.bfloat16)mk = torch.zeros(1, 64, 132, dtype=torch.float32)a[0, :, :7]  = torch.from_numpy(target).to(torch.bfloat16)mk[0, :, :7] = 1.0inner = dict(batch["inputs"]) if "inputs" in batch else dict(batch)inner["action"], inner["action_mask"] = a, mkreturn inner
```

# A.5 闭环评测链：quantize-once 与零改动部署

早期按前向 fake-quant 的评测慢 ~50 倍并触发 rollout 端 ZMQ 超时连环崩。修复是 `rl/scoped_quant.py` 的 **quantize-once**：加载时把 in-scope 线性层权重一次性替换为 NVFP4 反量化值，前向恢复原生 BF16 速度——数值与逐前向完全等价：

```python
def mark_scope(model):with torch.no_grad():for name, mod in model.named_modules():if isinstance(mod, nn.Linear) and inscope(name) and ...:mod.weight.data = fake_quant_nvfp4_torch(w).to(w.dtype)nn.Linear.forward = _ORIG   # 恢复原生前向；权重已携带量化噪声
```

烤盘（A.3.3）是它的持久化版本。eval server（`run_gr00t_server_fp4vla.py`）在此之上只加三个环境变量钩子：`FP4VLA_QUANT/SCOPE`（量化注入）与 `FP4VLA_LOG_DIR`（A.4.3 的 rollout 日志）。于是全部实验臂——五臂 PTQ、LoRA 合并产物、RWR 产物——都以同一形态进入闭环：**换 checkpoint 路径，不换代码**。

# A.6 复现索引

| 组件 | 位置 | 一条命令 |
|---|---|---|
| 校准采集 | quant/ptq/collector.py | `.venv/bin/python .../collector.py`（16 批×8 窗，189 层 Hessian 5.49GB） |
| 五臂烤盘 | quant/ptq/bake.py | `bake.py --recipe mixed --out weights/ptq_bakes/gr00t_ptq_mixed` |
| QAD/OPD-LoRA | rl/lora_qad.py | `GR00T_BASE_CKPT=<fp8臂> QAD_STEPS=500 ... lora_qad.py`（OPD 加 `QAD_OPD_KL_W=1.0`） |
| LoRA 合并 | rl/lora_merge_bake.py | `--ckpt <checkpoint-500> --out <部署目录>` |
| RWR 臂 | rl/lora_rwr.py + rwr_chain.sh | 采集（`FP4VLA_LOG_DIR=...`）→ 训练 → 评测 |
| 闭环全量 | groot-fsdp2/run_libero_eval_fp4vla.sh | `bash run_libero_eval_fp4vla.sh <CKPT> <TAG> 10` |
| 引擎侧 | patches/apxinf-fp4vla-engine.patch | `git apply` 于引擎主干（1387 行，adapter/算子/加载器/拓扑/变体全链） |

引擎补丁的完整文件清单：`cublaslt_fp4_adapter.cu`（GEMM+量化核+cast 核）、`build.rs`/`ffi/cublaslt.rs`（注册与声明）、`kernels/fp4.rs`（fp4_linear+测试）、`pi05/fp4_weights.rs`（工件加载器+往返测试）、`pi05/model/blocks/fp4.rs`（Fp4Blocks 混合拓扑）、`pi05/config.rs`/`model.rs`/`load.rs`（Nvfp4Static 变体接线）、外加 fp8 在 sm_120 的输出 dtype 修复。每个文件的 spike/验证脚本在 fp4vla 仓库 `spike/` 与 `quant/` 下成对出现。