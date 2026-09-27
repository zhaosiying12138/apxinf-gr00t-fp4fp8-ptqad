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
