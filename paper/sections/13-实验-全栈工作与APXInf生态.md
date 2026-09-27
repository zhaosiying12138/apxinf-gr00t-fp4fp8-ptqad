# §13 实验：我们的全栈工作与 APXInf 生态的角色

> 本章回答两个问题：**我们做了什么**（量化 → 校准 → 后训练恢复 → 闭环评测的全链自研
> 栈），以及 **APXInf 帮到了什么**（位级语义基准、推理引擎、serving 拓扑、工件产线）。
> 立场声明：APXInf 是本工作的推理引擎底座而非量化方法来源——量化方法学（校准、GPTQ、
> 混合分配）是我们按 §12 的公开文献自研实现的，但每一步都锚定在 APXInf 引擎的真实
> 语义上，保证离线实验的数字就是部署路径的数字。

## 13.1 全栈工作清单（本会话 09-27/28 新增部分加粗）

**量化与校准（§12）**
- NVFP4/FP8-E4M3 weight-only 量化器（PyTorch 侧，与引擎工件路径位级一致）
- 校准激活采集器：189 个 backbone Linear 的精确 Hessian（16 批×8 窗口，43k 行/层）
- GPTQ 的 NVFP4 块格式适配（块 scale 进入时从当前误差补偿值重算）
- AWQ 折叠映射全套实现（GQA 共享通道、GeGLU 折叠、SiLU 不可折叠点）+ 全站点搜索
- 五臂配方烤盘：rtn / calib / fp8 / mixed（NVIDIA 分配复刻）/ aggr

**后训练恢复（§13.2，训练栈）**
- **QAD-LoRA**：merged-quant 语义（y = x·Q(W+BA)ᵀ per-forward），训练=部署同函数，
  24GB 无 FSDP2（对比旧全量头 QAD 需要 FSDP2+CPU offload+流式保存三件套）
- **OPD-LoRA**：+ probe 缓存 teacher-KL（38 秥 BF16 teacher 探针，防 twin-forward 崩溃）
- **RWR-LoRA**：PPO 家族对照（成功加权自模仿；DPPO 式精确似然留作 future work）
- LoRA 合并烤盘：量化域内合并（fake_quant(W+Δ)），产出标准 eval checkpoint

**推理与评测（serving 栈）**
- 闭环评测：LIBERO-10 × 10 episodes 全量协议 + 3×5 mini 快筛协议
- quantize-once 语义烤盘（§12 的持久化版本）
- eval server 的 rollout 日志钩子（RWR 数据采集）

**引擎侧（APXInf 内，此前会话）**
- fp4 GEMM adapter（cuBLASLt E2M1×E2M1→F16 + VEC16_UE4M3 swizzled scales，maxrel 4.8e-4）
- π0.5 nvfp4_static 全链（加载器/Fp4Blocks 混合拓扑/工件完整性测试）
- PACKED 工件转换器（两树布局 + RMSNorm 折叠，99 张量 7.11× 压缩验证）
- π0.5 双精度闭环：BF16 90.0% vs nvfp4_static 0/100（真实 kernel 路径）

## 13.2 恢复阶梯实验设计（QAD/OPD/RWR vs PPO 家族）

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

预期（待回填）：[TBD-QAD]、[TBD-OPD]、[TBD-RWR]；基座 70.8%，BF16 上界 96.7%。

**恢复臂数据（09-28 实测）**：

| 臂 | 训练时长 | mini 闭环（3×5） | 10×10 全量 |
|---|---|---|---|
| fp8 基座（无恢复） | — | 70.8%（微波炉任务 0.125） | [TBD-base] |
| QAD-LoRA | **3.5 分钟**（500 步 @0.43s，loss 0.233→0.055） | **15/15 = 100%**（微波炉 0.125→1.0） | [TBD-QAD] |
| OPD-LoRA | 同预算（teacher-KL 实测 0.186→0.135 下降） | **15/15 = 100%** | [TBD-OPD] |
| RWR-LoRA | 同预算 + 25 分钟 rollout 采集 | [TBD-RWR] | — |

三臂语义学注记：v2 加性 LoRA（y = x·W_qᵀ + x·Aᵀ·Bᵀ·s）等价于"量化基座 + BF16 低秩
修正分支"（SVDQuant/EoRA 家族的训练侧形态）——梯度经低秩路径**精确**回传（无 STE
近似），部署合并 = W_q + Δ（BF16 加法，不重量化），训练与闭环评测**严格同一函数**；
且前向恢复原生速度（0.43s/步 vs 逐前向全矩阵量化合并的 75s/步，170×）。

## 13.3 APXInf 具体帮到了什么（可验证清单）

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
   为旋转类 PTQ（QuaRot/SpinQuant 一族）预备了离线折叠的工程位——这是未来把 §12.5
   方法落进引擎的路径。
5. **性能基线**。GR00T N1.7 BF16 引擎 33.7ms/29.7Hz（sm_120 首批 GeForce 数据）给出了
   量化部署的延迟参照系。

诚实的边界：GR00T 的校准/后训练实验目前跑在 PyTorch 侧（量化噪声以烤盘形式注入），
引擎侧的 GR00T fp4 kernel 化（对应 π0.5 的 nvfp4_static 路径）是下一步工作；π0.5 侧
已完成的引擎闭环证明该迁移路径可行。

## 13.4 产出物索引

- 代码：`quant/ptq/`（采集/量化/GPTQ/AWQ/烤盘/评测）、`rl/lora_qad.py`、`rl/lora_rwr.py`、
  `rl/lora_merge_bake.py`、`rl/overnight_chain.sh`
- 校准数据：`/mnt/c/fq_ptq_calib/calib.pt`（189 层 Hessian 5.49GB）
- 烤好的 checkpoint：`weights/ptq_bakes/gr00t_ptq_{rtn,calib,fp8,mixed,aggr}`
- 闭环结果：`runs/eval_mini_{mixed,fp8,calib,aggr}`、`runs/eval_full_mixed`
- 复现：每臂一条 bake 命令 + 一条 eval 命令（§12.7 表可直接映射）
