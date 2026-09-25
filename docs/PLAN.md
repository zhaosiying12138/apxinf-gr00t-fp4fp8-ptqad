# FP4-VLA 执行计划（v1，2026-09-24）

目标：在 RTX 5090 Laptop（sm_120）上完成 GR00T N1.7 / π0.5 的 NVFP4 量化研究，
产出一篇中文论文（单 HTML + 知乎导出包）与完全可复现的开源仓库。

## 里程碑与当前状态

| # | 里程碑 | 状态 | 产出 |
|---|---|---|---|
| M0 | 环境 + 仓库骨架 | ✅ | setup/00-03 脚本 |
| M1 | FP4 spike go/no-go | ✅ **GO** | spike/ + docs/spike-notes.md：NVFP4 506 TFLOPS（1.8×FP8）；swizzle 布局解码；MXFP4 不支持 |
| M2 | 引擎 sm_120 编译 + π0.5 BF16 冒烟 | 🔄 编译中 | 首批 sm_120 引擎数据 |
| M3 | GR00T N1.7 BF16 跑通 | ⏳ 权重下载中 | 首批 GeForce GR00T 数据（业界首批） |
| M4 | NVFP4 引擎路径 | ⏳ | quant/：权重转换、swizzle 打包、激活校准（挂到引擎 FP8 校准框架）、E0 parity |
| M5 | 敏感性分析 + 混合精度 | ⏳ | exp/：两阶段协议（筛查 3任务×20eps → 决赛 10×50），失效模式分类，误差-成功率传播链 |
| M6 | 系统测量 | ⏳ | 延迟分解、batch 1/4/8/16 吞吐、功耗/能耗、显存 |
| M7a | 恢复阶梯 PTQ→QAD→OPD(+PPO 对照)，head-LoRA 主线 | ⏳ | QAD ~2h / OPD ~8h / PPO ~8h；实验矩阵 R0-R8；**不等待任何外部依赖** |
| M7b | full-weight action head 臂（**E2 门已过 2026-09-25 01:54，冒烟授权、排队 GPU 窗口**；**E4 已定稿 2026-09-25 05:00：正式实验授权**；推荐配方 AC+b≥32（峰值 14.3GB@b32，offload GPU 代价 ~1%，瓶颈 CPU 优化器 ~6s/step）；复用入口 ~/codebase/groot-fsdp2/Isaac-GR00T 分支 fsdp2-single-gpu，配方 --use-fsdp2 --no-fsdp2-reshard-after-forward --fsdp2-activation-checkpointing --no-fsdp2-pin-memory；FSDP2+CPUOffload，**实测 1.62B 可训练参数**：DiT 1083M+VL-SA 201M+act_enc 227M+state_enc 55M+act_dec 38M） | ⏳⏸ | **门控**：姊妹项目(groot-fsdp2) E2 完成→10-min 冒烟；E4 完成→正式实验；R5b 扩为容量阶梯（scale-only/LoRA-r8/full-head）；优化器锁 AdamW（Muon 禁入，归因隔离）；交付纯 NVFP4 权重（训 W+STE 直接导出）。**显存修正（2026-09-24 param_audit 实测）**：AdamW fp32 m+v ~13GB（bf16-state 路径 ~6.5GB），offload 后 CPU RAM 需求 ~20-26GB（本机可用 47GB，注意与数据预载叠加）；frozen backbone+lm_head 1835M（bf16 3.5GB 常驻） |
| M8 | 论文 + 发布 | ⏳ | paper/paper.html + paper/zhihu/；数据恢复脚本；上传 |

## 关键技术决策（已定；2026-09-24 按顾问建议审计修订，见 docs/ADVICE-AUDIT.md）

1. **量化方法 = PTQ**：权重 block scale 直接转换（免数据），激活 scale 用 LIBERO 数据校准。
2. **恢复阶梯 = PTQ → QAD → OPD（+PPO 对照臂）**：
   - QAD（teacher-forced 蒸馏，λ=0）：冻结 VLM backbone 与量化 scale，只训 action-head LoRA
     （稠密 VLA 无 MoE，"冻结 MoE 之外权重"按敏感性结果移植为"冻结 backbone"）；
   - OPD（在策蒸馏，λ>0）：fake-quant 学生 rollout + ApxInf BF16 teacher 逐步纠偏，
     向量场匹配损失（GKD 的 λ 混合 + reverse-KL 论证迁移到 flow-matching 动作分布）；
   - PPO 对照臂（RLinf）：检验稠密监督 vs 稀疏奖励样本效率、以及"超越教师上限"（H4）。
   数学原理、假设 H1-H5 与实验矩阵 R0-R8 见 paper/sections/02-恢复理论与实验设计.md。
3. **NVFP4 落点 = 引擎新精度模式**：模仿现有 bf16/fp8/int8 的 runtime/executor/weight 三元组
   （monomorphized，见 apxinf/doc/gr00t-n1.7.md），GEMM 走 cuBLASLt block-scaled，
   scale 上传用已解码的 swizzle 公式（docs/spike-notes.md）。
4. **混合精度假说（M5 验证）**：backbone NVFP4 + action head FP8/BF16。
   spike 证据：计算受限形状 FP4 1.8× FP8；batch-1 GEMV FP4 反而慢于 FP8。
5. **评测协议**：LIBERO-10，双视角，与引擎 PR #42 对齐（GR00T 状态约定 8 维含双 gripper）。
   成功率报 Wilson 95% CI；动作误差统一 on-policy 口径；状态访问分布 TV 距离双口径
   （本体直方图 + VLM 嵌入 Wasserstein）。
6. **测量协议**：分块跑（每块 ≤2h），全程 nvidia-smi 功耗采样，报持续性能并注明 TGP。

## 权重与数据（weights/，gitignore，setup/03 恢复）

- `nvidia/GR00T-N1.7-LIBERO`（主模型，LIBERO 微调版，libero_10/ 已就绪 6.5GB 精简版）
  + `nvidia/Cosmos-Reason2-2B`（backbone，**已解锁 2026-09-25 02:1x：token 到位（~/.cache/huggingface/token，姊妹会话放的），Cosmos-Reason2-2B 经 fetch_cosmos2.sh 拉取完毕。GR00T 全线待 GPU 窗口：PyTorch 基线 → 引擎 BF16 → NVFP4 转换**）
- `lerobot/pi05_libero_base` + openpi norm_stats.json（第二模型，ungated ✓）
- LIBERO 数据集：评测时经 apxinf-robo CLI 拉取（EGL headless 渲染）

### 2026-09-24 执行顺序调整（已被 09-25 Cosmos 解锁取代；π0.5 先行成果保留）

π0.5 路线前置（ungated）：引擎冒烟 → bench → PyTorch 基线 → NVFP4 转换全在 π0.5 上先做通；
GR00T 路线在拿到 HF_TOKEN 后立即恢复（权重本体已就绪，只差 backbone 快照）。
论文双模型结构不变，仅执行顺序对调。

## 论文骨架（中文，单 HTML + 知乎包）

标题候选：《FP4-VLA：面向视觉-语言-动作模型的 NVFP4 量化与混合精度部署》
1. 引言：边缘 VLA 部署约束；Blackwell 消费卡 FP4 普及；问题="VLA 能否跑 FP4"
2. 相关工作：VLA 模型；LLM 量化（GPTQ/AWQ/SmoothQuant）；FP4 格式（NVFP4/TRT-LLM/torchao）；具身推理引擎（ApxInf/Embodied.cpp/vla-perf）
3. 背景：NVFP4 格式；GR00T N1.7（Cosmos-Reason2-2B + DiT action head）；ApxInf 引擎
4. 方法：PTQ recipe（含 swizzle 打包细节——spike 的逆向解码本身是贡献）；混合精度准则；恢复阶梯；引擎集成
5. 实验：E0 parity / E1 主表 / E2 敏感性 / E3 恢复（= R0-R8 矩阵，理论见第 3 章草稿）/ E4 系统
6. 讨论与局限
- 图表计划：架构图（HTML/SVG）、swizzle 布局可视化（动画：逻辑→物理重排）、
  精度-延迟-成功率三元组图、batch 吞吐曲线、失败模式截图拼图、（动画）rollout 对比
- 知乎导出：paper/zhihu/ 内 markdown + 图片资源相对路径引用

## 风险与降级

- 引擎在 sm_120 编译失败/跑不动 → 只修到 BF16 可用；FP4 走"离线量化 + cuBLASLt standalone 推理原型"（spike 已验证全部算子）
- L2 RL 恢复超预算 → 论文重心移到 M5/M6（敏感性 + 系统），L2 只报趋势
- 官方抢先发 FP4 → 我们的 swizzle 解码 + GeForce 数据 + 传播链分析仍有独立价值

### 2026-09-25 发现：引擎 FP8 路径在 sm_120 损坏（新贡献点）

- fp8_static 加载成功（calibration.json 约定：放模型根目录），前向触发
  `cuBLAS error: status 15`（NOT_SUPPORTED），graph capture 降级 eager 后仍失败
- 对照：spike/fp4_gemm_bench.cu 里同机器 cuBLASLt FP8 E4M3 GEMM 稳定 268 TFLOPS
  → 引擎 fp8 GEMM 的 tactic/算法选择（kernels/gemm/fp8.rs 2211 行，硬编码 algo）
  在 sm_120 上选到了不支持的组合
- 修复路径已知：把 fp8 GEMM 路由到 spike 验证过的 cuBLASLt provider 模式
  （scale-pointer + heuristic），本身就是给上游的第二个 sm_120 修复 PR
- 论文含义：E1 表 sm_120 行 BF16 ✓ / FP8 ✗(修复中) / NVFP4(ours) ✓ —— 差异化更强
- FP8 校准管线已全通：calibrate_pi05.py (--libero-suite, 32 obs/10 tasks,
  256 scales, results/calib/pi05_fp8_libero10.json)；patches: precision→
  model_variant, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1（LIBERO 元数据加载）

### FP8 sm_120 修复补丁方案（待 GPU 空闲验证，~30min）

根因（spike/fp8_probe 铁证）：sm_120 cuBLASLt 无 E4M3 输出的 GEMM。
引擎唯一无条件 F8E4M3 输出点：`apxinf-cuda/src/kernels/gemm/fp8.rs:460`
`geglu_tuning_key()`（Thor 定制 tactic 遗产）。
补丁：`key.output_dtype = if sm==110 { F8E4M3 } else { F16 }`；
重编（cargo 增量 ~10min）→ fp8_static bench 复测 → 若过则 E1 表 FP8 行补齐 +
上游 PR。注意 autotune 失败时的 vendor fallback（cublaslt_fp8_gemm_f16）为何没接住
——重编后若仍挂，在 fallback 处加日志。

### FP8 修复进展（2026-09-25 03:0x）：一处已修，Vendor 默认路径仍 status 15

- geglu_tuning_key E4M3→F16 条件化补丁已编入并验证：**未修复主故障**
- fp8_emulation_required 只救 <sm_89；sm_120 走原生 Vendor 默认路径
  （resolve_fused_plan default=TacticBackend::Vendor）直接 cuBLAS status 15，
  且无 "tactic failed" 日志 → 故障在 vendor 直调的布局/算法配置，需运行时
  GEMM-key 日志定位（下一步：fp8.rs vendor 调用处加 eprintln 打印 key）
- 论文口径：FP8@sm_120 标"引擎原生不支持（根因已定位：E4M3 GEMM 输出，
  probe 铁证），我们的 NVFP4 路径从设计上规避该限制（输出 F16）"
- 优先级下调至 fp4 GEMM provider 之后

### 2026-09-25 08:13 full-head 冒烟#4 PASS（M7b 验证环节完成）

batch=1、1/1 step @91.7s：模型加载+可训练 1.62B(51.54%)+FSDP2(37 fully_shard
单元, pin=True, reshard=True)+NVFP4 fake-quant 前向/反向+checkpoint-1 保存全链
路通过（日志 ~/fq_smoke4.log，产物 ~/fq_smoke_out/checkpoint-1）。正式 QAD
（b32+AC，~1000 steps）已在协调表申请窗口。

### fp4 3b 关键发现（2026-09-25 11:2x）：打包名工件需求

引擎 Gemma 执行器的 qkv/gate_up 是 PACKED 权重（k+q+v 拼接 / dual-geglu
interleaved），而离线工件按 checkpoint 原始名（k_proj/gate_proj...）量化——
语言层核心 GEMM 站点无法直接路由。行动项：
1. nvfp4_convert.py 加 --packed 模式：读引擎 Bf16Weights 的加载代码确定
   packing 布局（weights.rs 的 qkv 拼接序 + geglu interleave 序），
   在量化前先做同样的拼接，工件名用引擎内存名。
2. action head（DiT）无打包（action_in_proj 等独立名）→ 可先路由，
   3b 的 action 分支先行。
3. 已入库：language_layer_fp4（BF16 镜像，等打包工件）、
   action_layer_fp4（stub）、Fp4Blocks 视觉路径路由（编译通过）。

### fp4 调试战报（2026-09-25 下午，进行中）

症状：nvfp4_static 前向链路通但动作 vs BF16 完全去相关（corr 0.11）。
已排除：
- 工件本身（单层 site-check：numpy GEMM corr 0.9957 = fp4 噪声期望值）
- tscale 双重量化缺陷（已修：转换器强制 tscale=1，与硬件一致）
- gate_up interleaved 布局（已加守卫；实测 interleaved=false 无影响）
- 形状/dtype/algo（C++ spike 64x256x512 F16-out PASS 4.8e-4）
- 路由命中（探针确认 qkv/gate_up 正确命中，形状 [K,N]vs[N,K] 数学等价）
- workspace 尺寸（64MB/256MB 无差）
已锁定：**Rust 侧组合调用**——fp4_linear 单测复现（maxrel=inf，out[0]=-5076，
量化中间产物正常），C++ 同形状同 dtype 通过。嫌疑：CudaBuffer 内存池
（cudaMallocAsync stream-ordered）与 cublasLt 的交互 / 流句柄 / 懒建句柄。
下一步：fp4_linear 内改用非池化 CudaBuffer::alloc 二分验证。

### fp4 调试结案（2026-09-25 17:4x）：组合路径无 bug，测试参照有 bug

gold_check4 铁证：以反量化 A×反量化 B 为参照，硬件 GEMM(A_fp4,B_fp4) 的
corr=1.000000、maxrel=4.83e-4（f16 舍入级）。此前全部"失败"为测试参照错误：
1. gold_check2/Rust gold 单测用原始激活做参照（混入 9% 激活量化噪声+小分母
   → 假 maxrel 1508）；
2. Rust 自写编解码器测试另有 codec bug（假 maxrel=inf）。
含 10× 逐块 scale 变化的四象限实验全部 PASS（VAR×VAR 等）——逐块 scale 语义
正确。C++/Python/kernel 编码仅 33/16384 tie 舍入差（无害）。

**引擎 corr=0.11 的重新解读：真实的 fp4 噪声复合**——36 层 × (权重+激活) 双
9% 噪声逐层复合。这正是论文的动机数据点（R0/敏感性分析的 PTQ 基线）：全量
fp4 PTQ 去相关 → 恢复流水线（QAD/OPD）的价值主张。nvfp4_static 引擎正确。
