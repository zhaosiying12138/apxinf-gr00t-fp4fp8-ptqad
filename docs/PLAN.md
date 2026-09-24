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
| M7b | full-weight action head 臂（FSDP2+CPUOffload，**实测 1.62B 可训练参数**：DiT 1083M+VL-SA 201M+act_enc 227M+state_enc 55M+act_dec 38M） | ⏳⏸ | **门控**：姊妹项目(groot-fsdp2) E2 完成→10-min 冒烟；E4 完成→正式实验；R5b 扩为容量阶梯（scale-only/LoRA-r8/full-head）；优化器锁 AdamW（Muon 禁入，归因隔离）；交付纯 NVFP4 权重（训 W+STE 直接导出）。**显存修正（2026-09-24 param_audit 实测）**：AdamW fp32 m+v ~13GB（bf16-state 路径 ~6.5GB），offload 后 CPU RAM 需求 ~20-26GB（本机可用 47GB，注意与数据预载叠加）；frozen backbone+lm_head 1835M（bf16 3.5GB 常驻） |
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
  + `nvidia/Cosmos-Reason2-2B`（backbone，**gated 仓库——需要用户接受许可并提供 HF_TOKEN，当前硬阻塞 GR00T 全线**）
- `lerobot/pi05_libero_base` + openpi norm_stats.json（第二模型，ungated ✓）
- LIBERO 数据集：评测时经 apxinf-robo CLI 拉取（EGL headless 渲染）

### 2026-09-24 执行顺序调整（Cosmos gated 阻塞）

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
