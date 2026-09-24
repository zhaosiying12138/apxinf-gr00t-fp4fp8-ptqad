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
| M7 | 恢复阶梯 L1/L2 | ⏳ | L1 校准式 scale 学习（天级）；L2 蒸馏+PPO 一天预算闭环（rl/） |
| M8 | 论文 + 发布 | ⏳ | paper/paper.html + paper/zhihu/；数据恢复脚本；上传 |

## 关键技术决策（已定）

1. **量化方法 = PTQ**：权重 block scale 直接转换（免数据），激活 scale 用 LIBERO 数据校准；
   恢复阶梯 L1（scale 学习）/ L2（蒸馏 + 小规模 PPO, LoRA r8 仅 action head）。
2. **NVFP4 落点 = 引擎新精度模式**：模仿现有 bf16/fp8/int8 的 runtime/executor/weight 三元组
   （monomorphized，见 apxinf/doc/gr00t-n1.7.md），GEMM 走 cuBLASLt block-scaled，
   scale 上传用已解码的 swizzle 公式（docs/spike-notes.md）。
3. **混合精度假说（M5 验证）**：backbone NVFP4 + action head FP8/BF16。
   spike 证据：计算受限形状 FP4 1.8× FP8；batch-1 GEMV FP4 反而慢于 FP8。
4. **评测协议**：LIBERO-10，双视角，与引擎 PR #42 对齐（GR00T 状态约定 8 维含双 gripper）。
5. **测量协议**：分块跑（每块 ≤2h），全程 nvidia-smi 功耗采样，报持续性能并注明 TGP。

## 权重与数据（weights/，gitignore，setup/03 恢复）

- `nvidia/GR00T-N1.7-LIBERO`（主模型，LIBERO 微调版）+ `nvidia/Cosmos-Reason2-2B`（backbone processor 快照，加载必需）
- `lerobot/pi05_libero_base` + openpi norm_stats.json（第二模型）
- LIBERO 数据集：评测时经 apxinf-robo CLI 拉取（EGL headless 渲染）

## 论文骨架（中文，单 HTML + 知乎包）

标题候选：《FP4-VLA：面向视觉-语言-动作模型的 NVFP4 量化与混合精度部署》
1. 引言：边缘 VLA 部署约束；Blackwell 消费卡 FP4 普及；问题="VLA 能否跑 FP4"
2. 相关工作：VLA 模型；LLM 量化（GPTQ/AWQ/SmoothQuant）；FP4 格式（NVFP4/TRT-LLM/torchao）；具身推理引擎（ApxInf/Embodied.cpp/vla-perf）
3. 背景：NVFP4 格式；GR00T N1.7（Cosmos-Reason2-2B + DiT action head）；ApxInf 引擎
4. 方法：PTQ recipe（含 swizzle 打包细节——spike 的逆向解码本身是贡献）；混合精度准则；恢复阶梯；引擎集成
5. 实验：E0 parity / E1 主表 / E2 敏感性 / E3 恢复 / E4 系统
6. 讨论与局限
- 图表计划：架构图（HTML/SVG）、swizzle 布局可视化（动画：逻辑→物理重排）、
  精度-延迟-成功率三元组图、batch 吞吐曲线、失败模式截图拼图、（动画）rollout 对比
- 知乎导出：paper/zhihu/ 内 markdown + 图片资源相对路径引用

## 风险与降级

- 引擎在 sm_120 编译失败/跑不动 → 只修到 BF16 可用；FP4 走"离线量化 + cuBLASLt standalone 推理原型"（spike 已验证全部算子）
- L2 RL 恢复超预算 → 论文重心移到 M5/M6（敏感性 + 系统），L2 只报趋势
- 官方抢先发 FP4 → 我们的 swizzle 解码 + GeForce 数据 + 传播链分析仍有独立价值
