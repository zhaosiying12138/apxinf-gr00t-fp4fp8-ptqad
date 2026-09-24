# 计划调整指令（跨会话协调，2026-09-24）

> 来源：用户授权的协调会话。用户已确认「两篇独立论文 + 协作」方案（否决三合一）。
> 本文件即决策记录，读完后按此调整计划，并将相关变更同步进 docs/ADVICE-AUDIT.md、
> 实验矩阵与 `C:\Users\Admin1\.zcode\workspace\default\GPU-COORDINATION.md`。

## 决策变更
训练配方从「冻结 VLM backbone + action head LoRA(r8)」调整为**两阶段容量策略**：

1. **第一阶段（立即执行，不受任何外部依赖阻塞）**：维持 head-LoRA(r8) 主线，
   跑通 R0–R8 全流水线（PTQ → QAD → OPD）。这是论文的最小可发表单元。
2. **第二阶段（等待协作方就绪）**：新增 **full-weight action head** 训练臂，
   ~300–400M 可训练参数，AdamW 状态 4–6GB，需 FSDP2 + CPU offload 才能在
   单张 24GB 5090 上跑起来。基础设施来自姊妹项目（GR00T FSDP2 论文，
   WSL repo `~/codebase/groot-fsdp2/Isaac-GR00T`，已有 `gr00t-patches/fsdp2.py` 初版）。

## 等待点（写入 GPU-COORDINATION.md 和实验矩阵）
- full-head 臂**在姊妹项目完成 E2（FSDP2 offload 基本可用 + B₂ batch 解锁确认）
  之前不得启动**；E2 后先做 10-min 冒烟（fake_quant forward + 1 step backward）。
- 姊妹项目完成 **E4（offload 流水线 prefetch 扫描定稿）** 后，full-head 臂进入
  正式实验。
- 在此之前 R0–R8 的 LoRA 路线照常推进，不等待。

## 实验矩阵调整
- R5b 扩展为**容量阶梯消融**（三档）：仅 scale 修正 → head-LoRA(r8) →
  full head（FSDP2+offload）。若 full-head 在闭环成功率上显著胜出，
  主路径切换为 full-head，R5b 升级为主结果之一。
- **优化器一律锁定 AdamW**，Muon 禁止进入本论文任何主表/消融——理由：
  优化器变更会污染「恢复增益来自流水线」的归因（confound）。
  Muon 属于姊妹论文的贡献，仅在其论文中讨论。

## 交付物变化（full-head 情形）
- full-head 训练直接更新 W + STE，导出即**纯 NVFP4 权重**，无 LoRA 旁路
  张量，引擎端直接部署——这是 full-head 相比 LoRA 的核心工程论据，写进论文。
- LoRA 情形沿用 Q(W+BA) 折叠后重量化方案，作为对照保留。

## 工程复用
- 姊妹项目的 `phase0/param_audit.py` 改造为本项目的冻结断言脚本：
  自动验证 scale 冻结、backbone 冻结、可训练参数计数（QAD 的定义性约束）。
- 权重同源：`weights/GR00T-N1.7-LIBERO` 两项目共用。
- π0.5 侧 full-head 无现成 patches，排在 GR00T 之后。

## 论文定位
独立成篇。FSDP2/offload 作为「系统使能技术」一节 + 引用姊妹论文；
不与 FSDP2/Muon 合并成一篇。请更新 docs/ADVICE-AUDIT.md、实验矩阵、
假设列表（新增 H6：容量阶梯与闭环恢复的关系）和 GPU-COORDINATION.md。
