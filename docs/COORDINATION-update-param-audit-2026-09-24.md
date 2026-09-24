# 协调更新：full-head 臂参数量实测修正（2026-09-24，sess_c1b7735a）

> 触发：FSDP2 侧 Phase 0 参数审计完成（safetensors header 级实测）。
> 影响：你方 full-head QAD/OPD 臂的显存/RAM 规划数字需上调。

## 实测事实（nvidia/GR00T-N1.7-3B，全部 BF16）
- 总参数 3455M；**冻结 backbone+lm_head 1835M（bf16 3.5GB 常驻）**
- **可训练 action head 1620M（bf16 3.1GB）**，细目：DiT 16 层 1083M、VL-SA 201M、
  action_encoder W2/W3 227M、state_encoder 55M、action_decoder 38M、其余 ~16M
- 明细数据：`C:\Users\Admin1\.zcode\workspace\default\phase0\param_audit.json`

## 对你方的直接影响
1. 此前"~300–400M、AdamW 状态 4–6GB"的估计偏低 **4 倍**：fp32 m+v 实为 ~13GB
   （若走 bf16-state 路径 ~6.5GB）。full-head 臂 offload 后 CPU RAM 需求 ~20–26GB
   （本机可用 47GB，满足但请纳入排期考虑，避免与数据预载叠加）。
2. E2 等待点不变，但 full-head 冒烟的 GPU 显存预期请按 1.62B 口径重估
   （FSDP2 侧 B₂ 搜索将给出实测上限）。
3. 单卡微基准（同机实测，可供你参考）：BF16 GEMM 峰值 98.5 TFLOPS；
   PCIe pinned H2D 38.2 / D2H 23.7 GB/s；**unpinned 仅 3.5–4.5 GB/s（10× 惩罚）**——
   你的 ApxInf 引擎若涉及 host staging，pinned memory 是硬前提。

## FSDP2 侧进度同步
- Phase 0：环境搭建中（uv 锁曾被你方 baselines 安装占用，torch 已入共享 cache）；
  GEMM/PCIe 基准与参数审计已完成；模型权重已就位。
- E2 里程碑不变，达成时会在 GPU-COORDINATION.md 宣布。
