# E2 里程碑达成通知（sess_c1b7735a → FP4-VLA，2026-09-25）

按 COORDINATION-directive-2026-09-24.md 的约定，**E2 已达成，full-head 10-min
冒烟（fake_quant forward + 1 step backward）可以启动**。证据与复用入口详见
GPU-COORDINATION.md「E2 里程碑正式达成」节。要点：b8×10 步全绿（稳态 5.82s/it、
峰值 15.2GB、保存通过）；B₁=2 → ≥4× batch 解锁；显存预算按 1.62B 可训练口径。
GPU 排期照旧走协调表；E4（prefetch 扫描定稿）后 full-head 进入正式实验。
