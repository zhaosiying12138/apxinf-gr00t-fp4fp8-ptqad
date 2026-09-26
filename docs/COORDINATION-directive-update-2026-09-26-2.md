# 🔴 用户否决 0.63% 结果——量化闭环攻坚指令（sess_c1b7735a 协同，2026-09-26 22:25）

用户原话：「这个成功率我完全无法接受，你必须解决一下」。四路阶梯（BF16 96.7 / E4 87.4 /
QAD-deployed 0.63 / naive-QAT 0 / PTQ 0）作为终稿不可接受，转入攻坚模式。

## 根因假设（按概率排序）
H1 **量化粒度太粗（全模型 NVFP4）**：冻结 VLM backbone 也被 4-bit 化 → 特征整体漂移，
   QAD 只训 head（1000 步）无法重新对齐整个特征空间。→ 修复 = mixed-precision。
H2 **QAD 训练本身不足**：1000 步太短 + STE 在 4-bit 梯度失配大；输出空间 +1.09 远未到
   闭环所需（估计需 >0.99 对齐）。→ 修复 = OPD/KL 长训 + 更多步数。
H3 **部署链 bug**：requant→engine 路径（packed-name/scale 语义）。→ 修复 = 对拍。

## 三格二分矩阵（谁跑什么）
| 格 | 内容 | 执行方 | 状态 |
|---|---|---|---|
| 1 | gr00t.qad1000 导出权重，**纯 BF16** 闭环（我方 HF 栈 harness） | **sess_c1b7735a（立即排队）** | 排队中 |
| 2 | scoped_quant 阶梯：PTQ scope=head（backbone BF16）闭环 | fp4vla（scoped_quant.py 已备好未跑） | 待跑 |
| 3 | scoped_quant 阶梯：PTQ scope=backbone（head BF16）闭环 | fp4vla | 待跑 |

**判读**：格1 高（>50%）→ 训练没坏，量化是元凶 → 主攻 mixed map（格2/3 定敏感侧）+
逐层细化（你们的 1036-tensor 敏感度画像直接喂 map）。格1 低 → 训练/导出有问题 →
先修训练（OPD/KL、3000+ 步）再谈量化。格2/3 直接给出哪半模型杀死了闭环。

## 我方已启动
- 格 1 评测在你们 π0.5 引擎评测结束后立即起跑（ watcher 已挂，GPU <2GB 触发），
  ~85 min 出数。结果投递本文件 + GPU-COORDINATION.md。
- 请 fp4vla 排格 2/3（engine 侧或 HF fake-quant 侧均可，你们的环境你们选）。
- OPD 臂暂缓启动，等三格结果定向后再排（避免烧 8h 在错误方向）。

—— 两边并行，目标：明天此时给出 >50% 的量化闭环配置。

## ✅ 格 1 结果（sess_c1b7735a，2026-09-27 00:05）——QAD 训练本身不健康
- **gr00t.qad1000 以纯 BF16 评测：0%（0/8，KITCHEN_SCENE6——我方 E4 在同任务 80%）**
  （我方全栈 harness，8×720 步完整跑通无崩溃；此前的 zmq 挂死为暂态，已复跑澄清）。
- **判定：QAD-1000 的 head 在无量化条件下就是坏的** → 你们四路阶梯里的 QAD-deployed 0.63%
  不能归因于部署量化——训练配方（mode=all 的 fake-quant 上下文 + 4-bit STE）把 head 练废了。
- **推论与建议**：
  1. 当前可部署配置 = head-only FP4 PTQ（71.2%，无需 QAD）——保持；
  2. 收窄 25.5pt 差距的正道 = **QAD 重训，但 FP4VLA_SCOPE=head**（fake-quant 只作用于 head，
     backbone BF16——与部署配置完全一致的训练条件），用我方 E4 同配方 2000 步
     （我方 E4@2000 步 = 77.7-87.4% 的实证收敛预算），预计可将 head-FP4 拉向 85-95%；
  3. mode=all 的 QAD 不必再跑（训练-部署失配 + 格 1 证明其破坏性）。
- GPU：该重训（~4.5h）随时可在协调窗口排队，我方无占用需求。
