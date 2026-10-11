# W4A4 论文与证据包

这里保存离线论文 [`paper.html`](paper.html)、知乎稿 [`zhihu/article.md`](zhihu/article.md)、17 张 Ubuntu 执行截图、RTN W4A4 协议和逐回合证据。工程安装、编译、PTQ、QAD、OPD、闭环评测和发布命令以仓库根目录 [`README.md`](../README.md) 为准。

| 配置 | 闭环成功率 |
|---|---:|
| BF16 | 145/160（90.62%） |
| W4A4 PTQ | 138/160（86.25%） |
| PTQ + QAD | 145/160（90.62%） |
| continued-QAD | 104/160（65.00%） |
| PTQ + QAD + OPD | 144/160（90.00%） |

主结果来自 `final_results.json`、`paired_comparison.json`、`final_manifest.json` 和每个 held-out 臂的原始 rollout/server 日志。`evidence/action_diagnostics/` 保存可复算的完整动作诊断，`evidence/gptq_reference/` 保存绑定最终五臂清单的 GPTQ 未执行收据；不包含 GPTQ 结果。`validate_publication.py` 会拒绝缺失五臂、动作诊断或明确范围收据、旧协议 SHA、非 3840×2280 截图以及未完成状态。

截图登记见 [`evidence/captures.json`](evidence/captures.json) 与 [`evidence/retained_captures.json`](evidence/retained_captures.json)；图表和 HTML/知乎稿必须由根 README 的命令重建，不手工改生成文件。
