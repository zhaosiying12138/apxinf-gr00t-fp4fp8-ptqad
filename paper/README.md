# 中文论文审阅入口

**v12 五臂正式评测与完整动作诊断已完成并冻结；当前仍是审阅稿。七张发布截图待补，尚未通过最终发布校验。** 审阅包已包含完整新实验结果，七张旧截图及其来源附件均已排除。校准 GPTQ 对照未执行，不能据此主张超过先进 PTQ。

- [下载科学结果审阅包（非最终发布包，7 张截图待补）](https://github.com/zhaosiying12138/apxinf-gr00t-fp4fp8-ptqad/raw/refs/heads/main/paper/apxinf-gr00t-fp4fp8-ptqad-review.zip)：解压后打开 `paper/paper.html`，科学证据、图表和复现文件随包提供。
- [离线 HTML](paper.html)：中文正文、公式、图表、源码解读与复现附录。
- [知乎 Markdown](zhihu/article.md)：与 HTML 共用正文，已展示图片位于 `zhihu/images/`。
- [根 README](../README.md)：项目意义、完整指标、依赖、安装、原生编译、训练评测、截图登记与打包命令。
- [最终实验清单](evidence/final_manifest.json)、[五臂结果与配对统计](evidence/final_results.json)、[逐回合配对记录](evidence/paired_comparison.json)。
- [动作诊断](evidence/action_diagnostics/summary.json)、[编码预算](evidence/recipe_inventory.json)、[训练成本](evidence/training/costs.json)、[搜索与数据成本](evidence/search_costs/summary.json)、[环境和源码身份](evidence/runtime/manifest.json)。

## 已完成的实验结论

BF16、RTN W4A4 PTQ、QAD、continued-QAD 和 QAD+OPD 分别为 **145/160、138/160、145/160、104/160、144/160**。QAD 相对 PTQ 增加 **4.375 个百分点**，总成功率点估计回到 BF16 的 **90.625%**；这一差异的配对区间包含零，尚不能确认总体改善。

OPD 为 **90.000%**，未超过 QAD；相对同演示更新预算的 continued-QAD 则高 **25.000 个百分点**，95% 配对区间为 **[18.125, 31.875]**，Holm p=**9.16746×10⁻⁸**。这支持本设置下教师监督相对纯演示续训更好地保留已有闭环表现，不能解释为从 continued-QAD 检查点顺序恢复，也不证明跨训练种子稳定性或未见任务泛化。

主分支加独立 BF16 LoRA 的净参数编码预算为 **3.276×**。GR00T 的行为结果来自 Torch QDQ 数值模拟；该预算不等于当前 checkpoint 文件体积、实测显存或原生 APXInf 端到端加速。完整动作诊断已归档，使用十任务各八个训练分布观测，不能代替闭环成功率。

## 截图与发布状态

保留全部 **17 个 Ubuntu 截图环节**。十张未受本轮恢复实验影响的实拍继续展示；量化、教师演示、QAD、学生采集、教师缓存、OPD、策略服务七个环节保留文字图位，不展示对应旧图。待补截图只拍摄对已完成原始日志和产物的 CPU 核验，不重新训练、量化或评测。

科学结果与截图发布是两个不同的完成条件：`evidence/` 已安装当前 v12 的完整证据，正文中的成功率、动作指标和成本均为实测或明确标识的编码核算；截图登记未完成，因此不能将本稿称为最终发布包。

## 重建当前审阅格式

从仓库根目录执行；依赖安装见根 README。下面仅从当前已核验的正文、图表数据和截图登记重建 HTML、知乎 Markdown 与浏览器检查，不运行模型，也不生成 ZIP。`meta.json` 应继续明确标识审阅稿，七个 `refresh_pending` 图位保留不变。

```bash
python3 paper/make_figs.py
bash paper/figs/render_pngs.sh
uv run --with-requirements paper/requirements-build.txt python paper/build_html.py
python3 paper/export_zhihu.py
node paper/qa_browser.cjs
```

旧的 `paper/build_review.py` 是实验尚未完成时的结构稿构建器，会要求数值占位，并生成占位结果图；它不适用于当前已完成数据的审阅稿，**不要用它重建本版**。当前正式生成器与截图门控均保持有效；上述格式重建不表示 `validate_publication.py` 或最终打包已经通过。

## 正式发布

本目录当前不提供正式发布 ZIP。实验已经完成，不需再次评测。补齐并登记七张真实核验截图后，按根 README 执行最终正文/README生成、图表构建、离线与浏览器检查、`validate_publication.py` 及 `package_publication.py`。只有当前完整证据、截图来源及全部发布检查均通过，才生成正式发布包；审阅稿不能替代这一步验收。

## 构建与检查科学结果审阅包

该入口只核验已有 CPU 证据与当前文稿，不运行模型，也不替代最终发布门禁：

```bash
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/package_review.py
python paper/package_review.py --verify paper/apxinf-gr00t-fp4fp8-ptqad-review.zip
```

包内 REVIEW-MANIFEST.json 登记所有成员（清单自身除外）的 SHA-256，包外 validation/review-package.json 登记完整 ZIP 与清单哈希。原始 HTML 构建收据保留七张被省略旧图的输入身份，因此该精简审阅包不声称可独立重跑完整构建。下载链接使用仓库地址；正文、图表及 README 的本地证据链接可离线阅读。
