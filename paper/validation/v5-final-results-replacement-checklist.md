# v5 最终结果回填清单

本文档是最终结果到位前的出版回填契约。它只描述应从完整 v5 `final_manifest.json` 提取什么、替换哪些旧叙述，以及哪些结论不能提前写入。它不包含开发集分数，也不改变 active heldout 工件。

## 唯一数据源

先用只读适配器验收完整 run，再生成发布证据包：

```bash
python3 paper/extract_final_evidence.py \
  --run-dir <RECOVERY_ROOT> \
  --out <PUBLISHED>/final_results.json

python3 paper/materialize_final_evidence.py \
  --final-manifest <RECOVERY_ROOT>/final_manifest.json \
  --out <PUBLISHED> \
  --orchestrator-run <RECOVERY_ROOT>
```

主表只读取 `<PUBLISHED>/final_results.json` 的 `public_arms`；它必须包含完整的 `bf16`、`ptq`、`qad`、`qad_opd`。`control.continued_qad` 只作为第五个控制行，用来判断 OPD 的变化是否只是额外演示更新。不得从开发选择日志、旧 `paper/evidence/recipe_inventory.json`、目录名或截图文字拼接数字。

回填字段如下：

| 文中量 | 唯一字段 | 用法 |
|---|---|---|
| BF16 成功回合/率 | `public_arms.bf16.successes`, `success_rate` | 主表第 1 行 |
| PTQ 成功回合/率 | `public_arms.ptq.successes`, `success_rate` | 主表第 2 行 |
| QAD 成功回合/率 | `public_arms.qad.successes`, `success_rate` | 主表第 3 行 |
| QAD+OPD 成功回合/率 | `public_arms.qad_opd.successes`, `success_rate` | 主表第 4 行 |
| continued-QAD 成功回合/率 | `control.continued_qad.successes`, `success_rate` | 主表控制行/附录 |
| PTQ 相对 BF16 | `deltas.ptq_minus_bf16_pp` | 只能写成 PTQ 的百分点变化 |
| QAD 恢复量 | `deltas.qad_minus_ptq_pp` | 只能写成 QAD 相对 PTQ 的百分点变化 |
| OPD 相对 QAD | `deltas.qad_opd_minus_qad_pp` | 大于 0 才能写“进一步提升” |
| OPD 相对继续 QAD | `deltas.qad_opd_minus_continued_qad_pp` | 控制比较；不等同于 QAD 增量 |

每一行固定为 10 个任务、100 个 episode。正文可显示 `成功回合/100` 与百分数；逐任务分子/分母留在证据包和附录。只有适配器通过完整 episode、逐任务和配对身份检查后，才可删除红色 `xxx`。

## v5 选择规则与配方口径

正文和附录中所有“压力窗口”统一改为 v5 协议：开发分区相对 BF16 **至少下降 5 个百分点**，且绝对成功率不低于 30%；在合格候选中选择 FP4 覆盖最高者。该规则来自 `exp/recovery_protocol_v5_exploratory_fp4.json` 的 `selection.pressure_rule`，heldout 不能选择配方、学习率、OPD 权重或训练长度。旧的“20 个百分点”表述、v3 协议名和 v3 阶梯选择说明必须从公开正文删除或改为 v5。

最终配方名称和组成只读 `final_results.selected_recipe` 及发布包中的 `selected_recipe/category_memory.json`。当前候选若为 `calib_category`，其记账应明确写出：

- eligible 参数的 NVFP4 占比为 100%，FP8 为 0，BF16 为 0；
- 这是**纯 NVFP4 极限端点**，不是 FP4/FP8 混合结果；
- “FP4/FP8 混合”仍可用于描述本文的分配框架和候选搜索，但不能把该端点的 FP8 数量写成非零，也不能沿用旧 `fp8 → … → calib` 表格中的数字作为最终配方；
- 全体 checkpoint 分母与 eligible 分母必须同时标注。被排除的非 Linear 张量和 tied alias 不得悄悄计入 FP4 覆盖率。

净编码压缩比不能直接取 `category_memory` 的纯 PTQ 比值。应从选中配方的去 alias `target_full_bytes` 加上实际导出的 BF16 LoRA 旁路字节，再除以同口径 BF16 源字节；QAD 与 QAD+OPD 使用同一旁路账目。若发布包尚未记录旁路账目，保留红色占位并补齐账本，不用旧的 `2.0777×`、`2.6669×` 或 `2.8606×`。

## 文件替换地图

### 正文源（先改这里）

| 文件/位置 | 回填或清理 |
|---|---|
| `paper/sections/01-摘要与引言.md` 摘要结果段 | 用最终配方、FP4 覆盖率、净压缩比和四个公开臂的 heldout 率替换 `xxx`；保留 APXInf 延迟与闭环率分开报告。若选中纯端点，摘要明确“最终配方为纯 NVFP4 端点”。 |
| `paper/sections/03-方法.md` §3.3 阶梯表 | 该表只能作为格式/候选分配说明；删除旧 v3 数字或改成从最终配方账本生成的候选预算。不能让旧 `calib` 行看起来就是 `calib_category` 的最终结果。 |
| `paper/sections/04-实验.md` 设置表 | 从最终 manifest/训练 costs 回填学习率、OPD 权重和实际探针数；分区改为 v5（development 4–8、collection 20–23、heldout 30–39）。 |
| `paper/sections/04-实验.md` §4.4 | 以最终选中 recipe 的 category memory 和净旁路账本替换旧五级阶梯结果；压力规则改为 5pp。开发结论只说明选择依据，不写开发成功率作为论文结论。 |
| `paper/sections/04-实验.md` §4.5 | 保留一张五行表：BF16、PTQ、QAD、QAD+OPD、continued-QAD（控制）。四个主臂的数字来自 `public_arms`，控制行来自 `control`。差值按 `deltas` 回填。 |
| `paper/sections/04-实验.md` §4.6–4.7 | 净压缩比按旁路计账；训练时间、显存和教师成本从 `final_costs_summary.json` 回填，不从截图估读。 |
| `paper/sections/05-讨论与结论.md` | 用实际三组差值写结论。只有 `qad_opd_minus_qad_pp > 0` 且控制差值方向一致时，才能写“OPD 进一步提升”；否则写“本配置未观察到额外增益”，不以训练损失代替行为证据。 |
| `paper/sections/14-附录A-核心源码走读与APXInf框架解析.md` | 将 v3 协议名改为 v5；保留源码职责解释，更新分区和“由 v5 选择记录固定”的描述。 |
| `paper/sections/15-附录B-复现与证据索引.md` | 全部 v3/v3_high_fp4 命令、20pp 规则、旧入口名改为 v5 `exp/run_category_recovery.py`、5pp 规则和 v5 分区；归档指向 `final_manifest.json` 与发布适配器。 |

### 生成副本与项目说明

| 文件/位置 | 处理方式 |
|---|---|
| `paper/paper.html` | 不手改；正文源和图表回填后重新运行 `build_html.py`。生成前后检查无 `xxx`、v3 协议、旧阶梯数字和纯端点的伪 FP8。 |
| `paper/zhihu/article.md` | 不手改；运行 `export_zhihu.py` 从同一正文源生成。保留 17 张截图和图位，数字与 HTML 必须逐项一致。 |
| `paper/README.md` | 把“审阅稿/红色 xxx”说明改为最终发布状态；新增 v5 发布包入口和 `final_results.json` 来源，保留构建、浏览器检查和 17 张截图清单。 |
| `README.md` | 复现命令、协议路径和结果表统一指向 v5；结果表只保留四个公开臂与 continued-QAD 控制行，不展示旧 v3 结果。 |
| `paper/evidence/README.md` | 说明发布包由 v5 final manifest 生成，16 个 heldout JSON 的 5 臂结构，以及纯 NVFP4 endpoint 的 FP8=0 口径。 |
| `paper/evidence/recipe_inventory.json`、`make_figs.py` | 不把旧静态 inventory 当最终数字；图表改读 materialized selected recipe/category memory 和 final results。 |

## 图表与截图验收

17 张 Ubuntu 截图全部保留。受量化器、恢复训练或评测实现影响的截图必须按当前 v5 run 重拍并更新 sidecar/hash；仅未受影响的算子/环境截图可沿用已核验版本。截图只证明命令执行，不提供成功率或压缩比数字。图表中的主结果、预算和曲线必须与 `final_results.json`、选中配方账本和成本包一致，不能从截图 OCR 或旧 SVG 文本回填。

## 发布前静态门禁

在 HTML 和知乎稿生成后执行以下只读检查：

```bash
grep -RInE 'xxx|recovery_protocol_v3|v3_high_fp4|至少低20|20 个百分点|2\.0777|2\.6669|2\.8606' \
  paper/sections paper/paper.html paper/zhihu/article.md paper/README.md README.md
```

允许保留相关工作中与本文无关的外部论文数字；本文主表、配方表、图表和结论不得命中上述旧结果。最后再运行 HTML 浏览器 QA、`validate_publication.py` 和打包器；任何 placeholder、heldout 不完整、配对身份不一致或截图 hash 不匹配都应使发布失败。
