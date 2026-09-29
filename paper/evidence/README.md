# 发布证据与核验方法

本目录按下述契约组织证据。这里列出路径不表示对应实验已经完成；缺失、未完成或来源不一致的文件必须使发布校验失败。生成与归档命令见 [完整复现说明](../../docs/reproduce-ptqad.md)，构建流程见 [论文发布说明](../README.md)。

| 证据 | 如何核验 | 文件性质 |
|---|---|---|
| `recipe_inventory.json` | 核对冻结计划、逐张量分配、scale 开销和低秩残差；区分物理 checkpoint 副本与已知 tied alias 去重口径 | 根据真实权重索引、形状与校准元数据计算的编码预算，不是文件大小、显存或速度实测 |
| `paired_comparison.json` 与 `heldout_<arm>/{eval_manifest,task_results,summary}.json` | **共 16 个 JSON**：1 个五臂汇总和 5×3 个来源文件；每臂十任务×十集，逐集核对布尔结局、官方初态索引 10–19、状态哈希和实际分母 | `compare_recovery.py` 生成汇总；汇总及 15 个来源 JSON 从同一个完整 round 按原字节复制，不能重新序列化或修改绝对路径 |
| `frontier_comparison.json`、`frontier/` | 复核训练前冻结的前驱清单、开发选择、主五臂与全部 PTQ 前驱的原始逐任务日志、初态配对和含残差的净预算 | 原始 JSON、日志、源码/元数据按字节复制；`frontier/evidence_manifest.json` 独立记录 `original_absolute_path → published_path`、字节数和 SHA-256 |
| `training/costs.json`、`training/evidence_manifest.json` 及其目录 | 检查三段正式训练完成状态、500/100/100 步、共同 QAD 起点、PTQ 基座、rank/alpha、演示 batch，以及采集学生和 BF16 教师的身份；采集间隔读取实际 `capture_manifest.every_server_calls`，不采用底层 helper 的备用默认值 | runtime/recovery/Trainer/merge 元数据与轻量日志原字节复制；`costs.json` 是从这些证据推导的成本汇总，冒烟不计入正式对比 |
| `runtime/manifest.json` | 必须同时包含 `bf16`、`ptq`、`qad`、`continued_qad`、`qad_opd`；将实际评测 checkpoint、导出权重、教师、源码与协议散列互相核对 | `capture_runtime.py` 生成的本机身份记录；不由文件名推断模型版本或执行 dtype |
| `captures.json`、`captures/`、`retained_captures.json` | 新截图需真实命令成功、`accepted_not_black`、人工内容核验、原始 capture sidecar，以及 taskbar-only crop 的前后 SHA/尺寸关联 | 15 张新执行截图和 2 张获准保留的 BF16 图；两张保留图以 Git 可达锚点和对应原始 JSON 核验，只作流程展示，不作为本轮主表计时 |

五臂的 15 个来源 JSON 必须同时位于 `evidence/heldout_<arm>/`，供主比较验收；frontier 归档中的 `frontier/main/heldout_<arm>/` 是同一原始证据的另一份组织副本。仅复制顶层汇总或仅有 frontier 副本不足以完成主比较验收。原始日志中的绝对路径作为来源事实保留，公开复算通过路径映射定位归档副本。

训练计数须区分计划与观察。在正式 OPD 完成 100 个更新、每 4 次更新触发一次、每次含 16 个累积微批的配置下，`scheduled_teacher_backward_passes` 推导为 **400**；稀疏日志只在第 4、24、44、64、84 步打印，完整日志应解析出 **80** 条。两者不是两种训练预算，也不能把 400 写成逐次日志实测。这里计数的是学生对缓存教师速度目标的反传；BF16 教师标注使用 `no_grad`，没有教师参数更新。演示窗口次数同样由完成步数×配置 batch 推导，不等于互异样本数。

训练、教师标注、状态采集和导出各自保留 producer 的计时范围。训练包含保存检查点；教师标注包含加载与输入核验、排除缓存序列化。不能把分段字段相加称为独立测得的端到端耗时，也不能把同演示更新预算称为同总计算。权重、观察张量和教师缓存不随轻量包分发：收集器核对其实际散列，公开验证只能复算已归档元数据/日志并核对记录身份，不能重新核验未分发张量的内容。

成本包齐备后可仅用 CPU 验证，不需要原权重目录或 PyTorch：

```bash
python3 paper/collect_training_costs.py --verify-published paper/evidence/training
```

frontier 归档也从公开副本重新核对来源映射、原始日志、配对与预算。全部结果、截图、生成图、HTML 和知乎 Markdown 齐备后，运行 `python3 paper/validate_publication.py`；打包器再次执行验收，不使用旧的通过标志。截图外观样例已经获准，不因重拍重复请求确认；缺失的质量或裁剪证明不能从现有 PNG 倒推补写。
