# 发布证据与核验

本目录承载冻结 v12 RTN W4A4 实验的轻量证据和来源散列。主配方是全覆盖 NVFP4 W4A4 RTN；QAD、continued-QAD 和 OPD 都从同一量化基座分支。协议 SHA-256 为 `31addae911f92db65dda3c1c3145e8b798dca46c49a5ae12bbe10d232b5a4521`。正式结果只能由 v12 `final_manifest.json`、五臂 held-out 目录和配对比较生成，开发分区只用于选择恢复超参数。若这些文件尚未材料化，目录仍是发布目标说明，不能视为已完成证据。

## 必须存在的证据

| 证据 | 核验内容 |
|---|---|
| `protocol/` | `recovery_protocol_v12_rtn_w4a4.json` 及 SHA-256；声明 W4A4、RTN、分区和训练预算 |
| `selected_recipe/` | v12 RTN W4A4 的逐层配方、479 个 eligible 张量、格式缩放、padding 与 tied alias 去重口径 |
| `final_results.json` | 从 v12 完整 manifest 提取的五臂摘要；每臂 10 个任务、每任务 16 个 episode |
| `paired_comparison.json` | 五臂使用相同任务、初态身份和分母的逐任务配对比较，含置信区间与检验结果 |
| `heldout_arm/`、`heldout_raw_logs.json` | 五臂各自的 manifest、task results、summary，以及全部 50 份 rollout 日志与 50 份 server 日志；校验器重放原始日志并核对 reset 身份 |
| `training/` | QAD、continued-QAD、OPD 的训练请求、runtime metrics、合并清单和教师缓存来源 |
| `search_costs/` | 候选训练、开发评测、演示采集、学生采集、教师标注和选择收据；独立校验不依赖作者磁盘上的模型与缓存 |
| `captures.json`、`retained_captures.json`、`captures/` | 17 张 Ubuntu 执行截图及 SHA-256、尺寸、sidecar、脚本、日志和窗口绑定 |
| `recipe_inventory.json` | 从 v12 RTN 配方计算的完整编码预算；不把预算写成实测文件大小、显存或延迟 |

## 统计口径

v12 包含 479 个 eligible 权重张量，全部采用 NVFP4；469 个普通 Linear 与 7 个 CategorySpecificLinear 安装激活 QDQ，3 个 embedding/position 张量只量化权重。QAD/OPD 在 468 个实际动作前向普通 Linear 上挂 BF16 LoRA，类别层保持冻结。物理键和已知 tied alias 去重口径都记录在 `recipe_inventory.json`。恢复模型的净编码预算必须加入完整 BF16 LoRA 旁路。

## 运行边界

GR00T 闭环采用 Torch QDQ 数值路径；APXInf 原生算子与 π0.5 benchmark 单独核验。截图证明命令确实运行，不替代逐 episode 结果。未经同模型 packed GR00T 测量，不能把独立 APXInf 延迟写成 GR00T 端到端加速。

## CPU 验收

最终证据生成后，在仓库根目录运行：

    uv run --with-requirements paper/requirements-build.txt python paper/validate_publication.py
    uv run --with-requirements paper/requirements-build.txt python paper/package_publication.py

验证器会拒绝缺少 v12 五臂、分母不完整、配对身份不一致、占位符或截图哈希错误的发布包。
