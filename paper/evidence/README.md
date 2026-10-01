# 发布证据与核验

本目录只收录冻结 v11 W4A4 运行的轻量证据和来源散列。最终结果由恢复运行目录的 final_manifest.json、五个 heldout arm 评测目录和 paired_comparison.json 生成；开发分区只用于选择学习率与 OPD 权重。

## 必须存在的证据

| 证据 | 核验内容 |
|---|---|
| selected_recipe/ | v11 W4A4 的逐层配方、全 NVFP4 分配、尺度开销和 tied alias 去重口径 |
| final_results.json | 从最终 manifest 提取的五臂摘要；每臂 10 个任务、每任务 16 个 episode |
| paired_comparison.json | 五臂使用相同任务、初态身份和分母的逐任务配对比较 |
| heldout_arm/ | BF16、全 NVFP4 W4A4 PTQ、QAD、continued-QAD、QAD+OPD 各自的 manifest、task results 和 summary |
| training/ | QAD、continued-QAD、OPD 的训练请求、runtime metrics、合并清单和教师缓存来源 |
| captures.json、captures/ | 真实 Ubuntu 执行截图及其 SHA-256、尺寸和 sidecar |
| recipe_inventory.json | 从 v11 category recipe 计算的编码预算；不把预算当成实测文件大小或延迟 |

## 统计口径

v11 W4A4 包含 479 个候选张量，全部采用 NVFP4；476 个可执行 Linear 同时安装激活 QDQ，3 个 embedding/位置张量仅量化权重。物理键和已知 tied alias 去重口径都记录在 recipe_inventory.json。QAD+OPD 的净编码预算还需加入实际 BF16 LoRA 旁路，不能直接套用 PTQ 基座压缩比。

## 运行边界

GR00T 闭环采用全 NVFP4 权重、NVFP4 激活 QDQ 的 W4A4 数值路径；APXInf 原生算子与 π0.5 benchmark 单独核验。截图证明命令确实运行，不替代逐 episode 结果。

## CPU 验收

最终证据生成后，在仓库根目录运行：

    python3 paper/validate_publication.py
    python3 paper/package_publication.py

验证器会拒绝缺少五臂、分母不完整、配对身份不一致、占位符或截图哈希错误的发布包。
