# 数据/工件恢复清单（每个 gitignore 条目对应的再生命令）

| 路径 | 大小 | 恢复命令 | 说明 |
|---|---|---|---|
| third_party/ | ~600MB | setup/02 内 git clone --recursive | APXinf-robo+引擎子模块(main) |
| weights/pi05_libero_base | 14.5GB | setup/03 | lerobot/pi05_libero_base + tokenizer + norm_stats |
| weights/GR00T-N1.7-LIBERO | 6.5GB | setup/03（lean 过滤，仅推理权重） | nvidia/GR00T-N1.7-LIBERO |
| weights/Cosmos-Reason2-2B | ~8GB | HF cache（fetch_cosmos2.sh 或 hf download） | GR00T backbone |
| weights/pi05.nvfp4* | 0.1-2GB×5 | setup/05 步骤 6 或 quant/ 转换器 | 全量/packed/lang/act 消融工件 |
| results/ 大产物 | 不定 | 由对应 exp/ 脚本重生成 | 评测 ledger JSON 已提交 |
| baselines/.venv* | ~12GB | setup/05 步骤 5 | 双基线环境 |
| engine 构建 | ~4GB | setup/02（maturin） | patches/ 提供 1400 行引擎补丁 |

一键全恢复：`bash setup/05_restore_all.sh`（各步幂等可重入）
