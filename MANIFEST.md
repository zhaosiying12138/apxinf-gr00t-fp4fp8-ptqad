# 工件、依赖与再生成清单

本清单说明代码仓提供什么、哪些大文件需要下载或重新生成，以及如何验证它们的来源。实验入口见 [README](README.md) 和 [完整复现说明](docs/reproduce-ptqad.md)。安装完成、文件存在和一次完整实验通过，是三个不同的状态。

## 1. 随仓库保存的内容

| 内容 | 路径 | 核对方式 |
|---|---|---|
| 实验源码与协议 | `quant/`、`rl/`、`eval/`、`exp/recovery_protocol_v12_rtn_w4a4.json` | Git commit、运行时源码 SHA-256、实际命令 |
| 运行环境锁文件 | `setup/locks/` | `manifest.json` 中的依赖锁文件及补丁 SHA-256 |
| GR00T 运行补丁 | `patches/gr00t-recovery-runtime.patch` | 固定公开上游 commit 后，正向或反向 `git apply --check` |
| APXInf 运行补丁 | `patches/apxinf-fp4vla-engine.patch` | 在固定引擎子模块版本上检查，应用后再编译 |
| CPU 检查 | `tests/` | 量化语义、教师梯度与缓存、tied aliases、LoRA 导出等测试 |
| 可发布证据 | `results/` 与 `paper/evidence/` 中选定的 JSON、CSV、文本日志 | 对照原始运行目录、协议、完整任务覆盖和散列；目录本身不等于有效结果 |
| 图表与截图 | `paper/figs/`、`paper/figures.json` | 保留全部 17 个 `shot_*` 图位；按最终截图清单核对文件散列 |
| 中文发布稿 | `paper/sections/`、`paper/paper.html`、`paper/zhihu/` | 以 sections 为正文源，统一构建与发布完整性校验 |

## 2. 外部源码与独立环境

| 对象 | 默认路径 | 创建入口与固定来源 |
|---|---|---|
| GR00T 源码 | `third_party/Isaac-GR00T` | `setup/06_install_recovery.sh`；NVIDIA 上游 `51d4c89f72fda44cbf77285c6a8114b52676b8a1` 加仓内运行补丁 |
| 恢复训练 / 模型服务 | `$GR00T_REPO/.venv` | 06；Python 3.12.14、PyTorch 2.9.0+cu128、transformers 4.57.3；其余依赖见 `recovery-py312.txt` |
| LIBERO 仿真 | `$GR00T_REPO/.venv-libero` | 06；独立依赖见 `libero-py312.txt`，LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| FFmpeg 7 | `third_party/media7` | 06 通过现有 conda 安装 `media7-linux-64.explicit.txt`；验证包版本与 build |
| LIBERO 本机配置 | `third_party/libero-config/config.yaml` | 06 生成；通过 `LIBERO_CONFIG_PATH` 使用，不覆盖用户的 `~/.libero` |
| 本机环境变量 | `setup/recovery-env.sh` | 06 生成后 `source`；包含训练/仿真解释器、backbone 与 media 路径；不入库 |
| APXinf-robo / 引擎 | `third_party/apxinf-robo` 及其 `apxinf` 子模块 | `setup/05_restore_all.sh --with-engine`；分别固定 `63540bb02b926a99cf9700a7e59a9253f054806f`、`e07dbe98da914bc0a3df8c4bb4b7251c2a726f69` |
| APXInf 编译产物 | 上述目录中的 `.venv`、`target/wheel` | `setup/02_build_engine.sh`；先检查补丁、再构建和安装 |

安装命令：

```bash
bash setup/01_install_dev_tools.sh
bash setup/03_download_weights.sh
CONDA_EXE="$HOME/miniforge3/bin/conda" bash setup/06_install_recovery.sh
source setup/recovery-env.sh
```

`05_restore_all.sh` 可串联选定准备步骤，`--skip-download` 跳过权重下载，`--with-engine` 加入原生引擎构建。现有 checkout revision 不匹配时拒绝覆盖；任何命令失败都会终止。WSL NVIDIA 驱动、EGL/OpenGL 系统库及 conda 是外部前提；Rust 与 CUDA toolkit 用于原生引擎编译。

锁文件与源码补丁已检查，补丁已对固定上游源码做正反向应用校验，安装脚本已做语法检查；**尚未在另一台空白机器执行全套依赖安装**。这些检查不能替代模型加载或模拟器 smoke。

## 3. 模型与数据

| 对象 | 默认路径 | 再生成入口 / 身份 |
|---|---|---|
| GR00T LIBERO-10 基座 | `weights/GR00T-N1.7-LIBERO/libero_10` | 03 下载 `nvidia/GR00T-N1.7-LIBERO` 的 libero_10 子目录并排除优化器状态；运行产物记录逐 shard SHA-256 |
| Cosmos backbone 与处理器资源 | `weights/Cosmos-Reason2-2B` | 03 下载 `nvidia/Cosmos-Reason2-2B` |
| 命名兼容的本地 backbone 路径 | `weights/nvidia/Cosmos-Reason2-2B` | 06 创建命名链接；工厂要求路径含 `nvidia/Cosmos-Reason2`，不可消解成只有 snapshot hash 的路径 |
| π0.5 独立引擎基准资源 | `weights/pi05_libero_base` | 03 下载 `lerobot/pi05_libero_base`，并下载对应 tokenizer 和归一化统计 |
| 演示子集 | `$GR00T_REPO/demo_data/libero_demo` | 固定 GR00T 源码附带；5 个 episode、1,406 帧、3 个任务，当前管线产生 1,331 个有效窗口 |
| LIBERO 任务与官方初态库 | GR00T 的固定 LIBERO 子模块 | 06 初始化；评测记录任务名称、bank 文件散列、索引及恢复后的状态散列 |

本项目的演示子集不是完整十任务演示集。OPD 额外采集十任务的学生访问状态并获取教师标签，增加交互、标注与状态覆盖；不能将它和 continued-QAD 说成同数据或同总计算量。第三方权重和数据遵循各自许可。

## 4. 每次实验生成的大工件

设 `PTQAD_RUN_DIR` 为一个新的实验根目录。下列阶段均按 GPU 串行运行，已存在的阶段日志与目标目录不覆盖。

| 工件 | 相对实验根路径 | 再生成入口 | 完成身份与验收 |
|---|---|---|---|
| GPTQ 校准统计 | `results/.../supplement_gptq_capture_v12/`（仅补充实验） | GPTQ 补充脚本；主 RTN 配方不读取校准统计 | 校准窗口、层覆盖、输入权重身份和散列；不能把补充统计写成 RTN 配方依赖 |
| RTN W4A4 基座 | `selection_final/rtn_w4a4_category/` | v12 driver 的冻结压力配方 | `ptq_recipe.json`、`category_ptq_recipe.json`、bake manifest：479 个 NVFP4 张量、W4A4 安装和完整字节预算 |
| QAD 训练产物 | `recovery_v12/artifacts/train_qad_lr_*/` | `exp/run_w4a4_recovery.py` | Trainer 状态与 `recovery_manifest.json`：实际 2,000 步、micro/global/accum、scope、基座及 LoRA 身份 |
| QAD 稠密导出 | `recovery_v12/artifacts/merge_qad_lr_*/` | v12 driver merge 阶段 | `merge_manifest.json`：冻结 W 核验、rank/alpha、输出 dtype、唯一物理键及 shard 散列 |
| 学生访问的观测 | `teacher_supervision_v12_clean/collection/` | v12 driver 的 collection 阶段 | 来源为 QAD 学生，官方初态索引 20–23；按任务保存观测来源 |
| 教师缓存 | `recovery_v12/artifacts/teacher_cache/` | v12 driver 的 teacher-cache 阶段 | 固定教师、学生端点、随机条件与有效 16×7 动作掩码；缓存身份绑定 v12 协议 |
| 两臂继续训练及导出 | `recovery_v12/artifacts/{train_continued_qad,opd_*}` 及 merge 目录 | v12 driver，continued-QAD 与 OPD 各 2,000 步 | 同一起始 QAD A/B 与 RTN W4A4 base；分别验证训练、合并和教师项收据 |
| 五臂最终评测 | `recovery_v12/artifacts/heldout_round/heldout_<arm>/` | v12 driver 的 heldout 阶段 | 每臂十任务、每任务 16 回合、逐回合日志、`eval_manifest.json` 和 `task_results.json` |
| 正式闭环与前驱证据 | `paper/evidence/frontier/`、顶层比较 JSON | `compare_recovery.py` 由完整 round 生成；`paper/collect_frontier_evidence.py --frontier ...` 归档 | 五臂及冻结前驱的原始日志、每集初态配对、实际分母和字节来源映射 |
| 正式训练成本 | `paper/evidence/training/` | `paper/collect_training_costs.py`；`--verify-published` 可脱离私有权重复算 | 完成状态、共同起点、更新预算、原始计时范围、教师与学生身份及成本推导 |
| 原生算子与引擎测量 | `results/spike/`、`results/engine/` | `spike/run.sh`、`exp/bench_engine.py` | 独立记录设备、形状、后端、warmup、样本量与计时口径 |

校准 H 的 full-scope 形状预算上界约 13.82 GiB；实际文件大小由执行模块覆盖决定。权重、缓存和训练 checkpoint 的空间需求随配方数量和保存次数变化，不能用目标低比特预算估计它们的实际磁盘占用。当前 PTQ/恢复导出仍保存稠密原 dtype 张量。

## 5. 评测与发布的验收边界

正式 v12 协议固定 development 初态索引 4–8，teacher/collection 索引 20–23，heldout 索引 9–19 与 24–28（每臂 160 回合）。三者均覆盖十任务；heldout 同时检查与训练分区的初态索引、种子不重叠。逐集保存官方 bank、恢复状态和初态散列；环境初态可以配对，不据此假定不同策略的所有动作噪声逐步相同。

正式闭环比较直接核对每集布尔结局、固定官方初态与原始日志；缺失或不配对的任务不能生成完整比较。`paper/collect_reevaluation.py` 仅为通用 driver 保留单臂日志快照兼容接口，不是正式配对或发布验收入口。论文结果绑定源码、权重/数据、参数、输出工件、完整日志和统计推导；截图不能替代这些记录。

论文构建见 [paper/README.md](paper/README.md)。修改 `paper/sections/` 后生成 HTML 与知乎 Markdown，核对 17 个截图 key、图片路径、公式资源和证据索引。论文构建依赖另置于独立环境，不升级实验运行环境；生成发布包不会自动发表。
