# apxinf-gr00t-fp4fp8-ptqad

APXInf × GR00T：NVFP4 W4A4 与 QAD/OPD 量化域恢复

视觉—语言—动作模型（VLA）把图像、语言和机器人状态映射为动作；本项目研究如何在 **GR00T N1.7** 上把量化推进到 W4A4，同时保持 LIBERO 闭环任务能力。方法由四个环节组成：面向 NVFP4 块缩放格式的训练后量化（PTQ）、固定二级 scale 的 NVFP4 激活 QDQ、量化基座上的低秩恢复（QAD），以及在学生访问状态上加入教师约束的 OPD。QAD/OPD 的残差始终读取原始 BF16 输入，因此不会把恢复分支再次量化。APXInf 的原生算子和图执行作为独立工程验证；GR00T 闭环结果使用同一套固定初态和逐回合日志评测。

论文和可视化发布包是本仓库的主要入口：

- [中文论文 HTML](paper/paper.html)：离线单文件，含公式、图表和全部 Ubuntu 执行截图。
- [知乎 Markdown 发布稿](paper/zhihu/article.md)：与 HTML 共用正文和图片资源。
- [论文发布包说明](paper/README.md)：构建、验证、打包和证据索引。
- [复现说明](docs/reproduce-ptqad.md)：从环境安装到独立闭环评测的完整步骤。

## 研究贡献

1. **W4A4 PTQ。** 权重使用 NVFP4 E2M1，激活先转 F16，再按 16 元素块用 E4M3 scale 和 E2M1 格点 QDQ；二级 activation scale 固定为 1.0，训练与服务使用同一算术契约。
2. **量化域恢复。** 在固定 PTQ 基座上训练低秩 QAD，再用学生访问的观测和 BF16 教师速度训练 OPD；正式部署保留冻结 base 与 A/B adapter，避免 `Wq+BA` 合并后破坏 W4A4 语义。
3. **闭环评测。** 所有正式臂使用同一组 LIBERO-10 任务、官方初态和 episode 预算，报告逐回合结果、任务宏平均和配对身份。
4. **APXInf 验证。** APXInf 的 NVFP4 算子、布局和图重放单独验收；它们的延迟结果不替代 GR00T 的闭环成功率。
5. **可复现发布。** 协议、配置、权重来源、日志、图表和 17 张 Ubuntu 截图共同组成发布包。

## 结果状态

最终公开表只从冻结协议对应的 `final_manifest.json` 和 `paired_comparison.json` 生成。实验完成前保留“待回填”，不把开发集分数写入最终结论。

| 配置 | 目标编码压缩比（去别名） | FP4／去别名可量化元素 | LIBERO-10 闭环成功率 | 备注 |
|---|---:|---:|---:|---|
| BF16 基线 | 1.0000× | 0% | <span style="color:#c00">待回填</span> | 同一 v11 bank、同一 episode 协议 |
| 全 NVFP4 PTQ + W4A4 | <span style="color:#c00">待回填</span> | <span style="color:#c00">待回填</span> | <span style="color:#c00">待回填</span> | 469 个普通 Linear 与 7 个 CategorySpecificLinear 进入 W4A4 QDQ；3 个 embedding/位置参数仅有 NVFP4 权重 |
| PTQ + QAD（W4A4 基座） | <span style="color:#c00">待回填（含 BF16 adapter）</span> | 同 PTQ 基座 | <span style="color:#c00">待回填</span> | `base(QA(x)) + BF16 LoRA(x)` |
| PTQ + QAD + OPD（W4A4 基座） | <span style="color:#c00">待回填（含 BF16 adapter）</span> | 同 PTQ 基座 | <span style="color:#c00">待回填</span> | 学生访问状态上的教师速度蒸馏 |
| continued-QAD 对照 | <span style="color:#c00">待回填（含 BF16 adapter）</span> | 同 PTQ 基座 | <span style="color:#c00">待回填</span> | 与 OPD 使用相同追加更新预算 |

压缩比以去别名后的 BF16 模型权重字节数为分子，以目标编码中的 NVFP4 权重、E4M3 block scale、FP32 secondary scale、未量化张量及 BF16 adapter 的总字节数为分母。恢复臂多了 adapter，净压缩比不能照抄 PTQ 基座。成功率从 v11 的 `final_manifest.json` 所绑定的配对评测读取，压缩预算从同一模型的量化清单和 adapter 清单计算；快速 smoke 分数不会写入这张表。正式结果另附十任务逐任务分子/分母、160 回合总数和训练预算。

本文 GR00T 路径是 **数值 W4A4 仿真**：普通 NVFP4 base 的输入激活使用固定 scale 的 QDQ，QAD/OPD 残差以 BF16 保留。当前 GR00T 原生 executor 尚未提供与 APXInf π0.5 相同的 packed W4A4 loader，因此本文不把这条 Torch QDQ 路径称为原生 APXInf GR00T kernel；原生加速只在独立算子和 π0.5 图执行基准中报告。

## 运行环境

推荐在 WSL2 Ubuntu、单张支持 NVFP4 的 Blackwell GPU 上运行。当前锁定并验证的参考平台为 RTX 5090 Laptop 24 GB（`sm_120`）。

| 用途 | 依赖 |
|---|---|
| 量化、训练与服务 | Python 3.12.14、PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、transformers 4.57.3、torchcodec 0.8.0 |
| LIBERO 仿真 | 独立 Python 3.12.14、robosuite 1.4.0、MuJoCo 3.3.1、gym 0.25.2，以及 EGL/OpenGL 系统库 |
| 媒体 | FFmpeg 7.1.1（独立环境，供 torchcodec 和视频记录使用） |
| APXInf 原生引擎（可选） | CUDA toolkit、`nvcc`、cuBLASLt、Rust/Cargo、Linux C++ 编译器、Make，以及支持 `sm_120` 的 NVIDIA WSL 驱动 |
| 论文构建 | Python 3、uv、Node.js、Playwright、CairoSVG/Pillow 和中文字体 |

逐包版本、源码 revision、权重哈希和 FFmpeg 显式清单见 [setup/locks/manifest.json](setup/locks/manifest.json)。训练 Python、LIBERO Python 和 APXInf Python 必须分开；混用环境通常会导致 `numpy`、`torchcodec` 或 EGL 导入错误。

## 安装

从仓库根目录执行。大模型权重、Hessian、训练 checkpoint 和虚拟环境均不进入 Git。

```bash
git clone https://github.com/zhaosiying12138/apxinf-gr00t-fp4fp8-ptqad.git
cd apxinf-gr00t-fp4fp8-ptqad

sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install

bash setup/01_install_dev_tools.sh
bash setup/03_download_weights.sh --core-only
CONDA_EXE="$HOME/miniforge3/bin/conda" bash setup/06_install_recovery.sh
source setup/recovery-env.sh
```

`setup/06_install_recovery.sh` 会准备固定 revision 的 GR00T、训练环境 `.venv`、仿真环境 `.venv-libero`、媒体库和恢复补丁，并生成 `setup/recovery-env.sh`。脚本需要已安装的 conda；若其路径不同，修改 `CONDA_EXE`。安装后先检查环境和权重：

```bash
python3 setup/verify_weights.py --models gr00t cosmos
```

需要 APXInf 原生引擎时，再确认 CUDA/Rust 工具链后执行：

```bash
bash setup/05_restore_all.sh --with-engine
source setup/recovery-env.sh
```

`setup/05_restore_all.sh --with-engine` 会克隆固定 revision 的 APXInf-robo 并调用引擎构建脚本；已有完整权重时可追加 `--skip-download`。不要把本机 Hugging Face snapshot 直接改名成不含 `nvidia/Cosmos-Reason2` 的路径，GR00T 工厂会用该字符串选择 backbone。

## 复现流程

以下路径对应安装脚本的默认新机器布局；每次打开 shell 先加载生成的环境文件。自定义安装位置时，沿用其中的路径，不覆盖为默认值。所有正式运行都应使用新的 `PTQAD_RUN_DIR`，避免覆盖既有日志。

```bash
export PROJECT="$(pwd)"
source setup/recovery-env.sh
export GR00T_REPO="$PROJECT/third_party/Isaac-GR00T"
export PTQAD_PYTHON="$GR00T_REPO/.venv/bin/python"
export LIBERO_PYTHON="$GR00T_REPO/.venv-libero/bin/python"
export PTQAD_BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export QAD_DATASET="$GR00T_REPO/demo_data/libero_demo"
export PTQAD_MEDIA_LIB="$PROJECT/third_party/media7/lib"
export PTQAD_RUN_DIR="$PROJECT/results/mixed_pressure"
export PTQAD_PROTOCOL_FILE="$PROJECT/exp/recovery_protocol_v11_w4a4_category.json"
export PTQAD_SELECTION="$PROJECT/results/ptqad_20261003/v11_selection/selection.json"
export PTQAD_CAPTURE="$PROJECT/results/ptqad_20261003/w4a4_teacher_supervision"
export PROTOCOL_FILE="$PTQAD_PROTOCOL_FILE"
```

运行来源：本次实验机的仿真解释器为 `/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python`。这是已有环境的记录；新机器使用安装脚本生成的 `.venv-libero`，并按锁文件核对依赖。

v11 的压力臂 checkpoint 是大文件，不随 Git 发布。协议中的
`selection.pressure_candidate_checkpoints` 当前保存的是实验机路径；在另一台机器上，必须先取得同一份已完成的 NVFP4 PTQ checkpoint，
再在新的协议副本中把这两个字段一起改为本地绝对路径，并将 `PTQAD_PROTOCOL_FILE`、`PROTOCOL_FILE` 指向该副本。修改协议会改变其 SHA-256，必须使用全新的 `PTQAD_RUN_DIR`；脚本会在开始评测前校验 `category_ptq_recipe.json`、`category_bake_manifest.json` 和全 NVFP4 recipe 账本。

### 1. CPU/加载冒烟

```bash
"$PTQAD_PYTHON" -m unittest discover -s tests -v
"$PTQAD_PYTHON" -m py_compile \
  quant/fake_quant.py quant/ptq/bake.py \
  exp/run_w4a4_recovery.py exp/run_mixed_pressure_recovery.py
```

### 2. 全覆盖 W4A4 PTQ、QAD、OPD 与闭环

v11 入口不会隐式生成大体积 checkpoint。新机器首次运行时，先按 [完整构建步骤](docs/reproduce-ptqad.md) 生成 W4A4 兼容的 NVFP4 PTQ base，再把 v11 协议的 candidate checkpoint 路径改为本机目录。协议副本和结果目录必须全新；入口会校验 recipe、adapter 身份和 W4A4 activation contract。

```bash
"$PTQAD_PYTHON" exp/run_w4a4_recovery.py \
  --run-dir "$PTQAD_RUN_DIR" \
  --protocol-file "$PTQAD_PROTOCOL_FILE" \
  --ptq-selection "$PTQAD_SELECTION" \
  --base "$PTQAD_BASE" \
  --gr00t-repo "$GR00T_REPO" --python "$PTQAD_PYTHON" \
  --rollout-python "$LIBERO_PYTHON" \
  --dataset "$QAD_DATASET" \
  --capture-dataset "$PTQAD_CAPTURE" \
  --port-base 5890
```

该入口消费已经通过开发集审计的 `selection.json` 和已经完成的 BF16 教师采集目录，随后按协议顺序完成 QAD 学习率选择、学生状态采集、OPD 对照和 heldout 评测。首次运行前必须先按 [完整构建步骤](docs/reproduce-ptqad.md) 生成 W4A4 PTQ base、开发集 selection 和教师采集；已写入完成标记的阶段可以从同一 `--run-dir` 继续。运行中断留下的阶段必须先按日志和校验器审计，不能直接覆盖。最终结果以运行目录下的 `final_manifest.json` 为准。

如果开发集上的 OPD 没有超过两个对照，但仍需完成固定的 paired held-out 对比，可在续跑时显式追加 `--allow-opd-nonimprovement`。该开关只允许实验继续，不改变选择分数；selection 证据保留实际比较。最终报告 OPD 相对 QAD 和 continued-QAD 的两项差值、配对区间与条件检验，小幅点估计上升不直接写成独立增益。分析规则见 [统计分析说明](paper/analysis_plan_w4a4.json)。

当前 148 个演示窗口顺序读取，micro batch 为 1，每个完整更新累积 16 个样本，每轮尾批为 4。每 2,000 次更新实际读取 29,600 次演示样本；OPD 每四次更新加入探针，共增加 6,800 次探针前向/反传。continued-QAD 与 OPD 对齐演示和更新预算，额外教师计算单列。训练器保存的 `checkpoint-*` 必须经部署打包后评测；直接加载原始训练 checkpoint 会被入口拒绝，以防丢失低秩分支。

### 3. 原生 APXInf 验收（可选）

先按 [原生 π0.5 复现说明](docs/native-pi05.md) 准备 packed 模型、候选 wheel 和验收程序，并串行运行算子与模型检查。`run_native_graph_gates.sh` 必须显式传入本次模型路径和一个尚不存在的输出目录：

```bash
# RUN 为原生复现说明中本次打包的输出根目录；GPU 此时应空闲。
NATIVE_PI05_MODEL="$RUN/model" \
NATIVE_PY="$PROJECT/third_party/apxinf-robo/.venv/bin/python" \
PTQAD_GPU_EXCLUSIVE=1 \
  bash exp/run_native_graph_gates.sh "$PROJECT/results/native_graph_new_run"
```

原生算子结果与 GR00T PyTorch 闭环结果分开报告；不要用单个 GEMM 延迟推断闭环成功率。

## Ubuntu 运行截图

论文 HTML 保留全部 17 张紫色 Ubuntu 终端截图。每张截图对应一个真实命令，原始日志和 SHA-256 清单见 [截图清单](paper/evidence/captures.json)；README 展示四张代表性画面，完整图集见 [paper/paper.html](paper/paper.html)。

| 阶段 | 运行证据 |
|---|---|
| 校准与 H 统计 | ![校准短测](paper/figs/shot_collect.png) |
| NVFP4 写盘 | ![量化写盘短测](paper/figs/shot_bake.png) |
| QAD 训练 | ![QAD 短测](paper/figs/shot_qad.png) |
| OPD 教师辅助更新 | ![OPD 短测](paper/figs/shot_opd.png) |

<details>
<summary>17 张截图完整索引</summary>

| 编号 | 阶段 | 文件 |
|---:|---|---|
| 01 | 完整模型 2 窗口校准 | [shot_collect.png](paper/figs/shot_collect.png) |
| 02 | H 驱动的 NVFP4/FP8 写盘 | [shot_bake.png](paper/figs/shot_bake.png) |
| 03 | π0.5 单层 NVFP4 打包 | [shot_packed.png](paper/figs/shot_packed.png) |
| 04 | 有效动作掩码与探针梯度 | [shot_probe.png](paper/figs/shot_probe.png) |
| 05 | QAD 训练短测 | [shot_qad.png](paper/figs/shot_qad.png) |
| 06 | 学生访问观测的教师标注 | [shot_opdcache.png](paper/figs/shot_opdcache.png) |
| 07 | QAD→OPD 续训短测 | [shot_opd.png](paper/figs/shot_opd.png) |
| 08 | 激活重算与 LoRA 梯度 | [shot_qat.png](paper/figs/shot_qat.png) |
| 09 | 导出模型服务与健康 RPC | [shot_evalserver.png](paper/figs/shot_evalserver.png) |
| 10 | LIBERO 学生闭环与观测采集 | [shot_rollout.png](paper/figs/shot_rollout.png) |
| 11 | 量化格点与 checkpoint 校验 | [shot_verify.png](paper/figs/shot_verify.png) |
| 12 | FP8 描述符探针 | [shot_fp8probe.png](paper/figs/shot_fp8probe.png) |
| 13 | 9 种形状 GEMM 短测 | [shot_gemm.png](paper/figs/shot_gemm.png) |
| 14 | 18 个实际层形状算子短测 | [shot_opbench.png](paper/figs/shot_opbench.png) |
| 15 | GR00T BF16 执行流程 | [shot_gr00t.png](paper/figs/shot_gr00t.png) |
| 16 | π0.5 BF16 执行流程 | [shot_pi05.png](paper/figs/shot_pi05.png) |
| 17 | π0.5 NVFP4 完整路径 | [shot_nvfp4.png](paper/figs/shot_nvfp4.png) |

</details>

截图只证明对应步骤确实运行；闭环成功率、压缩比和延迟仍以证据 JSON、逐回合日志和独立基准为准。

## 目录

```text
quant/       NVFP4/FP8 格式、校准统计、PTQ 分配与权重写盘
rl/          QAD、学生状态采集、教师标注、OPD 与 LoRA 导出
eval/        GR00T 服务、固定初态 LIBERO 评测与逐回合日志
exp/         冻结协议、开发/恢复调度器、配方账本和原生基准
spike/       cuBLASLt/NVFP4 布局与算子探针
setup/       环境安装、权重下载、版本锁和引擎构建
tests/       CPU 数值、梯度、缓存及 checkpoint 契约测试
paper/       中文 HTML、知乎 Markdown、图表、17 张截图和证据索引
docs/        复现、权重溯源、原生引擎和恢复训练说明
results/     可解析实验日志；大模型和训练工件不入库
weights/     本地权重/训练工件目录，不入库
```

## 验证与发布

论文发布前在独立构建环境执行：

```bash
uv run --with-requirements paper/requirements-build.txt python paper/make_figs.py
bash paper/figs/render_pngs.sh
uv run --with-requirements paper/requirements-build.txt python paper/build_html.py
uv run --with-requirements paper/requirements-build.txt python paper/export_zhihu.py
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 node paper/qa_browser.cjs
uv run --with-requirements paper/requirements-build.txt python paper/validate_publication.py
uv run --with-requirements paper/requirements-build.txt python paper/package_publication.py
```

含有红色占位符、缺少 heldout 原始日志或截图哈希不匹配时，`validate_publication.py` 会拒绝发布。发布前应完成 HTML、知乎稿、截图和证据校验。

## 总结

本仓库把 NVFP4 W4A4 数值契约、模块级 PTQ、QAD/OPD 恢复和 LIBERO 闭环评测放在同一条复现链路中。读者可以从 HTML 和知乎稿阅读方法，从 v11 协议运行实验，从逐回合日志核对结果，并从 Ubuntu 截图确认关键步骤。

第三方代码、模型权重和数据遵循各自上游许可证。项目自身许可证将在正式发布时确定。
