# APXInf × GR00T：FP4/FP8 混合 PTQ 与量化域恢复

本项目研究一个面向真实机器人行为的低位宽量化问题：如何把 **GR00T N1.7** 的更多线性权重压到 NVIDIA NVFP4，同时保持 LIBERO 闭环任务能力。项目把 **APXInf 原生执行路径、面向块缩放格式的训练后量化（PTQ）、量化域低秩恢复（QAD）和学生状态蒸馏（OPD）** 放在同一条可审计流水线上，逐步记录校准统计、权重来源、训练成本、逐回合成功与原始终端截图。

论文和可视化发布包是本仓库的主要入口：

- [中文论文 HTML](paper/paper.html)：离线单文件，含公式、图表和全部 Ubuntu 执行截图。
- [知乎 Markdown 发布稿](paper/zhihu/article.md)：与 HTML 共用正文和图片资源。
- [论文发布包说明](paper/README.md)：构建、验证、打包和证据索引。
- [复现说明](docs/reproduce-ptqad.md)：从环境安装到独立闭环评测的完整步骤。

## 研究贡献

1. **混合精度 PTQ。** 以 NVFP4 的 16 元素块缩放格式为目标，结合 FP8 行缩放和模块敏感度分配；校准阶段在 FP32 中累积二阶统计，并用 GPTQ 补偿可观测层。
2. **量化域恢复。** 在固定 PTQ 基座上训练低秩适配器 QAD；再从学生真实访问的观测出发做一次 on-policy 蒸馏 OPD。QAD 和 OPD 都在部署所用的量化权重上训练，避免训练/推理函数不一致。
3. **闭环而非静态误差验收。** 所有正式臂使用同一组 LIBERO-10 任务、官方初态和 episode 预算，保留逐回合布尔结果、任务宏平均、配对身份和 SHA-256。
4. **推理生态验证。** APXInf 的 NVFP4 算子、布局和图重放作为独立执行路径验收；GR00T 行为结果由 PyTorch 策略服务执行，二者的测量口径分别记录。
5. **可追溯证据。** 每个数字都绑定协议、checkpoint、源码 revision、原始日志和结果数组；失败阶段保留日志，不以缺失记录替代成功或失败。

## 结果状态

最终公开表只从通过完整验收的 `final_manifest.json` 和 `paired_comparison.json` 生成。当前高 FP4 实验仍可能替换先前的候选，README 不展示旧实验结论，也不把开发集分数当作最终结果。

| 配置 | 目标编码压缩比 | FP4 覆盖率 | LIBERO-10 闭环成功率 | 备注 |
|---|---:|---:|---:|---|
| BF16 基线 | <span style="color:#c00">待回填</span> | 0% | <span style="color:#c00">待回填</span> | 固定基座 |
| 选定高 FP4 PTQ | <span style="color:#c00">待回填</span> | <span style="color:#c00">待回填</span> | <span style="color:#c00">待回填</span> | 只由开发分区按预注册规则选择 |
| PTQ + QAD | <span style="color:#c00">待回填</span> | 同 PTQ | <span style="color:#c00">待回填</span> | 低秩恢复 |
| PTQ + QAD + OPD | <span style="color:#c00">待回填</span> | 同 PTQ | <span style="color:#c00">待回填</span> | 学生状态蒸馏 |
| continued-QAD 对照 | <span style="color:#c00">待回填</span> | 同 PTQ | <span style="color:#c00">待回填</span> | 与 OPD 使用相同追加更新预算 |

结果回填必须同时写明：十任务逐任务分子/分母、100 回合总数、开发/heldout 分区、实际选中的配方身份、FP4/FP8/旁路的编码口径、QAD/OPD 训练步数与额外教师成本。若压力配方未达到协议规定的成功率下降窗口，恢复阶段应记录 `selected_recipe=null`，不能为了得到更高恢复增益而事后改阈值。

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

`setup/06_install_recovery.sh` 会准备固定 revision 的 GR00T、训练环境、LIBERO 环境、媒体库和恢复补丁。安装后先检查环境和权重：

```bash
bash setup/00_env_report.sh
python3 setup/verify_weights.py
```

需要 APXInf 原生引擎时，再确认 CUDA/Rust 工具链后执行：

```bash
bash setup/02_build_engine.sh
bash setup/05_restore_all.sh --with-engine
```

不需要重新下载权重时可使用 `--skip-download`；不要把本机 Hugging Face snapshot 直接改名成不含 `nvidia/Cosmos-Reason2` 的路径，GR00T 工厂会用该字符串选择 backbone。

## 复现流程

先设置路径。所有正式运行都应使用新的 `PTQAD_RUN_DIR`，避免覆盖既有日志。

```bash
export PROJECT="$(pwd)"
export GR00T_REPO="$PROJECT/third_party/Isaac-GR00T"
export PTQAD_PYTHON="$GR00T_REPO/.venv/bin/python"
export LIBERO_PYTHON="$GR00T_REPO/.venv-libero/bin/python"
export PTQAD_BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export QAD_DATASET="$GR00T_REPO/demo_data/libero_demo"
export PTQAD_MEDIA_LIB="$HOME/miniforge3/envs/media7/lib"
export PTQAD_RUN_DIR="$PROJECT/weights/reproductions/ptqad_run_01"
export PROTOCOL_FILE="$PROJECT/exp/recovery_protocol_v5_exploratory_fp4.json"
```

### 1. CPU/加载冒烟

```bash
"$PTQAD_PYTHON" -m unittest discover -s tests -v
"$PTQAD_PYTHON" -m py_compile \
  quant/ptq/category_fp4.py quant/ptq/collector_category.py \
  quant/ptq/bake_category.py exp/run_category_development.py \
  exp/run_category_recovery.py
```

### 2. 校准和高 FP4 配方

标准校准入口为 `exp/reproduce_ptqad.sh calibrate`。高 FP4 类别扩展使用冻结的 `recovery_protocol_v5_exploratory_fp4.json`，先以 `--dry-run` 检查路径，再运行开发分区：

下面两个父 checkpoint 由前置 `head_lang_vision`/`calib` PTQ 阶段生成，并放在当前运行目录的 `parents/` 下。

```bash
PTQAD_CAL_SCOPE=calib PTQ_CAL_WINDOWS=128 PTQ_CAL_BATCH=1 \
  bash exp/reproduce_ptqad.sh calibrate

"$PTQAD_PYTHON" -u exp/run_category_development.py \
  --run-dir "$PTQAD_RUN_DIR/development_v5" \
  --protocol-file "$PROTOCOL_FILE" \
  --base "$PTQAD_BASE" \
  --parent-head-lang-vision "$PTQAD_RUN_DIR/parents/head_lang_vision" \
  --parent-calib "$PTQAD_RUN_DIR/parents/calib" \
  --gr00t-repo "$GR00T_REPO" \
  --python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON" \
  --dataset "$QAD_DATASET" --media-lib "$PTQAD_MEDIA_LIB" \
  --calibration-windows 128 --calibration-batch 1 --dry-run

# 删除 --dry-run 后执行同一条命令。
```

开发阶段必须完成 BF16、`head_lang_vision_category` 和 `calib_category` 三臂，并由 `selection.json` 记录是否满足预注册压力窗口。开发分区不能替代最终 heldout。

### 3. QAD、OPD 和独立闭环

```bash
"$PTQAD_PYTHON" exp/run_category_recovery.py \
  --run-dir "$PTQAD_RUN_DIR/recovery_v5" \
  --ptq-selection "$PTQAD_RUN_DIR/development_v5/selection.json" \
  --protocol-file "$PROTOCOL_FILE" \
  --base "$PTQAD_BASE" --gr00t-repo "$GR00T_REPO" \
  --python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON" \
  --dataset "$QAD_DATASET" --validate-only

# 验证通过后，去掉 --validate-only 执行；阶段可用 --until qad_selection、
# --until opd_selection 或 --until all 逐步推进。
```

调度器会固定 QAD 学习率搜索、QAD 续训对照、学生状态采集、教师标注、OPD 权重搜索和 heldout 评测。最终只从协议绑定的 `final_manifest.json` 读取主表，不手工拼接目录中的旧结果。

### 4. 原生 APXInf 验收（可选）

```bash
bash spike/run.sh
bash exp/run_native_graph_gates.sh
```

原生算子结果与 GR00T PyTorch 闭环结果分开报告；不要用单个 GEMM 延迟推断闭环成功率。

## Ubuntu 运行截图

论文 HTML 保留全部 17 张紫色 Ubuntu 终端截图，每张截图都来自真实命令、带原始日志和 SHA-256 清单。README 展示其中四张代表性画面；完整图集可在 [paper/paper.html](paper/paper.html) 和 [截图清单](paper/evidence/captures.json) 中查看。受最终代码影响的截图会在最终实验完成后重新实拍，图位和数量保持不变。

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
python3 paper/export_zhihu.py
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 node paper/qa_browser.cjs
uv run --with-requirements paper/requirements-build.txt python paper/validate_publication.py
uv run --with-requirements paper/requirements-build.txt python paper/package_publication.py
```

含有红色占位符、缺少 heldout 原始日志或截图哈希不匹配时，`validate_publication.py` 应拒绝发布。最终交付前把经过验证的 README、HTML、知乎稿和轻量证据同步到官方 checkout，再由维护者提交和推送。

## 总结

本仓库把低位宽数值格式、模块级 PTQ、量化域恢复和真实环境行为放进同一条可复现链路。读者可以从 HTML 直接理解方法，从协议文件复现实验，从逐回合日志核对结论，从 Ubuntu 截图确认命令确实执行。最终论文只发布通过协议和证据门禁的最强结果；任何待测字段在证据齐备前都保持明确的“待回填”状态。

第三方代码、模型权重和数据遵循各自上游许可证。项目自身许可证将在正式发布时确定。
