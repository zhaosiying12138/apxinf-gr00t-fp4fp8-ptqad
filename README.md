# apxinf-gr00t-fp4fp8-ptqad

在 APXInf 生态下研究 GR00T N1.7 的 NVFP4/FP8 混合量化，以及量化后的 QAD 和学生状态教师蒸馏。项目关注的问题是：**提高 FP4 覆盖率后，用多少恢复训练和教师计算，能够保留多少闭环任务能力？**

VLA（视觉—语言—动作）策略根据图像、语言指令和机器人状态生成动作。量化造成的动作偏差会改变后续观测，因而需要在环境中反复执行策略，以任务成功率检验恢复效果。项目采用 LIBERO-10 闭环评测，并保留每个 episode 的结果和初态身份。

- 阅读：[中文 HTML](paper/paper.html) · [知乎 Markdown](paper/zhihu/article.md) · [发布包说明](paper/README.md)
- 复现：[运行说明](docs/reproduce-ptqad.md) · [恢复训练](rl/RECOVERY.md) · [高 FP4 协议文件](exp/recovery_protocol_v3_high_fp4.json)
- 核查：[配方账本](paper/evidence/recipe_inventory.json) · [环境锁](setup/locks/manifest.json) · [实验工件](results/ptqad_20260929)
- 上游：[APXInf](https://github.com/infinigence/ApxInf) · [APXinf-robo](https://github.com/RLinf/APXinf-robo)

## 项目实现

完整流程是 `checkpoint → 校准 → 混合量化 → QAD → 学生状态采集 → 教师标注 → 续训对照 → 独立闭环评测`。

| 阶段 | 做什么 | 主要产物 |
|---|---|---|
| PTQ | 用演示输入的二阶统计选择裁剪和 GPTQ 补偿，将各层分配到 NVFP4 或 FP8 | PTQ checkpoint、逐层配方、校准 manifest |
| QAD | 冻结 PTQ 基座，以演示流匹配损失训练 LoRA A/B | 适配器、恢复 manifest、稠密导出模型 |
| 学生状态蒸馏 | 保存 QAD 策略真实访问的观测，由冻结教师标注速度场 | 观察记录、可重放教师缓存 |
| 续训对照 | 从相同 QAD A/B 分出 continued-QAD 与 QAD→OPD，比较教师项的作用 | 两份续训模型及各自成本记录 |
| 闭环评测 | 在相同任务和官方初态上执行各策略 | 逐集布尔结果、十任务宏平均、配对统计 |

这里的 QAD 是演示监督的量化后低秩恢复；OPD 是单轮 on-policy 蒸馏，缓存来自采集时的学生分布。LoRA 使用 `W = W_PTQ + (alpha/rank) × B @ A`。教师与学生共享完整 40×132 流插值输入，蒸馏损失只覆盖处理器定义的 16×7 有效动作区域，以 112 个元素归一化。主演示损失和学生探针依次反向传播，教师模型在标注结束后释放。

工程同时提供 APXInf 原生 FP4 算子与 π0.5 模型路由，用于检验硬件格式、矩阵布局及图重放。GR00T 恢复链通过 PyTorch 稠密 checkpoint 评测，两类执行的口径列在文末。

## 环境与资源

参考平台为 Windows + WSL Ubuntu、RTX 5090 Laptop 24 GB（Blackwell `sm_120`），WSL 可见 24 个 CPU 线程、约 47 GiB 主机内存。完整 H 在 CPU 累积，按形状清点的存储上界约 13.82 GiB。GPU 阶段串行执行，先用短测核对当前机器的模型加载与显存。

| 环境 | 依赖 |
|---|---|
| 模型训练与服务 | Python 3.12.14、PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、transformers 4.57.3、torchcodec 0.8.0 |
| LIBERO 仿真 | 独立 Python 3.12.14 环境，robosuite 1.4.0、MuJoCo 3.3.1、gym 0.25.2，EGL/OpenGL 库 |
| 媒体 | 独立 FFmpeg 7.1.1 环境，为 torchcodec 与视频记录提供库和程序 |
| 原生引擎 | 独立 Python 环境、Rust/Cargo、Linux C++ 编译器、Make、CUDA toolkit 与 WSL NVIDIA 驱动 |

逐包版本和源码 revision 位于 [setup/locks](setup/locks)。原生编译须预装支持 `sm_120` 和 NVFP4 接口的 toolkit，包括 `nvcc`、头文件及 cuBLASLt。`setup/01_install_dev_tools.sh` 只检查或安装 uv；`setup/02_build_engine.sh` 在系统工具已就绪时应用补丁并构建/安装 wheel；`setup/05_restore_all.sh --with-engine` 串联准备步骤。

[只读工具链记录](paper/validation/native-toolchain-observation.json)保存当前观察到的 CUDA toolkit 13.3（V13.3.73）、Rust/Cargo 1.98.1，以及已有设备日志的 NVIDIA-SMI 610.53、KMD 610.74、CUDA UMD 13.3。训练 wheel 的 `+cu128` 表示其 CUDA 12.8 构建版本。已有 native 构建收据保存源码和产物哈希，但没有精确编译器版本；复现时应记录本机实际构建版本。

## 安装

从仓库根目录运行；`CONDA_EXE` 使用本机绝对路径。

```bash
git clone https://github.com/zhaosiying12138/apxinf-gr00t-fp4fp8-ptqad.git
cd apxinf-gr00t-fp4fp8-ptqad
# Ubuntu 首次准备 Git LFS；已安装时可省略安装命令。
sudo apt-get update
sudo apt-get install -y git-lfs
bash setup/01_install_dev_tools.sh
bash setup/03_download_weights.sh --core-only
CONDA_EXE="$HOME/miniforge3/bin/conda" bash setup/06_install_recovery.sh
source setup/recovery-env.sh

export PROJECT=$(pwd)
export PTQAD_BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export QAD_DATASET="$GR00T_REPO/demo_data/libero_demo"
export PTQAD_RUN_DIR="$PROJECT/weights/reproductions/ptqad_run_01"
```

`setup/06_install_recovery.sh` 创建 `third_party/Isaac-GR00T` 及独立训练、仿真和媒体环境，生成 `setup/recovery-env.sh`。它固定 GR00T commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`，应用 [运行补丁](patches/gr00t-recovery-runtime.patch)，并固定 LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`。已有 checkout 必须满足版本和补丁身份检查。

安装脚本用 Git LFS 仅拉取 `demo_data/libero_demo/**`，逐项验证 5 份 Parquet、10 段视频的 OID 和大小。演示集共 5 个 episode、1,406 帧、3 个任务，形成 1,331 个有效窗口；任务为双杯分盘、白杯与巧克力布丁摆放、杯子放入微波炉。窗口迭代允许重复抽样。

本地 backbone 使用 `weights/nvidia/Cosmos-Reason2-2B` 命名链接，上游工厂按其中的 `nvidia/Cosmos-Reason2` 字符串选择模型。保留这个命名路径，LIBERO 配置则由 `LIBERO_CONFIG_PATH` 指向项目目录。

[权重溯源](docs/weight-provenance.md)记录 23 项资源。Cosmos 与可选 π0.5 使用已有来源凭据的固定 HF revision；GR00T 使用七个文件内容均已核对的 revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`，并继续验证内容哈希。`--core-only` 下载核心 GR00T/Cosmos；π0.5 实验使用完整下载入口。已有权重时，整体准备脚本可加 `--skip-download`。

同时准备原生引擎可执行 `bash setup/05_restore_all.sh --with-engine`，系统 CUDA/Rust 工具链按前一节预先安装。

## 运行流程

以下命令展示当前维护入口的参数传递。128 窗口、500/100 次更新及教师权重是一组可执行示例；高 FP4 研究使用 `exp/recovery_protocol_v3_high_fp4.json`，其开发、采集和最终测试分区分别为 4–8、20–23、30–39。第一轮使用的 0–1、2–3、10–19 只作为探索证据保存，不能与新一轮最终结果混称。为每轮创建新的输出根目录，保留失败尝试的日志与身份。

### 1. 检查实现和模型加载

```bash
CUDA_VISIBLE_DEVICES= "$PTQAD_PYTHON" -m unittest discover -s tests -v

# 使用新目录；2 窗口只检查执行链路，不能充当正式校准集。
(cd "$GR00T_REPO"; "$PTQAD_PYTHON" "$PROJECT/quant/ptq/collector.py" \
  --base "$PTQAD_BASE" --dataset "$QAD_DATASET" \
  --out "$PTQAD_RUN_DIR/calibration_smoke" \
  --recipe calib --windows 2 --batch 1)
```

收集器检查完整模型的 16 层语言、32 层 DiT 与 4 层 VL，保留基座归一化统计，并记录源 shard 哈希、实际窗口数、逐层输入行数与调用次数。两窗口短测通过后再建立完整校准。

现有 CPU 收据记录 142 项通过、0 跳过、0 失败：主套件 96 项、发布证据检查 39 项、图表布局 7 项。命令、源码和日志哈希见 [验证目录](paper/validation)。发布与布局检查入口：

```bash
for suite in test_frontier_publication.py test_publication_guards.py test_training_publication.py test_frontier_layout.py; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PTQAD_PYTHON" -O -m unittest discover -s paper/validation -p "$suite" -v
done
```

### 2. 校准、生成配方并冻结开发选择

```bash
# 一份 full-scope H 支持后续整个阶梯；先完成校准，不预先评测所有配方。
PTQAD_CAL_SCOPE=calib PTQ_CAL_WINDOWS=128 PTQ_CAL_BATCH=1 \
  bash exp/reproduce_ptqad.sh calibrate
export DEV_ROOT="${PTQAD_EVAL_RUNS_ROOT:-$PTQAD_RUN_DIR/evaluations}/development"

# dry-run 只打印可能执行的命令，不读取大权重、不启动 GPU。
bash exp/run_development.sh --development-root "$DEV_ROOT" --dry-run
bash exp/run_development.sh --development-root "$DEV_ROOT"

# 在恢复训练与 heldout 之前冻结纯 PTQ 前驱参考组；仅使用开发集证据。
CUDA_VISIBLE_DEVICES= "$PTQAD_PYTHON" exp/recipe_inventory.py \
  --base "$PTQAD_BASE" --calibration-meta "$PTQAD_RUN_DIR/calibration/calib_meta.json" \
  --out "$DEV_ROOT/recipe_inventory.json"

python3 eval/compare_ptq_frontier.py freeze \
  --selection "$DEV_ROOT/selection.json" --development-root "$DEV_ROOT" \
  --budget "$DEV_ROOT/recipe_inventory.json" --out "$DEV_ROOT/frontier_plan.json"
```

示例按 `bf16 → fp8 → head_ffn → head_lang → head_lang_vision → calib` 逐级构建和评测，选择首个相对 BF16 开发集宏平均下降至少 0.10 的升级配方，未触发则选择最大覆盖。每一级保存 `comparison_after_<arm>.json`，最终写出 `selection.json`。新研究在开发阶段确定更高覆盖和恢复设置，再冻结独立评测清单。

`required` 模式核对完整 H、源配置、统计和权重身份；真实 Linear 缺 H 会失败，实际非 Linear 目标按记录回退 RTN。bake 对 embedding/lm_head 共享副本使用同一量化值。`recipe_inventory.py` 在本机重建预算；`frontier_plan.json` 固定已完成的纯 PTQ 前驱，供最终补充比较。

### 3. 训练 QAD 并导出

```bash
# 从已完成的开发集选择读取，不手填或根据最终测试更换配方。
export PTQAD_RECOVERY_RECIPE="$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["selection_uses_heldout"] is False and r["selected_recipe"]; print(r["selected_recipe"])' "$DEV_ROOT/selection.json")"
export QAD_LORA_SCOPE=head+lang_all
QAD_STEPS=500 QAD_MICRO_BATCH=1 QAD_GLOBAL_BATCH=16 QAD_LR=0.0001 QAD_ACTIVATION_CHECKPOINTING=1 \
  bash exp/reproduce_ptqad.sh train-qad merge-qad
```

示例采用 `head+lang_all`、rank=32、alpha=64，microbatch=1、累积 16 次得到有效演示 batch=16。`QAD_ACTIVATION_CHECKPOINTING=1` 为语言、DiT 和 VL 可训练块启用非重入激活重算。训练器检查首步 head/language 的 B 梯度，写出 `recovery_manifest.json` 和 `runtime_metrics.json`，记录实际更新数、dtype、墙钟时间及设备 allocated/reserved 峰值。

导出器只从训练 checkpoint 取 A/B，逐张量核对冻结权重与 PTQ base，按 base 的键集合及 dtype 写出完整模型。`merge_manifest.json` 保存 rank/alpha、配置/统计/配方散列、残差和输出 shard 身份。

### 4. 收集状态，比较两条续训分支

```bash
PTQ_BASE="$PTQAD_RUN_DIR/$PTQAD_RECOVERY_RECIPE" \
BF16_TEACHER="$PTQAD_BASE" \
QAD_ADAPTER="$PTQAD_RUN_DIR/train_qad/checkpoint-500" \
QAD_MERGED="$PTQAD_RUN_DIR/qad" \
ROUND_OUT="$PTQAD_RUN_DIR/round_01" \
EXTRA_STEPS=100 OPD_WEIGHT=1.0 OPD_EVERY=4 \
  bash rl/run_onpolicy_round.sh
```

脚本依次采集真实学生观测、生成教师标签、执行完整教师调度周期的短测，再训练和导出两条续训分支，最后评测五臂。两条分支精确继承同一 QAD A/B、同一 PTQ 基座，均重新初始化 Adam 和学习率计划。示例教师项每 4 次更新触发，触发更新内逐微批执行独立探针反向。

collection 覆盖十任务，默认最多保留 160 个观察探针；每个样本独立 collate，保存 BF16 输入舍入和共享 noise/time 的重放条件。其交互、教师标注及学生探针计算在成本表中单列。两条续训分支匹配的是演示与优化器更新预算。

### 5. 完成独立评测并归档

示例评测使用一个环境、每次执行 8 个动作步、每集上限 720 步。官方 bank 状态恢复后执行 10 个原始零动作稳定步，保存 bank、恢复状态及稳定后状态三项哈希。

| 分区 | 每任务回合 | 官方 bank 索引 | 起始 seed |
|---|---:|---|---:|
| development（高 FP4 v3） | 5 | 4–8 | 440000 |
| collection（高 FP4 v3） | 4 | 20–23 | 550000 |
| heldout（高 FP4 v3） | 10 | 30–39 | 660000 |

任务 seed 加 `1000 × task_index`，每集再加 episode index。入口按版本化协议核查参数，主比较为 BF16、相邻 PTQ、压力 PTQ、QAD、continued-QAD、QAD→OPD；所有任务完成后生成 `summary.json` 与 `paired_comparison.json`。旧 10–19 结果只用于第一轮探索审计，新研究结果单独归档。

五臂完成后，按已冻结计划评测纯 PTQ 前驱：

```bash
bash exp/run_ptq_frontier.sh --plan "$DEV_ROOT/frontier_plan.json" \
  --round "$PTQAD_RUN_DIR/round_01" --out "$PTQAD_RUN_DIR/frontier_01" --port 5595
```

在本轮独立发布 checkout 中，先放入冻结计划引用的预算原字节。以下输出目录使用新路径，归档器重算比较、核对初态与来源，保存原始 JSON、任务日志和成本；已有相同字节副本可以核验复用。

```bash
set -e
cmp "$DEV_ROOT/recipe_inventory.json" paper/evidence/recipe_inventory.json
python3 paper/collect_frontier_evidence.py --frontier "$PTQAD_RUN_DIR/frontier_01"
python3 paper/collect_pairing_evidence.py \
  --round "$PTQAD_RUN_DIR/round_01" --out paper/evidence
CUDA_VISIBLE_DEVICES= "$PTQAD_PYTHON" paper/collect_training_costs.py \
  --qad-dir "$PTQAD_RUN_DIR/train_qad" --qad-merged "$PTQAD_RUN_DIR/qad" \
  --round "$PTQAD_RUN_DIR/round_01" --out paper/evidence/training
python3 paper/collect_training_costs.py --verify-published paper/evidence/training
```

再绑定实际解释器与五份最终 checkpoint：

```bash
"$PTQAD_PYTHON" paper/capture_runtime.py \
  --out paper/evidence/runtime --gr00t "$GR00T_REPO" \
  --server-python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON" \
  --checkpoint "bf16=$PTQAD_BASE" \
  --checkpoint "ptq=$PTQAD_RUN_DIR/$PTQAD_RECOVERY_RECIPE" \
  --checkpoint "qad=$PTQAD_RUN_DIR/qad" \
  --checkpoint "continued_qad=$PTQAD_RUN_DIR/round_01/continued_qad_merged" \
  --checkpoint "qad_opd=$PTQAD_RUN_DIR/round_01/qad_opd_merged"
```

### 6. 原生算子与引擎实验

完成原生环境安装后，按 [native-pi05.md](docs/native-pi05.md) 准备 packed 模型并运行四项 GPU 验收：矩阵布局与 tensor scale、图重放、scale padding、完整 π0.5 `RequireGraph`。通过后安装候选 wheel，再测量对应源码和扩展。

```bash
bash spike/run.sh

# 完成 --with-engine 安装后，从仓库外目录运行以避免 Python 包遮蔽。
(cd /tmp; "$PROJECT/third_party/apxinf-robo/.venv/bin/python" \
  "$PROJECT/exp/bench_engine.py" --model-dir "$PTQAD_BASE" \
  --variant bf16 --extra-kwarg "backbone=$GR00T_BACKBONE_MODEL" \
  --warmup 10 --samples 30 --out "$PROJECT/results/engine/gr00t_bf16_run01.json")
```

`bench_engine.py` 在预热后和每个样本后记录只读 `execution_mode`，π0.5 graph 路径返回 `graph`，GR00T 返回 `cuda-graph`。PyTorch 参考入口和计时范围见 [baseline-timing.md](docs/baseline-timing.md)；π0.5 LeRobot 是可选参考环境。

## 当前结果

<!-- BEGIN CURRENT RESULTS -->

结果区将在新独立评测完整结束后统一生成。

<!-- END CURRENT RESULTS -->

## 配方账本

下表按已核对模型形状计算编码预算，描述精度分配与目标载荷。eligible 是 `.weight`、二维且输入宽度可被 16 整除的张量；其余参数保留源精度。它是配方账本，不是已测文件体积或新一轮行为结果。

| 配方 | 新增 FP4 范围 | FP4 / eligible 元素 | FP4 / 全部元素 | 目标完整编码压缩比 |
|---|---|---:|---:|---:|
| fp8 | 起点：动作头；backbone 为 FP8 | 45.96% | 41.16% | 2.1614× |
| head_ffn | 语言 FFN gate/up | 60.26% | 53.97% | 2.3014× |
| head_lang | 语言注意力 q/k/v | 65.03% | 58.24% | 2.3522× |
| head_lang_vision | 视觉 backbone | 79.41% | 71.12% | 2.5201× |
| calib | 剩余 eligible 张量 | 100.00% | 89.55% | 2.8063× |

账本去除了已明确识别的 tied lm_head 别名：eligible 分母为 2,815,557,632 个元素，完整分母为 3,144,016,000。物理 checkpoint 保存两份共享权重，其对应分母为 3,126,722,560 和 3,455,180,928；两种口径都在 [recipe_inventory.json](paper/evidence/recipe_inventory.json) 中公开。

预算包含 FP4 payload、每 16 元素的 E4M3 scale、张量全局 scale，以及 FP8 的行 scale；未包括对齐、激活和优化器。上表为 PTQ 基座预算。`head+lang_all`、rank=32 的 LoRA 另外含 58,580,992 个元素；若以 BF16 保存旁路，需增加 117,161,984 字节后再计算压缩比。恢复评测实际加载稠密原 dtype checkpoint；文件体积和推理耗时分别取自运行产物。

## Ubuntu 运行记录

保留全部 17 张截图。15 张新短测有 [capture 清单](paper/evidence/captures.json)，另两张 BF16 执行展示有 [保留清单](paper/evidence/retained_captures.json)。每张图按下面的实际模型、任务及样本范围阅读；原始命令、日志、图像哈希和裁剪链可沿清单核对。定量表由对应完整实验工件生成。

<details>
<summary>量化、校准与产物</summary>

**01 · 完整模型的 2 窗口校准短测**

![01 · 完整模型的 2 窗口校准短测](paper/figs/shot_collect.png)

完整 GR00T（语言 16 层、DiT 32 层、VL 4 层）以 batch=1 抽样 2 个演示窗口，得到 468 层二阶统计。H 在 GPU 上按 FP32 逐层计算并累加至 CPU；记录范围是本次 2 窗口抽样，窗口允许重复。

**02 · 2 窗口 H 驱动的 head_lang 写盘短测**

![02 · 2 窗口 H 驱动的 head_lang 写盘短测](paper/figs/shot_bake.png)

使用本次新采集的 2 窗口 H，以 `required` 模式完成 80 个 NVFP4-GPTQ、253 个 NVFP4-RTN 和 139 个 FP8 张量。输出为 BF16 容器中的量化格点值。终端 `2.6860×` 仅为 eligible 物理张量子集的目标编码预算；配方账本中的 `2.3522×` 包含未量化张量并对已知共享副本去重，使用完整纯 PTQ 模型口径。

**03 · π0.5 单个语言层的 NVFP4 打包短测**

![03 · π0.5 单个语言层的 NVFP4 打包短测](paper/figs/shot_packed.png)

从真实 checkpoint 打包第 0 个语言层的 qkv 与 gate_up 两个张量，包含归一化折叠、payload 与硬件 scale 布局。图中 `orig_bytes=289,406,976` 按 **FP32** 统计，`packed_bytes=40,697,856` 仅含 payload 与物理 scale；约 7.11× 是相对 FP32，同元素 BF16 参照约 3.56×；记录范围为这个单层子集。

**04 · 有效动作掩码与探针梯度的 CPU 短测**

![04 · 有效动作掩码与探针梯度的 CPU 短测](paper/figs/shot_probe.png)

微型模型验证掩码、随机状态恢复与串行探针反向；合成 processor/statistics 配置夹具验证 40×132 容器中的 16×7 有效区域（112 个元素）。测试对象为 CPU 夹具与微型模型。

</details>

<details>
<summary>恢复训练与教师标注</summary>

**05 · 完整模型 QAD 的 2 步短测**

![05 · 完整模型 QAD 的 2 步短测](paper/figs/shot_qad.png)

本次截图使用完整 GR00T、`head+lang_all`、rank=32、alpha=64，并开启激活重算；microbatch=1、梯度累积=2、有效 batch=2，共 2 次优化器更新与 4 次演示窗口抽样。日志与 checkpoint 对应该次 2 步执行。

**06 · 2 条真实学生访问观测的教师标注短测**

![06 · 2 条真实学生访问观测的教师标注短测](paper/figs/shot_opdcache.png)

从本次单任务闭环的 2 条真实学生访问观测生成 version-3 缓存。完整教师冻结，参数按 FP32 存储、使用 BF16 autocast；保留完整 40×132 插值端点和共享 noise/time 的重放种子，仅有效 16×7 输出参与后续 masked MSE。缓存规模为 2 个观察样本。

**07 · 从 QAD 适配器继续的 4 步教师辅助短测**

![07 · 从 QAD 适配器继续的 4 步教师辅助短测](paper/figs/shot_opd.png)

精确载入本次 QAD `checkpoint-2` 的 728 个 A/B 张量，以新优化器执行 4 次更新：microbatch=1、梯度累积=2、有效 batch=2，共 8 次演示窗口抽样。教师项权重为 1、每 4 步触发；第 4 步执行 2 次学生对缓存教师目标的串行探针反向，教师本体冻结且 `no_grad`。完整 `checkpoint-4` 已保存。记录范围为训练链路，额外探针计算另计。

**08 · 激活重算与 LoRA 梯度的 CPU 小模型检查**

![08 · 激活重算与 LoRA 梯度的 CPU 小模型检查](paper/figs/shot_qat.png)

真实 Qwen、DiT 和 VL 模块的小尺寸集成测试比较重算前后的输出及 LoRA 梯度；微型 dropout 模型另检验 RNG 恢复。图名 `shot_qat` 沿用文件 key，内容是激活重算测试。

</details>

<details>
<summary>闭环评测</summary>

**09 · 两步 QAD 导出后的策略服务端与健康 RPC**

![09 · 两步 QAD 导出后的策略服务端与健康 RPC](paper/figs/shot_evalserver.png)

将本次 `checkpoint-2` 的 LoRA 导出至独立 BF16 checkpoint，加载真实 LIBERO 策略服务端，执行 `ping` RPC 后正常关闭。记录范围为稠密模型导出、加载与服务健康检查。

**10 · 学生闭环与观测采集（1 任务 × 1 回合短测）**

![10 · 学生闭环与观测采集（1 任务 × 1 回合短测）](paper/figs/shot_rollout.png)

本次两步 QAD 学生在 LIBERO-10 第一个任务上运行单环境、单回合：官方 bank index=4、环境 seed=910000，先执行 10 个 raw-zero 稳定步。唯一计分回合成功，日志长度为 285 步，保存视频及 2 条真实访问观测；后续自动 reset 只留下初始化记录，不增加计分回合。该单回合使用独立的 smoke 分区。

</details>

<details>
<summary>数值格式、算子与原生引擎</summary>

**11 · 量化格点与最小 checkpoint 导出的 CPU 短测**

![11 · 量化格点与最小 checkpoint 导出的 CPU 短测](paper/figs/shot_verify.png)

覆盖量化格点、共享权重和最小跨分片 LoRA 导出契约。记录范围为数值与导出契约。

**12 · FP8 描述符可用性与同步返回状态**

![12 · FP8 描述符可用性与同步返回状态](paper/figs/shot_fp8probe.png)

GPU 探针初始化完整输入，记录所测组合的 heuristic、matmul 与 CUDA 同步返回状态。`NO ALGO` 表示该描述符组合没有可用算法。

**13 · 9 种形状的 GEMM 短测（每项 5 次）**

![13 · 9 种形状的 GEMM 短测（每项 5 次）](paper/figs/shot_gemm.png)

先执行 decoded-input 参考校验，再运行全部 9 种形状、每项 5 次的短 sweep。MXFP4 不可用项没有有效耗时；该图记录入口短测，性能表采用独立计时文件。

**14 · 18 个实际层形状的算子短测（预热 1、采样 3）**

![14 · 18 个实际层形状的算子短测（预热 1、采样 3）](paper/figs/shot_opbench.png)

全部 18 个实际层形状均保留，包括较慢形状。NVFP4 口径包含在线激活量化与 adapter；预热 1 次、采样 3 次用于入口短测。

**15 · GR00T BF16 执行流程展示（10 次采样）**

![15 · GR00T BF16 执行流程展示（10 次采样）](paper/figs/shot_gr00t.png)

该图保留模型加载与动作输出过程，采样数为 10。

**16 · π0.5 BF16 执行流程展示（10 次采样）**

![16 · π0.5 BF16 执行流程展示（10 次采样）](paper/figs/shot_pi05.png)

该图保留模型加载与动作输出过程，采样数为 10。 图中的变换统计提示随原始输出保留。

**17 · π0.5 NVFP4 完整模型路径短测（预热 1、采样 2）**

![17 · π0.5 NVFP4 完整模型路径短测（预热 1、采样 2）](paper/figs/shot_nvfp4.png)

使用独立、完整的 π0.5 packed checkpoint，记录原生 NVFP4 路径的实际 `execution_mode` 与输出有限性。该图使用独立 π0.5 模型，范围为预热 1 次、采样 2 次的入口检查。

</details>

## 目录

```text
setup/       安装脚本、逐包锁文件与环境配置
patches/     固定上游源码之上的运行补丁
quant/       数值格式、PTQ 分配、校准统计与权重写盘
rl/          QAD、学生观察采集、教师标注、LoRA 导出
eval/        仓内模型服务与固定初态的 LIBERO 评测
exp/         实验协议、串行执行入口、独立引擎基准
spike/       原生低精度算子验证与 microbenchmark
tests/       CPU 数值、梯度、缓存与 checkpoint 一致性检查
results/     实验日志和可解析的结果；大工件不入库
paper/       中文论文、图表、17 张截图、知乎发布包和证据
docs/        详细运行说明
weights/     本地权重与实验 checkpoint，不入库
third_party/ 固定版本上游源码及独立环境，不入库
```

## 测量范围与复现说明

| 内容 | 采用的范围 |
|---|---|
| GR00T 行为 | 量化格点基座与恢复后的稠密 checkpoint，由 PyTorch 策略服务执行闭环。 |
| 目标编码 | FP4/FP8 payload、scale、未量化参数与适用的低秩旁路；实际磁盘字节另取 shard。 |
| 原生性能 | 算子形状、模型调用和含前后处理的策略耗时分别计量，绑定实际 wheel、variant、warmup 与样本数。 |
| 数据与成本 | 演示窗口可重复抽样；OPD 额外使用学生交互、教师标注与探针计算，成本单独报告。 |
| 成功率与统计 | 十任务宏平均，同时保存逐任务分子/分母、micro 及探索性配对统计。配对身份是环境初态。 |
| 环境重建 | 源码补丁、锁文件与脚本已核验；在新机器上按加载、校准、训练、仿真阶段逐项验收。 |

这个工程把低精度格式、恢复训练和真实环境行为连到可追溯的产物上。最终评估同时回答任务能力、目标编码规模和恢复成本，原生性能作为独立执行路径报告。

第三方代码、权重与数据遵循各自上游许可。论文包提供可编辑 Markdown、离线 HTML 和图片，构建脚本只生成本地文件。
