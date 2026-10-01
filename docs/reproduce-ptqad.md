# v11 W4A4 category：PTQAD 实验复现

本文档只描述仓库当前的 **v11 W4A4 category** 主线：

`BF16 → 全覆盖 NVFP4 权重 PTQ → QAD → continued-QAD → QAD+OPD → LIBERO-10 闭环`。

v11 的目标是把 479 个可量化权重张量全部写成 NVFP4；其中 469 个普通 Linear 和 7 个 CategorySpecificLinear 在运行时使用 W4A4 激活 QDQ，另外 3 个 embedding/位置张量只有权重量化。QAD 与 OPD 保留独立的 BF16 低秩残差。本文档中的成功率只指最终协议的原始逐回合记录，不接受 smoke 或 development 分数代替。

## 1. 环境和依赖

推荐 WSL2 Ubuntu、支持 NVFP4 的 Blackwell GPU、NVIDIA 驱动和 CUDA。已核验的参考软件版本为 Python 3.12.14、PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、transformers 4.57.3、torchcodec 0.8.0、robosuite 1.4.0、MuJoCo 3.3.1、gym 0.25.2 和 FFmpeg 7.1.1。只有复现 APXInf 原生算子时才需要 Rust、nvcc、cuBLASLt 和 C++ 编译器。

完整逐包版本、源码 revision 和文件散列见 [setup/locks/manifest.json](../setup/locks/manifest.json)。训练环境和 LIBERO 仿真环境必须分开。

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

需要 APXInf 原生引擎时，再执行：

```bash
bash setup/05_restore_all.sh --with-engine
source setup/recovery-env.sh
```

安装脚本只准备依赖、固定 revision 和权重目录；它不会自动开始量化、训练或闭环评测。大模型权重、Hessian、训练 checkpoint 和教师缓存保留在本机，不进入 Git。

## 2. 统一路径和协议

每次打开新 shell 都先加载环境文件，然后为本轮实验创建新的结果目录。不要复用含有其他协议或失败阶段的目录。

```bash
export PROJECT="$(pwd)"
source setup/recovery-env.sh

export GR00T_REPO="$PROJECT/third_party/Isaac-GR00T"
export PTQAD_PYTHON="$GR00T_REPO/.venv/bin/python"
export LIBERO_PYTHON="$GR00T_REPO/.venv-libero/bin/python"
export PTQAD_BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export QAD_DATASET="$GR00T_REPO/demo_data/libero_demo"
export PTQAD_MEDIA_LIB="$PROJECT/third_party/media7/lib"

export V11_ROOT="$PROJECT/results/ptqad_v11"
export PTQAD_PROTOCOL_FILE="$V11_ROOT/recovery_protocol_v11_w4a4_category.local.json"
export DEV_DIR="$V11_ROOT/development"
export TRAIN_DIR="$V11_ROOT/training"
export RECOVERY_DIR="$V11_ROOT/recovery"
export SELECTION_DIR="$V11_ROOT/selection"
mkdir -p "$V11_ROOT"
cp "$PROJECT/exp/recovery_protocol_v11_w4a4_category.json" "$PTQAD_PROTOCOL_FILE"
```

| 分区 | 初态索引 | 每任务回合 | 用途 |
|---|---|---:|---|
| development | 4–8 | 5 | 只用于压力臂和恢复超参数选择 |
| teacher_supervision | 20–23 | 4 | 只用于 BF16 教师成功轨迹 |
| collection | 20–23 | 4 | 只用于 QAD 学生访问状态和 OPD 探针 |
| heldout | 9–19、24–28 | 16 | 五臂最终配对评测，共 160 回合/臂 |
| smoke | 0 | 1 | 只检查服务和张量契约 |

每个正式分区包含 10 个 LIBERO 任务。评测执行动作块的前 8 步，每个 episode 最多 720 个环境步；官方初态恢复后先执行 10 个零动作稳定步。heldout 的 160 回合必须在所有选择和训练完成后运行。

在新机器上，协议中的候选路径需要改成当前机器的绝对路径。协议内容一旦改变，SHA-256 就会改变；必须从新的 `V11_ROOT` 开始，不能把旧日志接到新协议上。

## 3. CPU 冒烟和源码检查

```bash
"$PTQAD_PYTHON" -m unittest discover -s tests -v
"$PTQAD_PYTHON" -m py_compile \
  quant/ptq/collector.py quant/ptq/collector_category.py \
  quant/ptq/bake.py quant/ptq/bake_category.py \
  rl/lora_qad.py rl/lora_merge_bake.py \
  exp/run_w4a4_recovery.py exp/make_w4a4_selection.py
```

smoke 只能使用协议的 index=0 分区。它证明服务能启动、W4A4 环境变量和输入输出契约有效；smoke 的成功率、延迟和日志不能进入最终结果表。

## 4. 128 窗口校准和全覆盖 W4A4 PTQ

校准器在真实 GR00T 前向中收集每层输入的 FP32 二阶矩 `XᵀX`，逐层把统计累积到 CPU。正式 v11 使用 128 个窗口、batch=1；2 窗口只用于检查依赖和路径。

```bash
(cd "$GR00T_REPO"; "$PTQAD_PYTHON" "$PROJECT/quant/ptq/collector.py" \
  --base "$PTQAD_BASE" --out "$V11_ROOT/calibration_smoke" \
  --dataset "$QAD_DATASET" --recipe calib --windows 2 --batch 1)

export PTQAD_RUN_DIR="$V11_ROOT/ptq_parent"
export PTQAD_CAL_SCOPE=calib
export PTQ_CAL_WINDOWS=128
export PTQ_CAL_BATCH=1
PTQAD_PROTOCOL_FILE="$PTQAD_PROTOCOL_FILE" \
  bash exp/reproduce_ptqad.sh calibrate

PTQAD_RUN_DIR="$V11_ROOT/ptq_parent" \
  bash exp/reproduce_ptqad.sh calib
```

量化账本 `ptq_recipe.json` 必须记录每个张量的请求方法、实际方法、校准散列、回退原因、scale 和共享 alias。压缩预算按编码载荷、scale、未量化张量和恢复旁路分别计账，不能把稠密 BF16 文件大小当成低位宽压缩比。

七个类别权重沿真实输入轴收集 128 个窗口，再由 `bake_category.py` 写回同一个 PTQ parent：

```bash
export PTQ_PARENT="$V11_ROOT/ptq_parent/calib"
export CATEGORY_CALIB_OUT="$V11_ROOT/category_calibration"
export CATEGORY_OUT="$V11_ROOT/w4a4_full_category"

PTQ_PARENT="$PTQ_PARENT" CATEGORY_CALIB_OUT="$CATEGORY_CALIB_OUT" \
CATEGORY_OUT="$CATEGORY_OUT" bash exp/rebuild_w4a4_category.sh

test -f "$CATEGORY_OUT/category_ptq_recipe.json"
test -f "$CATEGORY_OUT/category_bake_manifest.json"
```

把本机 checkpoint 写入 v11 协议副本：

```bash
"$PTQAD_PYTHON" - "$PTQAD_PROTOCOL_FILE" "$CATEGORY_OUT" <<'PY'
import json, pathlib, sys
protocol = pathlib.Path(sys.argv[1])
candidate = str(pathlib.Path(sys.argv[2]).resolve())
d = json.loads(protocol.read_text())
d["selection"]["pressure_candidate_checkpoints"]["all_nvfp4_gptq_category"] = candidate
d["quantization_scope"]["candidate_checkpoint"] = candidate
protocol.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
PY
```

## 5. development 压力筛选

v11 只注册 `all_nvfp4_gptq_category` 一个候选；不允许根据 heldout 分数改配方。两条 development 输出必须使用相同任务顺序和初态 pairing。

```bash
mkdir -p "$DEV_DIR"

FP4VLA_QUANT=0 FP4VLA_W4A4=0 FP4VLA_W4A4_ADAPTER=0 \
"$PTQAD_PYTHON" eval/run_recovery_eval.py \
  --checkpoint "$PTQAD_BASE" --out "$DEV_DIR/bf16" \
  --purpose development --seed 940000 --episodes 5 \
  --gr00t "$GR00T_REPO" --server-python "$PTQAD_PYTHON" \
  --rollout-python "$LIBERO_PYTHON" --port 5890 \
  --protocol-file "$PTQAD_PROTOCOL_FILE"

FP4VLA_QUANT=0 FP4VLA_W4A4=1 FP4VLA_W4A4_ADAPTER=0 \
"$PTQAD_PYTHON" eval/run_recovery_eval.py \
  --checkpoint "$CATEGORY_OUT" --out "$DEV_DIR/all_nvfp4_gptq_category" \
  --purpose development --seed 940000 --episodes 5 \
  --gr00t "$GR00T_REPO" --server-python "$PTQAD_PYTHON" \
  --rollout-python "$LIBERO_PYTHON" --port 5891 \
  --protocol-file "$PTQAD_PROTOCOL_FILE"

"$PTQAD_PYTHON" exp/make_w4a4_selection.py \
  --protocol-file "$PTQAD_PROTOCOL_FILE" \
  --bf16 "$DEV_DIR/bf16" \
  --candidate all_nvfp4_gptq_category \
  --candidate-output "$DEV_DIR/all_nvfp4_gptq_category" \
  --out "$SELECTION_DIR"
```

选择器会检查逐回合日志、reset identity、协议散列和 category recipe，并要求 PTQ 相对 BF16 的 micro success rate 落在协议声明的 5–60 个百分点压力窗口内，且 PTQ 绝对成功率不低于 30%。如果未通过，流程应停止并记录没有可恢复压力窗口，不能直接修改门槛。

## 6. BF16 教师成功轨迹

QAD 使用成功的 BF16 轨迹；这部分数据独立于 development、collection 和 heldout。每个任务使用 teacher_supervision 分区的 4 个初态：

```bash
mkdir -p "$TRAIN_DIR"

FP4VLA_QUANT=0 FP4VLA_W4A4=0 FP4VLA_W4A4_ADAPTER=0 \
"$PTQAD_PYTHON" eval/run_recovery_eval.py \
  --checkpoint "$PTQAD_BASE" --out "$TRAIN_DIR/teacher_supervision" \
  --purpose teacher_supervision --seed 950000 --episodes 4 \
  --gr00t "$GR00T_REPO" --server-python "$PTQAD_PYTHON" \
  --rollout-python "$LIBERO_PYTHON" --port 5892 \
  --protocol-file "$PTQAD_PROTOCOL_FILE"

"$PTQAD_PYTHON" exp/verify_teacher_replay.py \
  --root "$TRAIN_DIR/teacher_supervision" \
  --protocol-file "$PTQAD_PROTOCOL_FILE" \
  --teacher "$PTQAD_BASE"
```

验证器只接受成功 episode 的样本，并要求十个任务都至少有 2 个成功 episode。失败或未完成的样本会被拒绝；如果该分区不足，必须换用新的 `TRAIN_DIR` 重新运行，不能手工改写 capture manifest。恢复驱动的 `--capture-dataset` 参数应指向整个 `teacher_supervision` 目录，因为它同时包含评测 manifest、成功样本和 replay 审计元数据。

## 7. QAD、continued-QAD 和 OPD

恢复驱动从同一个冻结 PTQ checkpoint 开始，先训练两个 QAD 学习率候选，再按 development macro success rate 选择一个；随后从选中的 QAD checkpoint 分出 continued-QAD 和两个 OPD 权重候选。continued-QAD 与 OPD 使用相同追加更新预算，OPD 还读取 collection 分区的学生访问状态和 BF16 教师探针。

先做配置审计：

```bash
"$PTQAD_PYTHON" exp/run_w4a4_recovery.py \
  --run-dir "$RECOVERY_DIR" --protocol-file "$PTQAD_PROTOCOL_FILE" \
  --ptq-selection "$SELECTION_DIR/selection.json" \
  --base "$PTQAD_BASE" --gr00t-repo "$GR00T_REPO" \
  --python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON" \
  --dataset "$QAD_DATASET" \
  --capture-dataset "$TRAIN_DIR/teacher_supervision" \
  --port-base 5890 --validate-only
```

正式阶段按顺序执行；中断后从同一命令继续即可：

```bash
# QAD 学习率选择
"$PTQAD_PYTHON" exp/run_w4a4_recovery.py \
  --run-dir "$RECOVERY_DIR" --protocol-file "$PTQAD_PROTOCOL_FILE" \
  --ptq-selection "$SELECTION_DIR/selection.json" \
  --base "$PTQAD_BASE" --gr00t-repo "$GR00T_REPO" \
  --python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON" \
  --dataset "$QAD_DATASET" --capture-dataset "$TRAIN_DIR/teacher_supervision" \
  --port-base 5890 --until qad_selection

# collection、teacher probe cache、continued-QAD 和 OPD development
"$PTQAD_PYTHON" exp/run_w4a4_recovery.py \
  --run-dir "$RECOVERY_DIR" --protocol-file "$PTQAD_PROTOCOL_FILE" \
  --ptq-selection "$SELECTION_DIR/selection.json" \
  --base "$PTQAD_BASE" --gr00t-repo "$GR00T_REPO" \
  --python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON" \
  --dataset "$QAD_DATASET" --capture-dataset "$TRAIN_DIR/teacher_supervision" \
  --port-base 5890 --until opd_selection

# 五臂 heldout：BF16、PTQ、QAD、continued-QAD、QAD+OPD
"$PTQAD_PYTHON" exp/run_w4a4_recovery.py \
  --run-dir "$RECOVERY_DIR" --protocol-file "$PTQAD_PROTOCOL_FILE" \
  --ptq-selection "$SELECTION_DIR/selection.json" \
  --base "$PTQAD_BASE" --gr00t-repo "$GR00T_REPO" \
  --python "$PTQAD_PYTHON" --rollout-python "$LIBERO_PYTHON" \
  --dataset "$QAD_DATASET" --capture-dataset "$TRAIN_DIR/teacher_supervision" \
  --port-base 5890 --until all
```

最后一条命令在 `RECOVERY_DIR/artifacts/heldout_round/` 写出五个 heldout 目录和 `paired_comparison.json`，并在 `RECOVERY_DIR/final_manifest.json` 绑定协议、selection、QAD learning rate、OPD weight 和五个 checkpoint。每臂 10 个任务 × 16 回合，共 160 回合；主摘要同时报告 micro success rate 和十任务 macro success rate。v11 预注册的最小效应量为 PTQ 相对 BF16 下降至少 5 个百分点、QAD 相对 PTQ 恢复至少 5 个百分点、OPD 同时高于 QAD 和 continued-QAD 至少 5 个百分点；更小差异只能作为探索性结果。

如需审计阶段收据：

```bash
"$PTQAD_PYTHON" exp/audit_recovery_release.py \
  --run-dir "$RECOVERY_DIR" \
  --out "$V11_ROOT/recovery-audit.json"
```

## 8. 证据归档和论文构建

五臂评测完成后，先按 final manifest 复制证据，再构建文章；不要手工把分数写入 HTML 或 Markdown。

```bash
"$PTQAD_PYTHON" paper/extract_final_evidence.py \
  --run-dir "$RECOVERY_DIR" \
  --out "$PROJECT/paper/evidence/final_results.json"

uv run --with-requirements paper/requirements-build.txt python paper/build_html.py
uv run --with-requirements paper/requirements-build.txt python paper/export_zhihu.py
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 node paper/qa_browser.cjs
uv run --with-requirements paper/requirements-build.txt python paper/validate_publication.py
uv run --with-requirements paper/requirements-build.txt python paper/package_publication.py
```

结果未完成时，使用 `uv run --with-requirements paper/requirements-build.txt python paper/build_review.py`；审阅构建会保留 17 张 Ubuntu 截图，并用红色 `xxx` 占位。`validate_publication.py` 按设计拒绝该审阅稿。正式稿必须通过截图 SHA-256、逐回合配对、训练成本、图位和离线资源校验。

APXInf 原生算子和 π0.5 图执行是独立基准，可按 [docs/native-pi05.md](native-pi05.md) 复现。它们验证 NVFP4 编码、scale 布局、GEMM 和 graph replay，不等于 GR00T 恢复模型已经使用原生 APXInf packed executor；两类结果不能合并成一个延迟或成功率结论。

## 9. 失败处理

所有正式输出目录必须先不存在；脚本遇到已存在的阶段日志或身份不一致会停止。断电后先读取 `run_parameters.json`、`stages/*.json` 和最近日志，再用同一个 `--run-dir` 继续。协议、基座、教师缓存或 selection 改动后必须新建结果根目录。临时 H、失败 checkpoint 和重复副本可以在确认没有被 manifest 引用后删除，并保留删除收据；不要删除仍被 final manifest、selection 或论文证据清单引用的文件。
