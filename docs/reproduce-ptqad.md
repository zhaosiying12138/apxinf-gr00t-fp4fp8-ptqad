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

在仓库根目录加载环境文件，沿用安装脚本生成的源码、解释器、媒体库和backbone路径。下面以 `results/ptqad_v11` 为新实验根；已有同名运行时，为新实验另取名称。继续原有运行则保留原路径和协议，不重复复制协议模板。

```bash
export PROJECT="$(pwd)"
source setup/recovery-env.sh

export PTQAD_BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export QAD_DATASET="$GR00T_REPO/demo_data/libero_demo"

export V11_ROOT="$PROJECT/results/ptqad_v11"
export PTQAD_PROTOCOL_FILE="$V11_ROOT/recovery_protocol_v11_w4a4_category.local.json"
export DEV_DIR="$V11_ROOT/development"
export TRAIN_DIR="$V11_ROOT/training"
export RECOVERY_DIR="$V11_ROOT/recovery"
export SELECTION_DIR="$V11_ROOT/selection"
export PTQAD_RUN_DIR="$RECOVERY_DIR"
export PTQAD_SELECTION="$SELECTION_DIR/selection.json"
export PTQAD_CAPTURE="$TRAIN_DIR/teacher_supervision"
export PROTOCOL_FILE="$PTQAD_PROTOCOL_FILE"

# 仅首次创建本轮协议；继续已有运行时跳过这两行。
mkdir -p "$(dirname "$V11_ROOT")"
mkdir "$V11_ROOT" && cp "$PROJECT/exp/recovery_protocol_v11_w4a4_category.json" "$PTQAD_PROTOCOL_FILE"
```

安装脚本默认把GR00T放在 `third_party/Isaac-GR00T`，分别创建 `.venv` 与 `.venv-libero`；自定义安装位置时，`setup/recovery-env.sh` 会保存实际路径。README中的 `PTQAD_RUN_DIR`、`PTQAD_SELECTION`、`PTQAD_CAPTURE` 分别与本节的恢复目录、selection文件和教师采集目录对应。

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

export PTQAD_CAL_SCOPE=calib
export PTQ_CAL_WINDOWS=128
export PTQ_CAL_BATCH=1
PTQAD_RUN_DIR="$V11_ROOT/ptq_parent" \
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

QAD使用成功的BF16轨迹，其初态与development和heldout不重叠。teacher_supervision与学生collection共享20–23号初态，但使用不同策略和种子，均属于训练数据。每个任务使用teacher_supervision分区的4个初态：

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

验证器只接受成功episode的样本，并要求十个任务都至少有2个成功episode。失败或未完成的样本会被拒绝；未满足该条件时停止恢复链并检查日志。若因执行故障需要重采集，先保留原始记录，再使用新的输出目录，不能手工改写capture manifest。`--capture-dataset` 指向整个 `teacher_supervision` 目录，因为它同时包含评测manifest、成功样本和replay审计元数据；不要改为任意一组 `.pt` 文件。

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

正式阶段按顺序执行。正常完成前一条命令后，下一条会复核并复用已登记阶段；断电或进程中断按第9节处理，不能把重新执行命令等同于恢复优化器状态。

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

本节只在 `RECOVERY_DIR/final_manifest.json` 完成且第7节审计通过后执行。数字、编码预算、训练成本和截图来自同一组已选定模型。结果提取与排版是两个步骤；只运行 `build_html.py` 不会自动回填实验结果。

首先在全新的暂存目录归档证据。下面从最终清单读取真正入选的QAD checkpoint，不固定某个学习率候选：

```bash
export PUBLICATION_STAGE="$V11_ROOT/publication-stage"
mkdir "$PUBLICATION_STAGE"
"$PTQAD_PYTHON" - "$RECOVERY_DIR" "$PUBLICATION_STAGE" <<'PY'
import json, pathlib, subprocess, sys
run, stage = map(pathlib.Path, sys.argv[1:])
final = json.loads((run / "final_manifest.json").read_text())
qad = pathlib.Path(final["selected_qad_checkpoint_identity"]["path"])
subprocess.run([
    sys.executable, "paper/build_recipe_inventory_v11.py",
    "--checkpoint", final["selected_ptq_checkpoint"],
    "--recovery-manifest", str(qad / "recovery_manifest.json"),
    "--out", str(stage / "recipe_inventory.json"),
], check=True)
PY

"$PTQAD_PYTHON" paper/materialize_final_evidence.py \
  --final-manifest "$RECOVERY_DIR/final_manifest.json" \
  --orchestrator-run "$RECOVERY_DIR" \
  --recipe-inventory "$PUBLICATION_STAGE/recipe_inventory.json" \
  --out "$PUBLICATION_STAGE/bundle"

"$PTQAD_PYTHON" paper/render_recovery_results.py \
  --final-results "$PUBLICATION_STAGE/bundle/final_results.json" \
  --out "$PUBLICATION_STAGE/result-inserts"
```

归档器重新核对逐回合配对，复制选中配方与训练成本记录；插入稿包含固定五臂主表、逐任务计数和四项配对差值，保留实际正负方向。审阅 `result-inserts/` 后，将内容合入 `paper/sections/`，同步摘要和结论。成功率恢复、OPD相对continued-QAD的增量、目标编码预算和延迟各自解释，不能用一种指标代替另一种。

正式构建前，将现有的闭环发布证据移到本地归档，然后按原字节安装 `bundle/final_results.json` 到 `paper/evidence/final_results.json`，以及 `bundle/evidence/` 中的同名文件和目录到 `paper/evidence/`。保留独立原生基准和截图证据；归档中含有旧结果的文件不进入正式包。完成后生成唯一配方的最终结果图输入：

```bash
"$PTQAD_PYTHON" paper/build_final_frontier.py \
  --final-results paper/evidence/final_results.json \
  --paired-comparison paper/evidence/paired_comparison.json \
  --inventory paper/evidence/recipe_inventory.json \
  --out paper/evidence/frontier_comparison.json
```

接着重拍受本轮实现影响的Ubuntu截图，保留全部17个环节，核对画面、执行日志与选中模型，并用 `paper/record_capture.py` 更新截图清单。每张截图完成实拍和人工核对后，再解除 `paper/figures.json` 中对应的 `refresh_pending` 标记；不能只解除标记而沿用未更新的图片。运行来源也须重新采集；以下命令从五个真实评测manifest读取模型路径，输出目录必须全新：

```bash
"$PTQAD_PYTHON" - "$RECOVERY_DIR" "$PUBLICATION_STAGE" <<'PY'
import json, os, pathlib, subprocess, sys
run, stage = map(pathlib.Path, sys.argv[1:])
final = json.loads((run / "final_manifest.json").read_text())
round_dir = pathlib.Path(final["heldout_round"])
command = [sys.executable, "paper/capture_runtime.py",
           "--run-dir", str(run), "--out", str(stage / "runtime"),
           "--gr00t", os.environ["GR00T_REPO"],
           "--server-python", os.environ["PTQAD_PYTHON"],
           "--rollout-python", os.environ["LIBERO_PYTHON"]]
for arm in ("bf16", "ptq", "qad", "continued_qad", "qad_opd"):
    record = json.loads((round_dir / f"heldout_{arm}" / "eval_manifest.json").read_text())
    command += ["--checkpoint", f"{arm}={record['checkpoint']}"]
subprocess.run(command, check=True)
PY
```

审阅并安装新的 `runtime/manifest.json` 到 `paper/evidence/runtime/manifest.json`。删除正文中已完成项目的红色占位并将元信息改为正式稿后，依次生成图、两种正文、浏览器检查及发布包：

```bash
"$PTQAD_PYTHON" paper/make_figs.py
bash paper/figs/render_pngs.sh
uv run --with-requirements paper/requirements-build.txt python paper/build_html.py
uv run --with-requirements paper/requirements-build.txt python paper/export_zhihu.py
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 node paper/qa_browser.cjs
uv run --with-requirements paper/requirements-build.txt python paper/validate_publication.py
uv run --with-requirements paper/requirements-build.txt python paper/package_publication.py
```

结果未完成时，只使用 `uv run --with-requirements paper/requirements-build.txt python paper/build_review.py` 生成审阅包。它保留17个截图环节：已完成截图正常显示，待重拍环节使用文字占位，其未更新图片不进入 HTML、知乎稿或审阅 ZIP。未完成结果保留红色 `xxx`，正式发布校验会拒绝它。最终ZIP必须补齐17张实拍，并通过截图身份、逐回合配对、统计分析、训练成本、图位及离线资源检查；不能复用改稿前的浏览器验收报告。

APXInf 原生算子和 π0.5 图执行是独立基准，可按 [docs/native-pi05.md](native-pi05.md) 复现。它们验证 NVFP4 编码、scale 布局、GEMM 和 graph replay，不等于 GR00T 恢复模型已经使用原生 APXInf packed executor；两类结果不能合并成一个延迟或成功率结论。

## 9. 失败处理

新阶段的输出目录必须尚不存在。断电后先检查 `RECOVERY_DIR/run_manifest.json`、`stages/*.json` 和 `logs/`，按产物状态处理：

- 阶段有完成收据且校验通过：使用原来的命令和 `--run-dir`，驱动会复用该阶段并继续后续工作。
- 产物实际完整但缺少完成收据：审计后可追加 `--adopt-complete`；所有阶段校验通过才会补登记，它不会继续训练半成品。
- 阶段确实中断：保留日志和checkpoint，换一个全新的 `RECOVERY_DIR`，从冻结的selection、基座和教师数据重新执行恢复链。驱动拒绝覆盖不完整目录和已有日志。

训练入口使用 `save_only_model=True`，`checkpoint-*` 不含可用于恢复的优化器、调度器状态。`QAD_INIT_ADAPTER` 只是把A/B作为新训练的起点，并重建优化器；正式continued-QAD与OPD都采用这一约定。它不能把中断前后的步数拼成一次等价的连续训练。重新开始一条恢复链时同时更新 `PTQAD_RUN_DIR="$RECOVERY_DIR"`；改变协议、基座、教师数据或selection则必须使用新的实验根并重新生成相关身份记录。

临时H、失败checkpoint和重复副本只有在确认未被后续manifest引用后才能清理，并保留删除收据。正式W4A4服务从 `merge_manifest.json` 定位冻结base和原训练checkpoint中的A/B，因此这两份源工件必须保留；只留下dense merge目录不足以部署恢复模型。
