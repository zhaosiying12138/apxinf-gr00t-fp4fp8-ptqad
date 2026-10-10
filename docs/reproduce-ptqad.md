# v12 NVFP4 W4A4：RTN-PTQ、QAD 与 OPD 复现

本文档复现当前冻结的 v12 主线：**BF16 → 全覆盖 NVFP4 W4A4 RTN-PTQ → QAD → {continued-QAD，QAD+OPD}**。两个追加训练分支从同一 QAD 检查点分别开始，最后对五臂进行配对闭环评测。校准 GPTQ 对照未执行；主线 RTN 不采集 Hessian。下文训练和量化命令供独立复现，本轮发布已冻结这些阶段，不再重跑已完成实验。

479 个可量化权重张量全部采用 NVFP4；运行时 469 个普通 Linear 与 7 个 CategorySpecificLinear 使用 W4A4 激活 QDQ，另外 3 个非 Linear 张量只做权重量化。QAD/OPD 使用独立的 BF16 LoRA 旁路，七个 category 银行保持冻结。这里的闭环执行器是 PyTorch 数值参考；原生 APXInf 算子和 π0.5 的验收单独进行。

本页“从编译到执行”与根 README 使用同一个 [v12 工作流模板](../paper/readme_v12_workflow.md)，由 [`render_workflow()`](../paper/readme_v12_workflow.py) 展开三份补充模板；这里只调整文档相对链接，命令本身保持一致。所有命令均从仓库根目录执行。

## 环境、依赖与安装

目标平台为 x86_64 WSL2 Ubuntu 和 Blackwell GPU；恢复训练、LIBERO 仿真、APXInf 原生引擎使用独立环境。已记录的恢复与仿真版本包括 Python 3.12.14、PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、transformers 4.57.3、torchcodec 0.8.0、robosuite 1.4.0、MuJoCo 3.3.1、gym 0.25.2 和 FFmpeg 7.1.1。逐包版本、源码 revision 与散列以 [安装锁定清单](../setup/locks/manifest.json) 为准。

### 安装步骤

以下安装步骤以 x86_64 WSL2 Ubuntu 为目标。先在 Windows 安装支持该 Blackwell GPU 的 NVIDIA 驱动，并确认 WSL 中 `nvidia-smi` 可见 GPU；WSL 驱动按 [NVIDIA 官方指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)配置。恢复链需要系统 Python、Git LFS、curl、C/C++ 构建工具、EGL/OpenGL 和已安装的 conda。这里只提供锁定环境的重建步骤；本次证据没有覆盖一台空白系统的完整安装。

先安装 Ubuntu 通用依赖，再获取仓库：

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl git git-lfs bzip2 unzip time \
  python3 python3-venv python3-dev \
  build-essential pkg-config cmake ninja-build libssl-dev \
  libegl1 libgl1 libglx0 libglvnd0 libosmesa6 libglfw3 \
  libx11-6 libxext6 libxrender1
git lfs install
nvidia-smi

git clone https://github.com/zhaosiying12138/apxinf-gr00t-fp4fp8-ptqad.git
cd apxinf-gr00t-fp4fp8-ptqad
```

若已有 conda，直接把 `CONDA_EXE` 指向其可执行文件。否则可选用 [Miniforge 官方 release](https://github.com/conda-forge/miniforge/releases)安装：下面先解析一次 release 标签，再从同一 release 下载 x86_64 安装器和 SHA-256 文件，校验通过才执行。保留打印的版本与校验文件；它只用于创建媒体环境，FFmpeg 等实际包仍由仓库的 explicit lock 固定。

```bash
(
set -euo pipefail
test ! -e "$HOME/miniforge3" || { echo 'Miniforge prefix already exists'; exit 1; }
MINIFORGE_STAGE="$(mktemp -d -t fp4vla-miniforge-XXXXXX)"
MINIFORGE_RELEASE="$(curl -fLsS --retry 3 \
  https://api.github.com/repos/conda-forge/miniforge/releases/latest | \
  python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_RELEASE"
cd "$MINIFORGE_STAGE"
curl -fL --retry 3 "$MINIFORGE_URL/Miniforge3-Linux-x86_64.sh" \
  -o Miniforge3-Linux-x86_64.sh
curl -fL --retry 3 "$MINIFORGE_URL/Miniforge3-Linux-x86_64.sh.sha256" \
  -o Miniforge3-Linux-x86_64.sh.sha256
sha256sum --check Miniforge3-Linux-x86_64.sh.sha256
printf 'Miniforge release: %s; installer evidence: %s\n' "$MINIFORGE_RELEASE" "$MINIFORGE_STAGE"
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
)
export CONDA_EXE="$HOME/miniforge3/bin/conda"
"$CONDA_EXE" --version
```

接下来从仓库根目录执行。大模型权重、Hessian、训练 checkpoint 和虚拟环境均不进入 Git：

```bash
export CONDA_EXE="${CONDA_EXE:-$HOME/miniforge3/bin/conda}"
test -x "$CONDA_EXE" || { echo 'Set CONDA_EXE to an installed conda'; exit 1; }
bash setup/01_install_dev_tools.sh
export PATH="$HOME/.local/bin:$PATH"
bash setup/03_download_weights.sh --core-only
CONDA_EXE="$CONDA_EXE" bash setup/06_install_recovery.sh
source setup/recovery-env.sh
```

`setup/06_install_recovery.sh` 会准备固定 revision 的 GR00T、训练环境 `.venv`、仿真环境 `.venv-libero`、媒体库和恢复补丁，并生成 `setup/recovery-env.sh`。脚本需要已安装的 conda；若其路径不同，修改 `CONDA_EXE`。安装后先检查环境和权重：

```bash
python3 setup/verify_weights.py --models gr00t cosmos
```

仅复现 Torch QDQ 恢复与闭环不需要编译 APXInf。需要原生引擎、CUDA 算子或编译截图时，另外安装包含 `nvcc` 和 cuBLASLt、支持 `sm_120` 的 CUDA toolkit，以及 Rust/Cargo。CUDA 按 [官方 WSL toolkit 安装说明](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2)选择 toolkit-only 安装；WSL 不安装 Linux 显示驱动。Rust 按 [官方 rustup 安装说明](https://www.rust-lang.org/tools/install)安装。参考编译截图使用 `nvcc 13.3.73`；PyTorch wheel 的 cu128 runtime 与系统 toolkit 分别核验，不要求字符串相同。

下面只检查已经安装的原生工具链，检查成功后才启动构建：

```bash
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
(
set -euo pipefail
nvcc --version
nvcc --list-gpu-code | grep -Fx sm_120
rustc --version
cargo --version
c++ --version
make --version
bash setup/05_restore_all.sh --with-engine
)
source setup/recovery-env.sh
```

`setup/05_restore_all.sh --with-engine` 会克隆固定 revision 的 APXInf-robo 并调用引擎构建脚本；已有完整权重时可追加 `--skip-download`。不要把本机 Hugging Face snapshot 直接改名成不含 `nvidia/Cosmos-Reason2` 的路径，GR00T 工厂会用该字符串选择 backbone。

该封装入口会依次调用 `setup/01_install_dev_tools.sh`、`setup/03_download_weights.sh`、`setup/06_install_recovery.sh`，并在 `--with-engine` 时调用 `setup/02_build_engine.sh` 和 `setup/00_env_report.sh`。`setup/06_install_recovery.sh` 会按锁文件创建训练环境与独立的 `.venv-libero`；因此本项目不再单独运行旧的 `setup/04_install_libero.sh`，避免把 LIBERO 依赖装入 APXInf/训练环境。

## 协议与结果口径

主协议为 [`exp/recovery_protocol_v12_rtn_w4a4.json`](../exp/recovery_protocol_v12_rtn_w4a4.json)。独立复现需为新 checkpoint 创建本地协议副本；正式论文生成器只接受仓库冻结的发布协议。两者的用途与切换命令在下方第 2、4 步明确列出。

| 分区 | 初态索引 | 每任务回合 | 用途 |
|---|---|---:|---|
| development | 4–8 | 5 | 只用于压力臂和恢复超参数选择 |
| teacher_supervision | 20–23 | 4 | 只用于 BF16 教师成功轨迹 |
| collection | 20–23 | 4 | 只用于 QAD 学生访问状态和 OPD 探针 |
| heldout | 9–19、24–28 | 16 | 五臂最终配对评测，共 160 回合/臂 |
| smoke | 0 | 1 | 只检查服务和张量契约 |

每个正式分区包含 10 个 LIBERO 任务。评测执行动作块的前 8 步，每个 episode 最多 720 个环境步；官方初态恢复后先执行 10 个零动作稳定步。heldout 的 160 回合必须在所有选择和训练完成后运行。

须区分开发筛选与最终统计主张。开发集的压力筛选下限为 5 个百分点。冻结协议的 `primary_effect_minimum` 三个数值字段也写为 0.05，但同一字段的说明将 10 个百分点规定为最低主张；正文采用后者，更小差异作探索性报告。协议没有 `OPD−continued-QAD≥5pp` 的字段，也不能据此保证 160 回合的统计功效。保持冻结协议原样，以实际配对区间判断证据强度。

[固定分析计划](../paper/analysis_plan_w4a4.json)要求报告 PTQ−BF16、QAD−PTQ、OPD−QAD 和 OPD−continued-QAD 全部四项差值，无论其正负方向。95% 区间采用 20,000 次按任务分层的配对 bootstrap，另作双侧精确 McNemar 检验和 Holm 校正。结果只覆盖本次十个固定任务与一个训练种子；区间跨零时不能宣称已确认恢复或额外增益，也不能由这组初态评测推断未见任务泛化。

## 从编译到执行

以下命令均来自仓库实际脚本；大模型、校准缓存、训练 checkpoint 和教师张量不进入 Git。先定义新运行目录，避免把旧实验混入证据：

```bash
export PROJECT="$PWD"
source setup/recovery-env.sh
export PY="$PTQAD_PYTHON" GR00T="$GR00T_REPO" LIBERO_PY="$LIBERO_PYTHON"
export BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export DATASET="$GR00T/demo_data/libero_demo"
export RUN="$PROJECT/results/reruns/rtn_w4a4_release_$(date -u +%Y%m%dT%H%M%SZ)"
export PROTOCOL_SRC="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"
export PROTOCOL="$PROTOCOL_SRC"
export RECOVERY="$RUN/recovery_v12"
export PTQAD_ZMQ_TIMEOUT_MS=120000
export PTQAD_SERVER_READY_TIMEOUT_S=600
mkdir -p "$RUN"
```

### 0. CPU 契约和源码静态检查

```bash
"$PY" paper/run_cpu_checks.py --out "$PROJECT/paper/_build/cpu-checks-$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" -m py_compile quant/ptq/bake.py quant/ptq/bake_category.py \
  eval/run_recovery_eval.py exp/run_w4a4_recovery.py exp/make_w4a4_selection.py
```

### 1. W4A4 RTN PTQ 与 CategorySpecificLinear

`quant/ptq/bake.py` 的 `rtn` 配方不读取 Hessian，直接生成全 NVFP4 普通层；`bake_category.py --method rtn_all` 补齐七个 category bank。

```bash
export PURE_RTN="$RUN/pure_rtn" CATEGORY_OUT="$RUN/rtn_category"
"$PY" quant/ptq/bake.py --base "$BASE" --out "$PURE_RTN" --recipe rtn --calibration-mode none
"$PY" quant/ptq/bake_category.py --parent "$PURE_RTN" \
  --out "$CATEGORY_OUT" --method rtn_all --rtn-clip 1.0
```

### 2. development、教师轨迹和选择文件

冻结协议只允许 development 选择压力臂；教师轨迹和学生 collection 只用于训练，held-out 不得参与选择。

```bash
export DEV="$RUN/development" TEACHER="$RUN/teacher_supervision_v12_clean" COLLECTION="$RUN/collection" SELECTION="$RUN/selection_final"
mkdir -p "$DEV" "$COLLECTION" "$SELECTION"
# A new bake lives under RUN, so standalone reproduction uses a local protocol
# copy whose pressure checkpoint path and SHA are bound to this run. The
# checked-in PROTOCOL_SRC remains immutable and is the only input accepted by
# the formal publication writer.
cp "$PROTOCOL_SRC" "$RUN/recovery_protocol_v12_rtn_w4a4.local.json"
export PROTOCOL="$RUN/recovery_protocol_v12_rtn_w4a4.local.json"
"$PY" - "$PROTOCOL" "$CATEGORY_OUT" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["selection"]["pressure_candidate_checkpoints"]["rtn_w4a4_category"] = str(pathlib.Path(sys.argv[2]).resolve())
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
print("standalone protocol sha256:", hashlib.sha256(p.read_bytes()).hexdigest())
PY
FP4VLA_QUANT=0 FP4VLA_W4A4=0 "$PY" eval/run_recovery_eval.py --checkpoint "$BASE" --out "$DEV/bf16" --purpose development --seed 940000 --episodes 5 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6920 --protocol-file "$PROTOCOL"
FP4VLA_QUANT=0 FP4VLA_W4A4=1 "$PY" eval/run_recovery_eval.py --checkpoint "$CATEGORY_OUT" --out "$DEV/rtn_w4a4_category" --purpose development --seed 940000 --episodes 5 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6921 --protocol-file "$PROTOCOL"
python3 exp/make_w4a4_selection.py --protocol-file "$PROTOCOL" --bf16 "$DEV/bf16" --candidate rtn_w4a4_category --candidate-output "$DEV/rtn_w4a4_category" --out "$SELECTION"
FP4VLA_QUANT=0 FP4VLA_W4A4=0 "$PY" eval/run_recovery_eval.py --checkpoint "$BASE" --out "$TEACHER" --purpose teacher_supervision --seed 950000 --episodes 4 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6922 --protocol-file "$PROTOCOL"
"$PY" exp/verify_teacher_replay.py --root "$TEACHER" --protocol-file "$PROTOCOL" --teacher "$BASE"
```

### 3. QAD、continued-QAD、OPD 与五臂 held-out

本轮教师数据包含 37 条成功轨迹的 148 个窗口；每 4 次策略调用采一窗、每回合最多四窗，覆盖偏向回合前段。训练按固定路径顺序读取，micro batch=1、最多累积 16 个微批，每轮保留四窗尾批；完成 2,000 次更新时，按样本数、完成轮数及加载规则推导读取 29,600 窗。该数是含尾批的读取预算，并非独立轨迹数或逐样本计数。学生状态采集也采用相同上限，并保留失败回合的状态。

恢复驱动会依次执行两个 QAD 学习率、QAD 选择、学生状态 collection、教师缓存、continued-QAD、两个 OPD 权重和最终五臂评测。完整且身份匹配的阶段可复用；半途失败的评测必须保留故障记录并改用新输出目录。当前训练 checkpoint 不含优化器与调度器状态，不能把重启训练称为无损续跑；详见 [故障恢复规则](../docs/reproduce-ptqad.md)。

```bash
export PTQAD_ZMQ_TIMEOUT_MS=120000
"$PY" exp/run_w4a4_recovery.py --run-dir "$RECOVERY" --protocol-file "$PROTOCOL" \
  --ptq-selection "$SELECTION/selection.json" --base "$BASE" --gr00t-repo "$GR00T" \
  --python "$PY" --rollout-python "$LIBERO_PY" --dataset "$DATASET" \
  --capture-dataset "$TEACHER" --port-base 6920 --validate-only
"$PY" exp/run_w4a4_recovery.py --run-dir "$RECOVERY" --protocol-file "$PROTOCOL" \
  --ptq-selection "$SELECTION/selection.json" --base "$BASE" --gr00t-repo "$GR00T" \
  --python "$PY" --rollout-python "$LIBERO_PY" --dataset "$DATASET" \
  --capture-dataset "$TEACHER" --port-base 6920 --until all --allow-opd-nonimprovement
```

运行期间，可在已设置同一组路径变量的另一个终端查看记录进度。该命令只读日志与阶段收据；未完成记录本身不能证明进程仍在运行，日志步数也不是最终成功率。

```bash
python3 exp/high_fp4_status.py --development-root "$SELECTION" --recovery-root "$RECOVERY"
```

完整动作块的固定观测诊断入口见 [动作诊断说明](../docs/ACTION_CHUNK_DIAGNOSTICS.md)。它复用正式加载器，从最终清单解析五臂，验证实际初始噪声后比较有效动作区域；训练分区观测只用于拟合分析，不作为独立泛化证据。诊断须在训练和正式评测结束、GPU 空闲后串行执行，验证范围与运行状态以该文档为准。

### 4. 证据、图表、HTML 和知乎稿

只有冻结 v12 发布运行的 `final_manifest.json` complete 且五臂各 160 回合时才材料化；命令不会把超时或半成品写进正文。上一步独立复现使用了 `$RUN/recovery_protocol_v12_rtn_w4a4.local.json`，其 SHA 与仓库冻结协议不同；它可以用于验证方法，但不能直接喂给正式发布工具。正式发布必须切换到本次冻结 v12 运行目录，并令 `PROTOCOL="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"`。

```bash
export RECOVERY="$PROJECT/results/reruns/rtn_w4a4_release_20261006_01/recovery_v12"
export PROTOCOL="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"
export CATEGORY_OUT="$("$PY" - "$RECOVERY/final_manifest.json" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(pathlib.Path(m["selected_pressure_checkpoint"]).resolve())
PY
)"
python3 paper/extract_final_evidence.py --run-dir "$RECOVERY" --out /tmp/v12_final_results.json
export OPD_MERGE="$("$PY" - "$RECOVERY/final_manifest.json" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
p = m.get("selected_opd_model_identity", {}).get("path")
if not p:
    raise SystemExit("selected_opd_model_identity.path is missing")
print(pathlib.Path(p).resolve())
PY
)"
python3 paper/build_recipe_inventory_v12.py --checkpoint "$CATEGORY_OUT" --recovery-manifest "$OPD_MERGE/recovery_manifest.json" --recipe-name rtn_all --out /tmp/v12_recipe_inventory.json
python3 paper/materialize_final_evidence.py --final-manifest "$RECOVERY/final_manifest.json" --recipe-inventory /tmp/v12_recipe_inventory.json --out paper/_build/final_bundle_v12 --orchestrator-run "$RECOVERY"
python3 paper/install_final_evidence.py --bundle paper/_build/final_bundle_v12 --archive paper/_build/evidence_archive_v12_previous
python3 paper/install_final_evidence.py --verify paper/evidence
python3 paper/collect_search_costs.py --run-dir "$RECOVERY" --out paper/evidence/search_costs
python3 paper/collect_search_costs.py --verify paper/evidence/search_costs
python3 paper/build_final_frontier.py --final-results paper/evidence/final_results.json --paired-comparison paper/evidence/paired_comparison.json --inventory paper/evidence/recipe_inventory.json --out paper/evidence/frontier_comparison.json
```

安装器只迁移本轮科学证据和截图登记；上面的命令重新归档已完成的搜索成本。runtime、search_costs、action_diagnostics、GPTQ 未执行收据和 frontier_comparison 齐备后，再按下方补充证据步骤原子登记；缺项不会自动跳过。随后从最终五臂的评测清单读取 checkpoint，记录环境、软件包和源码哈希：

```bash
"$PY" - "$RECOVERY" "$GR00T" "$PY" "$LIBERO_PY" <<'PY'
import json, pathlib, subprocess, sys
run, groot, server_python, rollout_python = sys.argv[1:]
final = json.loads((pathlib.Path(run) / "final_manifest.json").read_text())
round_dir = pathlib.Path(final["heldout_round"])
command = [sys.executable, "paper/capture_runtime.py", "--run-dir", run,
           "--out", "paper/evidence/runtime", "--gr00t", groot,
           "--server-python", server_python, "--rollout-python", rollout_python,
           "--source-file", "exp/source_snapshots/9615ab64fd4a9cba46defe31ab1d0d58a4dd344a9d1dac1cc40f03497af9f00a/run_recovery_eval.py"]
for arm in ("bf16", "ptq", "qad", "continued_qad", "qad_opd"):
    record = json.loads((round_dir / f"heldout_{arm}" / "eval_manifest.json").read_text())
    command += ["--checkpoint", f"{arm}={record['checkpoint']}"]
subprocess.run(command, check=True)
PY
```

### 补充证据与冻结范围

本轮训练、新量化和搜索已于 2026 年 10 月 10 日冻结，未完成的 v12 开发集、held-out 正式评测与既定固定观测诊断可收尾。校准 GPTQ 补充未执行；不再收集 Hessian、不量化新权重、不启动 GPTQ 评测，不报告其成功率、配对统计或校准成本。范围决定保存在 [publication_scope_v12.json](../paper/publication_scope_v12.json)。

主五臂证据安装到 [paper/evidence/](../paper/evidence/) 后，用下面的 CPU 命令将该决定绑定到实际最终清单。它要求完整的当前五臂结果、准确的协议 SHA 和本次发布运行路径；已有目录、缺少范围决定或旧清单都会失败。

~~~bash
python3 paper/gptq_reference_publication.py --record-not-performed
python3 paper/gptq_reference_publication.py --verify
~~~

动作诊断保留已授权的首次评测复现入口：主五臂全部结束、GPU 空闲后，对冻结的检查点按既定观测和噪声分别评测一次，不更新参数、不新建量化权重或搜索候选。已经冻结的 80 个观测清单可复用，不重新 rollout；已有输出禁止覆盖。执行顺序与来源核验保留如下。当前运行中的开发集与 held-out 评测享有优先权。

~~~bash
set -euo pipefail
cd /home/zhaosiying/codebase/fp4vla
PTQAD_ROOT="$PWD"
PTQAD_GR00T=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
PTQAD_PY="$PTQAD_GR00T/.venv/bin/python"
PTQAD_RUN="$PTQAD_ROOT/results/reruns/rtn_w4a4_release_20261006_01"
PTQAD_DIAG="$PTQAD_ROOT/paper/_build/w4a4_action_diagnostics_v12"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -s "$PTQAD_RUN/recovery_v12/final_manifest.json"
python3 - "$PTQAD_RUN/recovery_v12/run_manifest.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1]))['status'] != 'complete':
    raise SystemExit('Main five-arm run has not completed')
PY
PTQAD_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$PTQAD_GPU_PIDS"
if [ ! -e "$PTQAD_DIAG/inputs.json" ]; then
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PY" \
    exp/action_chunk_diagnostics.py freeze \
    --capture-root "$PTQAD_RUN/teacher_supervision_v12_clean/observations" \
    --protocol-file exp/recovery_protocol_v12_rtn_w4a4.json \
    --per-task 8 --seed 2026100600 --out "$PTQAD_DIAG/inputs.json"
fi
for arm in bf16 ptq qad continued_qad qad_opd; do
  test ! -e "$PTQAD_DIAG/$arm"
  test ! -e "$PTQAD_DIAG/$arm.log"
  OMP_NUM_THREADS=1 "$PTQAD_PY" exp/action_chunk_diagnostics.py collect \
    --inputs "$PTQAD_DIAG/inputs.json" \
    --final-manifest "$PTQAD_RUN/recovery_v12/final_manifest.json" \
    --arm "$arm" --gr00t "$PTQAD_GR00T" --out "$PTQAD_DIAG/$arm" \
    > "$PTQAD_DIAG/$arm.log" 2>&1
done
~~~

诊断完成后，以下 CPU 命令只归档已有输出。发布动作误差结论仍要求五臂的完整有限数值张量、实际噪声身份和指标重算全部通过。如果源输出不存在，发布门禁继续失败，不能以 GPTQ 的未执行决定跳过动作证据检查。

~~~bash
PTQAD_ROOT="$PWD"
PTQAD_RUN="$PTQAD_ROOT/results/reruns/rtn_w4a4_release_20261006_01"
PTQAD_DIAG="$PTQAD_ROOT/paper/_build/w4a4_action_diagnostics_v12"
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt \
  python paper/collect_action_diagnostics.py collect \
  --source-root "$PTQAD_DIAG" \
  --final-manifest "$PTQAD_RUN/recovery_v12/final_manifest.json" \
  --out paper/evidence/action_diagnostics
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt \
  python paper/collect_action_diagnostics.py verify \
  --folder paper/evidence/action_diagnostics \
  --final-manifest paper/evidence/final_manifest.json
python3 paper/install_final_evidence.py --register-supplements paper/evidence --supplement-root paper/evidence
python3 paper/install_final_evidence.py --verify paper/evidence
~~~

补充登记仍要求 runtime、search_costs、action_diagnostics、gptq_reference 与 frontier_comparison 全部齐备；GPTQ 目录只含明确的未执行收据，禁止与测量结果混用。其余已声明的证据标准保持完整。requirements-evidence.txt 固定 Ubuntu x86_64、Python 3.12 的官方 CPU PyTorch 2.9.0 wheel 及 SHA，并使用 NumPy 1.26.4 重放指标；不需要 GPU，也不安装进恢复环境。

### 5. 截图登记

最终五臂完成后先生成本轮受控脚本，再用仓库内的 Windows/WSL 采集工具逐张运行；训练和评测进程必须退出，GPU 必须空闲。

```bash
CAPTURE_ROOT="$(dirname "$RECOVERY")/captures/w4a4_final"
python3 paper/prepare_w4a4_captures.py --mode verify-completed --final-manifest "$RECOVERY/final_manifest.json" --out "$CAPTURE_ROOT"
```

上一步已生成本轮只读截图脚本，下面直接执行；不要再次向同一输出目录运行生成器。

按 `plan_v12_completed.json` 的顺序执行七张：`shot_bake`、`shot_collect`、`shot_qad`、`shot_rollout`、`shot_opdcache`、`shot_opd`、`shot_evalserver`。它们在 CPU 上核验已完成实验的原始日志、元数据与哈希；**不训练、不量化、不采集、不生成缓存、不启动策略服务、不重复评测，也不查询 GPU**。截图展示核验命令的真实完整输出，不能标为“正在训练”或“现场闭环执行”。原有其余十张截图保留，最终仍为 17 个图位。

`legacy-smoke` 仅保留用于重建旧命令的来源记录；当前冻结发布不执行该模式，正式截图门禁也不接受它。只读模式不能在最终清单生成前截图，不接受上一轮结果作为回退。

在 Windows PowerShell 中，设置实际 WSL 项目路径。下面使用作者运行位置；独立复现时替换为自己的路径。每次只改变 `$Figure`，执行并查看一张图片后再进行下一张。

```powershell
$WslProject = '/home/zhaosiying/codebase/fp4vla'
$Repo = '\\wsl.localhost\Ubuntu' + $WslProject.Replace('/','\')
$WslCaptureRoot = "$WslProject/results/reruns/rtn_w4a4_release_20261006_01/captures/w4a4_final"
$CaptureTools = Join-Path $env:LOCALAPPDATA 'fp4vla-capture-v12'
if (-not (Test-Path $CaptureTools)) {
  Copy-Item -LiteralPath (Join-Path $Repo 'setup\windows_capture') -Destination $CaptureTools -Recurse
}
$OutDir = Join-Path $env:USERPROFILE 'shot\ptqad_v12_final'
$CapturePython = '/home/zhaosiying/.venvs/fp4vla-capture/bin/python'
$Figure = 'shot_bake'
# v12 后缀使窗口记录与先前批次的文件名不会冲突。
$OutPng = Join-Path $OutDir ($Figure + '_v12.png')
powershell.exe -ExecutionPolicy Bypass -File (Join-Path $CaptureTools 'capture_session.ps1') `
  -BashScript "$WslCaptureRoot/$Figure.sh" -OutPng $OutPng -TimeoutSec 900
if ($LASTEXITCODE -ne 0) { throw 'Capture failed; inspect the raw log before retrying.' }
$ToolsWsl = (& wsl.exe -d Ubuntu -- wslpath -a $CaptureTools.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve capture tools.' }
$ImageWsl = (& wsl.exe -d Ubuntu -- wslpath -a $OutPng.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve captured image.' }
wsl.exe -d Ubuntu -- $CapturePython "$ToolsWsl/crop_taskbar.py" $ImageWsl
if ($LASTEXITCODE -ne 0) { throw 'Crop failed.' }
```

裁剪解释器的安装、桌面条件和独立样张清单见 [截图工具说明](../setup/windows_capture/README.md)。作者沿用已批准的样张；其他机器应先确认自己的样张，并在独立复现副本中登记。检查命令、末尾输出、紫色主题、尺寸和遮挡后，才运行下面的登记命令。`CAPTURE_WINDOWS_DIR` 与 PowerShell 的 `$OutDir` 对应；`APPROVED_SAMPLE_PNG` 必须指向清单中已经批准的样张。

```bash
export CAPTURE_WINDOWS_DIR="/mnt/c/Users/Admin1/shot/ptqad_v12_final"
export APPROVED_SAMPLE_PNG="/mnt/c/Users/Admin1/shot/ptqad_20260929/sample_verified.png"
FIGURE=shot_bake
# 生成器已保存原始 plan 字节，登记前确认它未变化。
cmp "$CAPTURE_ROOT/plan.json" "$CAPTURE_ROOT/plan_v12_completed.json"
python3 paper/record_capture.py \
  --figure "$FIGURE" \
  --image "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.png" \
  --log "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.log" \
  --script "$CAPTURE_ROOT/$FIGURE.sh" \
  --sidecar "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.json" \
  --crop-manifest "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.crop.json" \
  --support-file "$CAPTURE_ROOT/common_v12.sh" \
  --support-file "$CAPTURE_ROOT/verify_completed_capture.py" \
  --support-file "$CAPTURE_ROOT/plan_v12_completed.json" \
  --support-file "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.png.window-binding.json" \
  --approved-sample "$APPROVED_SAMPLE_PNG" \
  --visually-verified
```

每次登记都校验截图与裁剪链，最后的发布校验还会核对 17 张图和本轮协议。核验过程的耗时不进入实验性能表；图中的指标只从最终原始证据读取。大体积观测和教师缓存张量不随轻量证据包公开；相关截图明确区分“核对已记录身份”与“重新检查张量字节”。

截图登记完毕后，再从同一套证据生成文稿与发布包。先前的输出目录或临时文件若已存在，工具会拒绝覆盖；复跑时使用新的 staging 名称并保留来源记录。

```bash
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/update_v12_release.py
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/write_readme_v12.py
sudo apt-get install -y libcairo2 fontconfig fonts-noto-cjk
node --version  # Node.js >=18; install from https://nodejs.org/ if absent.
npm install --prefix paper/_build/renderer --save-exact playwright@1.58.2
node paper/_build/renderer/node_modules/playwright/cli.js install --with-deps chromium
python3 paper/make_figs.py && bash paper/figs/render_pngs.sh
uv run --with-requirements paper/requirements-build.txt python paper/build_html.py
python3 paper/export_zhihu.py
node paper/qa_browser.cjs
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/validate_publication.py
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/package_publication.py --check
uv run --python 3.12 --with-requirements paper/requirements-evidence.txt python paper/package_publication.py
```

### 6. 原生 APXInf 验收（可选）

仅验证独立 CUDA 程序的编译时，先复制源码到新的 scratch 目录，避免覆盖已记录性能的二进制。以下命令对应[编译补充截图](../docs/images/ubuntu-compile.png)；实拍环境为 `nvcc 13.3.73`、`sm_120`，三个目标均编译成功，没有执行 GPU 基准。它是论文17图之外的编译补充图，不代表空白系统安装验收或新增性能结果。

```bash
(
set -euo pipefail
SPIKE_BUILD="$PROJECT/paper/_build/spike-compile-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$SPIKE_BUILD")"
mkdir "$SPIKE_BUILD"
cp "$PROJECT/spike/Makefile" "$PROJECT/spike/fp4_gemm_bench.cu" \
  "$PROJECT/spike/fp4_opbench.cu" "$PROJECT/spike/fp8_probe.cu" "$SPIKE_BUILD/"
make -C "$SPIKE_BUILD" -j1 ARCH=sm_120 \
  APXINF_FP4_ADAPTER="$PROJECT/third_party/apxinf-robo/apxinf/crates/apxinf-cuda/adapters/cublaslt_fp4_adapter.cu" \
  fp4_gemm_bench fp4_opbench fp8_probe
)
```

完成前面的 APXInf 安装后，按下面命令编译候选 wheel、准备 π0.5 packed 模型，再串行验收和计时。需要完整权重下载（不带 `--core-only`）；GPU 此时必须空闲。每个输出目录/文件必须是新的，失败时保留现场并改用新的运行名称。实现与计时边界见 [原生 π0.5 说明](../docs/native-pi05.md)。

<details>
<summary>展开：原生编译、四道验收、GEMM/单层与三条策略基准的完整命令</summary>

```bash
(
set -euo pipefail
cd "$PROJECT"
export BENCH_RUN="$PROJECT/results/independent_bench_new_run"
export RUN="$PROJECT/weights/native_pi05_new_run"
mkdir "$BENCH_RUN"
bash setup/03_download_weights.sh
export APX="$PROJECT/third_party/apxinf-robo/apxinf"
export ENGINE_PY="$PROJECT/third_party/apxinf-robo/.venv/bin/python"
source "$PROJECT/third_party/apxinf-robo/.venv/bin/activate"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="$HOME/.cuda-stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export APXINF_CUDA_ARCH=sm_120 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=target/wheel

# 1. 编译；尚未执行模型。
(
  cd "$APX"
  maturin build --release --features cuda --auditwheel skip \
    -m crates/apxinf-py/Cargo.toml
  cargo build --release -p apxinf-model --features cuda \
    --example pi05_fp4_graph_smoke
) > "$BENCH_RUN/build_engine.log" 2>&1
make -C "$PROJECT/spike" fp4_gemm_bench fp8_probe fp4_opbench \
  > "$BENCH_RUN/build_spike.log" 2>&1

# 2. CPU 打包及模型 overlay；两个阶段各自拒绝已有输出。
bash exp/prepare_native_pi05.sh pack \
  --base "$PROJECT/weights/pi05_libero_base" --out "$RUN" --chunk-rows 512
bash exp/prepare_native_pi05.sh model-overlay \
  --base "$PROJECT/weights/pi05_libero_base" --out "$RUN"

# 3. 四道 GPU 验收；全部通过才由入口安装候选 wheel。
NATIVE_PI05_MODEL="$RUN/model" NATIVE_PY="$ENGINE_PY" PTQAD_GPU_EXCLUSIVE=1 \
  bash exp/run_native_graph_gates.sh "$BENCH_RUN/native_graph"

# 4. 独立算子验收与计时。
./spike/fp8_probe > "$BENCH_RUN/fp8_probe.log" 2>&1
./spike/fp4_gemm_bench --verify --slayout=hw \
  > "$BENCH_RUN/gemm_verify.stdout" 2> "$BENCH_RUN/gemm_verify.log"
./spike/fp4_gemm_bench --slayout=hw --iters=50 \
  > "$BENCH_RUN/gemm.csv" 2> "$BENCH_RUN/gemm.log"
./spike/fp4_opbench --verify-only \
  > "$BENCH_RUN/opbench_verify.csv" 2> "$BENCH_RUN/opbench_verify.log"
./spike/fp4_opbench --warmup=10 --samples=30 \
  > "$BENCH_RUN/opbench.csv" 2> "$BENCH_RUN/opbench.log"

# 5. 三个独立模型进程依次退出，避免同时驻留。
cd /tmp
"$ENGINE_PY" "$PROJECT/exp/bench_engine.py" \
  --model-dir "$RUN/model" --variant bf16 --device cuda:0 --telemetry-device 0 \
  --warmup 10 --samples 30 --seed 7 --model-seed 0 \
  --out "$BENCH_RUN/pi05_bf16.json"
"$ENGINE_PY" "$PROJECT/exp/bench_engine.py" \
  --model-dir "$RUN/model" --variant nvfp4_static --device cuda:0 --telemetry-device 0 \
  --warmup 10 --samples 30 --seed 7 --model-seed 0 \
  --out "$BENCH_RUN/pi05_nvfp4.json"
"$ENGINE_PY" "$PROJECT/exp/bench_engine.py" \
  --model-dir "$BASE" --variant bf16 --device cuda:0 --telemetry-device 0 \
  --extra-kwarg "backbone=$GR00T_BACKBONE_MODEL" \
  --warmup 10 --samples 30 --seed 7 --model-seed 0 \
  --out "$BENCH_RUN/gr00t_bf16.json"
)
```

`.cuda-stubs` 只用于链接阶段的 `LIBRARY_PATH`，不要放入运行时 `LD_LIBRARY_PATH`。`PTQAD_GPU_EXCLUSIVE=1` 是串行使用 GPU 的声明，不会抢占其他作业。候选 wheel、完整模型验收程序与单层基准须来自同一份已应用项目补丁的 APXInf 源码。

</details>

### 7. PyTorch 参考基准（可选）

GR00T 复用恢复环境；π0.5 另用锁定的 LeRobot 环境。以下安装模板尚未在空白环境重新执行验证，锁文件记录的是实际测量环境；依赖来源与加载契约见 [PyTorch 基准说明](../docs/baseline-timing.md)。同样逐条串行运行：

<details>
<summary>展开：两个 PyTorch 参考入口与 π0.5 独立依赖安装</summary>

```bash
(
set -euo pipefail
cd "$PROJECT"
source setup/recovery-env.sh
BASELINE_RUN="$PROJECT/results/pytorch_baseline_new_run"
mkdir "$BASELINE_RUN"
bash setup/03_download_weights.sh
(
  cd "$GR00T_REPO"
  "$PTQAD_PYTHON" "$PROJECT/baselines/bench_gr00t_pt.py" \
    --checkpoint "$BASE" --backbone "$GR00T_BACKBONE_MODEL" \
    --device cuda:0 --seed 7 --warmup 10 --samples 50 \
    --out "$BASELINE_RUN/gr00t_bf16.json"
)

PI05_ENV="$PROJECT/.venv-pi05-baseline"
test ! -e "$PI05_ENV"
uv python install 3.12.14
uv venv --python 3.12.14 "$PI05_ENV"
uv pip install --python "$PI05_ENV/bin/python" --no-deps --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r setup/locks/pi05-baseline-py312.txt
CUDA_VISIBLE_DEVICES= "$PI05_ENV/bin/python" -c \
  'from lerobot.policies.pi05 import PI05Policy; print(PI05Policy.__module__)'
"$PI05_ENV/bin/python" "$PROJECT/baselines/bench_pi05_lerobot.py" \
  --checkpoint "$PROJECT/weights/pi05_libero_base" \
  --device cuda:0 --seed 7 --warmup 10 --samples 30 \
  --out "$BASELINE_RUN/pi05_bf16.json"
)
```

</details>

## 故障恢复

新阶段的输出目录必须尚不存在。断电后先检查 `RECOVERY/run_manifest.json`、`stages/*.json` 和 `logs/`，按产物状态处理：

- 阶段有完成收据且校验通过：使用原来的命令和 `--run-dir`，驱动会复用该阶段并继续后续工作。
- 产物实际完整但缺少完成收据：审计后可追加 `--adopt-complete`；所有阶段校验通过才会补登记，它不会继续训练半成品。
- 阶段确实中断：保留日志和checkpoint，换一个全新的 `RECOVERY`，从冻结的selection、基座和教师数据重新执行恢复链。驱动拒绝覆盖不完整目录和已有日志。

训练入口使用 `save_only_model=True`，`checkpoint-*` 不含可用于恢复的优化器、调度器状态。`QAD_INIT_ADAPTER` 只是把A/B作为新训练的起点，并重建优化器；正式continued-QAD与OPD都采用这一约定。它不能把中断前后的步数拼成一次等价的连续训练。重新开始一条恢复链时更新 `RECOVERY`；改变协议、基座、教师数据或selection则必须使用新的实验根并重新生成相关身份记录。

临时H、失败checkpoint和重复副本只有在确认未被后续manifest引用后才能清理，并保留删除收据。正式W4A4服务从 `merge_manifest.json` 定位冻结base和原训练checkpoint中的A/B，因此这两份源工件必须保留；只留下dense merge目录不足以部署恢复模型。
