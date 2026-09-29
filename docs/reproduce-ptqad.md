# 高 FP4 v3 实验复现

本流程从固定 GR00T 权重出发，完成 128 窗口校准、高 FP4 配方选择、QAD／OPD 开发搜索与独立闭环评测。协议统一使用 `exp/recovery_protocol_v3_high_fp4.json`，GPU 阶段串行。校准／bake 入口为 `exp/reproduce_ptqad.sh`，开发入口为 `exp/run_development.sh`，恢复入口为 `exp/run_high_fp4_v3.py`。

## 环境与依赖

训练和 LIBERO 仿真使用两个独立 Python 3.12.14 环境。训练环境固定 PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、transformers 4.57.3、torchcodec 0.8.0；仿真环境固定 robosuite 1.4.0、MuJoCo 3.3.1、gym 0.25.2。完整逐包版本、FFmpeg 7.1.1 的 conda 显式包清单和 SHA-256 见 `setup/locks/manifest.json`，不能混用 APXInf 的 Python 环境。

`74204f0` 是实验机上的本地 GR00T 提交，不能从 NVIDIA 仓库直接 checkout。安装脚本克隆可获取的上游 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`，再应用 `patches/gr00t-recovery-runtime.patch`，重建本次运行用到的源码；它包含严格完整权重加载、FSDP2/训练支持等运行修改，未包含个人启动脚本。LIBERO 固定到 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。

```bash
# 新机器需要 NVIDIA WSL 驱动、EGL/OpenGL 系统库和已安装的 conda。
# CUDA toolkit / Rust 只在构建独立 APXInf 引擎时需要。
bash setup/01_install_dev_tools.sh
bash setup/03_download_weights.sh --core-only
CONDA_EXE="$HOME/miniforge3/bin/conda" bash setup/06_install_recovery.sh
source setup/recovery-env.sh

# 可选整合入口；任何阶段失败都会退出，不会打印虚假的全部复现完成。
# bash setup/05_restore_all.sh --with-engine
```

06 会新建 `third_party/Isaac-GR00T/.venv` 与 `.venv-libero`，拒绝覆盖另一个 revision 的现存 GR00T checkout，验证源码补丁与逐包版本。FFmpeg 独立安装到 `third_party/media7`，LIBERO 路径配置写入 `third_party/libero-config`，不修改用户的 `~/.libero`。本轮工作已验证锁文件、补丁正反向应用和脚本语法；**未在另一台空白机器执行全套安装**。包版本一致也不等同于模型加载或闭环评测通过，仍需下列冒烟验证。03 是下载入口，`--core-only` 只准备 GR00T/Cosmos；不带该选项还准备可选 π0.5 参考资源。[权重来源说明](weight-provenance.md) 给出核验过的资源、HF revision 与内容哈希；GR00T 使用已核验内容相同的固定 revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`，下载后仍通过锁定的 SHA-256 检查；这不代表恢复了原始下载记录。

```bash
export PROJECT=$(pwd)
export PTQAD_BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export QAD_DATASET="$GR00T_REPO/demo_data/libero_demo"
export PTQAD_RUN_DIR="$PROJECT/weights/reproductions/ptqad_v3_run_01"
export PTQAD_PROTOCOL_FILE="$PROJECT/exp/recovery_protocol_v3_high_fp4.json"
export PROTOCOL_FILE="$PTQAD_PROTOCOL_FILE"
export DEV_ROOT="$PTQAD_RUN_DIR/evaluations/development"
export RECOVERY_ROOT="$PTQAD_RUN_DIR/recovery_v3"
```

复用实验机已有环境时，手动设置 `GR00T_REPO`、`PTQAD_PYTHON`、`LIBERO_PYTHON`、`PTQAD_MEDIA_LIB` 和 `LIBERO_CONFIG_PATH` 即可，不执行安装。`GR00T_BACKBONE_MODEL` 的本地路径必须包含字面字符串 **`nvidia/Cosmos-Reason2`**；运行工厂按这个字符串选择 Qwen3 backbone。安装脚本使用 `weights/nvidia/Cosmos-Reason2-2B` 链接；不要将它 `realpath` 成不含该字符串的 HF snapshot hash 路径。直接用远程模型名的离线加载还可能触发 tokenizer 的仓库查询，所以优先使用命名正确的本地目录。

`PTQAD_RUN_DIR` 必须是新实验目录。脚本记录源码 SHA-256、Git HEAD、未提交状态、基座与关键参数；每个阶段日志不可覆盖。正式评测直接调用仓内 `eval/run_recovery_eval.py`，不依赖实验机上未发布的 shell 文件。

固定源码中的 `libero_demo` 只有 5 个 episode、1,406 帧、3 个任务，当前管线产生 1,331 个有效窗口；它不是 LIBERO-10 全任务演示集。校准与 QAD 使用这个子集，OPD 则额外从十任务收集学生状态并获取教师标签。两组续训具有相同演示与更新预算，但并非相同数据、交互或总计算预算。

## 量化、校准与配方定义

正式开发选择按需构建 `head_lang_vision` 与 `calib` 两个压力候选。RTN、mixed、aggr 是独立机制参照，不属于 v3 压力选择集合。

`bake.py` 保存原 dtype 容器里的反量化数值，不生成原生低比特打包权重。输出 `ptq_recipe.json` 同时记录申请方法、实际方法、RTN 回退原因、每层误差、所有未量化张量及两个分母下的精度占比。压缩比为目标格式估算，完整 checkpoint 和符合谓词的线性参数分别计算，不能解释成输出文件体积或速度。

mixed 的 FFN 匹配包含真正嵌套的 `action_head.model.transformer_blocks.<i>.ff.net.{0.proj,2}`。`--calibration-mode required` 要求真实完整模型的校准元数据及一致的缓存哈希、源权重哈希；aggr 同样读取提供的校准缓存。没有 H 的层显式记录 RTN，不能将标签保留为 GPTQ。

收集器加载 checkpoint 的真实 16 层语言模型、32 层 DiT、4 层 VL 模块，通过 `training.start_from_checkpoint` 加载权重。每层的 `XᵀX` 在禁止 autocast 的 FP32 中计算，随后累积到 CPU；`calib_meta.json` 记录实际窗口数、每层输入行数、调用数和全部 base shard 哈希。

```bash
# 先用独立目录做 2 窗口冒烟；不能将该缓存冒充 128 窗口实验。
(cd "$GR00T_REPO"; "$PTQAD_PYTHON" "$PROJECT/quant/ptq/collector.py" \
  --base "$PTQAD_BASE" --out /tmp/ptqad_calib_smoke_unique \
  --dataset "$QAD_DATASET" --recipe mixed --windows 2 --batch 1)

# 主实验：128 窗口，microbatch 1；可设 PTQ_CAL_BATCH=2。
PTQAD_CAL_SCOPE=calib PTQ_CAL_WINDOWS=128 PTQ_CAL_BATCH=1 \
  bash exp/reproduce_ptqad.sh calibrate
```

`PTQAD_CAL_SCOPE=calib` 收集所有符合条件且实际执行的线性层，便于同一个缓存支持 mixed、aggr 与 calib；mixed 专用 scope 只收 mixed 请求 GPTQ 的层。按 checkpoint 形状计算，H 存储上界约为 mixed 6.03 GiB、aggr 8.35 GiB、calib 13.82 GiB，实际因 embedding 与未实例化输出头不挂线性层钩子而更低。最大单层 K=8192 的 FP32 H 为 256 MiB。全量 scope 的 CPU/GPU 传输及累计量约为 mixed 的 2.3 倍；实际时间应由冒烟测量外推，不能预填性能数字。

若单独研究传统配方，可在新实验目录执行 `bash exp/reproduce_ptqad.sh rtn mixed aggr`；其中 GPTQ 配方应使用 full-scope 校准缓存。这些可选实验不进入下文的正式选择序列。非线性模块的二维 embedding 权重和不在实例化模型中的张量不会获得 H，必须保留逐层回退清单，不能宣称每个权重均接受了 GPTQ。

## 嵌套压缩阶梯与配方选择

工程提供下列五级单调配方。每级保留前级全部 FP4 张量，新增语言／视觉层使用同一 full-H 缓存与数值格式；v3 压力实验固定比较最后两级：

| 配方 | 相比前一级新增 FP4 | FP4 / 去已知 tied alias 后 eligible 元素 | 目标完整存储压缩比 |
|---|---|---:|---:|
| fp8 | 起点：所有 eligible action head；backbone 为 FP8 | 45.96% | 2.1614× |
| head_ffn | 语言 gate/up | 60.26% | 2.3014× |
| head_lang | 语言 q/k/v | 65.03% | 2.3522× |
| head_lang_vision | 视觉 backbone | 79.41% | 2.5201× |
| calib | 剩余 eligible 张量，包括语言 o/down 和 embedding | 100.00% | 2.8063× |

`calib` 是混合精度阶梯的全候选 FP4 终点，可量化集合中不再保留 FP8；非候选张量与恢复旁路仍使用高精度。

分母包含 2,815,557,632 个 eligible 存储元素；完整分母 3,144,016,000。这里去掉了 311,164,928 个重复 lm_head 元素，仅对明确记录的 tied alias 去重，不混称磁盘文件大小或激活内存。原始物理 checkpoint 的两个分母分别为 3,126,722,560 和 3,455,180,928；`ptq_recipe.json` 同时保存两种口径。合并后的稠密恢复 checkpoint 仍按原 dtype 写盘，目标压缩预算不代表实际文件已经打包。

embedding 与 lm_head 在 GR00T backbone 加载时共享 Parameter。bake 检查源两份物理张量相等，以 embedding 为 canonical，lm_head 输出复制相同值；不独立 GPTQ 其中一份。非 Linear 模块由 collector 按真实模块类型记录为 `nonlinear_targets`，这些层可显式回退 RTN；`required` 模式缺少实际 Linear 的 H 则直接报错。

```bash
bash exp/run_development.sh \
  --run-dir "$PTQAD_RUN_DIR" --development-root "$DEV_ROOT" \
  --protocol-file "$PTQAD_PROTOCOL_FILE" --base "$PTQAD_BASE" \
  --gr00t "$GR00T_REPO" --server-python "$PTQAD_PYTHON" \
  --rollout-python "$LIBERO_PYTHON" --dry-run

# 检查打印的路径和配置后，执行同一计划。
bash exp/run_development.sh \
  --run-dir "$PTQAD_RUN_DIR" --development-root "$DEV_ROOT" \
  --protocol-file "$PTQAD_PROTOCOL_FILE" --base "$PTQAD_BASE" \
  --gr00t "$GR00T_REPO" --server-python "$PTQAD_PYTHON" \
  --rollout-python "$LIBERO_PYTHON"

CUDA_VISIBLE_DEVICES= "$PTQAD_PYTHON" exp/recipe_inventory.py \
  --base "$PTQAD_BASE" --calibration-meta "$PTQAD_RUN_DIR/calibration/calib_meta.json" \
  --out "$DEV_ROOT/recipe_inventory.json"

# 在任何恢复训练与最终评测之前冻结纯 PTQ 参考及预算。
python3 eval/compare_ptq_frontier.py freeze \
  --selection "$DEV_ROOT/selection.json" --development-root "$DEV_ROOT" \
  --budget "$DEV_ROOT/recipe_inventory.json" \
  --protocol-file "$PTQAD_PROTOCOL_FILE" --out "$DEV_ROOT/frontier_plan.json"
```

正式协议为 `exp/recovery_protocol_v3_high_fp4.json`。开发阶段依次评测 BF16、`head_lang_vision`、`calib`，两个压力候选都须完成。合格候选的开发成功率必须比 BF16 至少低 20 个百分点，同时不低于 30%；从中选择 FP4 覆盖最高者。若没有合格候选，保存 `selected_recipe=null` 并停止恢复流程，不自动改阈值或改选其他配方。

每个开发臂保存 checkpoint 身份、逐任务日志和 `comparison_after_<arm>.json`；两个候选都完成后写 `selection.json`。已有 bake 必须通过源／输出 shard、配置、统计与 required/full-scope/128 窗口身份检查。`--dry-run`只打印计划，不表示大权重或运行资源已验收。开发目录必须全新，失败目录保留供排查。

| 分区 | 每任务回合 | 官方初态索引 | 起始 seed |
|---|---:|---|---:|
| development | 5 | 4–8 | 440000 |
| collection | 4 | 20–23 | 550000 |
| heldout | 10 | 30–39 | 660000 |
| smoke | 1 | 0 | 770000 |

正式分区覆盖十任务，n_envs=1；任务 seed 加 1000 倍任务索引，环境每集再加 episode 索引。官方初态恢复后执行 10 个原始七维零动作稳定步，每次执行动作块前 8 步，每回合最多 720 个策略环境步。日志绑定 bank、恢复状态与稳定后状态的 SHA-256。

## QAD 开发搜索与基座锚定导出

恢复入口为 `exp/run_high_fp4_v3.py`。它核对 PTQ 选择与协议身份，再串行训练两个 QAD 候选：学习率 `5e-5` 和 `1e-4`，各 500 次优化器更新。二者使用同一 PTQ 基座、训练 seed=20260929、`head+lang_all`、rank=32、alpha=64、microbatch=1、累积 16 次及有效演示 batch=16，并启用非重入激活重算。每个候选评测开发集后，选择宏平均较高者；平分时选择较小学习率。每个候选的预算为 8,000 个演示窗口，窗口可以重复。

```bash
run_recovery_v3() {
  python3 "$PROJECT/exp/run_high_fp4_v3.py" \
    --run-dir "$RECOVERY_ROOT" --ptq-selection "$DEV_ROOT/selection.json" \
    --protocol-file "$PTQAD_PROTOCOL_FILE" --base "$PTQAD_BASE" \
    --gr00t-repo "$GR00T_REPO" --python "$PTQAD_PYTHON" \
    --rollout-python "$LIBERO_PYTHON" --dataset "$QAD_DATASET" "$@"
}

run_recovery_v3 --validate-only
run_recovery_v3 --until qad_selection
```

QAD 冻结量化基座，仅更新低秩 A/B。`lora_merge_bake.py --base` 必须指向训练实际 PTQ 基座；导出逐张量核对冻结 W，只读取 A/B，按 base 的完整键集合与 dtype 写盘。未实例化的 base 键仍保留，LoRA 张量不留在 index 或物理 safetensors 中。导出核对 rank、alpha、配置、统计和配方散列，`merge_manifest.json`绑定输入／输出 shard 身份。当前产物是稠密 BF16；合并末次舍入与独立旁路不保证逐位相同，闭环评测加载实际导出文件。

非重入逐块激活重算覆盖可训练语言、DiT 和 VL 模块。首步要求 head 及 language 的 LoRA B 有有限非零梯度；B 零初始化时 A 首步梯度可为零。`runtime_metrics.json`记录实际更新、dtype、脚本初始化至训练保存结束的墙钟时间和进程内 PyTorch allocated/reserved 峰值；不包含桌面占用。执行短测方法见 `rl/RECOVERY.md`。

## OPD 开发搜索与同预算 continued-QAD

从选中的同一 QAD A/B 出发，调度器创建 continued-QAD、OPD 权重 0.25、OPD 权重 1.0 三个续训候选。三者都重置优化器，沿用选中学习率、相同学习率计划与有效演示批量，各更新 100 步；每个候选的预算为 1,600 个演示窗口。OPD 每 4 次优化器更新启用教师项，触发更新内逐累积微批执行独立探针反向。两个 OPD 候选按开发宏平均选择，平分取较小权重。continued-QAD 始终保留为同起点、同额外演示更新预算的对照。

```bash
run_recovery_v3 --until opd_selection
```

调度器先用选中 QAD 模型完成 40 个 collection 回合，采集实际送入学生的观测，再请求 160 个教师探针。缓存必须标识 `student_rollout`，实际数量、十任务覆盖、学生与教师身份均由 metadata 验收。单个样本独立 collate，不沿第 0 维切分 Qwen 扁平图像块。归一化统计沿用基座。

学生动作只作为流插值端点，教师对共享观测、完整 40×132 端点、noise/time 预测速度场。MSE 仅覆盖 16×7 有效动作区域，以 112 个元素归一化。师生探针 eval 模式禁用 dropout，学生保留 autograd；主演示损失与探针顺序反传，控制显存。逐样本随机数重放在结束时恢复状态，不改变下一批演示训练序列。

一次采集对应单轮学生分布蒸馏，参数更新后不自动刷新缓存。演示更新预算相同不等于总算力相同：学生交互、教师标注与探针计算分别计量。OPD 执行短测须覆盖 4 步调度周期与触发更新中的 16 次有限探针反向，短测适配器不进入正式训练。

## 独立闭环与纯 PTQ 参考

```bash
run_recovery_v3 --until all

# 主五臂完成后，评测开发阶段已经冻结的另一项压力候选。
bash exp/run_ptq_frontier.sh --plan "$DEV_ROOT/frontier_plan.json" \
  --round "$RECOVERY_ROOT/artifacts/heldout_round" \
  --out "$RECOVERY_ROOT/frontier" --port 5595
```

主比较为 BF16、压力 PTQ、QAD、continued-QAD、QAD+OPD。各臂使用相同 100 回合 heldout、collection 身份、动作步长与环境起点；最终测试不选择配方、教师权重、训练长度或 checkpoint。每任务完成写入逐回合布尔结果，全部十任务完成后才生成 summary。服务异常或缺失记录须补跑验收，不直接记为成功或失败。

调度器把训练、合并、开发评测、采集和教师缓存放在 `RECOVERY_ROOT/artifacts/`，原始命令日志放在 `logs/`，阶段验收记录放在 `stages/`。`artifacts/select_qad_lr/selection.json` 与 `artifacts/select_opd_weight/selection.json` 保存开发选择。最终五臂位于 `artifacts/heldout_round/heldout_<arm>/`，全部完成后写出 `paired_comparison.json` 和根目录的 `final_manifest.json`。后者绑定实际选中模型、协议与两个选择记录，归档时按它读取路径，不凭目录名猜测所选配置。

对同一个 `RECOVERY_ROOT` 重复调用时，入口先核验已完成阶段的产物与身份，再继续后续阶段。未完成目录不会自动覆盖；保留失败日志后定位问题。`--adopt-complete` 仅接受重新验真的完整工件；`--cleanup-duplicates` 仅删除已确认与保留 checkpoint 逐字节相同的训练根目录 shard，并写删除收据。

为比较恢复后的净编码收益，相邻纯 PTQ 参考需在 heldout 开始前冻结，并执行相同 100 回合。参考清单、checkpoint、预算与 collection 身份一同保存，全部已声明参考进入比较，不能根据测试分数取舍。目标预算包含缩放、未量化张量与 BF16 低秩旁路，不等于稠密导出文件或实测显存。参考证据尚未齐备时，不给出已超过纯 PTQ 前沿的结论。

## 归档与发布

归档以 `final_manifest.json` 指向的五臂模型与完整评测为入口，另附开发选择、逐层配方、预算、三项训练成本和教师采集／标注成本。开发搜索的两个 QAD 与三个续训候选保留完整日志，搜索成本另列；主表的训练成本对应最终选中的 QAD、continued-QAD 与 OPD。原始 JSON 与逐任务日志按原字节保存，来源路径、大小和 SHA-256 进入证据清单；模型权重、观察张量和教师缓存留在本地，公开包保留核对后的身份。

`paper/collect_pairing_evidence.py` 重算五臂比较并归档 16 份顶层 JSON：五臂各自的 manifest、task_results、summary，加一份总比较。它在原目录、临时副本与目标目录复核，保留原字节。`paper/collect_training_costs.py` 核对训练与教师成本，`paper/capture_runtime.py` 记录两个解释器的包、源码与模型散列。归档器必须读取同一 v3 协议和实际选择记录。量化配方、合并输出、评测 checkpoint 及训练起点应逐项连通，不能只复制一个成功率表。

相邻纯 PTQ 比较使用开始 heldout 前冻结的参考清单，保存全部已声明参考的结果与净预算。其原始证据由 `paper/collect_frontier_evidence.py` 归档；缺少完整同协议参考时，压缩前沿图继续保留待测标记。完整发布命令与校验顺序见 `paper/README.md`。

以下命令在本次结果的发布目录执行；`frontier/`、`training/`、`runtime/` 以及五臂证据输出均须为新目录。`RECOVERY_ROOT` 仍指向已完成的实验，权重和缓存不搬入发布包。

```bash
cp -- "$DEV_ROOT/recipe_inventory.json" paper/evidence/recipe_inventory.json
cmp "$DEV_ROOT/recipe_inventory.json" paper/evidence/recipe_inventory.json

python3 paper/collect_frontier_evidence.py --frontier "$RECOVERY_ROOT/frontier"
python3 paper/collect_pairing_evidence.py \
  --round "$RECOVERY_ROOT/artifacts/heldout_round" --out paper/evidence
CUDA_VISIBLE_DEVICES= "$PTQAD_PYTHON" paper/collect_training_costs.py \
  --orchestrator-run "$RECOVERY_ROOT" --protocol "$PTQAD_PROTOCOL_FILE" \
  --out paper/evidence/training
python3 paper/collect_training_costs.py --verify-published paper/evidence/training
```

从最终评测 manifest 读取五臂的真实 checkpoint 路径，再记录环境和权重身份，避免手写路径指向未选中的候选：

```bash
"$PTQAD_PYTHON" - "$RECOVERY_ROOT/final_manifest.json" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

final = json.loads(Path(sys.argv[1]).read_text())
round_dir = Path(final["heldout_round"])
command = [sys.executable, "paper/capture_runtime.py",
           "--out", "paper/evidence/runtime",
           "--gr00t", os.environ["GR00T_REPO"],
           "--server-python", os.environ["PTQAD_PYTHON"],
           "--rollout-python", os.environ["LIBERO_PYTHON"]]
for arm in ("bf16", "ptq", "qad", "continued_qad", "qad_opd"):
    record = json.loads((round_dir / ("heldout_" + arm) / "eval_manifest.json").read_text())
    if record["protocol_sha256"] != final["protocol_sha256"]:
        raise RuntimeError("Protocol identity differs: " + arm)
    command += ["--checkpoint", arm + "=" + record["checkpoint"]]
subprocess.run(command, check=True)
PY
```

公开证据的输出目录须全新，或仅含经核对字节相同的副本。最终图表从同一套证据生成，再按“HTML／Markdown 构建、浏览器检查、发布验证、ZIP 打包”的顺序验收。`paper/sections/` 是两种正文的唯一来源，17 张真实运行截图均须保留。
