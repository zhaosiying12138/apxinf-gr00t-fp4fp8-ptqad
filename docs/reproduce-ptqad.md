# 修正实现的复现实验入口

这份说明对应新的量化器、配方分配、完整模型校准及恢复导出，不沿用已存在的结果。所有 GPU 阶段必须串行；单次命令完成并释放服务后再执行下一阶段。`exp/reproduce_ptqad.sh` 本身不会启动后台任务，默认 `set -Eeuo pipefail`，输出目录、阶段日志及评测标签重复时拒绝覆盖。

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
export PTQAD_RUN_DIR="$PROJECT/weights/reproductions/ptqad_new_run"
```

复用实验机已有环境时，手动设置 `GR00T_REPO`、`PTQAD_PYTHON`、`LIBERO_PYTHON`、`PTQAD_MEDIA_LIB` 和 `LIBERO_CONFIG_PATH` 即可，不执行安装。`GR00T_BACKBONE_MODEL` 的本地路径必须包含字面字符串 **`nvidia/Cosmos-Reason2`**；运行工厂按这个字符串选择 Qwen3 backbone。安装脚本使用 `weights/nvidia/Cosmos-Reason2-2B` 链接；不要将它 `realpath` 成不含该字符串的 HF snapshot hash 路径。直接用远程模型名的离线加载还可能触发 tokenizer 的仓库查询，所以优先使用命名正确的本地目录。

`PTQAD_RUN_DIR` 必须是新实验目录。脚本记录源码 SHA-256、Git HEAD、未提交状态、基座与关键参数；每个阶段日志不可覆盖。正式评测直接调用仓内 `eval/run_recovery_eval.py`，不依赖实验机上未发布的 shell 文件。

固定源码中的 `libero_demo` 只有 5 个 episode、1,406 帧、3 个任务，当前管线产生 1,331 个有效窗口；它不是 LIBERO-10 全任务演示集。校准与 QAD 使用这个子集，OPD 则额外从十任务收集学生状态并获取教师标签。两组续训具有相同演示与更新预算，但并非相同数据、交互或总计算预算。

## 量化、校准与配方定义

正式开发集选择使用下一节的串行阶梯入口；它按需构建配方，不要求提前生成所有权重。RTN、mixed、aggr 仍作为可选机制参照保留，不能代替阶梯中的前驱或用于事后更改选择规则。

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

五个传统配方用于机制参照；压缩边界实验使用下列单调阶梯。每一级保留前一级全部 FP4 张量，新增的语言/视觉层使用同一 full-H 缓存与相同数值格式：

| 配方 | 相比前一级新增 FP4 | FP4 / 去已知 tied alias 后 eligible 元素 | 目标完整存储压缩比 |
|---|---|---:|---:|
| fp8 | 起点：所有 eligible action head；backbone 为 FP8 | 45.96% | 2.1614× |
| head_ffn | 语言 gate/up | 60.26% | 2.3014× |
| head_lang | 语言 q/k/v | 65.03% | 2.3522× |
| head_lang_vision | 视觉 backbone | 79.41% | 2.5201× |
| calib | 剩余 eligible 张量，包括语言 o/down 和 embedding | 100.00% | 2.8063× |

分母包含 2,815,557,632 个 eligible 存储元素；完整分母 3,144,016,000。这里去掉了 311,164,928 个重复 lm_head 元素，仅对明确记录的 tied alias 去重，不混称磁盘文件大小或激活内存。原始物理 checkpoint 的两个分母分别为 3,126,722,560 和 3,455,180,928；`ptq_recipe.json` 同时保存两种口径。合并后的稠密恢复 checkpoint 仍按原 dtype 写盘，目标压缩预算不代表实际文件已经打包。

embedding 与 lm_head 在 GR00T backbone 加载时共享 Parameter。bake 检查源两份物理张量相等，以 embedding 为 canonical，lm_head 输出复制相同值；不独立 GPTQ 其中一份。非 Linear 模块由 collector 按真实模块类型记录为 `nonlinear_targets`，这些层可显式回退 RTN；`required` 模式缺少实际 Linear 的 H 则直接报错。

```bash
export DEV_ROOT="${PTQAD_EVAL_RUNS_ROOT:-$PTQAD_RUN_DIR/evaluations}/development"
bash exp/run_development.sh --development-root "$DEV_ROOT" --dry-run
bash exp/run_development.sh --development-root "$DEV_ROOT"

# 必须在恢复训练与最终测试开始前冻结补充纯 PTQ 比较计划。
CUDA_VISIBLE_DEVICES= "$PTQAD_PYTHON" exp/recipe_inventory.py \
  --base "$PTQAD_BASE" --calibration-meta "$PTQAD_RUN_DIR/calibration/calib_meta.json" \
  --out "$DEV_ROOT/recipe_inventory.json"

python3 eval/compare_ptq_frontier.py freeze \
  --selection "$DEV_ROOT/selection.json" --development-root "$DEV_ROOT" \
  --budget "$DEV_ROOT/recipe_inventory.json" --out "$DEV_ROOT/frontier_plan.json"
```

`run_development.sh` 按 BF16、fp8、head_ffn、head_lang、head_lang_vision、calib 的顺序逐臂执行。缺失的权重调用 `reproduce_ptqad.sh` 构建；已有权重必须通过完整状态、配方、基座、源/输出 shard SHA-256、配置及统计检查。每臂使用规范子目录名，并调用 `compare_development.audit_development` 验证全十任务与初态配对；首次选出配方即停止，保存 `selection.json`，不执行剩余阶梯。完整比较前缀分别保存在 `comparison_after_<arm>.json`。`--dry-run` 只打印可能执行的命令，明确不表示资源身份已通过检查。

`recipe_inventory.py` 在 CPU 上读取 safetensors 头部并流式校验源权重 SHA-256，不加载模型、权重张量或 H 缓存。它要求 128 窗口 full-scope 校准、完整 Linear 覆盖、相同源配置/统计/权重身份，再用真实 Linear 和 `rl/lora_scope.in_scope` 计算 rank=32、alpha=64、BF16 旁路预算。输出记录本机的 base 路径，拒绝覆盖；目录迁移允许，内容身份变化不允许。论文自带的 `paper/evidence/recipe_inventory.json` 是原实验记录，不作为新克隆的 freeze 输入。

开发集输出必须是不存在的新目录；失败或已完成目录均拒绝重用。若前一次 bake 失败，应保留证据并改用新的 `PTQAD_RUN_DIR`，不能将残留目录当完整工件。该入口也拒绝在当前实验的恢复训练目录出现后重新选择配方。`PTQAD_EVAL_RUNS_ROOT` 只改变开发集根路径，不改变训练权重目录或协议。

开发集 seed 330000、每任务 2 集，收集集 seed 110000、每任务 2 集，最终测试 seed 220000、每任务 10 集。三者都覆盖完整十任务，环境数固定 1，分别固定使用官方初始状态库索引 `[0,1]`、`[2,3]`、`[10..19]`；入口按 purpose 校验集数和索引，设置初态后执行 10 步稳定处理，记录 `libero10_official_bank_v1` 协议与实际状态哈希。选择规则见 `exp/recovery_protocol.json`：按阶梯顺序取第一个开发集宏平均比同协议 BF16 下降至少 0.10 的升级配方；若没有越过阈值，选择最大压缩 calib 并如实报告。不得用最终测试结果选择配方、教师权重、训练长度或 checkpoint。将选中的名称设为 `PTQAD_RECOVERY_RECIPE`；训练开始后脚本禁止改变它。

## QAD 与基座锚定导出

```bash
export PTQAD_RECOVERY_RECIPE="$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["selection_uses_heldout"] is False and r["selected_recipe"]; print(r["selected_recipe"])' "$DEV_ROOT/selection.json")"
export QAD_LORA_SCOPE=head+lang_all
QAD_STEPS=500 QAD_GLOBAL_BATCH=16 QAD_MICRO_BATCH=1 QAD_LR=0.0001 QAD_ACTIVATION_CHECKPOINTING=1 \
  bash exp/reproduce_ptqad.sh train-qad merge-qad
```

QAD 使用冻结量化基座与 BF16/FP32 计算中的低秩旁路，训练入口检查完整模型及归一化统计。`lora_merge_bake.py --base` 必须指向训练实际使用的冻结量化基座。导出逐张量验证训练保存的冻结 W 与 base 一致，只读取 A/B 残差，以原 base 的完整键集合和 dtype 写盘；额外的未实例化 base 键保留且记录，不把 FP32 训练容器直接当部署 dtype。

正式驱动与单轮脚本均默认 `QAD_ACTIVATION_CHECKPOINTING=1`，三条训练臂使用相同的非重入逐块重算设置。该路径已通过实际两步训练 smoke，但完整运行的资源用量以各输出目录的 `runtime_metrics.json` 为准：实际步数、总耗时、参数 dtype 和各设备的 PyTorch allocated/reserved 峰值均直接读取运行状态。教师缓存已有标注耗时字段，并记录自己的设备峰值；这两类进程内指标都不包含桌面显存。

导出还强制核对训练 manifest 中的 rank、alpha、base 路径和 config/statistics/recipe 哈希；不能用任意 alpha 生成看似正常的 checkpoint。输出 index 的 `total_size` 与实际物理张量逐一核对，LoRA 张量既不留在 index，也不留在 safetensors。`ptq_recipe.json`、base provenance、`recovery_manifest.json` 与 `merge_manifest.json` 共同绑定参数来源。当前导出是稠密浮点权重；合并最后一次 dtype 舍入不保证和独立旁路逐位相同，闭环需要评测实际导出文件。

## QAD 后的 OPD 与同预算 continued-QAD

两条继续训练分支必须从同一个 QAD A/B checkpoint 和同一个冻结量化 base 开始。不要以稠密合并权重重新初始化量化基座，否则引入额外舍入与不同优化问题。先收集 QAD 学生真实访问的观察，再用完整 BF16 教师离线标注；该缓存必须标识 `student_rollout` 来源。标注流程的具体入口由 `rl/capture_onpolicy.py` 与 `rl/opd_probe_cache.py` 的运行参数提供。

```bash
# 完整一轮：收集学生观察、教师标注、两条 100-step 继续训练、五臂最终测试。
PTQ_BASE="$PTQAD_RUN_DIR/$PTQAD_RECOVERY_RECIPE" \
BF16_TEACHER="$PTQAD_BASE" \
QAD_ADAPTER="$PTQAD_RUN_DIR/train_qad/checkpoint-500" \
QAD_MERGED="$PTQAD_RUN_DIR/qad" \
ROUND_OUT="$PTQAD_RUN_DIR/round_01" \
EXTRA_STEPS=100 bash rl/run_onpolicy_round.sh
```

也可以逐阶段执行 driver：`collect-qad` 后用 `rl/opd_probe_cache.py --input-dir <collection>/observations --teacher <BF16> --count 160 --out <新缓存.pt>` 标注，再设 `OPD_CACHE_PATH`，执行 `train-continued merge-continued train-opd merge-opd`。两条路径选一种，不能重复使用同名输出。

单轮脚本在正式续训前，先从初始 QAD 适配器执行四步 OPD 冒烟，覆盖默认 `OPD_EVERY=4` 的完整教师调度周期，使用正式 microbatch、梯度累积和教师缓存，并要求记录 16 次有限教师项反向传播，检查完整模型的顺序反向传播及保存过程。该适配器只用于执行验证；两条正式分支仍从原始 QAD 适配器开始。冒烟耗时单独记录，不计入两臂各 100 个优化器更新，也不参与闭环结果选择。教师缓存同时记录教师权重和实际采样观测的散列；请求 160 个探针时，实际数量以缓存 metadata 为准。

continued-QAD 不使用教师项，OPD 使用可微的学生预测 masked MSE：共享完整 40×132 插值条件，但仅对真实 16×7 动作区域取误差并以 112 个有效元素归一化；两臂保持优化步数、演示 batch、初始 A/B 和学习率一致，并分别记录增加的教师计算成本。教师项改善成功率是待检验问题，不能由训练损失或设计目的预先决定。若只采集一次学生状态，应准确称单轮学生分布蒸馏；它不是持续交互收集的多轮在线算法。

## 纯 PTQ 前驱的补充比较

`frontier_plan.json` 在训练之前由开发集选择及编码预算冻结，仅包括选中配方之前所有已完成的纯 PTQ 前驱；例如选中 head_lang 时，前驱为 fp8 与 head_ffn。计划绑定原始选择 JSON、权重和预算身份，不读取 heldout 来决定参考组。

主五臂全部完成后运行：

```bash
bash exp/run_ptq_frontier.sh --plan "$DEV_ROOT/frontier_plan.json" \
  --round "$PTQAD_RUN_DIR/round_01" --out "$PTQAD_RUN_DIR/frontier_01" --port 5595
```

该入口串行评测冻结的参考组，不覆盖主轮目录，不改变 `recovery_protocol.json`；所有参考必须使用同一 collection manifest、每任务 10 集 heldout 及逐集配对初态。比较器验证 100 集完整结果，分别报告成功率差及计入 LoRA 旁路后的净编码预算。不得因某个前驱表现较好而遗漏它，也不能只比较 PTQ 基座字节而忽略恢复旁路。

生成一轮新结果的发布包时，应在独立发布目录将该轮冻结计划所引用的预算原字节作为 `paper/evidence/recipe_inventory.json`，让图表与冻结比较使用同一份输入；同时重新归档证据并执行发布检查。不能修改计划中的哈希来适配另一份预算，也不要覆盖正在使用的已冻结实验包。

## 闭环与结果收集

`run_onpolicy_round.sh` 的最终五臂是 BF16、选择后的 PTQ、QAD、continued-QAD、QAD→OPD。每臂使用相同 heldout seed、任务、动作步长和 episode 预算；heldout 入口必须提供收集 manifest 来同时核验官方初态索引和 seed 不交叉。新 wrapper 每集记录 reset seed 和实际初始状态哈希，因此逐集对应关系可以按证据核验；不能只凭相同集数假定配对。

正式发布以完整五臂配对比较、冻结 PTQ 前驱比较和训练成本为同一套证据链。`run_onpolicy_round.sh` 完成时已经调用 `eval/compare_recovery.py` 生成 `round_01/paired_comparison.json`；不要重复运行会拒绝覆盖的生成命令。`collect_frontier_evidence.py` 会重算五臂与前驱比较，再按字节归档原始 JSON、逐任务日志和来源映射。

以下命令在为本轮新结果准备的发布 checkout 中执行：其中 `paper/evidence/recipe_inventory.json` 必须是冻结计划引用的预算原字节，`frontier/`、`training/`、`runtime/` 和顶层比较输出均须尚不存在。已有发布包保留不动；缺失、未完成或不配对的评测不能通过归档。

```bash
set -e
cmp "$DEV_ROOT/recipe_inventory.json" paper/evidence/recipe_inventory.json
python3 paper/collect_frontier_evidence.py --frontier "$PTQAD_RUN_DIR/frontier_01"
python3 - "$PTQAD_RUN_DIR/round_01/paired_comparison.json" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1])
with Path("paper/evidence/paired_comparison.json").open("xb") as stream:
    stream.write(source.read_bytes())
PY
CUDA_VISIBLE_DEVICES= "$PTQAD_PYTHON" paper/collect_training_costs.py \
  --qad-dir "$PTQAD_RUN_DIR/train_qad" --qad-merged "$PTQAD_RUN_DIR/qad" \
  --round "$PTQAD_RUN_DIR/round_01" --out paper/evidence/training
python3 paper/collect_training_costs.py --verify-published paper/evidence/training
```

配对比较保留每集布尔结果、实际分母、初态身份与逐任务结果。成本收据分别报告 QAD、两种续训、教师标注和状态采集的真实计时范围；等演示与更新预算不表示等总计算。公开验证可重算轻量元数据和日志；未随包分发的权重、观察张量和教师缓存只保留收集时核对的散列。

运行环境与五个最终 checkpoint 必须一并归档：

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
