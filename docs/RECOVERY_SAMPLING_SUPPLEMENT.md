# QAD/OPD 时间覆盖与状态分布补充实验

日期：2026-10-07。状态：完整候选采集、成对视图、QAD 训练来源检查、原 PTQ 证据的显式复用、阶段 B 配对计划、共用 QAD 的端点生成、教师缓存/训练准入及三臂串行开发入口已在独立分支 `experiment/full-trajectory-capture` 实现。[补充协议草案](../exp/recovery_sampling_protocol_draft.json)尚未冻结，GPU 补充实验尚未启动；真实推理、缓存、三臂训练仍待完成。本方案不修改正在运行的 v12 协议、训练代码或数据。它用于定位恢复不足的原因，不预设任一实验臂胜出。

## 已核实的输入限制

[`audit_capture_coverage.py`](../exp/audit_capture_coverage.py) 读取已结束的采集目录，核对任务清单、原始 rollout 日志哈希、逐回合长度、reset 身份和每个窗口的成功标签。它只用 CPU，不加载策略，也不推断抓取或放置阶段。

本轮[完整审计 JSON](evidence/recovery-capture-coverage-v12.json)确认：十任务共 40 回合，37 个成功回合各保留四窗，共 148 窗；三个失败回合的 12 窗另行保存为 rejected。成功来源回合长度为 164–487 步。采集器每四次服务查询取一窗，达到每回合四窗后不再保存；后续训练重新采样流噪声并不能增加环境观测覆盖。

从仓库根目录复查，输出必须位于原始证据目录以外：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  /home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T/.venv/bin/python \
  exp/audit_capture_coverage.py \
  --source results/reruns/rtn_w4a4_release_20261006_01/teacher_supervision_v12_clean \
  --out paper/_build/capture_coverage_teacher_v12.json
```

独立安装时把解释器替换为 `setup/recovery-env.sh` 中的 `PTQAD_PYTHON`。同一工具也支持已完成的 `collection` 目录：这时 `sample_*.pt` 包含失败回合中的学生状态，不能按教师成功筛选规则拒绝它们。报告将“保留窗口数”和“来自成功回合的窗口数”分别列出。

`server_call` 是任务级累计查询号，不能当作回合内步数。完整回合长度来自日志的 `episode_lengths`，不能用只在保存窗口时更新的 `capture_counts.calls` 代替。审计通过仅证明记录一致，不证明覆盖了某个操控阶段。

## 实验 A：固定四窗预算，只改变时间位置

使用现有 BF16 教师、W4A4 RTN 基座及训练初态 20–23。重新采集十任务各四回合，仍每四次查询保存候选，但持续到回合结束。同一批完整候选派生两份数据，避免两次 rollout 的随机差异混入采样比较。

两份数据的规则如下：

| 数据视图 | 确定性选择规则 | 共同条件 |
|---|---|---|
| `head4` | 每回合最早四个候选 | 同一批源轨迹、同一成功过滤、相同每回合窗口数 |
| `stratified4` | 候选按时间顺序等分四组，每组取中位候选；偶数长度取靠前者 | 不按损失、动作误差或成功率选择窗口 |

若一回合不足四个候选，两组均使用全部实际候选，不复制补齐。新一轮成功数和窗口数以采集结果为准，不能预填为 37/148。终止时必须确认最后一段候选仍在，不能仅增大一个不足以覆盖长回合的前缀配额。

这里的 `head4` 与 `stratified4` 是同一新采集中的成对比较，不是旧 v12 的逐样本复刻。新采集每回合从第 1、5、9…次查询重新计数；旧采集沿用任务级查询相位。新路径还复用原 collator 输出。因此不能将新 `head4` 与旧 v12 的差异全部归因于时间覆盖。本文补充实验中的“演示状态”特指 BF16 教师成功轨迹中的观测。

两份数据分别训练 QAD：同一初始化、rank=32、alpha=64、学习率 `5e-5`、相同随机种子和 2,000 次更新；不再搜索学习率。它们使用同一个新的补充协议，避免跨协议续训。先在预先固定的 development 分区各评测 50 回合，并报告逐任务差异；不能只比较训练损失。是否扩展后续实验由开发证据与动作诊断决定，不能依据 v12 held-out 分数改变采样规则。

## 实验 B：控制额外教师监督，比较观测来源

若开展本阶段，预先指定 `QAD-stratified4` 为共同起点，分别进行三项 2,000 次更新：

| 配置 | 训练目标 | 识别的因素 |
|---|---|---|
| continued-QAD | 演示流匹配 | 追加演示更新 |
| demo-state KD | 演示流匹配＋演示观测上的教师速度 MSE | 演示分布上的额外教师监督 |
| student-state OPD | 演示流匹配＋学生访问观测上的教师速度 MSE | 改用学生状态后的教师监督 |

两个 KD 臂固定 `lambda=0.25`、`opd_every=1`。三臂匹配优化器重置、学习率计划、演示顺序、实际窗口读取数、更新数、梯度裁剪和数值开关。两个 KD 臂另匹配教师标注数、探针前向/反传数、缓存循环方式与有效动作掩码。

学生用同一训练初态完成 40 回合，也从全时域选四窗，并保留失败状态。两个 KD 缓存按预先声明的任务/初态身份规则匹配数量；为保持演示状态的成功来源定义，只使用具有对应成功教师轨迹的身份，但不能再按学生成功与否过滤。这个匹配使机制对照只覆盖该共同身份集合，不声称使用了全部失败分布。

**两组动作端点都由同一个冻结 QAD 重新生成。** 在各自观测上统一运行正式 `get_action`，然后由同一个教师、相同噪声/时间采样规则标注。不能让 demo-state KD 用原演示端点、OPD 用学生端点，否则同时改变了观测分布与插值问题。端点生成复用 `exp/action_chunk_diagnostics.py::infer_chunk` 和正式服务加载器，先移除旧 `action/action_mask`，避免启用实时动作拼接。

诊断同时检查训练缓存与独立观测/独立噪声，分别报告速度误差和四步积分后的动作误差。更低缓存 MSE 不自动意味着更高闭环成功率。

## 实现、预算与执行顺序

现有入口的实现状态如下，不重新实现推理或训练：

1. 已实现 `rl/capture_onpolicy.py` 的完整候选模式：记录每回合每次查询，持续保存 `candidate_*.pt`；安全容量耗尽会记错并停止，不能当作完整采集。
2. 已实现 `eval/run_recovery_eval.py` 的显式协议开关及完整候选最终化；未配置时维持 v12 前缀采样。完整采集启动前记录实际权重及引用的 base/adapter 文件身份，完成时重新核对；派生视图拒绝缺少完成确认或已经变更的检查点。`rl/capture_sampling.py` 校验逐回合计数、reset 身份、标签及源文件哈希，再确定性选择两份视图。`exp/derive_capture_views.py` 检查十任务完整日志、协议和原始结果，生成等预算视图及来源映射，不生成新的评测结果。
3. 已接入现有 `verify_teacher_replay.py` 入口，通过独立的派生视图规则检查来源。检查器重算两份选择计划，逐张量核对派生文件与源候选，核对十任务原始日志、BF16 教师身份及协议。`legacy_audit_compatible=false` 仍保留，表示派生视图不伪装成旧原始 rollout。训练器会重新审计，并将报告摘要与调度请求比较；每个样本读取后、反序列化前再核对字节 SHA，避免检查后换入别的输入。这项训练加载保护仅用于新派生视图。
4. 已实现 `exp/ptq_sampling_reference.py`：在原协议下重新核查 PTQ 选择、原始开发日志与权重，再给新协议附上显式来源。新协议只允许到 `qad_dev` 或只读验证，不得用原分数授权新的后续恢复选择或 held-out 评测。
5. 已实现 `exp/probe_endpoint_plan.py` 的 CPU 计划：重新审计教师和学生的原始 rollout、两份视图与检查点，按共同任务/初态和查询顺序配对；保留学生失败，固定每对端点/速度种子，拒绝不匹配的有效动作掩码或任务预算。起点必须为同一教师 `stratified` 视图上训练的纯 QAD，核对数据、训练参数及完整步数收据，拒绝 `head`、continued-QAD 或 OPD 检查点误入。计划只记录后续推理输入，不产生动作、教师速度或成功率。
6. 已实现 `exp/probe_endpoint_bundle.py`：复用正式服务加载器和 `infer_chunk`，在两个来源的观测上由同一 QAD 生成完整动作端点；逐对核对实际初始噪声，而非只核对 seed。保留原始来源和有效掩码，pad 端点也完整保存。任何中断、来源变更或额外文件都会使数据包无法通过加载检查。尚未执行真实 GR00T GPU 端点生成。
7. 已扩展 `rl/opd_probe_cache.py` 的显式 `--endpoint-bundle` / `--endpoint-role` 接口；按计划使用全部样本和各自速度重放 seed，拒绝截断数量或覆盖 seed。演示状态仍标为 `teacher_rollout`，学生状态仍标为 `student_rollout`；动作端点策略另行记录，不能改写观测来源。
8. 已实现 `rl/endpoint_cache_guard.py` 并接入现有 `lora_qad.py` / `ProbeAnchor`。三个续训臂均检查同 QAD 起点、演示数据和追加预算；两个蒸馏臂再逐张量核对缓存输入、端点、种子和标注源码。检查在权重分支前执行，不能以误设零权重绕过。新缓存按审计 SHA 核验实际读入字节，再反序列化。旧 v12 协议不增加这些来源要求。
9. 新增 `exp/run_state_distillation_dev.py`，复用原 Driver 的训练、导出、开发评测与完成收据。它在首臂训练前预审三组输入，核对端点 QAD 使用的基座与原 PTQ 选择一致；只运行三项固定续训，不选择赢家、不执行 held-out。协议必须另行显式声明 `state_distillation_execution`；旧 `run_high_fp4_v3.py --until qad_dev` 的边界保持不变。

## 新接口的使用与验证范围

新协议的 `teacher_supervision` / `collection` 分区可显式加入以下字段；这只是接口示例，不是已冻结的实验协议。开发和测试分区禁止打开此开关。启动前会检查与协议中已声明的 development、heldout、smoke 的初态和 reset seed 隔离；这不能替代跨历史实验的初态使用账本。

```json
"capture_sampling": {
  "mode": "full_trajectory_candidates",
  "every_server_calls": 4,
  "safety_candidates_per_episode": 192
}
```

训练准入还要求在同一协议的顶层固定视图模式和每回合窗口数。未声明、模式不符或派生时使用其他窗口预算，均不能进入正式恢复训练：

```json
"capture_views": {
  "modes": ["head", "stratified"],
  "windows_per_episode": 4
}
```

192 是拒绝异常运行的安全容量，不是每回合选窗数。它覆盖 720 步回合中即使每步都请求策略的候选上限；正常每次执行八步动作时会保存更少候选。只支持一个环境和逐回合 reset 事件。捕获不额外运行 collator 或模型；CPU 桩模型测试验证了动作张量及 Python、NumPy、Torch RNG 状态不变，实际 GR00T 的同输入验证仍待 GPU 空闲后执行。

完成新的候选采集后，在仓库根目录执行以下 CPU 命令生成两份视图；路径由操作者填入新协议的实际采集输出，命令不启动 rollout 或训练：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PYTHON" \
  exp/derive_capture_views.py \
  --source "$FULL_CAPTURE_RUN" \
  --out "$CAPTURE_VIEWS_OUT" \
  --windows-per-episode 4
```

输出为 `head/observations/<task>/`、`stratified/observations/<task>/`、成对选择计划和 `views_manifest.json`。源候选保持原样；教师失败回合输出为 `rejected_*.pt`，学生失败状态仍保留为可训练样本。输出目录必须全新且位于源评测目录以外。若日志被改动、候选缺失、计数不全、协议身份不一致或任一任务未完成，工具拒绝生成完成收据。它证明查询序列覆盖和来源一致，不能证明抓取或放置等语义阶段已经覆盖。

准备训练时使用原来的 CPU 审计入口，必须只选一份视图；传入两份视图共同的父目录会被拒绝，避免把相同轨迹重复加入训练：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PYTHON" \
  exp/verify_teacher_replay.py \
  --root "$CAPTURE_VIEWS_OUT/head" \
  --protocol-file "$SAMPLING_PROTOCOL" \
  --teacher "$BF16_TEACHER"
```

`stratified` 使用同一入口；每任务至少需要两个实际贡献样本的成功教师回合。调度器已有 `--capture-dataset` 参数可传 `head` 或 `stratified`，并自动把指定教师和审计摘要交给训练器。阶段 A 必须限定在 `--until qad_dev`，两个数据臂分别使用独立输出目录和同一固定学习率。旧调度器的后续 collection/cache 阶段仍不自动串联阶段 B；不能把单个接口实现宣称为整套补充恢复链已跑通。直接启动训练器时，新视图必须提供与审计一致的 `QAD_CAPTURE_TEACHER` 和 `QAD_CAPTURE_AUDIT_SHA256`，缺少时会明确报错。

原 PTQ 选择已经可以通过草案的 `ptq_reference` 显式接入。它分别固定源协议、源选择文件和新协议的 SHA，要求量化格式、开发分区、评测契约及压力选择规则一致；原选择中的 `protocol_sha256` 保持原值，新增 `reference_provenance` 说明这是复用证据而非新评测。调度器在构造、冻结复查及恢复运行时重新核对这些身份，并限制 `--until qad_dev`。不声明这个字段时，原有同协议检查不变。

已用 CPU 在真实 v12 权重和原始日志上验证该入口，所引用的原开发证据仍为 BF16 46/50、RTN W4A4 41/50；它没有重新运行开发评测，也不是补充采样结果。草案中的 held-out 字段只为沿用协议结构，不能据此执行新的正式测试。新协议最终身份必须在启动补充采集之前冻结，不能边采集边改。

阶段 B 完整配对计划的入口如下。只有新的教师全时域视图、预定 QAD 和对应学生采集都完成后才能执行；命令本身只用 CPU，输出必须是源证据目录以外的新文件：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PYTHON" \
  exp/probe_endpoint_plan.py \
  --teacher-view "$TEACHER_VIEWS_OUT/stratified" \
  --student-view "$STUDENT_VIEWS_OUT/stratified" \
  --protocol-file "$SAMPLING_PROTOCOL" \
  --qad-checkpoint "$QAD_STRATIFIED_CHECKPOINT" \
  --teacher-checkpoint "$BF16_TEACHER" \
  --out "$ENDPOINT_PLAN_OUT"
```

计划匹配的是同一初态身份下的窗口顺序，不表示两个策略在同一时刻访问了同一个物理状态。教师与学生 rollout 使用各自预声明的 reset seed；二者的官方 bank 与初态字节必须相同。两类状态始终保留各自来源，后续端点则统一由冻结 QAD 生成。

下面是 GPU 空闲且阶段 A/学生采集完成后的执行入口，当前未执行。先生成一个同时包含两种来源的端点包，再分别运行两个教师标注进程；无需打开 RPC 服务或重跑 rollout。

```bash
"$PTQAD_PYTHON" exp/probe_endpoint_bundle.py \
  --plan "$ENDPOINT_PLAN_OUT" --out "$ENDPOINT_BUNDLE_OUT" --gr00t "$PTQAD_GR00T"

PTQAD_SOURCE_ROOT="$PWD"
cd "$PTQAD_GR00T"
"$PTQAD_PYTHON" "$PTQAD_SOURCE_ROOT/rl/opd_probe_cache.py" \
  --teacher "$BF16_TEACHER" --endpoint-bundle "$ENDPOINT_BUNDLE_OUT" \
  --endpoint-role teacher --dataset "$PTQAD_DEMO_DATASET" --out "$TEACHER_STATE_CACHE"
"$PTQAD_PYTHON" "$PTQAD_SOURCE_ROOT/rl/opd_probe_cache.py" \
  --teacher "$BF16_TEACHER" --endpoint-bundle "$ENDPOINT_BUNDLE_OUT" \
  --endpoint-role student --dataset "$PTQAD_DEMO_DATASET" --out "$STUDENT_STATE_CACHE"
cd "$PTQAD_SOURCE_ROOT"
```

上述变量均应使用绝对路径。缓存不接受 `--count` 或 `--seed`，避免两组预算被单独改变。训练继续复用 `rl/lora_qad.py`；新协议的 `state_distillation` 固定两个 KD 臂权重 0.25、追加 2,000 步、学习率 `5e-5` 和梯度裁剪 0.25。续训环境需声明 `QAD_ENDPOINT_ROLE=continued/teacher/student`。continued 臂通过 `QAD_ENDPOINT_BUNDLE` 绑定共同起点但不读取教师速度标签；另外两臂从 `OPD_CACHE_PATH` 追溯同一包。新入口会显式写入这些变量，不能只在调用者 shell 中导出后交给旧分支。

## 三臂开发入口

完成全时域采集、两份 QAD、学生采集、端点包和两个教师缓存后，先运行下面的 CPU 检查。`SOURCE_PTQ_SELECTION` 指向草案已经绑定的原选择文件，`STATE_DEV_OUT` 必须是源证据和模型目录之外的独立输出；检查会创建运行清单，但不会训练或评测。

```bash
STATE_DEV_ARGS=(
  --run-dir "$STATE_DEV_OUT"
  --protocol-file "$SAMPLING_PROTOCOL"
  --ptq-selection "$SOURCE_PTQ_SELECTION"
  --endpoint-bundle "$ENDPOINT_BUNDLE_OUT"
  --teacher-cache "$TEACHER_STATE_CACHE"
  --student-cache "$STUDENT_STATE_CACHE"
  --gr00t-repo "$PTQAD_GR00T"
  --python "$PTQAD_PYTHON"
  --rollout-python "$LIBERO_PYTHON"
  --dataset "$PTQAD_DEMO_DATASET"
)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PTQAD_PYTHON" \
  exp/run_state_distillation_dev.py "${STATE_DEV_ARGS[@]}" --validate-only
```

主链结束、协议已冻结且 CPU 检查通过后，再使用相同参数串行执行。该命令尚未在真实 GPU 产物上运行：

```bash
"$PTQAD_PYTHON" exp/run_state_distillation_dev.py "${STATE_DEV_ARGS[@]}"
```

入口固定执行 `continued_qad → teacher_state_kd → student_state_opd`，每臂依次训练、导出并在同一 development 分区评测；最后写入 `artifacts/paired_summary/paired_comparison.json`。报告含三臂逐任务计数和三组逐回合配对比较，全部标记为开发证据，不据此发布正式成功率。continued 与两个 KD 臂的额外 GPU 成本仍分别记录。

已完成阶段会重新核验证据后复用；无完成收据的中断产物会报错，不能假定自动恢复训练成功。`--adopt-complete` 只接纳通过完整验收的既有产物。原 PTQ 分数保留源协议 SHA；新入口的开发执行范围、共同 QAD、端点与缓存另外写入运行清单，不能冒称原分数是在补充协议下重新测得。独立 planner 只验证 QAD 来源链自洽，所选 PTQ 基座的对应关系还由此入口核验。

[流匹配重放审计](OPD_FLOW_REPLAY_AUDIT_20261007.md)未发现当前教师与学生辅助前向的 noise/time 错位。它验证的是相同缓存输入下的速度监督；不要求与主演示损失共享随机样本，也没有证明整个 CUDA 前向数值一致。统一端点生成策略去除了一项混杂因素，后续结论仍限定在匹配身份与预算下的状态来源效果。

此前 218 项 CPU 集成测试通过，覆盖完整候选采集、原始与派生来源审计、真实训练 Dataset/Collator 接线、PTQ 参考复用、端点计划与生成、教师缓存、训练准入、调度请求和既有收据兼容。端点串联测试仅用轻量模型代替完整 GR00T 加载，仍执行真实来源检查和完整动作采样入口；60 对观测均进入缓存与训练审计，30 个失败学生窗口保留。它不是机器人实验结果。测试还拒绝只重写收据协议 SHA 的跨协议复用，以及缓存检查后的文件替换。对现有真实教师数据的只读复查仍为十任务、148 个样本通过。另有十项旧 `exp.test_high_fp4_v3_cpu` 夹具错误在未修改的 HEAD driver 上同样复现：九项缺少未跟踪的历史 v8 协议文件，一项续训夹具缺少已有数值开关。它们未被计入本次通过项，也未通过放宽检查来修正。

三臂入口新增 12 项 CPU 测试，Driver 接线另有 12 项；共 24 项通过。测试替换实际训练、完整模型加载和机器人环境，在真实阶段收据及日志审计上验证执行顺序、共同输入、配对统计、续跑和篡改拒绝。CLI 可在不导入 Torch 的系统 Python 中显示帮助。它们验证调度行为，不代表真实 GPU 恢复已经完成。

完整上限是五次训练、共 10,000 次更新，加 80 个采集回合和 250 个开发评测回合。若两份 QAD 数据各为 148 窗，沿用当前加载与尾批规则，五臂合计 148,000 次演示窗口读取，两个 KD 臂合计 59,200 次额外探针前向/反传。实际预算必须由新样本数和训练收据计算；不能把匹配更新次数称为相同总 GPU 成本。

当前 v12 调度命令为 `--until all --allow-opd-nonimprovement`，因此先完成当前链，保留所有控制臂的真实结果。GPU 补充采集和训练串行排在其后；新协议与数据选择规则应在执行前冻结。上述开发补充不会自动获得新的独立测试集身份。若要把补充方案作为最终主结果，必须先明确确认集来源、所有候选的选择规则及多重比较口径；不能反复读取同一 held-out 并把最好一次称为独立验证。
