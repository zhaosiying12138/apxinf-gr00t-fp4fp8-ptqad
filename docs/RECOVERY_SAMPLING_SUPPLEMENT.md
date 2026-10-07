# QAD/OPD 时间覆盖与状态分布补充实验

日期：2026-10-07。状态：CPU 来源审计已完成；完整候选采集、成对数据视图及 QAD 训练来源检查已在独立分支 `experiment/full-trajectory-capture` 实现。下述 GPU 补充实验尚未启动，新协议尚未冻结，阶段 B 的端点生成与蒸馏缓存仍待接入。本方案不修改正在运行的 v12 协议、训练代码或数据。它用于定位恢复不足的原因，不预设任一实验臂胜出。

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

使用现有 BF16 教师、W4A4 RTN 基座及训练初态 20–23。重新采集十任务各四回合，仍每四次查询保存候选，但持续到回合结束。同一批完整候选派生两份数据，避免两次 rollout 的随机差异混入采样比较：

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
2. 已实现 `eval/run_recovery_eval.py` 的显式协议开关及完整候选最终化；未配置时维持 v12 前缀采样。`rl/capture_sampling.py` 校验逐回合计数、reset 身份、标签及源文件哈希，再确定性选择两份视图。`exp/derive_capture_views.py` 检查十任务完整日志、协议和原始结果，生成等预算视图及来源映射，不生成新的评测结果。
3. 已接入现有 `verify_teacher_replay.py` 入口，通过独立的派生视图规则检查来源。检查器重算两份选择计划，逐张量核对派生文件与源候选，核对十任务原始日志、BF16 教师身份及协议。`legacy_audit_compatible=false` 仍保留，表示派生视图不伪装成旧原始 rollout。训练器会重新审计，并将报告摘要与调度请求比较；每个样本读取后、反序列化前再核对字节 SHA，避免检查后换入别的输入。这项训练加载保护仅用于新派生视图。
4. 待扩展 `rl/opd_probe_cache.py`，明确支持捕获的演示状态类型，并验证端点来自指定 QAD；保留教师/学生状态来源，不能把教师样本改名伪装成学生 rollout。
5. 继续使用 `rl/lora_qad.py` 与 `ProbeAnchor`；新增臂均在同一补充协议内运行，不修改旧 adapter 元数据绕过身份检查。

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

`stratified` 使用同一入口；每任务至少需要两个实际贡献样本的成功教师回合。调度器已有 `--capture-dataset` 参数可传 `head` 或 `stratified`，并自动把指定教师和审计摘要交给训练器。阶段 A 必须限定在 `--until qad_dev`，两个数据臂分别使用独立输出目录和同一固定学习率。现有后续 collection/cache 调度尚不支持阶段 B 的成对端点包，不能把这一数据接入宣称为整套补充恢复链已跑通。直接启动训练器时，新视图必须提供与审计一致的 `QAD_CAPTURE_TEACHER` 和 `QAD_CAPTURE_AUDIT_SHA256`，缺少时会明确报错。

启动前还需解决一个独立的协议接入条件：现有调度器要求 PTQ selection、开发评测与当前协议 SHA 一致。加入新采样字段后，不能直接使用 v12 的 `selection_final` 启动新协议。可以复用其已核验 PTQ 权重，但选择证据必须通过新协议的开发评测取得，或通过另行实现、显式记录原协议与原日志身份的参考证据复用入口接入；不得改写旧收据的 SHA 来通过检查。该入口与阶段 B 尚未完成，当前也没有启动新 GPU 实验。

本次验证包括 110 项相关 CPU 测试：完整候选采集、原始与派生来源审计、真实训练 Dataset/Collator 接线、调度请求和既有收据兼容。测试验证 JSON 子进程传输前后的审计摘要一致，也拒绝只重写派生收据协议 SHA 的跨协议复用。对现有真实教师数据的只读复查仍为十任务、148 个样本通过；这不是新采样实验结果。另有十项旧 `exp.test_high_fp4_v3_cpu` 夹具错误在未修改的 HEAD driver 上同样复现：九项缺少未跟踪的历史 v8 协议文件，一项续训夹具缺少已有数值开关。它们未被计入通过的 110 项，也未通过放宽检查来修正。

完整上限是五次训练、共 10,000 次更新，加 80 个采集回合和 250 个开发评测回合。若两份 QAD 数据各为 148 窗，沿用当前加载与尾批规则，五臂合计 148,000 次演示窗口读取，两个 KD 臂合计 59,200 次额外探针前向/反传。实际预算必须由新样本数和训练收据计算；不能把匹配更新次数称为相同总 GPU 成本。

当前 v12 调度命令为 `--until all --allow-opd-nonimprovement`，因此先完成当前链，保留所有控制臂的真实结果。GPU 补充采集和训练串行排在其后；新协议与数据选择规则应在执行前冻结。上述开发补充不会自动获得新的独立测试集身份。若要把补充方案作为最终主结果，必须先明确确认集来源、所有候选的选择规则及多重比较口径；不能反复读取同一 held-out 并把最好一次称为独立验证。
