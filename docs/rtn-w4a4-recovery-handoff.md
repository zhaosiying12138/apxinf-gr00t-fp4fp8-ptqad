# RTN W4A4 压力链交接说明

本文档记录 2026-10-06 对全 NVFP4 W4A4 RTN 候选的代码与证据审计。它面向后续执行训练和正式评测的 agent；文档中的 development 分数只用于冻结候选，不能直接写入论文最终结果。

## 已确认的候选

候选目录为 `results/reruns/rtn_w4a4_pressure_20261006_01/rtn_category`。`category_ptq_recipe.json` 与 `category_bake_manifest.json` 均为 `complete`，479 个可量化张量的 NVFP4 占比为 1.0，FP8 占比为 0；服务日志显示 469 个普通 Linear 与 7 个 CategorySpecificLinear 均安装 W4A4 激活 QDQ。该目录不是 W4A16 配方。

开发评测目录为 `results/reruns/rtn_w4a4_pressure_20261006_01/dev_rtn`，协议分区为 seed `940000`、初态索引 `4,5,6,7,8`、每任务 5 回合。十个任务共 `41/50 = 82.00%`，逐任务分子为 `4,5,5,5,2,5,5,4,2,4`。所有任务都产生了完整的 `eval_manifest.json`、`task_results.json`、`summary.json`、rollout 日志、server 日志、视频和 reset 哈希。

可比的 BF16 development 收据 `results/ptqad_20261003/w4a4_dev_bf16_v11/summary.json` 为 `46/50 = 92.00%`，因此当前 RTN 候选在同一 development 分区下降 10 个百分点，满足“至少下降 5 个百分点且绝对成功率不低于 30%”的压力门槛。该 BF16 收据绑定旧 v11 协议哈希，不能直接冒充新 RTN 协议的证据；若建立新协议，应重新生成 BF16 收据，或者在审计器中明确记录并验证同分区的派生关系。

## 不能直接沿用的入口

`exp/recovery_protocol_v6_rtn_pressure.json` 把 RTN 写成 W4A16，使用 seed `440000/660000` 和 10 回合 held-out；它与当前 W4A4 候选以及 940000 development 运行不一致，不能用于这条实验链。

`exp/recovery_protocol_v11_w4a4_category.json` 是旧的 GPTQ W4A4 候选协议。它的候选名、PTQ checkpoint、协议 SHA 都绑定旧结果，也不能通过改名直接承载 RTN。正式链应冻结一个新的 v12 W4A4 RTN 协议。

## QAD/OPD 的实现边界

`rl/lora_qad.py` 只对 `nn.Linear` 安装 LoRA。`QAD_LORA_SCOPE=all_ordinary_linear` 覆盖普通 Linear（当前账本为 468 个参与动作前向的模块），但 7 个 `CategorySpecificLinear` 的 `[category,K,N]` 权重仍冻结。`rl/scoped_quant.py` 已覆盖这些层的 W4A4 激活路径，但不提供 category adapter。论文必须把“全量 W4A4”与“低秩恢复覆盖普通 Linear”分开表述；如果要恢复 category 层，需要先实现按 active bank 的 A/B 分支并增加单元测试，再重新训练。

`rl/probe_distill.py` 的 OPD 是带 action mask 的 velocity MSE，不是 KL。它每 `OPD_EVERY` 个 optimizer update 追加一次 probe backward，并按真实梯度累积步数归一化。若 `OPD_EVERY=4`，配置的 probe 权重长期平均只有四分之一；新协议必须预先冻结 `OPD_EVERY` 与候选权重，并让 continued-QAD 使用同样的演示更新数和优化器预算。

## 最短的可审计执行顺序

1. 新建 v12 协议，声明 `w4a4=true`、候选名 `rtn_w4a4_category`、普通算子 469、category 层 7、可量化张量 479；development 使用 940000 和 `[4,5,6,7,8]`，teacher/collection 使用 `[20,21,22,23]`，held-out 使用 16 个独立初态并冻结 seed。协议中固定 `all_ordinary_linear`、rank 32、alpha 64、QAD/continued 更新数、OPD 权重和 `OPD_EVERY`。
2. 在新协议下完成 BF16 development，并用 `exp/make_w4a4_selection.py` 将 BF16 与 RTN 原始目录绑定为新的 `selection.json`。选择脚本会核对协议 SHA、逐任务日志、reset pairing 和 checkpoint identity。
3. 重新运行 BF16 teacher supervision；`run_high_fp4_v3.py` 的 `validate_teacher_capture` 会调用 `exp/verify_teacher_replay.py` 检查协议 SHA、教师 checkpoint、十任务覆盖和成功样本。旧 v11 capture 不能仅通过复制目录复用。
4. 运行 `exp/run_w4a4_recovery.py --protocol-file <v12> --ptq-selection <selection.json>` 的 `qad_dev`、`qad_selection`、`recovery_dev` 和 `opd_selection` 阶段。训练入口会对每阶段写入 source、环境变量、checkpoint 和完成收据；任何已有但未完成的 stage 都必须换新目录，不得覆盖。
5. QAD 选定后，固定该 checkpoint 采集学生状态并生成 teacher probe cache；然后运行 continued-QAD 与 OPD。OPD 只有在 development 上同时超过 QAD 和同预算 continued-QAD 时才可进入正式主张；否则仍可完成 held-out，但论文必须如实报告无额外增益。
6. 选择冻结后再运行五臂 held-out。每臂必须完整覆盖十任务和协议规定的初态，最终用 `eval/compare_recovery.py` 生成配对比较，再用 `paper/extract_final_evidence.py` 和 `paper/materialize_final_evidence.py` 更新发布证据。旧 v11 的五臂结果、图表和截图在新证据完成前不得混入最终表格。

## 证据门禁

正式论文主张需要同一 held-out 分区同时满足：PTQ 相对 BF16 有预注册的退化窗口；QAD 高于 PTQ；OPD 高于 QAD，并且高于同预算 continued-QAD。每一项都要有完整的逐回合 Boolean 结果、reset 哈希、协议 SHA、checkpoint identity 和 server activation report。开发集只用于选择，不能替代这三项 held-out 对比。

