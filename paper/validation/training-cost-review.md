# 训练成本收集与公开复核检查

本记录检查计量口径、证据关联及拒收行为，不生成正式训练成本结果，不运行 GPU，也不修改训练程序。正式数值须由完成实验后的 `collect_training_costs.py` 读取原始产物生成。

## 计量口径

- 初始 QAD 与两个继续训练分支分别保留生产者的 wall time。训练计时从脚本导入开始，包含模型准备、训练和 checkpoint 保存；不同阶段的保存次数可能不同，因此该值是阶段耗时，不是纯计算的每步成本。
- 教师阶段耗时包含输入与 checkpoint 哈希、配置和观测读取、模型加载与标注，结束于 cache 序列化之前。收集阶段包含逐任务服务器加载、rollout、写出与关闭。不能把不同生产者的计时段之和称为另一次直接测得的端到端 wall time。
- merge 的 `elapsed_seconds` 仅按 `export_reported_seconds` 保存；其采样点早于全部元数据复制、输出验证和最终目录操作，不解释为完整导出进程耗时。
- `scheduled_teacher_backward_passes` 从完整 optimizer step 数、cadence 和 accumulation 推导。100 步、每 4 步、每步 16 个累积微批对应计划 400 次。当前日志策略只在第 4、24、44、64、84 步打印，每步 16 条，对应 80 条可观察日志。两者分列；计划数不能标作逐次日志实测数。这里是学生对缓存教师速度场目标反向传播，教师始终冻结。
- demonstration draws 从完成步数与有效 batch 推导，不代表独立 microbatch 计数，也不代表不同样本数量。CUDA allocator peak 是各进程的 PyTorch 指标；未记录的 collection peak 保持 null。

## 身份与公开边界

收集阶段校验三个训练导出的基座权重一致，训练 checkpoint 的非空 shard inventory 与实际字节哈希吻合，初始 QAD 导出与 rollout 学生相连，BF16 教师与 PTQ 原始来源相连，OPD manifest 的 cache 哈希与实际 cache 相连。公开发布检查进一步把三份导出、PTQ 基座、收集学生与教师关联到正式闭环评测的 checkpoint 身份，防止把另一轮训练成本附在当前结果上。

两个继续训练 manifest 指向同一个初始 adapter，且收集时的 adapter 字节与初始 QAD 导出记录一致。训练启动时没有另存一份初始 adapter 权重摘要，因此不能声称独立证明了两个进程启动瞬间的全部 adapter 字节。

轻量公开包原样保存 metadata、日志、协议与源文件，并校验映射中的每个字节摘要。`--verify-published` 只靠该包重算训练步数、draws、阶段耗时、分支耗时比、日志数与调度数，不需要原私有权重、观测或 cache。大张量不随包发布；公开复核验证其记录身份及内部关联，不能重新读取不存在的张量来重复收集时的内容哈希检查。被归档的 evaluation parser 仅在所有映射哈希通过后加载，且不会在证据目录生成 `__pycache__`。

## CPU 检查

2026-09-29 独立执行以下检查，均通过：

```sh
CUDA_VISIBLE_DEVICES= /usr/bin/python3 -O tests/test_training_costs.py
# 13 tests: metadata/log arithmetic, strict rejection, standalone public verifier
CUDA_VISIBLE_DEVICES= /usr/bin/python3 -O paper/validation/test_training_publication.py
# 8 tests: binding published costs to the evaluated checkpoints and input identities
```

独立 cadence 小模型调用真实 `install_sequential_probe`，检查 100 个 optimizer steps × 16 个累积微步：实际触发 400 次学生 probe backward，捕获 80 条打印记录，且 CUDA 未初始化。该结果位于 `training-cost-cadence-check.json`，仅验证调度代码，不替代正式实验日志。

拒收测试覆盖空 shard inventory、非法存储计数、错误计时范围、改变的证据字节、错误教师或学生身份、错误基座或恢复导出、公开 verifier 失败，以及删除原私有目录后的独立验证。正式成本包仍必须在实验全部完成后生成并通过发布检查。
