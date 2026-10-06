# 同覆盖 GPTQ 参考的复用审查

本次只读审查用于落实 QVLA 对照中发现的强 PTQ 基线缺口。没有读取候选的 held-out 成功率，没有运行量化、模型推理或闭环评测，也没有改变当前 v12 五臂协议。

**结论：现存 GPTQ 产物的底座和量化格式可与本轮对齐，但校准记录不足以核验初态分区独立性。不能直接将它认定为已通过本轮数据隔离检查的强基线。** 这表示缺少来源证据，并不表示已发现数据污染。

## 现存候选

```text
results/ptqad_20261003/w4a4_full_category
```

选择它的依据是 GPTQ 方法、完整覆盖和根 BF16 身份，不是闭环分数。两份权重分片现存；此次没有重算大权重文件哈希，因此尚未完成权重字节级复验。

| 核查项 | 查到的事实 | 结论边界 |
|---|---|---|
| 根模型 | `category_bake_manifest.json` 中的 `root_bf16` 与 v12 RTN 基座记录完全相同 | 来源声明一致，重新执行前仍须核对实际权重 |
| 普通部分 | 472 个 eligible 张量均为 NVFP4，实际方法为 468 GPTQ＋4 RTN | 不应称为所有张量均执行 GPTQ；别名和非 Linear 回退须保留说明 |
| category 部分 | 七个 category 张量的活动 LIBERO bank=2 使用 GPTQ，其余 217 个 bank 使用 RTN | 总计 479 个 eligible 权重张量均为 NVFP4；活动 bank 的误差补偿才参与当前机器人路径 |
| 数值表示 | E2M1 数据、每 16 元素一个 E4M3 块尺度、FP32 张量尺度；磁盘检查点保存反量化 BF16 值 | 这是数值参考产物，不是 packed NVFP4 部署文件 |
| 激活 | 正式服务可以安装与 v12 相同的 NVFP4 激活 QDQ，二级尺度固定为 1 | 预期 469 个普通算子＋7 个 category；本次未实际加载验证覆盖 |
| 普通校准 | 官方演示目录 `groot-fsdp2/Isaac-GR00T/demo_data/libero_demo`，128 窗口，seed=20260929 | 元数据明记 `windows_are_unique` 未断言，没有逐窗口/初态映射 |
| category 校准 | 同一演示目录，128 窗口，seed=20261003 | 没有逐回合及初态 bank 映射 |

因此，现有 metadata 无法证明校准避开本轮 held-out 的 9–19、24–28。不能因为路径名含 `demo` 就补写初态隔离结论，也不能反过来推断它使用了 held-out。

本次重算的小文件 SHA-256：

```text
ptq_recipe.json
64eed3cfc5ca735db7dd7e2550a33d66094589e673a0041a84a6da64a98a485a
category_ptq_recipe.json
9d47780ea0beccaacb50edde750fda87a304e8dbe5d6b7439f1afe880c5bb122
category_bake_manifest.json
97005d3685f4f34bb37b299038560ecdddce0d94c7225cc374beb3573a8d0912
```

## 建立可核验参考的最小路径

初次审查时，[`collector.py`](../quant/ptq/collector.py) 和 [`collector_category.py`](../quant/ptq/collector_category.py) 仅从 `--dataset` 构造迭代数据集，不能把 `capture_onpolicy.py` 的捕获文件直接当作 dataset 参数。随后已增加独立的 `--capture-manifest` 入口，使用方式和验收边界见[捕获校准说明](CAPTURED_PTQ_CALIBRATION.md)。

后续补充应另立协议，保持当前 v12 五臂不变，执行顺序如下：

1. 为两个既有 collector 增加冻结捕获清单适配，复用现有 Hessian 累积和 GPTQ 实现；不另写量化算法。使用本轮已核验的教师训练分区 20–23，记录源 capture manifest、每个输入文件 SHA、任务、回合、初态和样本选择规则。
2. 在原 BF16 上收集普通 Linear 的 H，生成新的全 NVFP4 GPTQ parent；保存各层实际 GPTQ/RTN 方法、回退原因和完整字节账本。
3. 在这个新 parent 上，对同一批冻结观测收集 category H，再量化七个 category 张量。不能把绑定旧 parent 的 category H 直接套到新 parent。
4. 先验证权重身份、实际激活覆盖与输入分区，再使用单独输出目录执行补充评测。模型、位宽及超参数应在读取补充 held-out 结果前固定，不利用主实验结果重新挑选参考。
5. 配对评测与 v12 采用相同任务、初态、seed 及执行约定。补充首先回答 GPTQ 相对 RTN 的贡献，以及选定恢复模型相对 GPTQ 的差异；独立报告校准数据量、计算成本和参数编码预算。统计对比与多重检验规则需在执行前写入补充协议。

本次已补齐输入适配与 CPU 校验；模型校准、重新量化和 GPTQ 补充闭环尚未执行。当前新增动作诊断仅测冻结观测上的完整动作差异，也不能替代 GPTQ 参考的闭环结果。即使补齐 GPTQ，对“超过 PTQ SOTA”的主张仍需更先进且同口径的可复现参照，不能把 GPTQ 自动等同于全部先进 PTQ 方法。
