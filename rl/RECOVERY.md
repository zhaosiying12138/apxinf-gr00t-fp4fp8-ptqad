# v12 W4A4 QAD/OPD 恢复协议

本文档只描述当前冻结的 v12 RTN W4A4 运行。量化基座使用全覆盖 NVFP4 权重和 NVFP4 激活 QDQ；QAD、continued-QAD 和 OPD 使用独立 BF16 LoRA 旁路。正式入口是 `exp/run_w4a4_recovery.py` 与 `exp/recovery_protocol_v12_rtn_w4a4.json`，旧协议和旧结果不属于当前复现流程。

## 训练合同

单 GPU 训练使用 `QAD_MICRO_BATCH=1`、`QAD_GLOBAL_BATCH=16`，累积步数为 16。QAD、continued-QAD 和每个 OPD 候选均为 2,000 个优化器更新，rank=32、alpha=64；QAD 学习率候选为 `5e-5` 与 `1e-4`，OPD 教师权重候选为 `0.25` 与 `1.0`。continued-QAD 从入选 QAD 的精确 A/B 参数开始，匹配演示读取数和更新数；OPD 只额外加入学生访问状态上的 BF16 教师速度 MSE。

QAD 主损失与 OPD 探针损失分开反向，避免把多个计算图叠在一起。教师和学生在探针前向中保持 eval 模式，随机噪声、时间采样、有效 16×7 动作掩码和归一化统计都写入缓存收据。训练输出必须包含 `runtime_metrics.json`、`recovery_manifest.json`、checkpoint 身份和实际 CUDA 峰值；缺少任一项都不能进入正式证据。

## 数据分区与评测

| 分区 | 每任务回合 | 官方初态索引 | 起始 seed | 用途 |
|---|---:|---|---:|---|
| development | 5 | 4–8 | 940000 | 选择压力和恢复超参数 |
| teacher supervision | 4 | 20–23 | 950000 | BF16 成功演示 |
| collection | 4 | 20–23 | 960000 | QAD 学生访问状态 |
| heldout | 16 | 9–19、24–28 | 970000 | 五臂正式闭环 |
| smoke | 1 | 0 | 980000 | 服务和数值合同检查 |

每个 episode 先执行 10 个稳定步，再由策略生成 16 步动作并执行前 8 步；单回合最多 720 步。held-out 五臂共享任务顺序、官方初态、seed stride 和服务器合同；配对只表示环境初态一致，不表示策略内部的每一步随机数完全相同。正式结果必须保留逐任务布尔结局、reset hash、真实分母和原始 rollout/server 日志。

## 运行命令

```bash
export PROJECT=$(pwd)
source "$PROJECT/setup/recovery-env.sh"
export PROTOCOL="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"
export RUN="$PROJECT/results/reruns/rtn_w4a4_release_20261006_01/recovery_v12"
python3 "$PROJECT/exp/run_w4a4_recovery.py" \
  --run-dir "$RUN" \
  --protocol-file "$PROTOCOL" \
  --ptq-selection "$PROJECT/results/reruns/rtn_w4a4_release_20261006_01/selection_final/selection.json" \
  --base "$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10" \
  --gr00t-repo "$GR00T_REPO" \
  --python "$PTQAD_PYTHON" \
  --rollout-python "$LIBERO_PYTHON" \
  --dataset "$GR00T_REPO/demo_data/libero_demo" \
  --capture-dataset "$PROJECT/results/reruns/rtn_w4a4_release_20261006_01/teacher_supervision_v12_clean" \
  --port-base 6920 --until all
```

`--until all` 必须由唯一 driver 顺序执行；不要并行启动第二个恢复进程。driver 完成后，先检查 `final_manifest.json`、五臂各 160 回合、配对比较和完整训练收据，再运行 `paper/extract_final_evidence.py`、`paper/materialize_final_evidence.py` 与发布校验器。若这些文件尚未生成，当前运行仍处于实验阶段，不能把本目录描述成已完成发布证据。
