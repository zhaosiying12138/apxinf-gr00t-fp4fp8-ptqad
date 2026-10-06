"""Tables from the independently verified captured-calibration GPTQ archive."""
from __future__ import annotations

from pathlib import Path

ARMS = ("bf16", "ptq", "gptq", "qad", "continued_qad", "qad_opd")
LABELS = {"bf16": "BF16", "ptq": "RTN W4A4 PTQ", "gptq": "校准 GPTQ W4A4 PTQ",
          "qad": "RTN + QAD", "continued_qad": "RTN + continued-QAD", "qad_opd": "RTN + QAD + OPD"}
CONTRASTS = {"gptq_vs_rtn": ("GPTQ − RTN", "ptq", "gptq"),
             "qad_vs_gptq": ("QAD − GPTQ", "gptq", "qad"),
             "opd_vs_gptq": ("OPD − GPTQ", "gptq", "qad_opd")}


def load_verified(paper: Path):
    from collect_gptq_reference import verify
    return verify(paper / "evidence/gptq_reference", main_evidence=paper / "evidence")


def render(report, *, detailed=False):
    data = report['comparison']
    if (set(data['arms']) != set(ARMS) or set(data['contrasts']) != set(CONTRASTS)
            or any(row['episodes'] != 160 or len(row['per_task']) != 10 for row in data['arms'].values())):
        raise ValueError('GPTQ publication requires the frozen six-arm comparison')
    rows = ["| 配置 | 成功回合 | 闭环成功率 |", "|---|---:|---:|"]
    for arm in ARMS:
        row = data['arms'][arm]
        rows.append(f"| {LABELS[arm]} | {row['successes']}/{row['episodes']} | {100 * row['macro_success_rate']:.2f}% |")
    pairs = ["| 比较 | 差值 / pp | 95% 配对区间 / pp | McNemar p | Holm p |",
             "|---|---:|---:|---:|---:|"]
    for key, (label, baseline, treatment) in CONTRASTS.items():
        row = data['contrasts'][key]
        if (row['baseline'], row['treatment']) != (baseline, treatment):
            raise ValueError('GPTQ contrast direction differs')
        lo, hi = row['pointwise_ci_pp']
        pairs.append(f"| {label} | {row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                     f"{row['exact_mcnemar_two_sided_p']:.6g} | {row['holm_adjusted_p']:.6g} |")
    methods = data['gptq_evidence']['actual_method_counts']
    memory = data['gptq_evidence']['memory']['known_tied_alias_deduplicated']
    gptq = data['arms']['gptq']
    results = ('\n'.join(rows) if detailed else
               f"GPTQ 参考成功 **{gptq['successes']}/{gptq['episodes']}（{100 * gptq['macro_success_rate']:.2f}%）**，"
               "与 RTN 及其恢复模型的差异如下：")
    text = ("为判断恢复是否优于同覆盖的校准 PTQ，另设一个 GPTQ 参考。它与主实验共享 BF16 源模型、"
            "479 个 NVFP4 权重张量、W4A4 激活约定及 160 个测试初态。校准使用训练分区的 148 个教师观测窗口；"
            "普通层先收集输入二阶统计并量化，category 层再在新 parent 上收集统计。"
            "该参考及三项比较在测试前固定，恢复模型沿用主实验开发集选出的检查点。\n\n"
            + results + '\n\n' + '\n'.join(pairs)
            + "\n\n区间为十个固定任务内的配对 bootstrap；Holm 仅校正这三项补充比较，"
            "不表示整篇论文的所有检验都得到统一校正。GPTQ 检验基于校准输入的层输出重构补偿的作用，"
            "不是 QVLA 动作敏感度方法的完整复现；这组结果的比较范围限于这里实现的参考。"
            + f"\n\n实际普通权重方法为 {methods['ordinary'].get('nvfp4_gptq', 0)} 项 GPTQ、"
            f"{methods['ordinary'].get('nvfp4_rtn', 0)} 项 RTN；活动 category bank 为 "
            f"{methods['active_category_banks'].get('nvfp4_gptq', 0)} 项 GPTQ、"
            f"{methods['active_category_banks'].get('nvfp4_rtn', 0)} 项 RTN。"
            "category 的选择依据同一校准统计上的误差，打平时保留 RTN；两种舍入方法的格式均为 NVFP4。"
            f"共享别名去重后的目标参数编码为 {memory['target_full_bytes']:,} B，"
            f"相对 BF16 源参数为 {memory['full_compression_x']:.3f}×；该纯 PTQ 参考没有 LoRA。")
    if not detailed:
        return text
    tasks = ["| 任务 | " + ' | '.join(LABELS[a] for a in ARMS) + ' |', '|---|' + '---:|' * len(ARMS)]
    for task in data['arms']['bf16']['per_task']:
        tasks.append('| `' + task + '` | ' + ' | '.join(
            f"{data['arms'][a]['per_task'][task]['successes']}/{data['arms'][a]['per_task'][task]['episodes']}"
            for a in ARMS) + ' |')
    descriptive = ["| 仅作描述的比较 | 差值 / pp | 仅左侧模型成功 | 仅右侧模型成功 |", "|---|---:|---:|---:|"]
    for row in data['descriptive_comparisons']:
        descriptive.append(f"| {LABELS[row['treatment']]} − {LABELS[row['baseline']]} | {row['difference_pp']:+.2f} | "
                           f"{row['only_treatment_success']} | {row['only_baseline_success']} |")
    costs = report['calibration_costs']
    timing = ["| 校准/量化阶段 | 已记录耗时 / s |", "|---|---:|"]
    for key, label in (("ordinary_collection_seconds", "普通层 H 收集"), ("ordinary_bake_seconds", "普通层量化"),
                       ("category_collection_seconds", "category H 收集"), ("category_bake_wall_seconds", "category 量化进程")):
        value = costs[key]
        timing.append(f"| {label} | " + ('未记录' if value is None else f'{value:.3f}') + ' |')
    return (text + '\n\n' + '\n'.join(tasks) + '\n\n' + '\n'.join(descriptive)
            + '\n\n左侧和右侧沿用减法表达式中的模型顺序；描述比较不报告显著性或等价性。'
            + '\n\n' + '\n'.join(timing)
            + '\n\n前三项沿用生产者的 elapsed_seconds，最后一项包含包装器记录的进程启动与日志写入。'
            '这些阶段计时不包含全部实验搜索和闭环评测成本，缺失项不按零处理。完整统计、方法分配、'
            '校准元数据与计时范围见 [`paper/evidence/gptq_reference/`](paper/evidence/gptq_reference/)。'
            '发布包可以从原始日志重算统计；未携带的权重和 Hessian 仅保留源端验收收据及字节哈希。')
