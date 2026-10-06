"""Render the verified action archive without executing a policy."""
from __future__ import annotations

from pathlib import Path

ARMS = ("ptq", "qad", "continued_qad", "qad_opd")
LABELS = {"ptq": "W4A4 PTQ", "qad": "QAD", "continued_qad": "continued-QAD", "qad_opd": "QAD + OPD"}
KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def load_verified(paper: Path):
    from collect_action_diagnostics import verify
    return verify(paper / "evidence/action_diagnostics",
                  final_manifest=paper / "evidence/final_manifest.json")


def render(report, *, detailed=False):
    spec = report['action_spec']
    if (report['integration_steps'] != 4 or spec['horizon'] != 16 or spec['dimensions'] != 7
            or tuple(spec['keys']) != KEYS or spec['key_dimensions'] != [1] * 7):
        raise ValueError('Action publication requires four integration steps and the LIBERO 16x7 contract')
    comparisons = report["comparisons"]
    if set(comparisons) != set(ARMS):
        raise ValueError("Action publication requires all four BF16 comparisons")
    for arm, row in comparisons.items():
        if (row["candidate"] != arm or row["reference"] != "bf16" or row["samples"] != 80
                or len(row["per_task"]) != 10 or not row["same_actual_noise_verified"]
                or row["interpretation"] != "training_distribution_fit_diagnostic"):
            raise ValueError("Action publication scope differs from the frozen v12 diagnostic")
    rows = ["| 配置（相对 BF16） | 归一化动作 MSE | 平移控制量 MSE | 旋转控制量 MSE | 夹爪指令不一致 |",
            "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        m = comparisons[arm]["task_macro"]
        position = sum(m[f"decoded_{k}_mse"] for k in ("x", "y", "z")) / 3
        rotation = sum(m[f"decoded_{k}_mse"] for k in ("roll", "pitch", "yaw")) / 3
        rows.append(f"| {LABELS[arm]} | {m['normalized_mse']:.6g} | {position:.6g} | {rotation:.6g} | "
                    f"{100 * m['libero_gripper_command_disagreement']:.2f}% |")
    text = ("固定十个任务各八个观测，共 80 个窗口；五臂使用相同观测和经数值核验的初始噪声，"
            "各自完成四步积分，再比较有效的 16×7 动作。观测来自已用于 QAD 的教师成功轨迹，"
            "因此这项测量解释训练分布上的动作拟合，不衡量未见状态泛化。\n\n"
            + "\n".join(rows)
            + "\n\n各项先在任务内平均，再对十个任务取宏平均。平移和旋转两列分别是反归一化后的 "
            "x/y/z 与 roll/pitch/yaw 三项 MSE 的均值，表示控制器输入差异，不是实测末端位姿误差。"
            "夹爪比较使用 LIBERO 的三值指令，保留中性点；指令不一致不等于抓取失败。"
            "这些误差也不替代闭环成功率。")
    if not detailed:
        return text
    metric_rows = ["| 指标（任务宏平均） | " + " | ".join(LABELS[a] for a in ARMS) + " |",
                   "|---|" + "---:|" * len(ARMS)]
    metrics = ["normalized_mse"] + [f"{space}_{key}_mse" for space in ("normalized", "decoded") for key in KEYS]
    metrics += ["libero_gripper_command_disagreement"]
    for metric in metrics:
        metric_rows.append("| `" + metric + "` | " + " | ".join(
            f"{comparisons[a]['task_macro'][metric]:.8g}" for a in ARMS) + " |")
    task_rows = ["| 任务（归一化动作 MSE） | " + " | ".join(LABELS[a] for a in ARMS) + " |",
                 "|---|" + "---:|" * len(ARMS)]
    for task in comparisons["ptq"]["per_task"]:
        task_rows.append("| `" + task + "` | " + " | ".join(
            f"{comparisons[a]['per_task'][task]['normalized_mse']:.8g}" for a in ARMS) + " |")
    return (text + "\n\n" + "\n".join(metric_rows) + "\n\n最后一行使用 0–1 比例；上方概览换算为百分比。"
            "\n\n" + "\n".join(task_rows)
            + "\n\n逐任务、逐观测的全部指标与原始动作见 "
            "[`paper/evidence/action_diagnostics/summary.json`](paper/evidence/action_diagnostics/summary.json) "
            "及该目录下的 `runs/`；`manifest.json` 记录来源、源码和逐文件哈希。")
