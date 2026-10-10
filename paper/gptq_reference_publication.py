"""Tables from the independently verified captured-calibration GPTQ archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v12_publication_contract import PROTOCOL_SHA256, validate_v12_results

ARMS = ("bf16", "ptq", "gptq", "qad", "continued_qad", "qad_opd")
LABELS = {"bf16": "BF16", "ptq": "RTN W4A4 PTQ", "gptq": "校准 GPTQ W4A4 PTQ",
          "qad": "RTN + QAD", "continued_qad": "RTN + continued-QAD", "qad_opd": "RTN + QAD + OPD"}
CONTRASTS = {"gptq_vs_rtn": ("GPTQ − RTN", "ptq", "gptq"),
             "qad_vs_gptq": ("QAD − GPTQ", "gptq", "qad"),
             "opd_vs_gptq": ("OPD − GPTQ", "gptq", "qad_opd")}


SCOPE_FILE = "publication_scope_v12.json"
RELEASE_RUN = "results/reruns/rtn_w4a4_release_20261006_01/recovery_v12"
OMISSION_FILE = "not_performed.json"
NOT_PERFORMED_TEXT = (
    "主实验采用同一 RTN W4A4 基座，以比较 QAD 与 OPD 的恢复作用。"
    "本文未执行同覆盖校准 GPTQ 对照，因此不能据此判断恢复方法是否优于校准 PTQ。"
    "未执行范围及最终清单绑定见 "
    "[paper/publication_scope_v12.json](paper/publication_scope_v12.json) 和 "
    "[paper/evidence/gptq_reference/not_performed.json](paper/evidence/gptq_reference/not_performed.json)。"
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(path):
    _need(path.is_file() and not path.is_symlink(), "Missing regular GPTQ scope input: " + str(path))
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _read(path):
    _identity(path)
    return json.loads(path.read_text(encoding="utf-8"))


def omission_receipt(paper: Path):
    """Bind the explicit user decision to complete, current v12 evidence."""
    scope_path = paper / SCOPE_FILE
    scope = _read(scope_path)
    _need(set(scope) == {"format", "protocol_sha256", "run_dir", "decision_date",
                         "decision_source", "gptq_reference"},
          "Unexpected publication scope fields")
    _need(scope["format"] == "v12_publication_scope_v1" and
          scope["protocol_sha256"] == PROTOCOL_SHA256 and scope["run_dir"] == RELEASE_RUN,
          "GPTQ scope is not the frozen v12 release")
    _need(scope["decision_source"] == "user_instruction" and
          scope["decision_date"] == "2026-10-10",
          "GPTQ scope requires the explicit user freeze decision")
    decision = scope["gptq_reference"]
    _need(isinstance(decision, dict) and
          set(decision) == {"status", "reason_code", "reason"} and
          decision["status"] == "not_performed" and
          decision["reason_code"] == "training_quantization_and_search_frozen" and
          isinstance(decision["reason"], str) and bool(decision["reason"].strip()),
          "Unrecognized GPTQ non-performance decision")
    final_path = paper / "evidence/final_manifest.json"
    results_path = paper / "evidence/final_results.json"
    results = _read(results_path)
    validate_v12_results(results, root=paper.parent)
    original = results["source"]
    _need(Path(original.get("run_dir", "")).as_posix().endswith("/" + RELEASE_RUN),
          "GPTQ decision belongs to another release run")
    identity = _identity(final_path)
    _need(all(original.get("final_manifest", {}).get(k) == v for k, v in identity.items()),
          "GPTQ receipt final manifest differs from extracted results")
    final = _read(final_path)
    _need(final.get("format") == "w4a4_recovery_v12_final_manifest" and
          final.get("protocol_sha256") == PROTOCOL_SHA256 and
          final.get("selection_uses_heldout") is False and
          set(final.get("required_arms", [])) == {"bf16", "ptq", "qad", "continued_qad", "qad_opd"},
          "GPTQ non-performance receipt requires complete v12 final evidence")
    return {
        "format": "gptq_reference_not_performed_v1",
        "status": "not_performed",
        "protocol_sha256": PROTOCOL_SHA256,
        "run_dir": RELEASE_RUN,
        "decision": {"path": SCOPE_FILE, **_identity(scope_path)},
        "reason_code": decision["reason_code"],
        "final_manifest": identity,
        "final_results": _identity(results_path),
    }


def record_not_performed(paper: Path):
    """Create an omission receipt only; never manufacture a measured archive."""
    receipt = omission_receipt(paper)
    folder = paper / "evidence/gptq_reference"
    _need(not folder.exists() and not folder.is_symlink(),
          "Refusing an existing GPTQ supplement directory")
    folder.mkdir()
    (folder / OMISSION_FILE).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    return receipt


def load_verified(paper: Path):
    scope_path = paper / SCOPE_FILE
    if scope_path.exists() or scope_path.is_symlink():
        expected = omission_receipt(paper)
        folder = paper / "evidence/gptq_reference"
        _need(folder.is_dir() and not folder.is_symlink(),
              "Explicit GPTQ decision requires its registered non-performance receipt")
        _need({path.name for path in folder.iterdir()} == {OMISSION_FILE},
              "GPTQ non-performance receipt cannot coexist with measured or unknown artifacts")
        receipt_path = folder / OMISSION_FILE
        _need(_read(receipt_path) == expected, "GPTQ non-performance receipt is stale or changed")
        return {"status": "not_performed", "decision": expected,
                "raw_logs_replayed": False, "paired_statistics_recomputed": False,
                "tensor_contents_reverified_offline": False,
                "files": [scope_path, receipt_path, paper / "evidence/final_manifest.json",
                          paper / "evidence/final_results.json"]}
    # There is no missing-evidence fallback. Actual measured references retain
    # their original full raw-log, provenance and statistical replay checks.
    from collect_gptq_reference import verify
    return verify(paper / "evidence/gptq_reference", main_evidence=paper / "evidence")


def render(report, *, detailed=False):
    if report.get("status") == "not_performed":
        _need(report.get("decision", {}).get("status") == "not_performed",
              "Missing explicit GPTQ non-performance decision")
        return NOT_PERFORMED_TEXT
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-not-performed", action="store_true",
                        help="Bind the explicit frozen scope to installed complete final evidence")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--paper", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.record_not_performed == args.verify:
        parser.error("choose exactly one of --record-not-performed or --verify")
    report = record_not_performed(args.paper) if args.record_not_performed else load_verified(args.paper)
    print(json.dumps({key: value for key, value in report.items() if key != "files"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
