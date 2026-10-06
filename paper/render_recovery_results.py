#!/usr/bin/env python3
"""Prepare Chinese result inserts from a completed, reverified v12 RTN run.

Writes new reviewable Markdown files only. It never selects an experimental
arm, fills absent measurements, or overwrites the manuscript or run evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER))
import collect_pairing_evidence
import extract_final_evidence
import v12_publication_contract

LABELS = {"bf16": "BF16", "ptq": "W4A4 PTQ", "qad": "PTQ + QAD",
          "continued_qad": "PTQ + continued-QAD", "qad_opd": "PTQ + QAD + OPD"}
CONTRASTS = {"ptq_vs_bf16": "PTQ − BF16", "qad_vs_ptq": "QAD − PTQ",
             "opd_vs_qad": "OPD − QAD", "opd_vs_continued_qad": "OPD − continued-QAD"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_identity(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def load_verified(path):
    path = Path(path).resolve(strict=True)
    first_identity = file_identity(path)
    final = json.loads(path.read_text(encoding="utf-8"))
    require(final.get("format") == "publication_final_results_v1" and final.get("status") == "complete",
            "final results are not complete publication evidence")
    v12_publication_contract.validate_v12_results(final, root=PAPER.parent)
    run = Path(final["source"]["run_dir"])
    pair_path = Path(final["source"]["heldout_comparison"]["path"])
    protocol = final["source"]["protocol"]["path"]
    files = [path, run / "final_manifest.json", pair_path, Path(protocol),
             PAPER / "analysis_plan_w4a4.json", PAPER / "paired_uncertainty.py", Path(__file__),
             Path(extract_final_evidence.__file__), Path(collect_pairing_evidence.__file__),
             Path(v12_publication_contract.__file__)]
    files += [pair_path.parent / name for name in collect_pairing_evidence.NAMES[1:]]
    # Snapshot before the expensive verification, not after it. The first
    # identity also binds the bytes from which dependency paths were parsed.
    identities = [file_identity(p) for p in files]
    require(identities[0] == first_identity, "final results changed while reading dependencies")
    require(final == extract_final_evidence.extract(run),
            "final results differ from freshly verified v12 evidence")
    paired = json.loads(pair_path.read_text(encoding="utf-8"))
    # Recompute pairing and accounting from all original five-arm JSON files.
    # This does not claim to replace the release audit of raw task logs.
    collect_pairing_evidence.audit(pair_path.parent, paired, protocol)
    require(all(file_identity(row["path"]) == row for row in identities),
            "result sources changed during verification")
    return final, identities


def render(final):
    arms = {**final["public_arms"], **final["control"]}
    require(set(arms) == set(LABELS), "all five arms, including continued-QAD, are required")
    require(all(row["episodes"] == 160 for row in arms.values()), "expected 160 episodes per arm")
    uncertainty = final["uncertainty"]
    require(set(uncertainty["contrasts"]) == set(CONTRASTS), "all four fixed contrasts are required")
    table = ["| 配置 | 成功回合 | 成功率 |", "|---|---:|---:|"]
    for name, label in LABELS.items():
        row = arms[name]
        table.append(f"| {label} | {row['successes']}/{row['episodes']} | {100*row['success_rate']:.2f}% |")
    effects = ["| 比较（前者减后者） | 差值/百分点 | 95% 配对区间 | 不一致回合数 | 条件双侧 p | Holm 校正 p |",
               "|---|---:|---:|---:|---:|---:|"]
    for name, label in CONTRASTS.items():
        row = uncertainty["contrasts"][name]
        lo, hi = row["pointwise_ci_pp"]
        effects.append(f"| {label} | {row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                       f"{row['discordant_episodes']} | {row['exact_mcnemar_two_sided_p']:.4g} | "
                       f"{row['holm_adjusted_p']:.4g} |")
    effects.append("\n区间为固定十任务内的配对 bootstrap 百分位区间，各比较分别为 95%，并非同时置信区间。"
                   "检验依赖任务内配对标签可交换及不一致方向独立的条件；Holm 校正保留这些前提。"
                   "这些结果不推断未见任务泛化，也不证明等效性。")
    if any(row["zero_discordance_warning"] for row in uncertainty["contrasts"].values()):
        effects.append("\n至少一项比较没有不一致回合：其零宽经验区间不代表总体差值已被精确确定。")
    tasks = list(arms["bf16"]["per_task"])
    require(len(tasks) == 10 and all(set(a["per_task"]) == set(tasks) for a in arms.values()),
            "all arms must contain the same ten tasks")
    per_task = ["| 任务 | " + " | ".join(LABELS.values()) + " |",
                "|---|" + "---:|" * len(LABELS)]
    for task in tasks:
        cells = [f"{arms[name]['per_task'][task]['successes']}/{arms[name]['per_task'][task]['episodes']}"
                 for name in LABELS]
        per_task.append("| `" + task + "` | " + " | ".join(cells) + " |")
    # All five scores are reported in a fixed order, regardless of outcome.
    summary = ("在 LIBERO-10 的同一组独立初态上，每臂完成 160 回合。" +
               "、".join(f"{LABELS[name]} 为 {arms[name]['successes']}/160（{100*arms[name]['success_rate']:.2f}%）"
                        for name in LABELS) + "。\n\n")
    summary += "；".join(f"{label} 的差值为 {uncertainty['contrasts'][name]['difference_pp']:+.2f} 个百分点"
                         for name, label in CONTRASTS.items()) + "。效果解释须结合配对区间、同预算对照和额外教师计算成本。\n"
    return {"main_table.md": "\n".join(table) + "\n",
            "paired_effects.md": "\n".join(effects) + "\n",
            "per_task.md": "\n".join(per_task) + "\n", "summary.md": summary}


def build(final_results, out):
    output = Path(out).absolute()
    require(not output.exists(), f"refusing existing result inserts: {output}")
    final, sources = load_verified(final_results)
    fragments = render(final)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="result-inserts-", dir=output.parent) as tmp:
        stage = Path(tmp) / "inserts"
        stage.mkdir()
        for name, text in fragments.items():
            (stage / name).write_text(text, encoding="utf-8")
        require(all(file_identity(row["path"]) == row for row in sources),
                "result sources changed during rendering")
        manifest = {"format": "v12_recovery_result_inserts_v1", "sources": sources,
                    "scope": "Verified numerical inserts for editorial review; raw-log, training, storage, screenshots and final publication validation remain separate requirements.",
                    "outputs": {name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
                                for name in fragments}}
        (stage / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        require(not output.exists(), f"output appeared during rendering: {output}")
        stage.rename(output)
    return {"output": str(output), "files": list(fragments)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.final_results, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
