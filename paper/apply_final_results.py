#!/usr/bin/env python3
"""Safely backfill verified v11 numbers into the public manuscript sources.

The script is intentionally fail-closed.  It first revalidates the complete
``publication_final_results_v1`` evidence through :mod:`render_recovery_results`,
then checks that the target files still contain the review placeholders this
script was written for.  Without ``--apply`` it prints a unified diff and does
not write anything.  With ``--apply`` it stages the replacements, rechecks the
source hashes, and rolls back already replaced files if a later replacement
fails.

This script does not select checkpoints, infer missing arms, or alter figures,
capture manifests, training-cost evidence, or the generated HTML/知乎 output.
Run the builders and publication validator after applying the inserts.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path
import re
import tempfile
import sys

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(PAPER))
from render_recovery_results import LABELS, CONTRASTS, load_verified  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def result_values(final: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    arms = {**final["public_arms"], **final["control"]}
    require(set(arms) == set(LABELS), "verified evidence lacks one of the five arms")
    require(all(row["episodes"] == 160 for row in arms.values()),
            "backfill requires 160 episodes for every arm")
    contrasts = final["uncertainty"]["contrasts"]
    require(set(contrasts) == set(CONTRASTS),
            "verified evidence lacks one of the four fixed contrasts")
    return arms, contrasts


def replace_once(source: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    require(count == 1, f"expected exactly one {label} placeholder row")
    return updated


def render_readme(source: str, arms: dict, contrasts: dict) -> str:
    # The four cells after the fixed columns are replaced as a unit.  Anchoring
    # the row labels prevents unrelated red placeholders elsewhere in README
    # from being consumed.
    row_labels = {
        "bf16": "BF16 基线",
        "ptq": "全 NVFP4 PTQ + W4A4",
        "qad": "PTQ + QAD（W4A4 基座）",
        "continued_qad": "continued-QAD 对照",
        "qad_opd": "PTQ + QAD + OPD（W4A4 基座）",
    }
    for name in LABELS:
        row = arms[name]
        value = f"{row['successes']}/160（{pct(row['success_rate'])}）"
        row_label = row_labels[name]
        pattern = rf"(?m)^\| {re.escape(row_label)} \|([^\n]*?)\| <span style=\"color:#c00\">(?:待回填|xxx/160（xxx%）)</span> \|"
        replacement = f"| {row_label} |\\1| {value} |"
        source = replace_once(source, pattern, replacement, f"README {name} success")

    metric_patterns = {
        "micro_count": r"(?m)^\| micro success（/160） \|.*$",
        "micro_rate": r"(?m)^\| micro success（%） \|.*$",
        "macro_rate": r"(?m)^\| 十任务 macro success（%） \|.*$",
    }
    require(all(re.search(p, source) for p in metric_patterns.values()),
            "README metric table is missing")
    order = list(LABELS)
    cells = lambda key: " | ".join(
        (f"{arms[name]['successes']}/160" if key == "micro_count" else pct(arms[name]['success_rate']))
        for name in order)
    source = re.sub(metric_patterns["micro_count"], f"| micro success（/160） | {cells('micro_count')} |", source, count=1)
    source = re.sub(metric_patterns["micro_rate"], f"| micro success（%） | {cells('micro_rate')} |", source, count=1)
    source = re.sub(metric_patterns["macro_rate"], f"| 十任务 macro success（%） | {' | '.join(pct(arms[n]['macro_success_rate']) for n in order)} |", source, count=1)

    contrast_rows = {
        "ptq_vs_bf16": "PTQ − BF16", "qad_vs_ptq": "QAD − PTQ",
        "opd_vs_qad": "OPD − QAD", "opd_vs_continued_qad": "OPD − continued-QAD",
    }
    for key, label in contrast_rows.items():
        row = contrasts[key]
        lo, hi = row["pointwise_ci_pp"]
        cells = f"{row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {row['exact_mcnemar_two_sided_p']:.4g} | {row['holm_adjusted_p']:.4g}"
        pattern = rf"(?m)^\| {re.escape(label)} \|.*$"
        source = replace_once(source, pattern, f"| {label} | {cells} |", f"README {label} comparison")

    source = source.replace(
        "中文论文 HTML](paper/paper.html)：离线单文件，含公式、图表和 Ubuntu 执行图集；当前为结果待回填的审阅稿。",
        "中文论文 HTML](paper/paper.html)：离线单文件，含公式、图表、Ubuntu 执行图集和已核验的 v11 结果。",
    )
    source = source.replace(
        "编码预算已由量化配方与完整 adapter 清单核验。闭环成功率仍待冻结协议对应的 `final_manifest.json` 和 `paired_comparison.json` 完成后回填。",
        "编码预算、闭环成功率和配对统计均由冻结 v11 协议对应的 `final_manifest.json` 与 `paired_comparison.json` 回填，并在发布校验中复核。",
    )
    return source


def render_experiment(source: str, arms: dict, contrasts: dict) -> str:
    labels = {
        "bf16": "BF16", "ptq": "W4A4 PTQ", "qad": "PTQ + QAD",
        "continued_qad": "PTQ + continued-QAD", "qad_opd": "PTQ + QAD + OPD",
    }
    for name, label in labels.items():
        row = arms[name]
        value = f"{row['successes']}/160（{pct(row['success_rate'])}）"
        pattern = rf"(?m)^\| {re.escape(label)} \|([^\n]*?)\| <span style=\"color:#b42318\">xxx/160（xxx%）</span> \|"
        source = replace_once(source, pattern, f"| {label} |\\1| {value} |", f"实验 {name} success")
    rows = {
        "ptq_vs_bf16": "PTQ − BF16", "qad_vs_ptq": "QAD − PTQ",
        "opd_vs_qad": "OPD − QAD", "opd_vs_continued_qad": "OPD − continued-QAD",
    }
    for key, label in rows.items():
        row = contrasts[key]
        lo, hi = row["pointwise_ci_pp"]
        pattern = rf"(?m)^\| {re.escape(label)} \| <span style=\"color:#b42318\">xxx</span> \| <span style=\"color:#b42318\">\[xxx, xxx\]</span> \|"
        source = replace_once(source, pattern, f"| {label} | {row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |", f"实验 {label} comparison")
    return source


def build_updates(final_results: Path, readme: Path, experiment: Path) -> dict[Path, str]:
    final, _ = load_verified(final_results)
    arms, contrasts = result_values(final)
    return {
        readme: render_readme(readme.read_text(encoding="utf-8"), arms, contrasts),
        experiment: render_experiment(experiment.read_text(encoding="utf-8"), arms, contrasts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-results", required=True, type=Path,
                        help="verified final_evidence.json from extract_final_evidence.py")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--experiment", type=Path, default=PAPER / "sections" / "04-实验.md")
    parser.add_argument("--apply", action="store_true",
                        help="write the verified replacements; default is diff-only")
    args = parser.parse_args()
    targets = [args.readme.resolve(strict=True), args.experiment.resolve(strict=True)]
    final_results = args.final_results.resolve(strict=True)
    before = {path: (sha(path), path.read_text(encoding="utf-8")) for path in targets}
    updates = build_updates(final_results, *targets)
    changed = False
    for path in targets:
        diff = difflib.unified_diff(before[path][1].splitlines(True), updates[path].splitlines(True),
                                    fromfile=str(path), tofile=str(path) + " (verified backfill)")
        text = "".join(diff)
        if text:
            changed = True
            print(text, end="")
    if not args.apply:
        print("[apply-final-results] dry run: no files written")
        return 0
    require(changed, "verified results produce no source changes; refusing a no-op apply")
    # Re-read and re-hash before committing so a concurrent editor cannot be
    # silently overwritten.  Writes are staged in the target directories and
    # then replaced one by one only after every target passes this check.
    for path, (digest, _) in before.items():
        require(sha(path) == digest, f"target changed during rendering: {path}")
    staged = []
    backups = []
    try:
        for path in targets:
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            with open(fd, "w", encoding="utf-8", newline="") as stream:
                stream.write(updates[path])
            staged.append((path, Path(name)))
            backup_fd, backup_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".bak", dir=path.parent)
            with open(backup_fd, "wb") as stream:
                stream.write(path.read_bytes())
            backups.append((path, Path(backup_name)))
        for path, temp in staged:
            temp.replace(path)
    except Exception:
        for path, backup in backups:
            if backup.exists():
                backup.replace(path)
        raise
    finally:
        for _, temp in staged:
            temp.unlink(missing_ok=True)
        for _, backup in backups:
            backup.unlink(missing_ok=True)
    print("[apply-final-results] applied verified numbers to:", ", ".join(str(p) for p in targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
