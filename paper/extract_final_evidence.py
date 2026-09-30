#!/usr/bin/env python3
"""Extract publication numbers from one completed recovery final manifest.

This is a read-only adapter at the experiment boundary.  It accepts only a
completed ``final_manifest.json`` and the heldout comparison named by that
manifest.  Development evaluations, candidate scores, and training logs are
never inspected or copied into the output.  The output is a small, stable
mapping for the paper builder; it contains the four public arms and keeps the
continued-QAD arm under ``control`` for the appendix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")
PUBLIC_ARMS = ("bf16", "ptq", "qad", "qad_opd")
FINAL_FORMAT_RE = re.compile(r"high_fp4_[a-z0-9_]+_final_manifest\Z")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{label} is not finite")
    return float(value)


def _validate_arm(name: str, row: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(row, dict), f"{name} arm is not an object")
    successes, count = row.get("successes"), row.get("count")
    _require(type(successes) is int and type(count) is int, f"{name} arm totals are not integers")
    _require(count == 100 and 0 <= successes <= count, f"{name} is not a complete 100-episode heldout arm")
    per_task = row.get("per_task")
    _require(isinstance(per_task, dict) and len(per_task) == 10, f"{name} lacks ten task totals")
    task_successes = 0
    task_episodes = 0
    for task, task_row in per_task.items():
        _require(isinstance(task_row, dict), f"{name}/{task} is not an object")
        k, n = task_row.get("successes"), task_row.get("episodes")
        _require(type(k) is int and type(n) is int and n == 10 and 0 <= k <= n,
                 f"{name}/{task} is not a complete ten-episode task")
        rate = _number(task_row.get("success_rate"), f"{name}/{task}.success_rate")
        _require(math.isclose(rate, k / n, rel_tol=0, abs_tol=1e-12),
                 f"{name}/{task} success rate disagrees with totals")
        task_successes += k
        task_episodes += n
    _require(task_successes == successes and task_episodes == count,
             f"{name} task totals disagree with arm totals")
    rate = _number(row.get("macro_success_rate"), f"{name}.macro_success_rate")
    expected_macro = sum(float(task_row["success_rate"]) for task_row in per_task.values()) / 10
    _require(math.isclose(rate, expected_macro, rel_tol=0, abs_tol=1e-12),
             f"{name} macro success rate disagrees with task totals")
    episodes = row.get("episodes")
    _require(isinstance(episodes, list) and len(episodes) == count,
             f"{name} lacks the complete heldout episode list")
    _require(all(isinstance(e, dict) and type(e.get("success")) is bool for e in episodes),
             f"{name} episode outcomes are incomplete")
    _require(sum(bool(e["success"]) for e in episodes) == successes,
             f"{name} episode outcomes disagree with totals")
    return {
        "successes": successes,
        "episodes": count,
        "success_rate": successes / count,
        "macro_success_rate": rate,
        "per_task": per_task,
    }


def _validate_source_identity(path: Path, identity: dict[str, Any], label: str) -> None:
    _require(isinstance(identity, dict), f"{label} identity is missing")
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    _require(type(identity.get("bytes")) is int and identity["bytes"] == path.stat().st_size,
             f"{label} byte count disagrees")
    _require(identity.get("sha256") == _sha256(path), f"{label} SHA-256 disagrees")


def extract(run_dir: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    """Validate a completed run and return the public heldout mapping.

    No development path is opened.  The only source files read are the final
    manifest, its declared protocol (for identity), and its heldout comparison.
    """
    root = Path(run_dir).resolve(strict=True)
    final_path = root / "final_manifest.json"
    _require(final_path.is_file() and not final_path.is_symlink(),
             f"completed final_manifest.json is missing: {final_path}")
    final = _read(final_path)
    _require(isinstance(final.get("format"), str) and FINAL_FORMAT_RE.fullmatch(final["format"]),
             f"unsupported or incomplete final manifest format: {final.get('format')!r}")
    _require(final.get("selection_uses_heldout") is False,
             "final selection must be independent of heldout results")
    _require(final.get("selected_pressure_recipe"), "final manifest has no selected pressure recipe")
    _require(set(final.get("required_arms", [])) == set(ARMS),
             "final manifest does not declare the complete five-arm comparison")

    protocol_file = Path(final.get("protocol_file", ""))
    _require(protocol_file.is_file(), f"final protocol is missing: {protocol_file}")
    protocol_sha = final.get("protocol_sha256")
    _require(isinstance(protocol_sha, str) and protocol_sha == _sha256(protocol_file),
             "final protocol SHA-256 disagrees with final manifest")

    round_dir = Path(final.get("heldout_round", ""))
    comparison_path = round_dir / "paired_comparison.json"
    declared_comparison = final.get("heldout_comparison")
    _validate_source_identity(comparison_path, declared_comparison, "heldout comparison")
    comparison = _read(comparison_path)
    _require(comparison.get("environment_pairing_verified") is True,
             "heldout environment pairing is not verified")
    _require(comparison.get("protocol_consistency_verified") is True,
             "heldout protocol consistency is not verified")
    _require(comparison.get("source_accounting_verified") is True,
             "heldout source accounting is not verified")
    arm_rows = comparison.get("arms")
    _require(isinstance(arm_rows, dict) and set(arm_rows) == set(ARMS),
             "heldout comparison does not contain exactly the five required arms")
    validated = {name: _validate_arm(name, arm_rows[name]) for name in ARMS}

    ptq_rate = validated["ptq"]["success_rate"]
    qad_rate = validated["qad"]["success_rate"]
    opd_rate = validated["qad_opd"]["success_rate"]
    control_rate = validated["continued_qad"]["success_rate"]
    result = {
        "format": "publication_final_results_v1",
        "status": "complete",
        "scope": "heldout-only; development scores and candidate logs excluded",
        "source": {
            "run_dir": str(root),
            "final_manifest": {"path": str(final_path), "sha256": _sha256(final_path),
                                "bytes": final_path.stat().st_size},
            "protocol": {"path": str(protocol_file), "sha256": protocol_sha},
            "heldout_comparison": {"path": str(comparison_path),
                                    "sha256": declared_comparison["sha256"],
                                    "bytes": declared_comparison["bytes"]},
        },
        "selected_recipe": final["selected_pressure_recipe"],
        "selected_ptq_checkpoint": final.get("selected_ptq_checkpoint"),
        "public_arms": {name: validated[name] for name in PUBLIC_ARMS},
        "control": {"continued_qad": validated["continued_qad"]},
        "deltas": {
            "ptq_minus_bf16_pp": (ptq_rate - validated["bf16"]["success_rate"]) * 100,
            "qad_minus_ptq_pp": (qad_rate - ptq_rate) * 100,
            "qad_opd_minus_qad_pp": (opd_rate - qad_rate) * 100,
            "qad_opd_minus_continued_qad_pp": (opd_rate - control_rate) * 100,
        },
        "selection": {
            "selection_uses_heldout": False,
            "selected_qad_learning_rate": final.get("selected_qad_learning_rate"),
            "selected_opd_weight": final.get("selected_opd_weight"),
        },
    }
    if out is not None:
        target = Path(out).absolute()
        _require(not target.exists(), f"refusing to overwrite existing output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="completed recovery run containing final_manifest.json")
    parser.add_argument("--out", help="new JSON output path; existing files are rejected")
    args = parser.parse_args()
    try:
        print(json.dumps(extract(args.run_dir, args.out), ensure_ascii=False, indent=2))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[extract-final-evidence] ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
