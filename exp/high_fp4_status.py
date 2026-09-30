#!/usr/bin/env python3
"""Read-only progress for the selected high-FP4 study; not publication evidence."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def evaluation(folder):
    if not folder.exists():
        return {"status": "not_started"}
    manifest = load(folder / "eval_manifest.json") or {}
    rows = load(folder / "task_results.json") or {}
    completed = [row for row in rows.values()
                 if row.get("returncode") == 0 and row.get("results")
                 and all(type(value) is bool for value in row["results"])]
    successes = sum(sum(row["results"]) for row in completed)
    episodes = sum(len(row["results"]) for row in completed)
    summary = load(folder / "summary.json") or {}
    complete = (len(completed) == manifest.get("task_count") == summary.get("tasks_complete")
                and episodes == summary.get("total_episodes")
                and successes == summary.get("total_successes"))
    result = {"status": "complete" if complete else "incomplete",
              "completed_tasks": len(completed),
              "completed_task_successes": successes,
              "completed_task_episodes": episodes}
    if complete:
        return result
    logs = [p for p in folder.glob("*.log") if not p.name.endswith(".server.log")]
    if logs:
        latest = max(logs, key=lambda p: p.stat().st_mtime_ns)
        with latest.open("rb") as stream:
            stream.seek(max(0, latest.stat().st_size - 20000))
            tail = stream.read().decode(errors="replace")
        result["latest_task"] = latest.stem
        progress = re.findall(r"Episodes:\s*[^\r\n]*?\|\s*(\d+)/(\d+)", tail)
        if progress:
            result["latest_task_episode_progress"] = list(map(int, progress[-1]))
        if "Traceback (most recent call last)" in tail:
            result["traceback_in_latest_log"] = True
    return result


def status(development, recovery):
    selection = load(development.parent / "exploratory_selection_v5.json") or load(development / "selection.json")
    candidates = (selection or {}).get("arms", {})
    if not candidates:
        candidates = ("bf16", "head_lang_vision_category", "calib_category")
    result = {"scope": "progress_only_not_final_results",
              "note": "Incomplete records do not prove that a process is currently running.",
              "paths": {"development": str(development), "recovery": str(recovery)},
              "development": {arm: evaluation(development / arm)
                              for arm in candidates}}
    if selection is not None:
        result["ptq_selection"] = {key: selection.get(key) for key in
                                   ("selected_recipe", "protocol_sha256", "selection_uses_heldout")}
    manifest = load(recovery / "run_manifest.json")
    if manifest is not None:
        result["recovery"] = {key: manifest.get(key) for key in
                              ("status", "selected_recipe", "last_completed_stage")}
        stages = {}
        for root in (recovery / "artifacts", recovery / "work"):
            for folder in sorted(root.glob("*")):
                if (folder / "eval_manifest.json").exists():
                    stages[folder.name] = evaluation(folder)
                elif (folder / "runtime_metrics.json").exists():
                    metrics = load(folder / "runtime_metrics.json") or {}
                    stages[folder.name] = {key: metrics.get(key) for key in
                                           ("status", "global_steps", "requested_optimizer_steps")}
        result["recovery"]["stages"] = stages
        result["heldout"] = {arm: evaluation(recovery / "artifacts" / "heldout_round" / f"heldout_{arm}")
                             for arm in ("bf16", "ptq", "qad", "continued_qad", "qad_opd")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "results/ptqad_20260929"
    parser.add_argument("--development-root", type=Path,
                        default=base / "high_fp4_v4_retry5/development")
    parser.add_argument("--recovery-root", type=Path, default=base / "exploratory_recovery_v5_retry")
    args = parser.parse_args()
    print(json.dumps(status(args.development_root, args.recovery_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
