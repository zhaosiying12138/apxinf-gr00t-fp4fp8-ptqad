#!/usr/bin/env python3
"""Read-only progress for the current frozen W4A4 recovery study.

Log counters describe recorded work, not process liveness or final evidence.
Explicit root arguments can still inspect another run without modifying it.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/reruns/rtn_w4a4_release_20261006_01"


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
    # Current selections carry each audited development arm beside selection.json.
    # Retain explicit inspection of the older layout without choosing it by default.
    selection_path = development / "selection.json"
    if not selection_path.is_file() and (development / "v11_selection" / "selection.json").is_file():
        selection_path = development / "v11_selection" / "selection.json"
    selection = load(selection_path)
    candidates = (selection or {}).get("arms", {})
    if not candidates:
        candidates = ("bf16", "rtn_w4a4_category")
    legacy_parent = development.parent if development.name == "v11_selection" else development
    legacy_names = {"bf16": "w4a4_dev_bf16_v11",
                    "all_nvfp4_gptq_category": "w4a4_dev_full_category"}
    candidate_dirs = {}
    for arm in candidates:
        direct = selection_path.parent / arm
        legacy = legacy_parent / legacy_names.get(arm, arm)
        candidate_dirs[arm] = direct if direct.is_dir() or not legacy.is_dir() else legacy
    result = {"scope": "progress_only_not_final_results",
              "note": "Incomplete records do not prove that a process is currently running.",
              "paths": {"development": str(development), "recovery": str(recovery)},
              "development": {arm: evaluation(candidate_dirs[arm])
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
                elif folder.name.startswith("train_"):
                    log = recovery / "logs" / (folder.name + ".log")
                    if log.is_file():
                        with log.open("rb") as stream:
                            offset = max(0, log.stat().st_size - 65536)
                            stream.seek(offset)
                            tail = stream.read().decode(errors="replace")
                        if offset:
                            # A truncated loader prefix must not look like an
                            # unlabelled training bar at the start of the tail.
                            tail = re.sub(r"^[^\r\n]*", "", tail, count=1)
                        request = load(folder / "orchestrator_training_request.json") or {}
                        environment = request.get("environment")
                        value = environment.get("QAD_STEPS") if isinstance(environment, dict) else None
                        expected_steps = int(str(value)) if re.fullmatch(r"[0-9]+", str(value)) else 0
                        # Only Trainer's unlabelled bar records optimizer steps.
                        # Model loading and dataset initialization have labels.
                        progress = []
                        for line in tail.splitlines():
                            match = re.match(
                                r"[ \t]*\d{1,3}%\|[^\r\n]*?\|[ \t]*(\d+)/(\d+)[ \t]*\[",
                                line,
                            )
                            if match:
                                done, total = map(int, match.groups())
                                if (0 <= done <= total and total > 0
                                        and (expected_steps <= 0 or total == expected_steps)):
                                    progress.append((done, total))
                        stages[folder.name] = {
                            "status": "incomplete",
                            "log": str(log),
                            "last_log_update_unix": log.stat().st_mtime,
                        }
                        if expected_steps > 0:
                            stages[folder.name]["requested_optimizer_steps"] = expected_steps
                        if progress:
                            done, total = progress[-1]
                            stages[folder.name].update(
                                logged_optimizer_steps=done,
                                requested_optimizer_steps=total,
                            )
        result["recovery"]["stages"] = stages
        result["heldout"] = {arm: evaluation(recovery / "artifacts" / "heldout_round" / f"heldout_{arm}")
                             for arm in ("bf16", "ptq", "qad", "continued_qad", "qad_opd")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = DEFAULT_RUN / "recovery_v12"
    default_development = DEFAULT_RUN / "selection_final"
    parser.add_argument("--development-root", type=Path,
                        default=default_development)
    parser.add_argument("--recovery-root", type=Path, default=base)
    args = parser.parse_args()
    print(json.dumps(status(args.development_root, args.recovery_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
