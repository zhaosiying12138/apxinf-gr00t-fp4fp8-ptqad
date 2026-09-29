#!/usr/bin/env python3
"""Snapshot and audit existing LIBERO logs. Never launches inference or a GPU job.

Example:
  python3 paper/collect_reevaluation.py audit20260929_bf16
  python3 paper/collect_reevaluation.py audit20260929_qad \
    --compare paper/evidence/reevaluation_audit20260929_bf16.json

Incomplete runs are written with explicit coverage and exit code 2. Pass
--allow-partial to accept a partial snapshot (the JSON remains incomplete).
Every invocation preserves a new timestamped log snapshot. The top-level
JSON is a latest pointer by content; old JSON and log bytes remain archived.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys


TASKS = [
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_task(data: bytes, task: str) -> dict:
    """Require exactly one valid terminal result; never infer failure from silence."""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    result_lines = [line.partition("results:")[2].strip()
                    for line in lines if line.startswith("results:")]
    row = {"task": task, "status": "missing_result", "successes": None,
           "episodes": None, "success_rate": None, "issues": []}
    headers = re.findall(r"Running collecting (\d+) episodes .* with (\d+) vec envs", text)
    row["collection_headers"] = [
        {"target_episodes_after_env_minimum": int(n), "parallel_envs": int(e)}
        for n, e in headers
    ]
    if not result_lines:
        row["last_nonempty_lines"] = [line for line in lines if line.strip()][-8:]
        return row
    if len(result_lines) != 1:
        row.update(status="ambiguous_results", issues=[f"{len(result_lines)} result lines; not merged"])
        return row
    try:
        value = ast.literal_eval(result_lines[0])
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("result must be a (task, bool-list, info-dict) tuple")
        name, outcomes, info = value
        if not isinstance(name, str) or name.rsplit("/", 1)[-1] != task:
            raise ValueError(f"task mismatch: {name!r}")
        if not isinstance(outcomes, list) or not outcomes or any(type(v) is not bool for v in outcomes):
            raise ValueError("outcomes must be a nonempty list of actual booleans")
        if not isinstance(info, dict):
            raise ValueError("episode info must be a dict")
        for field in ("episode_lengths", "episode_rewards"):
            if field in info and (not isinstance(info[field], list) or len(info[field]) != len(outcomes)):
                raise ValueError(f"{field} length differs from outcome count")
        if "episode_lengths" in info and any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in info["episode_lengths"]):
            raise ValueError("episode lengths must be positive finite numbers")
        if headers and len(outcomes) < int(headers[-1][0]):
            raise ValueError("terminal result has fewer episodes than declared collection target")
        rate = sum(outcomes) / len(outcomes)
        printed = re.findall(r"^success rate:\s*([0-9.eE+-]+)\s*$", text, re.M)
        if printed and not math.isclose(float(printed[-1]), rate, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("printed success rate disagrees with terminal boolean list")
        # Validate serializability here so a malformed info dict cannot masquerade
        # as a successful parse and then break the snapshot write later.
        json.dumps(info, allow_nan=False)
        row.update(status="complete", successes=sum(outcomes), episodes=len(outcomes),
                   success_rate=rate, outcomes=outcomes, episode_info=info)
    except (ValueError, TypeError, SyntaxError, OverflowError) as exc:
        row.update(status="invalid_result", issues=[str(exc)])
    return row


def aggregate(rows: list[dict], expected: int) -> dict:
    valid = [row for row in rows if row["status"] == "complete"]
    n = sum(row["episodes"] for row in valid)
    s = sum(row["successes"] for row in valid)
    observed_macro = sum(row["success_rate"] for row in valid) / len(valid) if valid else None
    return {
        "expected_tasks": expected,
        "completed_tasks": len(valid),
        "coverage_fraction": len(valid) / expected,
        "complete": len(valid) == expected,
        "successes_observed": s,
        "episodes_observed": n,
        "micro_success_rate_observed": s / n if n else None,
        "macro_success_rate_observed_tasks": observed_macro,
        "macro_success_rate_full_task_set": observed_macro if len(valid) == expected else None,
        "missing_or_invalid_tasks": [row["task"] for row in rows if row["status"] != "complete"],
        "aggregation_note": "Micro weights by actual completed episode count; macro weights tasks equally. Missing tasks are not scored as zero. Observed rates of an incomplete run are not full-suite rates.",
    }


def compare_shared(current: dict, previous: dict, source: Path, data: bytes) -> dict:
    a = {r["task"]: r for r in previous["tasks"] if r["status"] == "complete"}
    b = {r["task"]: r for r in current["tasks"] if r["status"] == "complete"}
    shared = sorted(a.keys() & b.keys())
    pairs = [{"task": t, "reference_successes": a[t]["successes"],
              "reference_episodes": a[t]["episodes"], "reference_rate": a[t]["success_rate"],
              "current_successes": b[t]["successes"], "current_episodes": b[t]["episodes"],
              "current_rate": b[t]["success_rate"],
              "difference_percentage_points": 100 * (b[t]["success_rate"] - a[t]["success_rate"])}
             for t in shared]
    return {
        "reference_json": str(source.resolve()), "reference_sha256": sha256(data),
        "reference_tag": previous.get("tag"), "shared_task_count": len(shared),
        "reference_only_tasks": sorted(a.keys() - b.keys()),
        "current_only_tasks": sorted(b.keys() - a.keys()),
        "shared_tasks": pairs,
        "macro_difference_percentage_points_shared_tasks": (
            sum(p["difference_percentage_points"] for p in pairs) / len(pairs) if pairs else None),
        "reference_shared_micro_rate": (
            sum(a[t]["successes"] for t in shared) / sum(a[t]["episodes"] for t in shared)
            if shared else None),
        "current_shared_micro_rate": (
            sum(b[t]["successes"] for t in shared) / sum(b[t]["episodes"] for t in shared)
            if shared else None),
        "episode_pairing_verified": False,
        "note": "Descriptive same-task comparison only. Episode IDs, initial states and seeds are not paired by this collector; no paired-episode significance or equivalence claim is made.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Tag used in runs/eval_<tag>; only letters, numbers, underscore or hyphen")
    parser.add_argument("--runs-root", type=Path,
                        default=Path("/home/zhaosiying/codebase/groot-fsdp2/runs"))
    parser.add_argument("--source-dir", type=Path, help="Override runs/eval_<tag>")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "evidence")
    parser.add_argument("--requested-episodes", type=int, default=10,
                        help="Caller-supplied request for provenance; actual denominator always comes from logs")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--compare", type=Path, help="Previous JSON emitted by this collector")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("unsafe tag; use only letters, numbers, underscore or hyphen")
    if args.requested_episodes < 1:
        parser.error("requested episode count must be positive")
    source = (args.source_dir or args.runs_root / ("eval_" + args.tag)).resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")
    output = args.output_root.resolve()
    if output == source or source in output.parents:
        parser.error("output must not be inside the source log directory")
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
    archive = output / "reevaluation" / args.tag / stamp
    archive.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in sorted(source.glob("*.log")):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        (archive / path.name).write_bytes(data)
        files[path.name] = {
            "data": data, "source": str(path), "bytes": len(data),
            "sha256": sha256(data), "stable_during_read": stable,
            "snapshot": str((archive / path.name).relative_to(output)),
        }
    rows = []
    for task in TASKS:
        name = task + ".log"
        if name not in files:
            rows.append({"task": task, "status": "missing_file", "successes": None,
                         "episodes": None, "success_rate": None, "issues": []})
            continue
        record = files[name]
        row = parse_task(record["data"], task)
        row["source"] = {k: v for k, v in record.items() if k != "data"}
        if not record["stable_during_read"]:
            row.update(status="file_changed_during_read", successes=None, episodes=None, success_rate=None)
            row["issues"].append("File changed during read; excluded from aggregate. Collect again after completion.")
        rows.append(row)
    summary = {
        "schema_version": 1, "tag": args.tag, "collected_at_utc": now.isoformat(),
        "source_directory": str(source), "snapshot_directory": str(archive.relative_to(output)),
        "collector_sha256": sha256(Path(__file__).read_bytes()),
        "requested_episodes_per_task_caller_supplied": args.requested_episodes,
        "tasks": rows, "summary": aggregate(rows, len(TASKS)),
        "files": [{k: v for k, v in record.items() if k != "data"} for record in files.values()],
        "unrecognized_task_logs": sorted(set(files) - {t + ".log" for t in TASKS}
                                          - {t + ".server.log" for t in TASKS} - {"server.log"}),
    }
    metadata = []
    for name in ("eval_manifest.json", "task_results.json", "summary.json"):
        path = source / name
        if not path.is_file():
            continue
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        destination = archive / "source_metadata" / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(data)
        metadata.append({"source": str(path), "snapshot": str(destination.relative_to(output)),
                         "sha256": sha256(data), "bytes": len(data),
                         "stable_during_read": (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)})
    summary["source_protocol_metadata"] = metadata
    if args.compare:
        data = args.compare.read_bytes()
        summary["same_task_comparison"] = compare_shared(summary, json.loads(data), args.compare, data)
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (archive / "summary.json").write_text(encoded, encoding="utf-8")
    target = output / f"reevaluation_{args.tag}.json"
    temporary = output / f".reevaluation_{args.tag}_{stamp}.tmp"
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"output": str(target), "snapshot": str(archive), **summary["summary"]},
                     ensure_ascii=False, indent=2))
    return 0 if summary["summary"]["complete"] or args.allow_partial else 2


if __name__ == "__main__":
    sys.exit(main())
