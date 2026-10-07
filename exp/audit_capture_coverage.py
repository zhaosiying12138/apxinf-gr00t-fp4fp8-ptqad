#!/usr/bin/env python3
"""Audit finalized recovery captures on CPU without changing source evidence.

Both teacher_supervision and collection are supported. Accepted means a
sample_*.pt file retained for training: collection deliberately retains failed
episodes, whereas teacher_supervision renames their files to rejected_*.pt.
The audit reports recorded query positions and never infers manipulation stages.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def identity(path: Path, root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def bounds(values) -> list[int] | None:
    values = list(values)
    return [min(values), max(values)] if values else None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def audit(source: Path) -> dict:
    """Read source/eval_manifest.json, finalized captures, and original logs."""
    import torch

    require(not torch.cuda.is_initialized(), "Run this CPU audit before initializing CUDA")
    source = Path(source).resolve()
    evaluation = read_json(source / "eval_manifest.json")
    purpose = evaluation.get("purpose")
    require(purpose in {"teacher_supervision", "collection"}, "Unsupported capture purpose")
    source_kind = "teacher_rollout" if purpose == "teacher_supervision" else "student_rollout"
    results = read_json(source / "task_results.json")
    folders = sorted(p for p in (source / "observations").iterdir() if p.is_dir())
    require(set(results) == {p.name for p in folders} == set(evaluation["tasks"]),
            "Capture tasks differ from evaluation/tasks results")
    rows, all_lengths, successful_lengths, accepted_lengths, calls = [], [], [], [], []
    metadata_fields, accepted_episode_counts = set(), Counter()
    for folder in folders:
        task = folder.name
        manifest_path = folder / "capture_manifest.json"
        counts_path = folder / "capture_counts.json"
        event_path = folder / "reset_events.jsonl"
        manifest, counts = read_json(manifest_path), read_json(counts_path)
        require(manifest.get("finalized") is True and manifest.get("finalized_purpose") == purpose
                and manifest.get("purpose") == purpose and manifest.get("task_name") == task
                and manifest.get("source_kind") == source_kind,
                f"Capture purpose/finalization identity differs for {task}")
        require(manifest.get("protocol_sha256") == evaluation.get("protocol_sha256"),
                f"Capture protocol identity differs for {task}")
        log_path = source / (task + ".log")
        log_identity = identity(log_path, source)
        require(results[task].get("log_sha256") == log_identity["sha256"],
                f"Raw log hash differs from task results for {task}")
        final_lines = [(i, line) for i, line in enumerate(log_path.read_text().splitlines(), 1)
                       if line.startswith("results:")]
        require(len(final_lines) == 1, f"Expected one final results tuple for {task}")
        line_number, line = final_lines[0]
        result = ast.literal_eval(line.split(":", 1)[1].strip())
        require(result[0] == "libero_sim/" + task, f"Raw log task identity differs for {task}")
        success, lengths = result[1], result[2]["episode_lengths"]
        resets = results[task]["resets"]
        require(len(success) == len(lengths) == len(resets) == evaluation["episodes"],
                f"Episode counts disagree for {task}")
        require(all(type(x) is bool for x in success) and success == results[task]["results"],
                f"Episode success labels disagree for {task}")
        require(all(type(x) is int and x > 0 for x in lengths), f"Invalid episode lengths for {task}")
        events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
        require(manifest.get("reset_event_file_sha256") == identity(event_path, source)["sha256"],
                f"Reset event file hash differs for {task}")
        for i, reset in enumerate(resets):
            require(reset.get("task_name") == task and reset.get("episode_index") == i
                    and sum(event == reset for event in events) == 1,
                    f"Episode reset identity disagrees for {task}:{i}")
        episodes = [{"episode_index": i, "success": ok, "episode_length": length,
                     "accepted_samples": [], "rejected_samples": [],
                     "accepted_server_calls": [], "rejected_server_calls": []}
                    for i, (ok, length) in enumerate(zip(success, lengths))]
        indexes, sample_records = set(), []
        for kind, pattern in (("accepted", "sample_*.pt"), ("rejected", "rejected_*.pt")):
            for path in sorted(folder.glob(pattern)):
                sample = torch.load(path, map_location="cpu", weights_only=True)
                metadata_fields.update(sample)
                episode, index, call = (sample.get(k) for k in ("episode_index", "capture_index", "server_call"))
                require(type(episode) is int and 0 <= episode < len(episodes),
                        f"Invalid sample episode identity: {path}")
                require(type(index) is int and index >= 0 and index not in indexes,
                        f"Invalid or duplicate capture_index: {path}")
                require(path.stem.split("_", 1)[1].isdigit()
                        and int(path.stem.split("_", 1)[1]) == index,
                        f"Filename and capture_index differ: {path}")
                require(type(call) is int and call > 0, f"Invalid server_call: {path}")
                require(type(sample.get("episode_success")) is bool
                        and sample["episode_success"] == success[episode],
                        f"Sample success label disagrees: {path}")
                reset = resets[episode]
                require(sample.get("task_name") == task and sample.get("reset_identity") == reset
                        and sample.get("episode_seed") == reset["seed"]
                        and sample.get("init_state_index") == reset["init_state_index"]
                        and sample.get("episode_result_index") == episode,
                        f"Sample episode/reset identity disagrees: {path}")
                require(sample.get("source_kind") == source_kind,
                        f"Sample source kind disagrees: {path}")
                should_reject = purpose == "teacher_supervision" and not success[episode]
                require((kind == "rejected") == should_reject,
                        f"Sample filename disagrees with purpose/success filtering: {path}")
                indexes.add(index)
                record = {"file": identity(path, source), "server_call": call,
                          "capture_index": index, "episode_index": episode,
                          "episode_success": success[episode],
                          "init_state_index": sample["init_state_index"], "source_kind": source_kind}
                sample_records.append(record)
                episodes[episode][kind + "_samples"].append(record)
                episodes[episode][kind + "_server_calls"].append(call)
                if kind == "accepted":
                    accepted_episode_counts[f"{task}:{episode}"] += 1
                    calls.append(call)
        saved = counts.get("saved")
        require(type(saved) is int and saved > 0 and indexes == set(range(saved)),
                f"Missing capture_index or count mismatch for {task}")
        ordered = sorted(sample_records, key=lambda x: x["capture_index"])
        require(all(a["server_call"] < b["server_call"] and a["episode_index"] <= b["episode_index"]
                    for a, b in zip(ordered, ordered[1:])),
                f"Capture chronology disagrees for {task}")
        successful_windows = sum(record["episode_success"] for record in ordered)
        rejected = sum(len(e["rejected_samples"]) for e in episodes)
        # The existing finalizer's accepted_samples field counts successful
        # windows, including for collection, where failed windows are retained.
        expected = {"total_samples": saved, "accepted_samples": successful_windows,
                    "unsuccessful_samples": saved - successful_windows,
                    "rejected_samples": rejected, "scored_episode_count": len(success),
                    "successful_episode_count": sum(success)}
        require(all(manifest.get(k) == v for k, v in expected.items()),
                f"Finalized capture counts disagree for {task}")
        for episode in episodes:
            for kind in ("accepted", "rejected"):
                episode[kind + "_server_calls"].sort()
            all_lengths.append(episode["episode_length"])
            if episode["success"]:
                successful_lengths.append(episode["episode_length"])
            if episode["accepted_samples"]:
                accepted_lengths.append(episode["episode_length"])
        rows.append({"task": task, "source_files": [log_identity] + [identity(p, source) for p in (
            manifest_path, counts_path, event_path)], "episode_lengths_log_line": line_number,
            "capture_configuration": {k: manifest.get(k) for k in (
                "every_server_calls", "per_task_limit", "total_limit", "per_episode_limit",
                "source_kind", "action_endpoint", "protocol_sha256")},
            "accepted_windows": saved - rejected, "rejected_windows": rejected,
            "successful_windows": successful_windows, "successful_episodes": sum(success),
            "episodes": episodes})
    require(not torch.cuda.is_initialized(), "CPU audit unexpectedly initialized CUDA")
    return {"format": "capture_coverage_audit_v1", "source": str(source), "purpose": purpose,
            "device": "cpu", "cuda_initialized": False,
            "source_files": [identity(source / name, source) for name in (
                "eval_manifest.json", "task_results.json")],
            "metadata_fields": sorted(metadata_fields),
            "interpretation_limits": [
                "accepted_windows means sample_*.pt retained for training; collection includes failed episodes",
                "manifest.accepted_samples counts successful windows, not all retained collection windows",
                "server_call is task-global; capture_index counts saved windows within a task",
                "no episode-relative action step or manipulation stage is inferred from server_call",
                "episode_lengths are verified against the hashed original rollout log results tuple",
                "capture_counts.calls only updates when saving a window and is not the full trajectory length"],
            "summary": {"tasks": len(rows),
                "accepted_windows": sum(r["accepted_windows"] for r in rows),
                "rejected_windows": sum(r["rejected_windows"] for r in rows),
                "successful_windows": sum(r["successful_windows"] for r in rows),
                "scored_episodes": len(all_lengths), "successful_episodes": len(successful_lengths),
                "accepted_source_episodes": len(accepted_episode_counts),
                "accepted_windows_per_episode_range": bounds(accepted_episode_counts.values()),
                "all_episode_length_range": bounds(all_lengths),
                "successful_episode_length_range": bounds(successful_lengths),
                "accepted_source_episode_length_range": bounds(accepted_lengths),
                "accepted_global_server_call_range": bounds(calls)}, "tasks": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Finalized evaluation capture directory")
    parser.add_argument("--out", type=Path, help="Optional JSON report, outside the source directory; otherwise stdout")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    if args.out:
        output = args.out.resolve()
        require(output != source and source not in output.parents,
                "Audit output must be outside the source evidence directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = audit(source)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.out:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
        print(json.dumps({"output": str(output), "sha256": identity(output, output.parent)["sha256"],
                          "summary": result["summary"]}, indent=2))
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
