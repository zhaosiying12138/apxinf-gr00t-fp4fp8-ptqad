"""Derive matched head/stratified views from a completed full-candidate rollout.

CPU only. Candidates remain untouched. This produces data views, not a new
evaluation receipt, and does not declare compatibility with legacy auditors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.run_recovery_eval import (TASKS, capture_sampling_config, parse_log,
                                    protocol_entry, validate_capture_partition_separation,
                                    validate_resets)
from rl.capture_sampling import materialize_capture_view, plan_capture_view


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plan_views(source, windows_per_episode=4):
    """Validate completed rollout and both plans before creating any output."""
    source = Path(source).resolve()
    manifest = read(source / "eval_manifest.json")
    summary = read(source / "summary.json")
    results = read(source / "task_results.json")
    purpose = manifest.get("purpose")
    require(purpose in ("teacher_supervision", "collection"), "Source must be a training capture")
    require(manifest.get("tasks") == TASKS and set(results) == set(TASKS), "All ten tasks are required")
    require(manifest.get("n_envs") == 1, "Full capture requires one environment")
    protocol = Path(manifest["protocol_file"])
    require(sha(protocol) == manifest.get("protocol_sha256"), "Source protocol changed")
    partition = protocol_entry(protocol, purpose)
    validate_capture_partition_separation(protocol, purpose)
    episodes = manifest["episodes"]
    require(type(episodes) is int and episodes > 0, "Invalid episode count")
    require(partition.get("episodes_per_task", episodes) == episodes
            and manifest["init_state_indices"] == partition["init_state_indices"]
            and len(partition["init_state_indices"]) == episodes
            and partition.get("seed", manifest["seed"]) == manifest["seed"],
            "Source partition differs from protocol")
    config = capture_sampling_config(partition, purpose, episodes, manifest["max_episode_steps"])
    require(config is not None and manifest.get("capture_sampling") == config,
            "Source lacks matching explicit full-candidate configuration")
    plans = {"head": [], "stratified": []}
    total_successes = 0
    for index, task in enumerate(TASKS):
        result = results[task]
        raw = parse_log(source / f"{task}.log")
        require(result.get("returncode") == 0 and raw["episodes"] == episodes,
                f"Incomplete task: {task}")
        require(all(result.get(key) == value for key, value in raw.items()),
                f"Raw rollout disagrees with task result: {task}")
        seed = manifest["seed"] + 1000 * index
        require(result.get("seed") == seed, f"Task seed differs: {task}")
        validate_resets(raw, seed, manifest["init_state_indices"])
        total_successes += raw["successes"]
        directory = source / "observations" / task
        capture = read(directory / "capture_manifest.json")
        require(capture.get("protocol_sha256") == manifest["protocol_sha256"]
                and capture.get("student_checkpoint") == manifest["checkpoint"]
                and capture.get("seed") == seed
                and capture.get("init_state_indices") == ",".join(map(str, manifest["init_state_indices"])),
                f"Capture identity differs from eval manifest: {task}")
        require(capture.get("scored_result") == {"results": raw["results"], "resets": raw["resets"]},
                f"Candidate success labels differ from rollout: {task}")
        require(capture.get("every_server_calls") == config["every_server_calls"]
                and capture.get("per_episode_limit") == config["safety_candidates_per_episode"]
                and capture.get("per_task_limit") == config["safety_candidates_per_task"]
                and capture.get("total_limit") == config["safety_candidates_per_task"],
                f"Runtime sampling differs from protocol: {task}")
        for mode in plans:
            plans[mode].append(plan_capture_view(
                directory, task, mode=mode, windows_per_episode=windows_per_episode))
    require(summary.get("tasks_complete") == 10
            and summary.get("total_episodes") == 10 * episodes
            and summary.get("total_successes") == total_successes
            and summary.get("purpose") == purpose, "Source summary is incomplete or inconsistent")
    for head, stratified in zip(plans["head"], plans["stratified"]):
        identity = lambda plan: [(x["episode_index"], x["episode_success"], x["accepted_for_training"])
                                 for x in plan["selected"]]
        require(identity(head) == identity(stratified), "Views must share episode identities and budgets")
    return {"schema": "fp4vla_paired_capture_views_plan_v1", "source_directory": str(source),
            "source_eval_manifest_sha256": sha(source / "eval_manifest.json"),
            "source_task_results_sha256": sha(source / "task_results.json"),
            "source_summary_sha256": sha(source / "summary.json"),
            "protocol_sha256": manifest["protocol_sha256"],
            "windows_per_episode": windows_per_episode, "plans": plans}


def derive_views(source, output, windows_per_episode=4):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source != output and source not in output.parents and output not in source.parents,
            "View output must be separate from the original evaluation")
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    plan = plan_views(source, windows_per_episode)
    output.mkdir(parents=True)
    (output / "paired_selection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    inventory = {}
    for mode, task_plans in plan["plans"].items():
        views = [materialize_capture_view(p, output / mode / "observations" / p["task_name"])
                 for p in task_plans]
        inventory[mode] = {"tasks": len(views),
                           "accepted_samples": sum(v["accepted_samples"] for v in views),
                           "rejected_samples": sum(v["rejected_samples"] for v in views)}
    receipt = {"schema": "fp4vla_paired_capture_views_v1", "status": "complete",
               "source_directory": str(source), "protocol_sha256": plan["protocol_sha256"],
               "paired_plan_sha256": sha(output / "paired_selection_plan.json"), "views": inventory,
               "legacy_audit_compatible": False,
               "scope": "CPU data views from one rollout; no inference, training or new evaluation"}
    (output / "views_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Completed full-candidate evaluation directory")
    parser.add_argument("--out", required=True, help="New directory outside source")
    parser.add_argument("--windows-per-episode", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(derive_views(args.source, args.out, args.windows_per_episode), indent=2))


if __name__ == "__main__":
    main()
