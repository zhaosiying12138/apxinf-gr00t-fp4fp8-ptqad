"""Derive matched head/stratified views from a completed full-candidate rollout.

CPU only. Candidates remain untouched. This produces data views, not a new
evaluation receipt, and does not declare compatibility with legacy auditors.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.run_recovery_eval import (TASKS, capture_sampling_config, parse_log,
                                    protocol_entry, validate_capture_partition_separation,
                                    validate_resets)
from rl.capture_sampling import materialize_capture_view, plan_capture_view
from rl.checkpoint_identity import checkpoint_files


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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
    require(manifest.get("checkpoint_files") == checkpoint_files(manifest["checkpoint"]),
            "Capture checkpoint bytes differ from the identity frozen before rollout")
    require(summary.get("checkpoint_files_verified_unchanged") is True,
            "Capture did not certify unchanged checkpoint bytes through rollout completion")
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


def audit_paired_views(paired_root, protocol_file):
    """Verify both data views against the external protocol and source rollout.

    This common provenance check permits teacher or collection sources. It
    does not declare collection states to be successful teacher supervision.
    """
    from rl.capture_sampling import verify_capture_view
    paired, protocol_file = (Path(x).resolve() for x in (paired_root, protocol_file))
    receipt = read(paired / "views_manifest.json")
    require(receipt.get("schema") == "fp4vla_paired_capture_views_v1"
            and receipt.get("status") == "complete", "Paired views are not complete")
    protocol = read(protocol_file)
    view_config = protocol.get("capture_views")
    require(isinstance(view_config, dict)
            and set(view_config) == {"modes", "windows_per_episode"}
            and view_config.get("modes") == ["head", "stratified"]
            and type(view_config.get("windows_per_episode")) is int
            and view_config["windows_per_episode"] > 0,
            "Protocol must freeze capture_views modes and windows_per_episode")
    protocol_sha = sha(protocol_file)
    require(receipt.get("protocol_sha256") == protocol_sha, "Derived view protocol differs")
    source = Path(receipt["source_directory"]).resolve()
    require(source != paired and source not in paired.parents and paired not in source.parents,
            "Paired views must be separate from source evaluation")
    plan = plan_views(source, view_config["windows_per_episode"])
    require(plan["protocol_sha256"] == protocol_sha,
            "Original rollout protocol differs from the requested training protocol")
    require(read(paired / "paired_selection_plan.json") == plan
            and receipt.get("paired_plan_sha256") == sha(paired / "paired_selection_plan.json"),
            "Paired selection plan or original source changed")
    evaluation = read(source / "eval_manifest.json")
    require(evaluation.get("protocol_sha256") == protocol_sha,
            "Source evaluation protocol differs from the requested training protocol")
    inventories, views, all_training_files = {}, {}, set()
    for selection, task_plans in plan["plans"].items():
        inventories[selection] = {"tasks": 0, "accepted_samples": 0, "rejected_samples": 0}
        views[selection] = {}
        for task_plan in task_plans:
            directory = paired / selection / "observations" / task_plan["task_name"]
            manifest = verify_capture_view(task_plan, directory)
            inventories[selection]["tasks"] += 1
            for key in ("accepted_samples", "rejected_samples"):
                inventories[selection][key] += manifest[key]
            all_training_files.update(directory / row["output_filename"] for row in task_plan["selected"]
                                      if row["accepted_for_training"])
            views[selection][task_plan["task_name"]] = manifest
    require(receipt.get("views") == inventories, "Paired view inventory differs")
    require(set(paired.rglob("sample_*.pt")) == all_training_files,
            "Unexpected training samples outside the declared paired views")
    return {"plan": plan, "receipt": receipt, "evaluation": evaluation, "views": views}


def audit_training_view(root, protocol_file, teacher, minimum_episodes=2):
    """Verify successful teacher supervision through both views and the original rollout."""
    import torch
    root, protocol_file, teacher = (Path(x).resolve() for x in (root, protocol_file, teacher))
    view = root.parent if root.name == "observations" else root
    mode = view.name
    require(mode in ("head", "stratified") and root in (view, view / "observations"),
            "Training root must select one head/stratified view or its observations child")
    require(type(minimum_episodes) is int and minimum_episodes > 0, "Invalid minimum episode count")
    paired = view.parent
    verified = audit_paired_views(paired, protocol_file)
    plan, receipt, evaluation = (verified[key] for key in ("plan", "receipt", "evaluation"))
    protocol_sha = sha(protocol_file)
    source = Path(plan["source_directory"])
    view_config = read(protocol_file)["capture_views"]
    require(evaluation.get("purpose") == "teacher_supervision"
            and evaluation.get("checkpoint") == str(teacher),
            "QAD views must come from the specified teacher's successful training rollouts")
    require(not any((teacher / name).exists() for name in (
        "ptq_recipe.json", "category_ptq_recipe.json", "merge_manifest.json")),
        "Teacher view requires an unquantized BF16 teacher checkpoint")
    variables = evaluation.get("environment_summary", {}).get("variables", {})
    require(all(variables.get(key) == "0" for key in (
        "FP4VLA_QUANT", "FP4VLA_W4A4", "FP4VLA_W4A4_ADAPTER",
        "FP4VLA_SATURATE_F16_ACTIVATIONS")), "Teacher capture lacks explicit BF16 execution provenance")
    statistics_sha, config_sha = sha(teacher / "statistics.json"), sha(teacher / "config.json")
    teacher_weights = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                       for p in sorted(teacher.glob("*.safetensors"))}
    require(bool(teacher_weights), "Teacher has no weight shards")
    for selection, task_views in verified["views"].items():
        for task, manifest in task_views.items():
            for key, value in {"source_kind": "teacher_rollout", "checkpoint_role": "teacher",
                               "student_checkpoint": str(teacher),
                               "student_statistics_sha256": statistics_sha,
                               "student_config_sha256": config_sha}.items():
                require(manifest.get(key) == value, f"Teacher view mismatch: {task}/{key}")
    observations = view / "observations"
    tasks = {}
    required_inputs = {"embodiment_id", "state", "input_ids", "attention_mask",
                       "pixel_values", "image_grid_thw", "action", "action_mask"}
    for task in TASKS:
        folder = observations / task
        manifest = verified["views"][mode][task]
        episodes, files = Counter(), []
        for record in manifest["selected_sources"]:
            if not record["accepted_for_training"]:
                continue
            path = folder / record["output_filename"]
            sample_bytes = path.read_bytes()
            require(hashlib.sha256(sample_bytes).hexdigest() == record["output_sha256"],
                    f"Sample changed during audit: {path}")
            sample = torch.load(io.BytesIO(sample_bytes), map_location="cpu", weights_only=True)
            require(sample.get("episode_success") is True and sample.get("source_kind") == "teacher_rollout"
                    and sample.get("checkpoint_role") == "teacher"
                    and sample.get("student_statistics_sha256") == statistics_sha,
                    f"Invalid successful teacher sample: {path}")
            inputs = sample.get("inputs")
            require(isinstance(inputs, dict) and set(inputs) == required_inputs,
                    f"Captured training input keys differ: {path}")
            for key, tensor in inputs.items():
                require(torch.is_tensor(tensor), f"Non-tensor training input: {path}/{key}")
                if tensor.is_floating_point():
                    require(bool(torch.isfinite(tensor).all()), f"Nonfinite training input: {path}/{key}")
            action, mask = inputs["action"], inputs["action_mask"]
            require(action.shape == mask.shape and action.ndim == 3 and action.shape[0] == 1,
                    f"Invalid action/mask shape: {path}")
            require(bool(((mask == 0) | (mask == 1)).all()) and bool(mask.any()), f"Invalid action mask: {path}")
            episodes[sample["episode_index"]] += 1
            files.append({"path": str(path.relative_to(observations)), "sha256": record["output_sha256"]})
        require(len(episodes) >= minimum_episodes,
                f"Too few successful episodes with samples: {task}: {len(episodes)} < {minimum_episodes}")
        tasks[task] = {"samples": len(files), "contributing_successful_episodes": len(episodes),
                       "samples_per_episode": {str(key): value for key, value in sorted(episodes.items())},
                       "source_files": files,
                       "capture_manifest_sha256": sha(folder / "capture_manifest.json")}
    return {"format": "derived_teacher_replay_audit_v1", "status": "verified",
            "evaluation": str(source), "observations": str(observations), "view": str(view),
            "selection_mode": mode, "windows_per_episode": view_config["windows_per_episode"],
            "protocol_sha256": protocol_sha, "teacher": str(teacher), "teacher_weights": teacher_weights,
            "minimum_contributing_successful_episodes": minimum_episodes,
            "task_count": len(tasks), "sample_count": sum(t["samples"] for t in tasks.values()),
            "tasks": tasks, "paired_views_manifest_sha256": sha(paired / "views_manifest.json"),
            "paired_plan_sha256": receipt["paired_plan_sha256"],
            "source_eval_manifest_sha256": plan["source_eval_manifest_sha256"],
            "source_task_results_sha256": plan["source_task_results_sha256"],
            "implementation_sha256": sha(__file__),
            "view_verifier_sha256": sha(Path(__file__).resolve().parents[1] / "rl/capture_sampling.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Completed full-candidate evaluation directory")
    parser.add_argument("--out", required=True, help="New directory outside source")
    parser.add_argument("--windows-per-episode", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(derive_views(args.source, args.out, args.windows_per_episode), indent=2))


if __name__ == "__main__":
    main()
