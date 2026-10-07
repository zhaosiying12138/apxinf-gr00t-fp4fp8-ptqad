"""Freeze matched teacher/student observations for later QAD endpoint inference.

CPU only. This validates the original rollouts and both derived views, records
identities and seeds, and never generates actions, caches, or training results.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.run_recovery_eval import TASKS, validate_recovery_checkpoint
from exp.action_chunk_diagnostics import checkpoint_files
from exp.derive_capture_views import audit_paired_views, audit_training_view


SCHEMA = "fp4vla_probe_endpoint_plan_v1"
CONFIG_FIELDS = {
    "view_mode", "pairing_rule", "identity_order", "window_pairing", "windows_per_identity",
    "minimum_common_identities_per_task", "minimum_pairs_per_task", "maximum_pairs_per_task",
    "endpoint_seed", "velocity_seed", "seed_rule",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def endpoint_config(protocol):
    config = protocol.get("endpoint_distillation")
    require(isinstance(config, dict) and set(config) == CONFIG_FIELDS,
            "Protocol must declare the exact endpoint_distillation fields")
    require(config["view_mode"] in ("head", "stratified"), "Invalid endpoint view mode")
    for field, expected in {
        "pairing_rule": "task_init_state_teacher_success_student_any",
        "identity_order": "task_order_then_init_state_index",
        "window_pairing": "episode_call_rank_prefix_min",
        "seed_rule": "base_plus_global_pair_index",
    }.items():
        require(config[field] == expected, "Unsupported endpoint rule: " + field)
    for field in ("windows_per_identity", "minimum_common_identities_per_task",
                  "minimum_pairs_per_task", "maximum_pairs_per_task"):
        require(type(config[field]) is int and config[field] > 0, "Invalid endpoint budget: " + field)
    require(config["minimum_pairs_per_task"] <= config["maximum_pairs_per_task"],
            "Endpoint minimum exceeds maximum task budget")
    require(config["windows_per_identity"] == protocol.get("capture_views", {}).get("windows_per_episode"),
            "Endpoint windows differ from the frozen capture view budget")
    for field in ("endpoint_seed", "velocity_seed"):
        require(type(config[field]) is int and 0 <= config[field] < 2**63, "Invalid endpoint seed: " + field)
        require(config[field] + len(TASKS) * config["maximum_pairs_per_task"] - 1 < 2**63,
                "Endpoint seed range exceeds signed 64-bit range")
    return config


def view_root(root, mode):
    root = Path(root).expanduser().resolve()
    view = root.parent if root.name == "observations" else root
    require(view.name == mode and root in (view, view / "observations"),
            "Both sources must select the protocol's single head/stratified mode")
    return view


def qad_training_identity(training, recovery, protocol, protocol_sha, teacher_view, teacher_checkpoint):
    """Verify the completed, uncontinued QAD-stratified training receipt."""
    from exp.run_high_fp4_v3 import capture_dataset_id, model_id
    selection = protocol.get("selection", {})
    integer_fields = ("qad_optimizer_steps", "rank", "effective_demo_batch", "train_seed", "opd_every")
    require(all(type(selection.get(key)) is int and selection[key] >= (0 if key == "train_seed" else 1)
                for key in integer_fields),
            "Protocol lacks explicit QAD training settings")
    rates = selection.get("qad_learning_rates")
    require(isinstance(rates, list) and len(rates) == 1 and type(rates[0]) in (float, int)
            and math.isfinite(rates[0]) and rates[0] > 0 and isinstance(selection.get("recovery_scope"), str)
            and type(selection.get("alpha")) in (int, float)
            and math.isfinite(selection["alpha"]) and selection["alpha"] > 0,
            "Endpoint QAD requires a single fixed learning rate, alpha and scope")
    steps, batch = selection["qad_optimizer_steps"], selection["effective_demo_batch"]
    require(training.name == f"checkpoint-{steps}", "Endpoint QAD is not the completed protocol checkpoint")
    stage = training.parent
    expected = {"rank": selection["rank"], "alpha": selection["alpha"],
                "scope": selection["recovery_scope"], "train_seed": selection["train_seed"],
                "seed": selection["train_seed"], "protocol_sha256": protocol_sha,
                "probe_weight": 0.0, "probe_every": selection["opd_every"],
                "initial_adapter": None, "optimizer_resumed": False,
                "probe_cache": None, "probe_cache_sha256": None, "probe_action_mask": None,
                "objective": "demo flow loss", "micro_batch": 1,
                "gradient_accumulation_steps": batch, "effective_global_batch": batch,
                "w4a4_enabled": True, "execution_mode": "W4A4 numerical QDQ + BF16 LoRA residual"}
    require(all(key in recovery and recovery[key] == value for key, value in expected.items())
            and recovery["initial_adapter"] is None and recovery["optimizer_resumed"] is False
            and type(recovery["probe_weight"]) in (int, float),
            "Endpoint policy must be pure QAD with the frozen protocol training settings")
    require(read(stage / "recovery_manifest.json") == recovery,
            "Endpoint checkpoint and completed training recovery manifest differ")
    metrics_path = stage / "runtime_metrics.json"
    metrics = read(metrics_path)
    require(metrics.get("status") == "completed" and metrics.get("global_steps") == steps
            and metrics.get("requested_optimizer_steps") == steps,
            "Endpoint QAD training did not complete the protocol update budget")
    capture_value = recovery.get("capture_dataset")
    require(isinstance(capture_value, str) and bool(capture_value), "QAD lacks its captured training dataset")
    capture = Path(capture_value).resolve()
    anchor = teacher_view.parent / "stratified"
    require(capture in (anchor, anchor / "observations"),
            "Endpoint QAD must have trained on this teacher's stratified view")
    audit = audit_training_view(capture, Path(recovery["protocol_file"]), teacher_checkpoint, minimum_episodes=2)
    require(audit.get("protocol_sha256") == protocol_sha, "QAD capture audit uses another protocol")
    dataset = capture_dataset_id(capture)
    dataset_sha, audit_sha = canonical_sha(dataset), canonical_sha(audit)
    require(recovery.get("capture_dataset_sha256") == dataset_sha
            and recovery.get("capture_dataset_samples") == audit["sample_count"],
            "QAD captured dataset identity differs from the audited stratified view")
    request_path = stage / "orchestrator_training_request.json"
    request = read(request_path)
    require(request.get("protocol_sha256") == protocol_sha
            and request.get("base_identity") == model_id(Path(recovery["base"]))
            and request.get("initial_adapter_identity") is None and request.get("cache_identity") is None
            and request.get("capture_dataset_identity") == dataset
            and request.get("teacher_capture_identity") == audit,
            "QAD training request source/audit identities differ")
    env = request.get("environment", {})
    actual_env = recovery.get("environment_summary", {}).get("variables", {})
    expected_env = {"QAD_STEPS": steps, "QAD_LR": rates[0], "QAD_LORA_R": selection["rank"],
                    "QAD_LORA_ALPHA": selection["alpha"], "QAD_LORA_SCOPE": selection["recovery_scope"],
                    "TRAIN_SEED": selection["train_seed"], "OPD_EVERY": selection["opd_every"],
                    "QAD_OPD_MSE_W": 0.0, "FP4VLA_QUANT": "0", "FP4VLA_W4A4": "1",
                    "FP4VLA_W4A4_ADAPTER": "0", "FP4VLA_SATURATE_F16_ACTIVATIONS": "0",
                    "QAD_W4A4": "1", "QAD_ACTIVATION_CHECKPOINTING": "1"}
    for key, value in expected_env.items():
        recorded = env.get(key)
        require(isinstance(recorded, str) and recorded == actual_env.get(key),
                "QAD requested/actual training environment differs: " + key)
        try:
            matches = float(recorded) == value if isinstance(value, (int, float)) else recorded == value
        except (TypeError, ValueError):
            matches = False
        require(matches, "QAD training setting differs from protocol: " + key)
    require(env.get("QAD_OUT") == str(stage) and env.get("QAD_CAPTURE_DATASET") == str(capture)
            and env.get("QAD_CAPTURE_DATASET_SHA256") == dataset_sha
            and env.get("QAD_CAPTURE_TEACHER") == str(teacher_checkpoint)
            and env.get("QAD_CAPTURE_AUDIT_SHA256") == audit_sha
            and env.get("QAD_GLOBAL_BATCH") == str(batch) and env.get("QAD_MICRO_BATCH") == "1"
            and env.get("QAD_INIT_ADAPTER") is None and env.get("OPD_CACHE_PATH") is None,
            "QAD training request does not bind the pure stratified starting point")
    return {"training_stage": str(stage), "capture_dataset": str(capture),
            "capture_dataset_sha256": dataset_sha, "capture_audit_sha256": audit_sha,
            "training_request": {"path": str(request_path), "sha256": sha(request_path)},
            "runtime_metrics": {"path": str(metrics_path), "sha256": sha(metrics_path)},
            "optimizer_steps": steps, "learning_rate": rates[0]}


def qad_identity(checkpoint, protocol_sha, protocol, teacher_view, teacher_checkpoint):
    """Bind the actual deployed base and adapter, not merely an export path."""
    checkpoint = Path(checkpoint).resolve()
    validate_recovery_checkpoint(checkpoint)
    require((checkpoint / "merge_manifest.json").is_file(), "Endpoints require a deployed QAD adapter")
    merge = read(checkpoint / "merge_manifest.json")
    files = checkpoint_files(checkpoint)
    base, training = (Path(merge[key]).resolve() for key in ("base", "training_checkpoint"))
    for root, field in ((base, "base_weights"), (training, "training_weights")):
        actual = {p.name: files[str(p)] for p in sorted(root.glob("*.safetensors"))}
        require(merge.get(field) == actual and bool(actual), "QAD deployment weights changed: " + field)
    recovery_path = Path(merge.get("recovery_manifest_source", "")).resolve()
    require(recovery_path in (training / "recovery_manifest.json", training.parent / "recovery_manifest.json"),
            "QAD recovery manifest is outside the training checkpoint")
    recovery = read(recovery_path)
    require(merge.get("recovery_manifest_sha256") == sha(recovery_path)
            and merge.get("recovery_manifest") == recovery
            and read(checkpoint / "recovery_manifest.json") == recovery,
            "QAD recovery manifest binding differs")
    require(recovery.get("protocol_sha256") == protocol_sha
            and recovery.get("w4a4_enabled") is True
            and Path(recovery.get("base", "")).resolve() == base,
            "QAD must use this protocol and its frozen W4A4 base")
    for field, filename in (("base_config_sha256", "config.json"),
                            ("base_statistics_sha256", "statistics.json"),
                            ("base_recipe_sha256", "ptq_recipe.json")):
        require(recovery.get(field) == sha(base / filename), "QAD base metadata changed: " + filename)
    for filename in ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json"):
        require(sha(checkpoint / filename) == sha(base / filename),
                "QAD export and actual loader base metadata differ: " + filename)
    training_identity = qad_training_identity(training, recovery, protocol, protocol_sha, teacher_view, teacher_checkpoint)
    return {"checkpoint": str(checkpoint), "checkpoint_files": files,
            "base": str(base), "training_checkpoint": str(training),
            "recovery_manifest_sha256": sha(recovery_path), "training_identity": training_identity}


def source_samples(audit, view, checkpoint, purpose, protocol_sha, checkpoint_identity):
    """Load only audited view bytes, retaining observation provenance unchanged."""
    import torch
    evaluation = audit["evaluation"]
    kind, role = (("teacher_rollout", "teacher") if purpose == "teacher_supervision"
                  else ("student_rollout", "student"))
    require(evaluation.get("purpose") == purpose and evaluation.get("checkpoint") == str(checkpoint)
            and evaluation.get("protocol_sha256") == protocol_sha,
            "Observation source purpose/checkpoint/protocol differs")
    if purpose == "collection":
        require(evaluation.get("checkpoint_files") == checkpoint_identity,
                "Collection lacks matching capture-time QAD checkpoint_files")
        variables = evaluation.get("environment_summary", {}).get("variables", {})
        require(all(variables.get(key) == value for key, value in {
            "FP4VLA_QUANT": "0", "FP4VLA_W4A4": "1", "FP4VLA_W4A4_ADAPTER": "1",
            "FP4VLA_SATURATE_F16_ACTIVATIONS": "1"}.items()),
            "Collection lacks explicit W4A4 QAD execution provenance")
    stats, config_sha = sha(checkpoint / "statistics.json"), sha(checkpoint / "config.json")
    selected = audit["views"][view.name]
    require(set(selected) == set(TASKS), "Source must contain all ten tasks")
    groups = defaultdict(list)
    for task in TASKS:
        capture = selected[task]
        for key, value in {"source_kind": kind, "checkpoint_role": role,
                           "student_checkpoint": str(checkpoint), "student_statistics_sha256": stats,
                           "student_config_sha256": config_sha}.items():
            require(capture.get(key) == value, "Capture identity differs: " + task + "/" + key)
        for record in capture["selected_sources"]:
            if not record["accepted_for_training"]:
                require(purpose == "teacher_supervision" and record["episode_success"] is False,
                        "Student states must not be filtered by success")
                continue
            path = view / "observations" / task / record["output_filename"]
            content = path.read_bytes()
            require(hashlib.sha256(content).hexdigest() == record["output_sha256"],
                    "Observation changed after view audit: " + str(path))
            sample = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            require(sample.get("source_kind") == kind and sample.get("checkpoint_role") == role
                    and sample.get("student_checkpoint") == str(checkpoint)
                    and sample.get("student_statistics_sha256") == stats and sample.get("task_name") == task,
                    "Observation source identity differs: " + str(path))
            inputs = sample.get("inputs")
            required_inputs = {"embodiment_id", "state", "input_ids", "attention_mask",
                               "pixel_values", "image_grid_thw", "action", "action_mask"}
            require(isinstance(inputs, dict) and set(inputs) == required_inputs,
                    "Observation model inputs differ")
            require(all(torch.is_tensor(value) and (not value.is_floating_point()
                        or bool(torch.isfinite(value).all())) for value in inputs.values()),
                    "Observation contains invalid/nonfinite tensors")
            action, mask = inputs["action"], inputs["action_mask"]
            require(action.ndim == 3 and action.shape[0] == 1 and action.shape == mask.shape
                    and bool(((mask == 0) | (mask == 1)).all()) and bool(mask.any()),
                    "Observation action/mask differs")
            require(type(sample.get("episode_success")) is bool
                    and (purpose != "teacher_supervision" or sample["episode_success"] is True),
                    "Invalid observation success provenance")
            state = sample.get("init_state_index")
            require(type(state) is int and sample.get("reset_identity", {}).get("init_state_index") == state,
                    "Observation initial-state identity differs")
            require(type(sample.get("episode_call")) is int and sample["episode_call"] > 0,
                    "Observation lacks a query rank")
            groups[(task, state)].append({"path": str(path), "sha256": record["output_sha256"],
                "source_kind": kind, "source_checkpoint": str(checkpoint),
                "action_contract": {"shape": list(action.shape), "mask_dtype": str(mask.dtype),
                    "mask_sha256": hashlib.sha256(mask.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()},
                "source_candidate": {"path": str(Path(capture["source_directory"]) / record["source_filename"]),
                                     "sha256": record["source_sha256"]},
                "provenance": {key: value for key, value in sample.items() if key != "inputs"}})
    for items in groups.values():
        items.sort(key=lambda item: item["provenance"]["episode_call"])
        require(len({x["provenance"]["episode_call"] for x in items}) == len(items)
                and len({x["provenance"]["episode_index"] for x in items}) == 1,
                "Repeated reset identity or query rank in observation source")
    return groups


def freeze_endpoint_plan(teacher_view, student_view, protocol_file, qad_checkpoint, *, teacher_checkpoint, out=None):
    protocol_file = Path(protocol_file).expanduser().resolve()
    teacher_checkpoint, qad_checkpoint = (Path(x).expanduser().resolve() for x in (teacher_checkpoint, qad_checkpoint))
    destination = Path(out).expanduser().resolve() if out is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError("Refusing existing endpoint plan output: " + str(destination))
    protocol_sha = sha(protocol_file)
    protocol = read(protocol_file)
    config = endpoint_config(protocol)
    teacher_view, student_view = (view_root(x, config["view_mode"]) for x in (teacher_view, student_view))
    require(teacher_view.parent != student_view.parent, "Teacher and student must have distinct rollout sources")
    teacher_audit = audit_paired_views(teacher_view.parent, protocol_file)
    student_audit = audit_paired_views(student_view.parent, protocol_file)
    training_audit = audit_training_view(teacher_view, protocol_file, teacher_checkpoint,
                                        minimum_episodes=config["minimum_common_identities_per_task"])
    qad = qad_identity(qad_checkpoint, protocol_sha, protocol, teacher_view, teacher_checkpoint)
    teacher_files = checkpoint_files(teacher_checkpoint)
    require(sha(teacher_checkpoint / "statistics.json") == sha(qad_checkpoint / "statistics.json"),
            "Teacher and QAD observation normalization differs")
    teachers = source_samples(teacher_audit, teacher_view, teacher_checkpoint, "teacher_supervision",
                              protocol_sha, teacher_files)
    students = source_samples(student_audit, student_view, qad_checkpoint, "collection",
                              protocol_sha, qad["checkpoint_files"])
    pairs, budgets = [], {}
    for task in TASKS:
        teacher_states = {state for name, state in teachers if name == task}
        student_states = {state for name, state in students if name == task}
        common = sorted(teacher_states & student_states)
        require(len(common) >= config["minimum_common_identities_per_task"],
                "Insufficient common teacher-success/student identities: " + task)
        identities = []
        task_pairs = 0
        for state in common:
            left, right = teachers[(task, state)], students[(task, state)]
            left_reset, right_reset = (items[0]["provenance"]["reset_identity"] for items in (left, right))
            require(left_reset.get("init_state_bank_sha256") == right_reset.get("init_state_bank_sha256")
                    and isinstance(left_reset.get("init_state_bank_sha256"), str)
                    and len(left_reset["init_state_bank_sha256"]) == 64,
                    "Paired initial-state bank identity differs: " + task)
            require(left_reset.get("initial_state_sha256") == right_reset.get("initial_state_sha256"),
                    "Paired initial-state bytes differ: " + task)
            count = min(len(left), len(right), config["windows_per_identity"])
            require(count > 0, "Empty common initial-state identity")
            for rank in range(count):
                require(left[rank]["action_contract"] == right[rank]["action_contract"],
                        "Paired action shape or valid-action mask differs: " + task)
                index = len(pairs)
                pairs.append({"pair_index": index, "task_name": task, "init_state_index": state,
                    "init_state_bank_sha256": left_reset["init_state_bank_sha256"], "window_rank": rank,
                    "endpoint_seed": config["endpoint_seed"] + index,
                    "velocity_seed": config["velocity_seed"] + index,
                    "teacher": left[rank], "student": right[rank]})
            identities.append({"init_state_index": state, "teacher_windows": len(left),
                               "student_windows": len(right), "paired_windows": count,
                               "student_episode_success": right[0]["provenance"]["episode_success"]})
            task_pairs += count
        require(config["minimum_pairs_per_task"] <= task_pairs <= config["maximum_pairs_per_task"],
                "Matched task budget is outside the protocol bounds: " + task)
        budgets[task] = {"pairs": task_pairs, "common_identities": len(common), "identities": identities,
                        "teacher_success_without_student": sorted(teacher_states - student_states),
                        "student_without_successful_teacher": sorted(student_states - teacher_states)}
    sources = {}
    for role, view, audit in (("teacher", teacher_view, teacher_audit), ("student", student_view, student_audit)):
        evaluation_root = Path(audit["plan"]["source_directory"])
        evidence_paths = [evaluation_root / name for name in ("eval_manifest.json", "task_results.json", "summary.json")]
        evidence_paths.extend(evaluation_root / (task + ".log") for task in TASKS)
        sources[role] = {"view": str(view), "source_directory": audit["plan"]["source_directory"],
                         "evaluation_files": {str(path): sha(path) for path in evidence_paths},
                         "paired_views_manifest_sha256": sha(view.parent / "views_manifest.json"),
                         "paired_plan_sha256": sha(view.parent / "paired_selection_plan.json"),
                         "audited_chain_sha256": canonical_sha(audit), "source_kind": (
                             "teacher_rollout" if role == "teacher" else "student_rollout")}
    protected = {teacher_view.parent, student_view.parent, teacher_checkpoint, qad_checkpoint,
                 Path(qad["base"]), Path(qad["training_checkpoint"])}
    protected.update(Path(record["source_directory"]).resolve() for record in sources.values())
    if destination is not None:
        require(destination != protocol_file and all(destination != root and root not in destination.parents
                                                    for root in protected),
                "Endpoint plan output must be outside source evidence and checkpoints")
    require(sha(protocol_file) == protocol_sha, "Protocol changed during endpoint planning")
    require(checkpoint_files(qad_checkpoint) == qad["checkpoint_files"], "QAD changed during endpoint planning")
    require(checkpoint_files(teacher_checkpoint) == teacher_files, "Teacher changed during endpoint planning")
    for key in ("training_request", "runtime_metrics"):
        record = qad["training_identity"][key]
        require(sha(record["path"]) == record["sha256"], "QAD training receipt changed during endpoint planning")
    plan = {"schema": SCHEMA, "status": "frozen_cpu_plan", "protocol_file": str(protocol_file),
            "protocol_sha256": protocol_sha, "endpoint_distillation": config, "task_order": TASKS,
            "qad": qad, "teacher_checkpoint": str(teacher_checkpoint), "teacher_checkpoint_files": teacher_files,
            "teacher_training_audit_sha256": canonical_sha(training_audit), "sources": sources,
            "pairs_per_arm": len(pairs), "task_budgets": budgets, "pairs": pairs,
            "implementation_sha256": sha(__file__),
            "scope": "matched observation and seed plan only; no endpoint inference or teacher velocity labels",
            "population": "common task/initial-state identities with successful teacher trajectories; student failures retained"}
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-view", required=True)
    parser.add_argument("--student-view", required=True)
    parser.add_argument("--protocol-file", required=True)
    parser.add_argument("--qad-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--out", help="New JSON file outside source evidence; omitted prints only")
    args = parser.parse_args()
    result = freeze_endpoint_plan(args.teacher_view, args.student_view, args.protocol_file,
                                  args.qad_checkpoint, teacher_checkpoint=args.teacher_checkpoint, out=args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
