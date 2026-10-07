"""Paired GR00T action diagnostics, separate from training and held-out scoring.

freeze/compare are CPU-only. collect runs ONE checkpoint through the existing
formal server loader, with its network server replaced by a local callback.
Captured training endpoints are removed before get_action to avoid enabling RTC.
Actual first-step noise, rather than a seed alone, certifies paired inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
import io
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
from checkpoint_identity import checkpoint_files
from gr00t_runtime import action_mask, libero_action_spec
from probe_distill import file_sha256, replay_context, require_full_model, tensor_tree


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_inputs(capture_root, protocol_file, per_task, seed):
    """Freeze observations without selecting them using model predictions."""
    import torch
    capture_root, protocol_file = Path(capture_root).resolve(), Path(protocol_file).resolve()
    require(type(per_task) is int and per_task > 0, "per_task must be positive")
    require(type(seed) is int and 0 <= seed < 2**63, "seed must be a nonnegative integer")
    protocol = read(protocol_file)
    evaluation = read(capture_root.parent / "eval_manifest.json")
    purpose = evaluation.get("purpose")
    require(purpose in ("teacher_supervision", "collection", "development"),
            "Diagnostics must not consume held-out or undeclared captures")
    require(evaluation.get("protocol_sha256") == file_sha256(protocol_file), "Protocol identity differs")
    partition = protocol["partitions"][purpose]
    allowed = set(partition["init_state_indices"])
    require(not allowed.intersection(protocol["partitions"]["heldout"]["init_state_indices"]),
            "Diagnostic partition overlaps held-out")
    require(evaluation.get("init_state_indices") == partition["init_state_indices"], "Partition differs")
    tasks = evaluation.get("tasks", [])
    require(tasks and len(set(tasks)) == len(tasks), "Missing or duplicate task identities")
    samples, stats = [], set()
    for task in tasks:
        folder = capture_root / task
        capture = read(folder / "capture_manifest.json")
        require(capture.get("finalized") is True, f"Capture not finalized: {task}")
        require(capture.get("protocol_sha256") == file_sha256(protocol_file), f"Capture protocol differs: {task}")
        require(capture.get("task_name") == task, f"Capture task differs: {task}")
        paths = sorted(folder.glob("sample_*.pt"))
        require(len(paths) >= per_task, f"Too few observations for {task}: {len(paths)} < {per_task}")
        # Fixed filename order; never rank by loss, success difference or outputs.
        for path in paths[:per_task]:
            raw = torch.load(path, map_location="cpu", weights_only=True)
            require(raw.get("task_name") == task, f"Sample task differs: {path}")
            require(raw.get("init_state_index") in allowed, f"Sample outside partition: {path}")
            require(raw.get("reset_identity", {}).get("init_state_index") == raw["init_state_index"],
                    f"Sample reset identity differs: {path}")
            stats.add(raw.get("student_statistics_sha256"))
            samples.append({"path": str(path), "sha256": file_sha256(path), "task": task,
                            "seed": seed + len(samples), "init_state_index": raw["init_state_index"],
                            "episode_index": raw["episode_index"], "server_call": raw["server_call"]})
    require(len(stats) == 1 and None not in stats, "Capture normalization identities differ")
    return {"format": "gr00t_action_inputs_v1", "protocol_file": str(protocol_file),
            "protocol_sha256": file_sha256(protocol_file),
            "capture_root": str(capture_root), "purpose": purpose,
            "interpretation": ("training_distribution_fit_diagnostic" if purpose != "development"
                               else "development_distribution_diagnostic_not_heldout"),
            "statistics_sha256": next(iter(stats)), "samples_per_task": per_task,
            "selection": "first filenames per task; frozen before action measurements",
            "seed": seed, "samples": samples, "implementation_sha256": file_sha256(__file__)}


def infer_chunk(model, raw_inputs, seed, spec, *, record_trace=False):
    """Call production get_action, capturing its actual initial integration state."""
    import torch
    require(raw_inputs["action"].shape[0] == 1, "Expected independently collated batch of one")
    expected_mask = action_mask(raw_inputs["action"], spec)
    require(torch.equal(raw_inputs["action_mask"].float(), expected_mask), "Unexpected valid-action mask")
    # Presence of action in action_input enables real-time chunking/inpainting.
    inputs = tensor_tree({k: v for k, v in raw_inputs.items() if k not in ("action", "action_mask")})
    noise, calls = [], []
    states, times, velocities = [], [], []

    def capture_initial(module, args):
        calls.append(1)
        if not noise:
            noise.append(args[0].detach().cpu().clone())
        if record_trace:
            require(len(args) >= 2, "Trace requires the production action-encoder time input")
            states.append(args[0].detach().cpu().clone())
            times.append(args[1].detach().cpu().clone())

    def capture_velocity(module, args, output):
        require(torch.is_tensor(output) and output.ndim == 3, "Unexpected action decoder output")
        velocities.append(output[:, -expected_mask.shape[1]:].detach().cpu().clone())

    hook = model.action_head.action_encoder.register_forward_pre_hook(capture_initial)
    velocity_hook = None
    try:
        if record_trace:
            velocity_hook = model.action_head.action_decoder.register_forward_hook(capture_velocity)
        # Gr00tPolicy._get_action uses inference_mode with BF16 parameters and
        # no outer autocast. Preserve that convention instead of training's.
        with replay_context(model, seed, "none"), torch.inference_mode():
            output = model.get_action(inputs)["action_pred"].detach().float().cpu()
    finally:
        hook.remove()
        if velocity_hook is not None:
            velocity_hook.remove()
    steps = int(model.action_head.num_inference_timesteps)
    require(len(calls) == steps and len(noise) == 1, "Incomplete integration/noise trace")
    require(output.shape == expected_mask.shape == noise[0].shape, "Action/noise shape differs")
    require(bool(torch.isfinite(output).all()) and bool(torch.isfinite(noise[0]).all()), "Nonfinite inference")
    result = {"action_pred": output, "initial_noise": noise[0], "integration_steps": steps}
    if record_trace:
        require(len(states) == len(times) == len(velocities) == steps, "Incomplete integration trace")
        require(all(x.shape == noise[0].shape and bool(torch.isfinite(x).all())
                    for x in states + velocities), "Nonfinite or malformed integration trace")
        buckets = int(model.action_head.num_timestep_buckets)
        require(all(t.shape == (1,) and t.dtype == torch.int64 and t.item() == int(i / steps * buckets)
                    for i, t in enumerate(times)), "Observed time buckets differ from production schedule")
        result["integration_trace"] = {
            "states": torch.stack(states), "time_buckets": torch.stack(times),
            "velocities": torch.stack(velocities),
            "scope": "Each policy's own integration states and velocities; not shared-state field error or robot rollout drift"}
    return result


def decode_chunk(policy, action, processor_config):
    """Use the production decoder only when raw states are not required."""
    import torch
    configs = processor_config["processor_kwargs"]["modality_configs"]["libero_sim"]["action"]["action_configs"]
    if not configs or any(c.get("rep") != "ABSOLUTE" for c in configs):
        return None
    result = policy.processor.decode_action(action.numpy(), policy.embodiment_tag, state=None)
    decoded = {key: torch.as_tensor(value).float() for key, value in result.items()}
    require(all(bool(torch.isfinite(value).all()) for value in decoded.values()), "Nonfinite decoded action")
    return decoded


def bind_final_checkpoint(final_path, arm, inputs):
    """Resolve labels from the completed run and verify the weights it selected."""
    final = read(final_path)
    require(final.get("format") == "w4a4_recovery_v12_final_manifest", "Expected completed v12 final manifest")
    if inputs.get("format") == "gr00t_action_inputs_v2":
        from exp.independent_action_protocol import validate_diagnostic_protocol
        reference = validate_diagnostic_protocol(inputs["protocol_file"], final_manifest=final_path)
        require(inputs.get("purpose") == "diagnostics"
                and inputs["protocol_sha256"] == reference["diagnostic_protocol_sha256"]
                and inputs.get("original_model_protocol_sha256") == reference["original_model_protocol_sha256"]
                and inputs.get("final_reference") == reference["source_final"],
                "Independent diagnostic model reference differs")
        require(final["protocol_sha256"] == reference["original_model_protocol_sha256"],
                "Original model protocol differs from diagnostic reference")
    else:
        require(final["protocol_sha256"] == inputs["protocol_sha256"], "Final and diagnostic protocols differ")
    require(file_sha256(final["selection_file"]) == final["selection_sha256"], "PTQ selection changed")
    selection = read(final["selection_file"])
    require(selection["protocol_sha256"] == final["protocol_sha256"], "Selection protocol differs")
    base_id = selection["arms"][final["selected_pressure_recipe"]]["model_identity"]
    identity = (selection["arms"]["bf16"]["model_identity"] if arm == "bf16" else
                base_id if arm == "ptq" else final[{
                    "qad": "selected_qad_model_identity", "continued_qad": "selected_continued_model_identity",
                    "qad_opd": "selected_opd_model_identity"}[arm]])
    checkpoint = Path(identity["path"]).resolve()
    files = checkpoint_files(checkpoint)

    def verify_model(record):
        root = Path(record["path"]).resolve()
        require(set(p.name for p in root.glob("*.safetensors")) == {r["name"] for r in record["shards"]},
                "Selected model shard set differs")
        for name, sha in record["metadata"].items():
            require(files[str(root / name)]["sha256"] == sha, "Selected model metadata changed: " + name)
        for shard in record["shards"]:
            require(files[str(root / shard["name"])] == {k: shard[k] for k in ("bytes", "sha256")},
                    "Selected model weights changed")

    verify_model(identity)
    merge_path = checkpoint / "merge_manifest.json"
    if merge_path.is_file():
        merge = read(merge_path)
        require(Path(merge["base"]).resolve() == Path(base_id["path"]).resolve(), "Adapter base differs from selected PTQ")
        verify_model(base_id)
        for root_key, weights_key in (("base", "base_weights"), ("training_checkpoint", "training_weights")):
            root = Path(merge[root_key]).resolve()
            require(merge.get(weights_key), "Missing deployment weight identities")
            require(set(p.name for p in root.glob("*.safetensors")) == set(merge[weights_key]), "Deployment shard set differs")
            for name, record in merge[weights_key].items():
                require(files[str(root / name)] == record, "Deployment weight identity changed")
    return checkpoint, files


def validate_input_manifest(manifest):
    """Reaudit either legacy training observations or separate diagnostic captures."""
    if manifest.get("format") == "gr00t_action_inputs_v2":
        from exp.independent_action_inputs import freeze_diagnostic_inputs
        actual = freeze_diagnostic_inputs(manifest["capture_root"], manifest["protocol_file"])
    else:
        require(manifest.get("format") == "gr00t_action_inputs_v1", "Unknown input manifest")
        actual = freeze_inputs(manifest["capture_root"], manifest["protocol_file"],
                               manifest["samples_per_task"], manifest["seed"])
    require(manifest == actual, "Frozen inputs no longer match their declared source/partition")
    return manifest


def collect(args):
    """One fresh process per arm; uses the same loader as formal evaluation."""
    import torch
    manifest_path, final_path = Path(args.inputs).resolve(), Path(args.final_manifest).resolve()
    out = Path(args.out).resolve()
    require(not out.exists(), f"Refusing to overwrite diagnostic output: {out}")
    manifest = read(manifest_path)
    validate_input_manifest(manifest)
    trace_enabled = manifest["format"] == "gr00t_action_inputs_v2"
    checkpoint, weights = bind_final_checkpoint(final_path, args.arm, manifest)
    require(manifest["statistics_sha256"] == file_sha256(checkpoint / "statistics.json"),
            "Checkpoint and observations use different normalization")
    for sample in manifest["samples"]:
        require(file_sha256(sample["path"]) == sample["sha256"], "Frozen observation changed")
    sys.path.insert(0, str(ROOT))
    from eval.run_recovery_eval import validate_recovery_checkpoint
    validate_recovery_checkpoint(checkpoint)
    quantized = any((checkpoint / n).is_file() for n in
                    ("ptq_recipe.json", "category_ptq_recipe.json", "merge_manifest.json"))
    require((args.arm == "bf16") != quantized, "Arm label disagrees with checkpoint format")
    adapter = (checkpoint / "merge_manifest.json").is_file()
    require(adapter == (args.arm in ("qad", "continued_qad", "qad_opd")), "Arm adapter identity differs")
    # Match run_recovery_eval's explicit environment; remove capture/logging
    # exports so diagnostics cannot mutate a training dataset accidentally.
    for key in list(os.environ):
        if key.startswith(("OPD_CAPTURE_", "FP4VLA_CAPTURE_")) or key == "FP4VLA_LOG_DIR":
            os.environ.pop(key)
    os.environ.update({"HF_HUB_OFFLINE": "1", "FP4VLA_QUANT": "0", "FP4VLA_SCOPE": "all",
                       "FP4VLA_W4A4": str(int(quantized)), "FP4VLA_W4A4_ADAPTER": str(int(adapter)),
                       "FP4VLA_SATURATE_F16_ACTIVATIONS": str(int(quantized)),
                       "GR00T_EVAL_SEED": str(manifest["seed"])})
    gr00t = Path(args.gr00t).resolve()
    sys.path.insert(0, str(gr00t))
    sys.path.insert(0, str(ROOT / "eval"))
    os.chdir(gr00t)
    from gr00t.policy import server_client
    records = []
    action_spec = libero_action_spec(checkpoint)
    processor_config = read(checkpoint / "processor_config.json")
    actual_base = Path(read(checkpoint / "merge_manifest.json")["base"]).resolve() if adapter else checkpoint
    for name in ("statistics.json", "processor_config.json", "embodiment_id.json"):
        require(file_sha256(actual_base / name) == file_sha256(checkpoint / name),
                "Export and actual loader base disagree: " + name)
    source_paths = [Path(__file__), ROOT / "eval/serve_recovery.py", ROOT / "eval/run_gr00t_server_fp4vla.py",
                    ROOT / "rl/probe_distill.py", ROOT / "rl/checkpoint_identity.py",
                    ROOT / "rl/scoped_quant.py", ROOT / "rl/w4a4_deploy.py",
                    ROOT / "rl/w4a4_lora.py", ROOT / "quant/native_activation.py",
                    gr00t / "gr00t/eval/sim/LIBERO/libero_env.py"]
    if trace_enabled:
        source_paths.extend(ROOT / "exp" / name for name in (
            "independent_action_inputs.py", "independent_action_protocol.py"))

    class LocalActionRunner:
        def __init__(self, *, policy, **kwargs):
            self.policy = policy
            require_full_model(policy.model)
            from gr00t.data.utils import parse_modality_configs
            expected_modalities = parse_modality_configs({"libero_sim": processor_config[
                "processor_kwargs"]["modality_configs"]["libero_sim"]})["libero_sim"]
            require(policy.processor.modality_configs["libero_sim"] == expected_modalities,
                    "Actual processor modality configuration differs")
            expected_stats = read(actual_base / "statistics.json")["libero_sim"]
            require(policy.processor.state_action_processor.statistics.get("libero_sim") == expected_stats,
                    "Actual decoder normalization differs")
            mapping = read(actual_base / "embodiment_id.json")
            require(all(policy.processor.embodiment_id_mapping.get(k) == v for k, v in mapping.items()),
                    "Actual processor embodiment mapping differs")
            coverage = sum(bool(getattr(m, "_fp4vla_w4a4", False)) for m in policy.model.modules())
            expected_coverage = read(manifest["protocol_file"])["quantization_scope"]["activation_linear_count"] if quantized else 0
            require(coverage == expected_coverage, "Actual W4A4 activation coverage differs")
            source_paths.extend(Path(inspect.getfile(type(item))).resolve() for item in (
                policy, policy.model, policy.processor, policy.processor.state_action_processor))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run(self):
            for sample in manifest["samples"]:
                content = Path(sample["path"]).read_bytes()
                import hashlib
                require(hashlib.sha256(content).hexdigest() == sample["sha256"], "Observation changed before inference")
                raw = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
                result = infer_chunk(self.policy.model, raw["inputs"], sample["seed"], action_spec,
                                     record_trace=trace_enabled)
                result["decoded"] = decode_chunk(self.policy, result["action_pred"], processor_config)
                records.append({"sample": sample, **result})
                print(f"[action-diagnostic] {args.arm} {len(records)}/{len(manifest['samples'])}", flush=True)

    original_server, original_argv = server_client.PolicyServer, sys.argv
    server_client.PolicyServer = LocalActionRunner
    try:
        # serve_recovery also validates deployment manifests and applies the
        # same determinism setup as production. No RPC port is opened.
        sys.argv = [str(ROOT / "eval/serve_recovery.py"), "--model-path", str(checkpoint),
                    "--embodiment-tag", "LIBERO_PANDA"]
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        server_client.PolicyServer, sys.argv = original_server, original_argv
    require(len(records) == len(manifest["samples"]) > 0, "Incomplete diagnostic run")
    require(checkpoint_files(checkpoint) == weights, "Checkpoint changed during diagnostic inference")
    out.mkdir(parents=True)
    sources = {str(p): file_sha256(p) for p in source_paths}
    final_sha = file_sha256(final_path)
    torch.save({"format": "gr00t_action_outputs_v1", "arm": args.arm,
                "inputs_sha256": file_sha256(manifest_path), "input_manifest": manifest,
                "final_manifest_sha256": final_sha, "source_files": sources,
                "action_spec": action_spec, "records": records}, out / "actions.pt")
    write_new(out / "manifest.json", {"format": "gr00t_action_run_v1", "status": "complete",
              "arm": args.arm, "checkpoint": str(checkpoint), "checkpoint_files": weights,
              "final_manifest_sha256": final_sha,
              "inputs_sha256": file_sha256(manifest_path), "actions_sha256": file_sha256(out / "actions.pt"),
              "source_files": sources,
              "interpretation": manifest["interpretation"], "sample_count": len(records),
              "decoded_units": "processor output/controller coordinates; not measured end-effector displacement",
              "gripper_metric": "LIBERO -sign(2*x-1), preserving neutral x=0.5; not grasp success"})


def integration_metrics(reference, candidate, mask):
    """Describe own-path integration drift; never call it shared-state velocity error."""
    import torch
    left, right = reference.get("integration_trace"), candidate.get("integration_trace")
    require((left is None) == (right is None), "Only one arm records integration traces")
    if left is None:
        return {}
    steps = reference["integration_steps"]
    expected_shape = (steps, *reference["action_pred"].shape)
    for record, trace in ((reference, left), (candidate, right)):
        require(set(trace) == {"states", "velocities", "time_buckets", "scope"}, "Unexpected integration trace fields")
        require(trace["scope"] == "Each policy's own integration states and velocities; not shared-state field error or robot rollout drift",
                "Integration trace has a different measurement scope")
        for key in ("states", "velocities"):
            require(trace[key].shape == expected_shape and bool(torch.isfinite(trace[key]).all()),
                    "Malformed or nonfinite integration trace")
        require(trace["time_buckets"].shape == (steps, 1) and trace["time_buckets"].dtype == torch.int64,
                "Invalid integration time buckets")
        require(trace["states"].dtype == record["initial_noise"].dtype
                and torch.equal(trace["states"][0], record["initial_noise"]),
                "Integration trace does not start at the recorded initial noise")
    require(torch.equal(left["time_buckets"], right["time_buckets"]), "Integration time schedules differ")
    return {f"integration_{label}_step_{i + 1}_mse": float(
            (left[key][i].double() - right[key][i].double()).square()[mask].mean())
            for i in range(steps) for key, label in (("states", "state"), ("velocities", "own_path_velocity"))}


def compare(reference, candidate):
    """Validate actual common noise and exclude padding from every action metric."""
    import torch
    require(reference.get("format") == candidate.get("format") == "gr00t_action_outputs_v1", "Unknown output format")
    require(reference["arm"] == "bf16" and candidate["arm"] != "bf16", "Expected BF16 reference and one candidate")
    require(reference["inputs_sha256"] == candidate["inputs_sha256"], "Input manifests differ")
    require(reference["final_manifest_sha256"] == candidate["final_manifest_sha256"], "Selected runs differ")
    require(reference["source_files"] == candidate["source_files"], "Diagnostic/inference source versions differ")
    require(reference["input_manifest"] == candidate["input_manifest"], "Input manifest contents differ")
    require(reference["action_spec"] == candidate["action_spec"], "Action contracts differ")
    spec = reference["action_spec"]
    rr, cr = reference["records"], candidate["records"]
    expected = reference["input_manifest"]["samples"]
    require(len(rr) == len(cr) == len(expected) > 0, "Incomplete or empty records")
    metrics = []
    for ref, cur, sample in zip(rr, cr, expected):
        require(ref["sample"] == cur["sample"] == sample, "Sample ordering/identity differs")
        require(ref["integration_steps"] == cur["integration_steps"] > 0, "Integration schedules differ")
        noise_a, noise_b = ref["initial_noise"], cur["initial_noise"]
        require(noise_a.dtype == noise_b.dtype and torch.equal(noise_a, noise_b), "Actual initial noise differs")
        a, b = ref["action_pred"], cur["action_pred"]
        require(a.shape == b.shape == noise_a.shape, "Action shapes differ")
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "Nonfinite action")
        mask = action_mask(a, spec).bool()
        squared = (a.double() - b.double()).square()
        values = {"normalized_mse": float(squared[mask].mean())}
        if reference["input_manifest"].get("format") == "gr00t_action_inputs_v2":
            require("integration_trace" in ref and "integration_trace" in cur,
                    "Independent diagnostics require full integration traces")
        values.update(integration_metrics(ref, cur, mask))
        offset = 0
        for key, width in zip(spec["keys"], spec["key_dimensions"]):
            group = squared[:, :spec["horizon"], offset:offset + width]
            values[f"normalized_{key}_mse"] = float(group.mean())
            offset += width
        require(offset == spec["dimensions"], "Malformed action groups")
        da, db = ref["decoded"], cur["decoded"]
        require((da is None) == (db is None), "Only one arm provides decoded actions")
        if da is not None:
            require(set(da) == set(db) == set(spec["keys"]), "Decoded keys differ")
            for key, width in zip(spec["keys"], spec["key_dimensions"]):
                x, y = da[key].double(), db[key].double()
                require(x.shape == y.shape == (1, spec["horizon"], width), "Decoded shape differs")
                require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), "Nonfinite decoded action")
                values[f"decoded_{key}_mse"] = float((x - y).square().mean())
            if "gripper" in da:
                # Match LIBERO normalize_gripper_action + invert_gripper_action.
                # x=0.5 yields neutral 0; it must not be assigned to either bin.
                command_a = -(2 * da["gripper"] - 1).sign()
                command_b = -(2 * db["gripper"] - 1).sign()
                values["libero_gripper_command_disagreement"] = float((command_a != command_b).double().mean())
        metrics.append({"task": sample["task"], "sample_sha256": sample["sha256"],
                        "seed": sample["seed"], "metrics": values})
    keys = set(metrics[0]["metrics"])
    require(all(set(m["metrics"]) == keys for m in metrics), "Inconsistent decoded coverage")
    per_task = defaultdict(list)
    for row in metrics:
        per_task[row["task"]].append(row["metrics"])
    task_means = {t: {k: sum(r[k] for r in rows) / len(rows) for k in sorted(keys)}
                  for t, rows in per_task.items()}
    return {"candidate": candidate["arm"], "reference": "bf16", "samples": len(metrics),
            "final_manifest_sha256": reference["final_manifest_sha256"],
            "interpretation": reference["input_manifest"]["interpretation"],
            "same_actual_noise_verified": True, "per_sample": metrics, "per_task": task_means,
            "task_macro": {k: sum(v[k] for v in task_means.values()) / len(task_means) for k in sorted(keys)}}


def load_outputs(directory):
    import torch
    directory = Path(directory)
    receipt = read(directory / "manifest.json")
    require(receipt.get("status") == "complete", "Run is incomplete")
    require(receipt["actions_sha256"] == file_sha256(directory / "actions.pt"), "Action output changed")
    payload = torch.load(directory / "actions.pt", map_location="cpu", weights_only=True)
    require(payload["arm"] == receipt["arm"] and payload["inputs_sha256"] == receipt["inputs_sha256"], "Receipt identity differs")
    require(payload["final_manifest_sha256"] == receipt["final_manifest_sha256"] and
            payload["source_files"] == receipt["source_files"], "Receipt final/source binding differs")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--capture-root", required=True)
    freeze.add_argument("--protocol-file", required=True)
    freeze.add_argument("--per-task", type=int, default=8)
    freeze.add_argument("--seed", type=int, default=2026100600)
    freeze.add_argument("--out", required=True)
    independent = subs.add_parser("freeze-independent")
    independent.add_argument("--capture-root", required=True)
    independent.add_argument("--protocol-file", required=True)
    independent.add_argument("--out", required=True)
    run = subs.add_parser("collect")
    run.add_argument("--inputs", required=True)
    run.add_argument("--final-manifest", required=True)
    run.add_argument("--arm", choices=("bf16", "ptq", "qad", "continued_qad", "qad_opd"), required=True)
    run.add_argument("--gr00t", required=True)
    run.add_argument("--out", required=True)
    report = subs.add_parser("compare")
    report.add_argument("--reference", required=True)
    report.add_argument("--candidate", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        write_new(args.out, freeze_inputs(args.capture_root, args.protocol_file, args.per_task, args.seed))
    elif args.command == "freeze-independent":
        from exp.independent_action_inputs import freeze_diagnostic_inputs
        write_new(args.out, freeze_diagnostic_inputs(args.capture_root, args.protocol_file))
    elif args.command == "collect":
        collect(args)
    else:
        write_new(args.out, compare(load_outputs(args.reference), load_outputs(args.candidate)))


if __name__ == "__main__":
    main()
