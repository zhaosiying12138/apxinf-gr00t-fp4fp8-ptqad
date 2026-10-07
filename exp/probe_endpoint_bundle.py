"""Generate and audit paired QAD action endpoints without changing observation origins.

Generation uses the formal GR00T loader and complete get_action integration.
Loading/auditing is CPU only and never substitutes cached endpoints for inference.
"""
import argparse
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp.action_chunk_diagnostics import infer_chunk
from exp.probe_endpoint_plan import SCHEMA as PLAN_SCHEMA, canonical_sha, freeze_endpoint_plan
from rl.capture_sampling import _require_same_sample
from rl.gr00t_runtime import action_mask, libero_action_spec
from rl.probe_distill import file_sha256, require_full_model, tensor_tree

SCHEMA = "fp4vla_probe_endpoint_bundle_v1"
TRACE_SCHEMA = "fp4vla_probe_endpoint_trace_v1"
ROLES = ("teacher", "student")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verified_bytes(path, record):
    content = Path(path).read_bytes()
    require(hashlib.sha256(content).hexdigest() == record["sha256"]
            and ("bytes" not in record or len(content) == record["bytes"]),
            "Evidence bytes changed: " + str(path))
    return content


def load_tensor_file(path, record):
    import torch
    return torch.load(io.BytesIO(verified_bytes(path, record)), map_location="cpu", weights_only=True)


def validated_plan(path, record=None):
    path = Path(path).resolve()
    content = path.read_bytes() if record is None else verified_bytes(path, record)
    plan = json.loads(content)
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("status") == "frozen_cpu_plan",
            "Expected a complete frozen endpoint plan")
    fresh = freeze_endpoint_plan(plan["sources"]["teacher"]["view"], plan["sources"]["student"]["view"],
        plan["protocol_file"], plan["qad"]["checkpoint"], teacher_checkpoint=plan["teacher_checkpoint"])
    require(plan == fresh, "Endpoint plan no longer matches its audited sources and QAD")
    require(path.read_bytes() == content, "Endpoint plan changed during audit")
    return plan, content


def implementation_files():
    names = ("exp/probe_endpoint_bundle.py", "exp/probe_endpoint_plan.py", "exp/action_chunk_diagnostics.py",
             "exp/derive_capture_views.py", "rl/capture_sampling.py", "rl/checkpoint_identity.py",
             "rl/probe_distill.py", "rl/gr00t_runtime.py", "eval/serve_recovery.py",
             "eval/run_gr00t_server_fp4vla.py", "rl/scoped_quant.py", "rl/w4a4_deploy.py",
             "rl/w4a4_lora.py", "quant/native_activation.py")
    return {str(ROOT / name): file_sha256(ROOT / name) for name in names}


def _run_qad_policy(plan, gr00t, callback):
    """Use the formal loader with a local callback instead of opening RPC."""
    checkpoint = Path(plan["qad"]["checkpoint"])
    gr00t = Path(gr00t).expanduser().resolve()
    require(gr00t.is_dir(), "GR00T repository does not exist")
    processor_config = read_json(checkpoint / "processor_config.json")
    actual_base = Path(plan["qad"]["base"])
    expected_stats = read_json(actual_base / "statistics.json")["libero_sim"]
    mapping = read_json(actual_base / "embodiment_id.json")
    expected_count = read_json(plan["protocol_file"])["quantization_scope"]["activation_linear_count"]
    original_env, original_cwd, original_path, original_argv = dict(os.environ), Path.cwd(), list(sys.path), sys.argv
    for key in list(os.environ):
        if key.startswith(("OPD_CAPTURE_", "FP4VLA_CAPTURE_")) or key == "FP4VLA_LOG_DIR":
            os.environ.pop(key)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "FP4VLA_QUANT": "0",
        "FP4VLA_SCOPE": "all", "FP4VLA_W4A4": "1", "FP4VLA_W4A4_ADAPTER": "1",
        "FP4VLA_SATURATE_F16_ACTIVATIONS": "1",
        "GR00T_EVAL_SEED": str(plan["endpoint_distillation"]["endpoint_seed"])})
    sys.path.insert(0, str(gr00t))
    sys.path.insert(0, str(ROOT / "eval"))
    os.chdir(gr00t)
    server_client, original_server = None, None
    runtime, executions = {}, []
    try:
        from gr00t.policy import server_client
        from gr00t.data.utils import parse_modality_configs
        expected_modalities = parse_modality_configs({"libero_sim": processor_config[
            "processor_kwargs"]["modality_configs"]["libero_sim"]})["libero_sim"]

        class LocalEndpointRunner:
            def __init__(self, *, policy, **kwargs):
                self.policy = policy
                architecture = require_full_model(policy.model)
                require(policy.processor.modality_configs["libero_sim"] == expected_modalities,
                        "Loaded QAD processor modalities differ")
                require(policy.processor.state_action_processor.statistics.get("libero_sim") == expected_stats,
                        "Loaded QAD normalization differs")
                require(all(policy.processor.embodiment_id_mapping.get(key) == value for key, value in mapping.items()),
                        "Loaded QAD embodiment mapping differs")
                count = sum(bool(getattr(module, "_fp4vla_w4a4", False)) for module in policy.model.modules())
                require(count == expected_count, "Loaded QAD W4A4 activation coverage differs")
                parameter = next(policy.model.parameters())
                sources = {str(Path(inspect.getfile(type(item))).resolve()) for item in (
                    policy, policy.model, policy.processor, policy.processor.state_action_processor,
                    policy.model.action_head, policy.model.action_head.action_encoder)}
                runtime.update({"loader": "serve_recovery_local_callback", "architecture": architecture,
                    "w4a4_activation_count": count, "model_parameter_dtype": str(parameter.dtype),
                    "device": str(parameter.device), "rpc_opened": False,
                    "source_files": {path: file_sha256(path) for path in sorted(sources)}})

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run(self):
                require(not executions, "Formal loader executed endpoint generation more than once")
                executions.append(True)
                callback(self.policy.model)

        original_server = server_client.PolicyServer
        server_client.PolicyServer = LocalEndpointRunner
        sys.argv = [str(ROOT / "eval/serve_recovery.py"), "--model-path", str(checkpoint),
                    "--embodiment-tag", "LIBERO_PANDA"]
        runpy.run_path(sys.argv[0], run_name="__main__")
        require(len(executions) == 1 and runtime, "Formal loader did not execute the endpoint callback")
        return runtime
    finally:
        if server_client is not None and original_server is not None:
            server_client.PolicyServer = original_server
        sys.argv, sys.path[:] = original_argv, original_path
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)


def _spec(plan):
    spec = libero_action_spec(plan["qad"]["checkpoint"])
    require(spec["horizon"] == 16 and spec["dimensions"] == 7,
            "Endpoint bundle requires the LIBERO 16 by 7 valid-action contract")
    require(spec == libero_action_spec(plan["teacher_checkpoint"]), "Teacher and QAD action contracts differ")
    return spec


def _endpoint_metadata(plan, pair, role, plan_sha, trace_record):
    return {"pair_index": pair["pair_index"], "endpoint_seed": pair["endpoint_seed"],
            "velocity_seed": pair["velocity_seed"], "role": role, "plan_sha256": plan_sha,
            "policy_checkpoint": plan["qad"]["checkpoint"],
            "policy_files_sha256": canonical_sha(plan["qad"]["checkpoint_files"]),
            "source_observation": {"path": pair[role]["path"], "sha256": pair[role]["sha256"]},
            "trace": trace_record}


def _same_noise(left, right):
    import torch
    require(torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
            and left.shape == right.shape and torch.equal(left, right),
            "Paired endpoint inference did not use identical actual initial noise")


def _validate_trace(trace, pair, spec):
    import torch
    require(trace.get("schema") == TRACE_SCHEMA and trace.get("pair_index") == pair["pair_index"]
            and trace.get("endpoint_seed") == pair["endpoint_seed"], "Endpoint trace identity differs")
    require(set(trace) == {"schema", "pair_index", "endpoint_seed", "teacher", "student"},
            "Endpoint trace fields differ")
    for role in ROLES:
        result = trace[role]
        require(set(result) == {"action_pred", "initial_noise", "integration_steps"}, "Incomplete endpoint trace")
        action, noise = result["action_pred"], result["initial_noise"]
        require(torch.is_tensor(action) and torch.is_tensor(noise)
                and action.device.type == noise.device.type == "cpu" and action.dtype == torch.float32
                and list(action.shape) == pair[role]["action_contract"]["shape"] == list(noise.shape)
                and bool(torch.isfinite(action).all()) and bool(torch.isfinite(noise).all()),
                "Endpoint trace has invalid action/noise values or shapes")
        require(type(result["integration_steps"]) is int and result["integration_steps"] > 0,
                "Endpoint trace lacks completed integration")
        action_mask(action, spec)
    require(trace["teacher"]["integration_steps"] == trace["student"]["integration_steps"],
            "Endpoint integration schedules differ between roles")
    _same_noise(trace["teacher"]["initial_noise"], trace["student"]["initial_noise"])


def _check_original(raw, source):
    require("endpoint" not in raw, "Observation already contains relabeled endpoint metadata")
    _require_same_sample({key: value for key, value in raw.items() if key != "inputs"},
                         source["provenance"], "source.provenance")


def _output_root(out, plan, plan_file):
    raw_out = Path(out).expanduser()
    require(not raw_out.is_symlink(), "Endpoint output must not be a symlink")
    root = raw_out.resolve()
    if root.exists():
        raise FileExistsError("Refusing existing endpoint bundle: " + str(root))
    protected = [Path(plan["teacher_checkpoint"]), Path(plan["qad"]["checkpoint"]), Path(plan["qad"]["base"]),
                 Path(plan["qad"]["training_checkpoint"]).parent]
    protected.extend(Path(source["source_directory"]) for source in plan["sources"].values())
    protected.extend(Path(source["view"]).parent for source in plan["sources"].values())
    require(all(root != path and path not in root.parents and root not in path.parents for path in protected)
            and root not in Path(plan_file).parents and root not in Path(plan["protocol_file"]).parents,
            "Endpoint bundle must be outside source evidence and checkpoints")
    return root


def generate_endpoint_bundle(plan_file, out, *, gr00t):
    """Generate both endpoint roles with one loaded, frozen QAD policy."""
    import torch
    source_plan = identity(plan_file)
    plan, plan_bytes = validated_plan(plan_file, source_plan)
    root = _output_root(out, plan, Path(plan_file).resolve())
    spec, sources = _spec(plan), implementation_files()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    root.mkdir(parents=True)
    for role in (*ROLES, "traces"):
        (root / role).mkdir()
    (root / "plan.json").write_bytes(plan_bytes)
    write_json(root / "incomplete.json", {"schema": SCHEMA, "status": "incomplete", "plan_sha256": plan_sha})
    records = []

    def collect(model):
        require(not records, "Endpoint callback must execute exactly once")
        for pair in plan["pairs"]:
            require(pair["pair_index"] == len(records), "Endpoint plan pair ordering differs")
            originals, results = {}, {}
            for role in ROLES:
                source = pair[role]
                original = load_tensor_file(source["path"], source)
                _check_original(original, source)
                expected = action_mask(original["inputs"]["action"], spec)
                _require_same_sample(original["inputs"]["action_mask"], expected, role + ".original_mask")
                originals[role] = original
                results[role] = infer_chunk(model, original["inputs"], pair["endpoint_seed"], spec)
            trace = {"schema": TRACE_SCHEMA, "pair_index": pair["pair_index"],
                     "endpoint_seed": pair["endpoint_seed"], **results}
            _validate_trace(trace, pair, spec)
            filename = f"pair_{pair['pair_index']:06d}.pt"
            torch.save(tensor_tree(trace), root / "traces" / filename)
            trace_id = {"path": "traces/" + filename,
                        **{key: value for key, value in identity(root / "traces" / filename).items() if key != "path"}}
            record = {"pair_index": pair["pair_index"], "trace": trace_id}
            for role in ROLES:
                raw = tensor_tree(originals[role])
                raw["inputs"]["action"] = tensor_tree(results[role]["action_pred"])
                raw["inputs"]["action_mask"] = action_mask(raw["inputs"]["action"], spec)
                raw["endpoint"] = _endpoint_metadata(plan, pair, role, plan_sha, trace_id)
                sample_path = root / role / f"sample_{pair['pair_index']:06d}.pt"
                torch.save(raw, sample_path)
                record[role] = {"path": sample_path.relative_to(root).as_posix(),
                    **{key: value for key, value in identity(sample_path).items() if key != "path"}}
            records.append(record)
            print(f"[endpoint-bundle] paired endpoints {len(records)}/{plan['pairs_per_arm']}", flush=True)

    runtime = _run_qad_policy(plan, gr00t, collect)
    require(len(records) == plan["pairs_per_arm"] > 0, "Endpoint generation was incomplete")
    fresh, _ = validated_plan(plan_file, source_plan)
    require(fresh == plan and implementation_files() == sources, "Endpoint generation sources changed")
    for path, digest in runtime["source_files"].items():
        require(file_sha256(path) == digest, "Loaded model implementation changed during endpoint generation")
    manifest = {"schema": SCHEMA, "status": "complete", "plan_sha256": plan_sha, "source_plan": source_plan,
        "protocol_sha256": plan["protocol_sha256"], "endpoint_policy_checkpoint": plan["qad"]["checkpoint"],
        "endpoint_policy_files": plan["qad"]["checkpoint_files"], "pairs_per_arm": len(records),
        "task_budgets": plan["task_budgets"], "action_spec": spec,
        "implementation_files": sources, "runtime": runtime, "records": records,
        "scope": "frozen QAD action endpoints on audited teacher/student observations; no new rollouts or velocity labels"}
    (root / "incomplete.json").unlink()
    write_json(root / "manifest.json", manifest)
    return manifest


def load_endpoint_bundle(root, role, *, teacher_checkpoint):
    """Reaudit both roles on CPU, then return the requested role in pair order."""
    require(role in ROLES, "Endpoint role must be teacher or student")
    require(not Path(root).is_symlink(), "Endpoint bundle must not be a symlink")
    root = Path(root).expanduser().resolve()
    manifest_identity = identity(root / "manifest.json")
    manifest_content = verified_bytes(root / "manifest.json", manifest_identity)
    manifest = json.loads(manifest_content)
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete",
            "Endpoint bundle is incomplete or has an unknown schema")
    require(manifest.get("implementation_files") == implementation_files(), "Endpoint implementation identity changed")
    plan, content = validated_plan(root / "plan.json", {"sha256": manifest["plan_sha256"]})
    require(Path(teacher_checkpoint).expanduser().resolve() == Path(plan["teacher_checkpoint"]),
            "Endpoint bundle BF16 teacher differs")
    source_plan = manifest["source_plan"]
    require(verified_bytes(source_plan["path"], source_plan) == content, "Original and copied endpoint plan differ")
    spec = _spec(plan)
    require(manifest.get("action_spec") == spec and manifest.get("protocol_sha256") == plan["protocol_sha256"]
            and manifest.get("endpoint_policy_checkpoint") == plan["qad"]["checkpoint"]
            and manifest.get("endpoint_policy_files") == plan["qad"]["checkpoint_files"]
            and manifest.get("pairs_per_arm") == plan["pairs_per_arm"]
            and manifest.get("task_budgets") == plan["task_budgets"], "Endpoint bundle contract differs from frozen plan")
    runtime = manifest.get("runtime", {})
    require(runtime.get("loader") == "serve_recovery_local_callback" and runtime.get("rpc_opened") is False
            and isinstance(runtime.get("source_files"), dict) and bool(runtime["source_files"]),
            "Endpoint bundle lacks the formal local loader identity")
    for path, digest in runtime["source_files"].items():
        require(file_sha256(path) == digest, "Endpoint runtime implementation changed")
    records = manifest.get("records")
    require(isinstance(records, list) and len(records) == len(plan["pairs"]), "Endpoint bundle pair count differs")
    expected_files = {"manifest.json", "plan.json"}
    for index in range(len(records)):
        expected_files.add(f"traces/pair_{index:06d}.pt")
        expected_files.update(f"{item}/sample_{index:06d}.pt" for item in ROLES)
    actual = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in actual), "Endpoint bundle contains symlinks")
    require({path.relative_to(root).as_posix() for path in actual if path.is_file()} == expected_files
            and {path.relative_to(root).as_posix() for path in actual if path.is_dir()} == {*ROLES, "traces"},
            "Endpoint bundle is incomplete or contains extra files")
    raw_samples, observation_files, endpoint_files = [], [], []
    for pair, record in zip(plan["pairs"], records):
        index = pair["pair_index"]
        require(record.get("pair_index") == index and set(record) == {"pair_index", "trace", *ROLES},
                "Endpoint record ordering or fields differ")
        require(record["trace"]["path"] == f"traces/pair_{index:06d}.pt", "Endpoint trace path differs")
        trace = load_tensor_file(root / record["trace"]["path"], record["trace"])
        _validate_trace(trace, pair, spec)
        for current_role in ROLES:
            sample_record = record[current_role]
            require(sample_record["path"] == f"{current_role}/sample_{index:06d}.pt", "Endpoint sample path differs")
            path = root / sample_record["path"]
            raw = load_tensor_file(path, sample_record)
            source = pair[current_role]
            original = load_tensor_file(source["path"], source)
            _check_original(original, source)
            expected_provenance = {**source["provenance"], "endpoint": _endpoint_metadata(
                plan, pair, current_role, manifest["plan_sha256"], record["trace"])}
            _require_same_sample({key: value for key, value in raw.items() if key != "inputs"},
                                 expected_provenance, current_role + ".provenance")
            expected_inputs = {**original["inputs"], "action": trace[current_role]["action_pred"]}
            expected_mask = action_mask(expected_inputs["action"], spec)
            _require_same_sample(original["inputs"]["action_mask"], expected_mask, current_role + ".original_mask")
            expected_inputs["action_mask"] = expected_mask
            _require_same_sample(raw["inputs"], expected_inputs, current_role + ".inputs")
            if current_role == role:
                raw_samples.append(raw)
                observation_files.append(identity(source["path"]))
                endpoint_files.append({"path": str(path), "bytes": sample_record["bytes"], "sha256": sample_record["sha256"]})
    provenance = {"schema": SCHEMA, "root": str(root), "role": role,
        "plan_sha256": manifest["plan_sha256"], "protocol_sha256": plan["protocol_sha256"],
        "protocol_file": plan["protocol_file"], "teacher_checkpoint": plan["teacher_checkpoint"],
        "endpoint_policy_checkpoint": plan["qad"]["checkpoint"], "endpoint_policy_files": plan["qad"]["checkpoint_files"],
        "endpoint_policy_training_checkpoint": plan["qad"]["training_checkpoint"],
        "endpoint_policy_training_identity": plan["qad"]["training_identity"],
        "source_kind": plan["sources"][role]["source_kind"], "pairs_per_arm": plan["pairs_per_arm"],
        "task_budgets": plan["task_budgets"], "action_spec": spec,
        "source_observation_files": observation_files, "source_endpoint_files": endpoint_files,
        "bundle_manifest": manifest_identity, "implementation_files": manifest["implementation_files"]}
    require(verified_bytes(root / "manifest.json", manifest_identity) == manifest_content,
            "Endpoint bundle manifest changed during audit")
    return raw_samples, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gr00t", required=True)
    args = parser.parse_args()
    result = generate_endpoint_bundle(args.plan, args.out, gr00t=args.gr00t)
    print(json.dumps({"status": result["status"], "pairs_per_arm": result["pairs_per_arm"],
                      "plan_sha256": result["plan_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
