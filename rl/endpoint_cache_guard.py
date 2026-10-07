"""Bind matched endpoint caches to the intended continuation before training.

Legacy protocols keep their existing cache path. The explicit matched-state
protocol requires a complete endpoint bundle, its frozen QAD starting adapter,
and unchanged demonstration data for either teacher-state KD or student OPD.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal_tensors(left, right):
    """Compare tensor bytes, including dtype; a changed padded endpoint matters."""
    import torch
    return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
            and left.shape == right.shape and torch.equal(
                left.detach().cpu().contiguous().view(torch.uint8),
                right.detach().cpu().contiguous().view(torch.uint8)))


def state_distillation_config(protocol):
    config = protocol.get("state_distillation")
    require(isinstance(config, dict) and set(config) == {
        "arms", "starting_qad_view", "teacher_velocity_weight", "probe_every",
        "optimizer_steps", "learning_rate", "max_grad_norm"},
        "Protocol must freeze the matched state-distillation training settings")
    require(config["arms"] == ["continued_qad", "teacher_state_kd", "student_state_opd"]
            and config["starting_qad_view"] == "stratified", "Unexpected matched recovery arms or QAD view")
    for key in ("teacher_velocity_weight", "learning_rate", "max_grad_norm"):
        require(type(config[key]) in (int, float) and math.isfinite(config[key]) and config[key] > 0,
                "Invalid matched recovery setting: " + key)
    for key in ("probe_every", "optimizer_steps"):
        require(type(config[key]) is int and config[key] > 0, "Invalid matched recovery budget: " + key)
    selection = protocol.get("selection", {})
    require(config["optimizer_steps"] == selection.get("continuation_optimizer_steps")
            and config["probe_every"] == selection.get("opd_every")
            and selection.get("qad_learning_rates") == [config["learning_rate"]],
            "Matched recovery budgets differ from the frozen selection settings")
    return config


def prepare_endpoint_training_cache(cache_path, protocol_file, env=None):
    """Audit a new matched cache and the actual training invocation on CPU.

Return None for legacy protocols. The full source audit is delegated to the
bundle reader, and every cached input is checked against its endpoint bytes.
The returned cache hash must also be checked when ProbeAnchor loads the file.
"""
    import torch
    env = os.environ if env is None else env
    protocol_file, cache_path = (Path(x).expanduser().resolve() for x in (protocol_file, cache_path))
    protocol = read(protocol_file)
    if "endpoint_distillation" not in protocol and "state_distillation" not in protocol:
        require(not env.get("QAD_ENDPOINT_ROLE"), "Endpoint role requires a matched-state protocol")
        return None
    config = state_distillation_config(protocol)
    role = env.get("QAD_ENDPOINT_ROLE")
    require(role in ("teacher", "student"), "Matched cache requires QAD_ENDPOINT_ROLE=teacher or student")
    content = cache_path.read_bytes()
    cache_sha = hashlib.sha256(content).hexdigest()
    cache = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    require(cache.get("version") == 3, "Matched training requires the masked velocity cache format")
    metadata = cache.get("metadata", {})
    reference = metadata.get("endpoint_bundle")
    require(isinstance(reference, dict) and reference.get("schema") == "fp4vla_probe_endpoint_bundle_v1"
            and reference.get("role") == role and isinstance(reference.get("root"), str),
            "Teacher cache lacks the intended matched endpoint bundle")
    teacher_value = env.get("QAD_CAPTURE_TEACHER")
    require(isinstance(teacher_value, str) and bool(teacher_value), "Matched training requires the original BF16 teacher")
    teacher = Path(teacher_value).expanduser().resolve()
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from exp.probe_endpoint_bundle import load_endpoint_bundle
    raw_samples, provenance = load_endpoint_bundle(reference["root"], role, teacher_checkpoint=teacher)
    require(reference == provenance, "Cache endpoint bundle provenance changed")
    require(provenance.get("protocol_sha256") == sha(protocol_file), "Endpoint cache and training protocols differ")
    kind = "teacher_rollout" if role == "teacher" else "student_rollout"
    require(metadata.get("source_kind") == provenance.get("source_kind") == kind,
            "Cache observation source kind differs from the requested arm")
    require(metadata.get("teacher") == str(teacher)
            and metadata.get("teacher_config_sha256") == sha(teacher / "config.json")
            and metadata.get("teacher_statistics_sha256") == sha(teacher / "statistics.json"),
            "Cache velocity teacher or normalization changed")
    teacher_weights = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                       for p in sorted(teacher.glob("*.safetensors"))}
    require(bool(teacher_weights) and metadata.get("teacher_weights") == teacher_weights,
            "Cache velocity teacher weights changed")
    require(metadata.get("model_dtype") == "float32" and metadata.get("autocast_dtype") == "bfloat16"
            and metadata.get("eval_mode") is True, "Matched velocity labels use a different numerical convention")
    for field, filename in (("labeling_implementation_sha256", "opd_probe_cache.py"),
                            ("replay_implementation_sha256", "probe_distill.py")):
        require(metadata.get(field) == sha(Path(__file__).with_name(filename)),
                "Teacher cache implementation changed: " + filename)
    samples = cache.get("samples")
    count = provenance.get("pairs_per_arm")
    require(type(count) is int and count > 0 and isinstance(samples, list)
            and len(samples) == len(raw_samples) == count
            and metadata.get("count") == metadata.get("requested_count") == count,
            "Matched teacher cache does not cover the complete frozen paired budget")
    for stored, raw in zip(samples, raw_samples):
        expected_provenance = {key: value for key, value in raw.items() if key != "inputs"}
        require(stored.get("provenance") == expected_provenance,
                "Teacher cache changed endpoint/observation provenance")
        require(stored.get("seed") == raw["endpoint"]["velocity_seed"], "Cache velocity seed changed")
        inputs = stored.get("inputs", {})
        require(isinstance(inputs, dict) and set(inputs) == set(raw["inputs"])
                and all(equal_tensors(inputs[key], value) for key, value in raw["inputs"].items()),
                "Teacher cache inputs differ from the frozen QAD endpoint")
        prediction = stored.get("pred")
        require(torch.is_tensor(prediction) and prediction.shape == inputs["action"].shape
                and prediction.dtype == torch.float32 and bool(torch.isfinite(prediction).all()),
                "Invalid teacher velocity prediction")
    initial = verify_continuation_invocation(provenance, protocol, config, env,
                                            config["teacher_velocity_weight"])
    require(sha(cache_path) == cache_sha, "Teacher cache changed during training audit")
    return {"schema": "fp4vla_endpoint_training_cache_audit_v1", "status": "verified",
            "cache_path": str(cache_path), "cache_sha256": cache_sha,
            "protocol_sha256": provenance["protocol_sha256"], "role": role, "source_kind": kind,
            "endpoint_bundle": provenance, "sample_count": count,
            "initial_adapter": initial, "state_distillation": config}


def verify_continuation_invocation(provenance, protocol, config, env, weight):
    """Common starting point, demonstration budget and numerics for all arms."""
    initial = env.get("QAD_INIT_ADAPTER")
    require(isinstance(initial, str) and bool(initial)
            and str(Path(initial).expanduser().resolve()) == provenance.get("endpoint_policy_training_checkpoint"),
            "Matched continuation must start from the endpoint policy's exact QAD adapter")
    identity = provenance.get("endpoint_policy_training_identity", {})
    data = env.get("QAD_CAPTURE_DATASET")
    require(isinstance(data, str) and bool(data)
            and str(Path(data).expanduser().resolve()) == identity.get("capture_dataset")
            and env.get("QAD_CAPTURE_DATASET_SHA256") == identity.get("capture_dataset_sha256")
            and env.get("QAD_CAPTURE_AUDIT_SHA256") == identity.get("capture_audit_sha256"),
            "Matched continuation must reuse the frozen QAD demonstration data")
    recovery = read(Path(initial).expanduser().resolve() / "recovery_manifest.json")
    base_value = env.get("GR00T_BASE_CKPT")
    require(isinstance(base_value, str) and bool(base_value)
            and str(Path(base_value).expanduser().resolve()) == recovery.get("base"),
            "Matched continuation changed the frozen quantized base")
    selection = protocol["selection"]
    numeric = {"QAD_STEPS": config["optimizer_steps"], "QAD_LR": config["learning_rate"],
               "QAD_OPD_MSE_W": weight, "OPD_EVERY": config["probe_every"],
               "QAD_MAX_GRAD_NORM": config["max_grad_norm"], "TRAIN_SEED": selection["train_seed"],
               "QAD_LORA_R": selection["rank"], "QAD_LORA_ALPHA": selection["alpha"],
               "QAD_GLOBAL_BATCH": selection["effective_demo_batch"], "QAD_MICRO_BATCH": 1}
    for key, value in numeric.items():
        try:
            matches = float(env[key]) == value
        except (KeyError, TypeError, ValueError):
            matches = False
        require(matches, "Matched recovery invocation differs from protocol: " + key)
    for key, value in {"QAD_LORA_SCOPE": selection["recovery_scope"], "QAD_W4A4": "1",
                       "FP4VLA_QUANT": "0", "FP4VLA_W4A4": "1", "FP4VLA_W4A4_ADAPTER": "1",
                       "FP4VLA_SATURATE_F16_ACTIVATIONS": "1", "QAD_ACTIVATION_CHECKPOINTING": "1"}.items():
        require(env.get(key) == value, "Matched recovery numerical mode differs: " + key)
    return str(Path(initial).expanduser().resolve())


def prepare_endpoint_training(protocol_file, weight, env=None):
    """Dispatch before the KD branch so zero/NaN cannot bypass arm validation."""
    env = os.environ if env is None else env
    protocol_file = Path(protocol_file).expanduser().resolve()
    protocol = read(protocol_file)
    role = env.get("QAD_ENDPOINT_ROLE")
    if "endpoint_distillation" not in protocol and "state_distillation" not in protocol:
        require(not role and not env.get("QAD_ENDPOINT_BUNDLE"), "Endpoint role requires a matched-state protocol")
        return None
    config = state_distillation_config(protocol)
    require(type(weight) in (int, float) and math.isfinite(weight) and weight >= 0,
            "Matched recovery weight must be finite and nonnegative")
    if not role:
        require(weight == 0 and not env.get("QAD_INIT_ADAPTER") and not env.get("OPD_CACHE_PATH")
                and not env.get("QAD_ENDPOINT_BUNDLE"),
                "Matched continuation requires an explicit role; only initial pure QAD may omit it")
        return None
    if role in ("teacher", "student"):
        require(weight == config["teacher_velocity_weight"] and env.get("OPD_CACHE_PATH"),
                "Matched KD role requires its positive protocol weight and teacher cache")
        return prepare_endpoint_training_cache(env["OPD_CACHE_PATH"], protocol_file, env)
    require(role == "continued", "Unknown matched recovery role")
    require(weight == 0 and not env.get("OPD_CACHE_PATH"), "Continued-QAD must not use teacher velocity labels")
    bundle, teacher_value = env.get("QAD_ENDPOINT_BUNDLE"), env.get("QAD_CAPTURE_TEACHER")
    require(isinstance(bundle, str) and bool(bundle) and isinstance(teacher_value, str) and bool(teacher_value),
            "Continued-QAD requires the common endpoint bundle and BF16 teacher identity")
    teacher = Path(teacher_value).expanduser().resolve()
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from exp.probe_endpoint_bundle import load_endpoint_bundle
    _, provenance = load_endpoint_bundle(bundle, "teacher", teacher_checkpoint=teacher)
    require(provenance.get("protocol_sha256") == sha(protocol_file), "Endpoint bundle and training protocols differ")
    initial = verify_continuation_invocation(provenance, protocol, config, env, 0.0)
    return {"schema": "fp4vla_endpoint_training_cache_audit_v1", "status": "verified",
            "protocol_sha256": provenance["protocol_sha256"], "role": role,
            "endpoint_bundle": provenance, "sample_count": 0, "cache_path": None,
            "initial_adapter": initial, "state_distillation": config}
