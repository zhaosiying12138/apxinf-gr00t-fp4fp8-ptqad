"""Label independently collated observations with the complete frozen teacher.

Run from GR00T. --input-dir consumes capture_onpolicy.py outputs. An explicit
--endpoint-bundle preserves the observation source while using common frozen
QAD action endpoints and per-pair replay seeds. With neither option this is a
demo/offline probe. FP32 parameters and BF16 autocast match training defaults.
"""
import argparse
import json
from itertools import zip_longest
import os
from pathlib import Path
import sys
import time

import torch
from probe_distill import CACHE_VERSION, file_sha256, flow_signature, prediction, replay_context, require_full_model, tensor_tree
from gr00t_runtime import configure_libero_data, verify_libero_statistics, libero_action_spec, action_mask
from runtime_metrics import cuda_memory_peaks

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_COUNT = 8
DEFAULT_SEED = 20260929


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default=os.environ.get("OPD_TEACHER", str(PROJECT / "weights/GR00T-N1.7-LIBERO/libero_10")))
    ap.add_argument("--out", default=os.environ.get("OPD_CACHE_PATH", str(Path(os.environ.get("OPD_CACHE_OUT", "/mnt/c/fq_opd_probes_v2")) / "teacher_probes.pt")))
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--input-dir", help="Captured student observations, one sample_*.pt per observation")
    source.add_argument("--endpoint-bundle", help="Complete paired observation/QAD-endpoint bundle")
    ap.add_argument("--endpoint-role", choices=("teacher", "student"), help="Observation source role inside the paired bundle")
    ap.add_argument("--dataset", default=os.environ.get("QAD_DATASET", "./demo_data/libero_demo"))
    ap.add_argument("--count", type=int, help="Legacy input/demo sample count (default: 8); forbidden for endpoint bundles")
    ap.add_argument("--seed", type=int, help="Legacy input/demo seed (default: 20260929); forbidden for endpoint bundles")
    ap.add_argument("--model-dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--autocast-dtype", choices=("bfloat16", "none"), default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if bool(args.endpoint_bundle) != bool(args.endpoint_role):
        ap.error("--endpoint-bundle and --endpoint-role must be supplied together")
    if args.endpoint_bundle:
        if args.count is not None or args.seed is not None:
            ap.error("Endpoint bundles freeze all pairs and velocity seeds; --count/--seed cannot override them")
    else:
        args.count = DEFAULT_COUNT if args.count is None else args.count
        args.seed = DEFAULT_SEED if args.seed is None else args.seed
        if args.count < 1:
            ap.error("--count must be positive")
    return args


def prepare_observation_source(teacher, *, input_dir=None, endpoint_bundle=None,
                               endpoint_role=None, count=None, seed=None):
    """Load and audit a captured source without loading a model or touching CUDA."""
    teacher = Path(teacher).resolve()
    if bool(input_dir) == bool(endpoint_bundle):
        raise ValueError("Select exactly one input directory or endpoint bundle")
    if endpoint_bundle:
        if endpoint_role not in ("teacher", "student") or count is not None or seed is not None:
            raise ValueError("Endpoint bundles require a role and forbid count/seed overrides")
        sys.path.insert(0, str(PROJECT))
        from exp.probe_endpoint_bundle import load_endpoint_bundle
        raw_samples, provenance = load_endpoint_bundle(
            endpoint_bundle, endpoint_role, teacher_checkpoint=teacher)
        validate_endpoint_budget(raw_samples, provenance)
        return raw_samples, {
            "source_kind": provenance["source_kind"],
            "source_observation_files": provenance["source_observation_files"],
            "source_endpoint_files": provenance["source_endpoint_files"],
            "endpoint_bundle": provenance,
            "seed": None, "seed_policy": "frozen per-pair endpoint.velocity_seed",
            "count": len(raw_samples), "requested_count": provenance["pairs_per_arm"],
        }
    if endpoint_role is not None:
        raise ValueError("Endpoint role requires an endpoint bundle")
    count = DEFAULT_COUNT if count is None else count
    seed = DEFAULT_SEED if seed is None else seed
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("Legacy count must be positive and seed must be an integer")
    groups = {}
    for path in sorted(Path(input_dir).rglob("sample_*.pt")):
        groups.setdefault(str(path.parent), []).append(path)
    # Round-robin folders before truncation retains the original smoke behavior.
    paths = [p for row in zip_longest(*groups.values()) for p in row if p is not None][:count]
    if not paths:
        raise ValueError("No captured observations found")
    source_files = [{"path": str(p.resolve()), "bytes": p.stat().st_size,
                     "sha256": file_sha256(p)} for p in paths]
    raw_samples = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    if any(s.get("source_kind") != "student_rollout" for s in raw_samples):
        raise ValueError("Input is not a captured student rollout")
    student_values = [s.get("student_checkpoint") for s in raw_samples]
    if any(not isinstance(value, str) or not value for value in student_values):
        raise ValueError("Captured observations lack student checkpoint identity")
    student_paths = {str(Path(value).resolve()) for value in student_values}
    if len(student_paths) != 1:
        raise ValueError("Captured observations do not share one student checkpoint")
    student = Path(next(iter(student_paths)))
    student_weights = {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                       for p in sorted(student.glob("*.safetensors"))}
    if not student_weights:
        raise ValueError("Student checkpoint has no safetensors weights")
    stats_hash = file_sha256(teacher / "statistics.json")
    if any(s["student_statistics_sha256"] != stats_hash for s in raw_samples):
        raise ValueError("Student/teacher normalization differs; cannot reuse normalized rollout inputs")
    return raw_samples, {
        "source_kind": "student_rollout", "source_observation_files": source_files,
        "seed": seed, "count": len(raw_samples), "requested_count": count,
        "student_checkpoint": str(student), "student_weights": student_weights,
        "student_config_sha256": file_sha256(student / "config.json"),
        "student_statistics_sha256": file_sha256(student / "statistics.json"),
    }


def validate_endpoint_budget(raw_samples, provenance):
    """Reject clipping/reordering between a verified bundle and teacher labeling."""
    from collections import Counter
    role = provenance.get("role")
    source_kind = {"teacher": "teacher_rollout", "student": "student_rollout"}.get(role)
    if (not source_kind or provenance.get("source_kind") != source_kind
            or type(provenance.get("pairs_per_arm")) is not int
            or not raw_samples or len(raw_samples) != provenance["pairs_per_arm"]):
        raise ValueError("Endpoint bundle source role or full paired budget differs")
    for index, raw in enumerate(raw_samples):
        endpoint = raw.get("endpoint", {})
        if raw.get("source_kind") != source_kind or endpoint.get("pair_index") != index:
            raise ValueError("Endpoint observation source or pair order differs")
        for key in ("endpoint_seed", "velocity_seed"):
            if type(endpoint.get(key)) is not int or not 0 <= endpoint[key] < 2**63:
                raise ValueError("Endpoint pair lacks a valid frozen " + key)
    observed = Counter(raw.get("task_name") for raw in raw_samples)
    budgets = provenance.get("task_budgets")
    if not isinstance(budgets, dict) or set(observed) != set(budgets) or any(
            type(budgets[task].get("pairs")) is not int or budgets[task]["pairs"] != count
            for task, count in observed.items()):
        raise ValueError("Endpoint task budgets differ from all planned pairs")
    for field in ("source_observation_files", "source_endpoint_files"):
        if not isinstance(provenance.get(field), list) or len(provenance[field]) != len(raw_samples):
            raise ValueError("Endpoint source file budget differs: " + field)


def label_observations(model, raw_samples, *, action_spec, device="cpu",
                       autocast_dtype="bfloat16", seed=None, endpoint_bundle=None,
                       progress=None):
    """Replay full-teacher velocity targets; independent of GR00T loading/I/O."""
    require_full_model(model)
    if endpoint_bundle is not None:
        if seed is not None:
            raise ValueError("Endpoint velocity seeds cannot be overridden")
        validate_endpoint_budget(raw_samples, endpoint_bundle)
    else:
        seed = DEFAULT_SEED if seed is None else seed
    samples = []
    for index, raw in enumerate(raw_samples):
        replay_seed = raw["endpoint"]["velocity_seed"] if endpoint_bundle is not None else seed + index
        inputs = tensor_tree(raw["inputs"])
        valid = action_mask(inputs["action"], action_spec)
        if "action_mask" in inputs:
            supplied = inputs["action_mask"]
            if (supplied.shape != valid.shape or not torch.isfinite(supplied).all()
                    or not ((supplied == 0) | (supplied == 1)).all()):
                raise ValueError("Probe action mask must be finite, binary, and match the action shape")
            if endpoint_bundle is not None:
                if supplied.dtype != torch.float32 or not torch.equal(supplied, valid):
                    raise ValueError("Endpoint action mask differs from the frozen LIBERO action extent")
            else:
                valid *= supplied.float()
        elif endpoint_bundle is not None:
            raise ValueError("Endpoint inputs lack their frozen action mask")
        if valid.sum() <= 0:
            raise ValueError("Probe has no valid action elements")
        if endpoint_bundle is None:
            inputs["action_mask"] = valid
        if inputs["action"].shape[0] != 1:
            raise ValueError("Each input must be independently collated as a microbatch of one")
        if not torch.isfinite(inputs["action"]).all():
            raise ValueError("Nonfinite action endpoint; refusing cache")
        with replay_context(model, replay_seed, autocast_dtype), torch.no_grad():
            pred = prediction(model(tensor_tree(inputs, device))).float().cpu()
        if pred.shape != inputs["action"].shape or not torch.isfinite(pred).all():
            raise ValueError(f"Invalid teacher velocity at probe {index}; refusing cache")
        samples.append({"inputs": inputs, "pred": pred, "seed": replay_seed,
                        "provenance": {k: v for k, v in raw.items() if k != "inputs"}})
        if progress is not None:
            progress(index, len(raw_samples), pred)
    return samples


def use_local_hf_metadata() -> None:
    """Keep teacher labeling offline when the checkpoint is fully local.

    Transformers may query ``huggingface_hub.model_info`` while constructing
    the Qwen tokenizer even with ``local_files_only=True``.  The query only
    detects a Mistral-specific regex; GR00T's cached Qwen tokenizer does not
    need it, so an empty tag response preserves the local tokenizer bytes.
    """
    if os.environ.get("PTQAD_LOCAL_HF_METADATA", "1") != "1":
        return
    import huggingface_hub
    from types import SimpleNamespace
    huggingface_hub.model_info = lambda *args, **kwargs: SimpleNamespace(tags=[])


def main(argv=None):
    args = parse_args(argv)
    outpath = Path(args.out)
    if outpath.exists():
        raise FileExistsError(f"Refusing to overwrite teacher evidence: {outpath}")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    use_local_hf_metadata()
    sys.path.insert(0, os.getcwd())
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model import MODEL_REGISTRY
    from transformers import AutoModel

    start = time.time()
    teacher = Path(args.teacher).resolve()
    teacher_weights = {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                       for p in sorted(teacher.glob("*.safetensors"))}
    if not teacher_weights:
        raise ValueError("Teacher checkpoint has no safetensors weights")
    # model_path is NOT a pipeline load argument; start_from_checkpoint is.
    config = get_default_config().load_dict({"data": {
        "download_cache": False, "override_pretraining_statistics": False,
        "datasets": [{"dataset_paths": [args.dataset], "mix_ratio": 1.0,
                      "embodiment_tag": "libero_sim"}]}})
    config.model = Gr00tN1d7Config.from_pretrained(teacher, local_files_only=True)
    config.model.model_name = os.environ.get("GR00T_BACKBONE_MODEL", config.model.model_name)
    config.model.use_relative_action = True
    config.training.start_from_checkpoint = str(teacher)
    config.training.use_fsdp2 = False
    config.training.use_wandb = False
    config.training.transformers_local_files_only = True
    config.load_config_path = None
    configure_libero_data(config, teacher)
    action_spec = libero_action_spec(teacher)

    if args.input_dir or args.endpoint_bundle:
        raw_samples, source_metadata = prepare_observation_source(
            teacher, input_dir=args.input_dir, endpoint_bundle=args.endpoint_bundle,
            endpoint_role=args.endpoint_role, count=args.count, seed=args.seed)
        model, info = AutoModel.from_pretrained(
            teacher, config=config.model, local_files_only=True,
            transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
            output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Teacher weight loading was not exact: {info}")
    else:
        pipeline = MODEL_REGISTRY.get(type(config.model))(config, outpath.parent)
        pipeline.setup()
        verify_libero_statistics(pipeline.processor, teacher)
        model = pipeline.return_model()
        dataset, _ = pipeline.return_dataset()
        collator = pipeline.return_collator()
        iterator = iter(dataset)
        raw_samples = [{"inputs": tensor_tree(collator([next(iterator)])["inputs"]),
                        "source_kind": "demo", "sample_index": i}
                       for i in range(args.count)]
        source_metadata = {"source_kind": "demo", "source_observation_files": [],
                           "seed": args.seed, "count": len(raw_samples), "requested_count": args.count}

    arch = require_full_model(model)
    model = model.to(device=args.device, dtype=getattr(torch, args.model_dtype)).eval()
    model.requires_grad_(False)
    samples = label_observations(
        model, raw_samples, action_spec=action_spec, device=args.device,
        autocast_dtype=args.autocast_dtype, seed=args.seed,
        endpoint_bundle=source_metadata.get("endpoint_bundle"),
        progress=lambda i, n, pred: print(f"[probe-cache] {i + 1}/{n} pred={tuple(pred.shape)}", flush=True))
    metadata = {"teacher": str(teacher), "teacher_config_sha256": file_sha256(teacher / "config.json"),
                "teacher_statistics_sha256": file_sha256(teacher / "statistics.json"),
                "teacher_weights": teacher_weights,
                "labeling_implementation_sha256": file_sha256(__file__),
                "replay_implementation_sha256": file_sha256(Path(__file__).with_name("probe_distill.py")),
                "architecture": arch, "model_dtype": args.model_dtype,
                "flow_config": flow_signature(model),
                "autocast_dtype": args.autocast_dtype, "eval_mode": True,
                **source_metadata,
                "objective": "masked velocity MSE at shared full interpolated action/noise/time",
                "action_mask": action_spec,
                "cuda_peak_memory": cuda_memory_peaks(),
                "timing_scope": "checkpoint loading and teacher labeling, before cache serialization",
                "cuda_peak_scope": "PyTorch allocator peaks for this labeling process",
                "elapsed_seconds": time.time() - start}
    torch.save({"version": CACHE_VERSION, "metadata": metadata, "samples": samples}, outpath)
    outpath.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[probe-cache] saved {outpath} source={source_metadata['source_kind']} architecture={arch}", flush=True)


if __name__ == "__main__":
    main()
