"""Auditable AutoPolicy.infer timing for local pi0.5 and GR00T checkpoints.

This executable needs exclusive GPU access. --help and unit tests are CPU-only.
Use a new --out JSON; --tag remains a results/engine/<tag>.json shorthand.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("bf16", "fp8_static", "int8_dynamic", "nvfp4_static", "fp8", "int8")


def publish_new(path, result):
    """Expose a complete JSON atomically, without replacing a previous run."""
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".bench-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic and fails if the destination exists
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def identity(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"path": str(path.absolute()), "resolved_path": str(path.resolve()),
            "bytes": after.st_size, "sha256": h.hexdigest()}


def directory_identity(folder):
    folder = Path(folder)
    files = {p.name: identity(p) for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".yaml", ".txt")}
    if not files:
        raise ValueError(f"No auditable model/asset files: {folder}")
    return {"path": str(folder.absolute()), "files": files}


def parse_extra(values):
    parsed = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep or not key or key in parsed:
            raise ValueError(f"Use unique key=value extras: {value}")
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            item = raw
        if isinstance(item, (dict, list)) or item is None:
            raise ValueError(f"Only scalar kwargs accepted: {key}")
        parsed[key] = item
    return parsed


def prepare_config(args):
    config = json.loads((args.model_dir / "config.json").read_text())
    kind = next((config[k] for k in ("type", "model_type", "model") if isinstance(config.get(k), str)), "")
    family = {"pi05": "pi05", "Gr00tN1d7": "gr00t", "gr00tn1d7": "gr00t", "gr00t": "gr00t"}.get(kind)
    if family is None:
        raise ValueError(f"This audited benchmark supports pi05/GR00T, got {kind!r}")
    kwargs = parse_extra(args.extra_kwarg)
    if any(key in kwargs for key in ("model_runner", "model_type", "metadata", "checkpoint")):
        raise ValueError("Model identity overrides are not accepted by the audited benchmark")
    for key, wanted in (("device", args.device),):
        if key in kwargs and kwargs[key] != wanted:
            raise ValueError(f"--extra-kwarg {key} conflicts with --{key}")
        kwargs[key] = wanted
    kwargs.setdefault("seed", args.model_seed)
    if family == "pi05":
        if args.variant not in ("bf16", "fp8_static", "int8_dynamic", "nvfp4_static"):
            raise ValueError("pi05 needs bf16/fp8_static/int8_dynamic/nvfp4_static")
        if "precision" in kwargs or ("model_variant" in kwargs and kwargs["model_variant"] != args.variant):
            raise ValueError("pi05 variant conflicts with extra precision/model_variant")
        kwargs["model_variant"] = args.variant
        kwargs.setdefault("norm_stats", str(args.model_dir / "norm_stats.json"))
        kwargs.setdefault("tokenizer_path", str(args.model_dir / "paligemma_tokenizer.model"))
        # Explicit local assets prevent environment-dependent tokenizer or identity-normalizer fallback.
        for key in ("norm_stats", "tokenizer_path"):
            if not Path(kwargs[key]).is_file():
                raise FileNotFoundError(f"Required pi05 asset {key}: {kwargs[key]}")
    else:
        if args.variant not in ("bf16", "fp8", "int8"):
            raise ValueError("GR00T needs explicit bf16/fp8/int8 precision")
        if "model_variant" in kwargs or ("precision" in kwargs and kwargs["precision"] != args.variant):
            raise ValueError("GR00T variant conflicts with extra precision/model_variant")
        kwargs["precision"] = args.variant
        kwargs.setdefault("backbone", os.environ.get("GR00T_BACKBONE_MODEL", ""))
        if not kwargs["backbone"] or not Path(kwargs["backbone"]).is_dir():
            raise ValueError("GR00T requires --extra-kwarg backbone=PATH or GR00T_BACKBONE_MODEL")
    for key in ("seed", "num_views", "num_flow_steps", "action_horizon", "action_dim"):
        if key in kwargs and (type(kwargs[key]) is not int or kwargs[key] < (0 if key == "seed" else 1)):
            raise ValueError(f"Invalid integer kwarg: {key}")
    return family, kwargs


def verify_variant(policy, family, requested):
    md = dict(policy.metadata)
    actual = getattr(policy.model_runner, "model_variant", None)
    if family == "pi05":
        if actual != requested or md.get("model_variant") != requested:
            raise ValueError(f"Native loaded variant {actual!r} differs from requested {requested!r}")
        return {"status": "native_model_variant_verified", "native_model_variant": actual,
                "policy_value": md["model_variant"], "limitation": None}
    if md.get("precision") != requested or (actual is not None and actual != requested):
        raise ValueError("GR00T precision metadata disagrees with explicit constructor selection")
    return {"status": "native_model_variant_verified" if actual is not None else "explicit_precision_metadata_only",
            "native_model_variant": actual, "policy_value": md["precision"],
            "limitation": None if actual is not None else "GR00T runtime does not expose its native model variant/dtype inventory; precision is an explicit constructor contract, not an independently inspected parameter dtype."}


def check_inference(result, expected_shape=None):
    if not isinstance(result, dict) or "actions" not in result:
        raise TypeError("Expected policy result with actions")
    action = np.asarray(result["actions"])
    if action.ndim != 2 or not action.size or not np.isfinite(action).all():
        raise ValueError("Actions must be a finite, nonempty horizon x action array")
    if expected_shape is not None and list(action.shape) != expected_shape:
        raise ValueError("Action shape changed during benchmark")
    if "normalized_actions" in result and not np.isfinite(np.asarray(result["normalized_actions"])).all():
        raise ValueError("Nonfinite normalized actions")
    timing = result.get("timing", {})
    for key in ("model_ms", "total_ms"):
        value = timing.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid timing {key}: {value!r}")
    if timing["model_ms"] > timing["total_ms"]:
        raise ValueError("Model time exceeds its containing total time")
    return list(action.shape)


def packed_identity(model_dir, checkpoint):
    folder = model_dir / "fp4"
    producer_file = folder / "producer_manifest.json"
    producer = json.loads(producer_file.read_text())
    if producer.get("status") != "complete":
        raise ValueError("NVFP4 artifact lacks accepted producer manifest")
    source = checkpoint["files"].get("model.safetensors")
    if source is None or source["sha256"] != producer.get("source_sha256"):
        raise ValueError("Packed source checkpoint hash mismatch")
    manifest = identity(folder / "manifest.json")
    if manifest["sha256"] != producer.get("manifest_sha256"):
        raise ValueError("Packed manifest hash mismatch")
    files = {}
    for name, record in producer["files"].items():
        if Path(name).name != name:
            raise ValueError("Packed manifest must use flat file names")
        actual = identity(folder / name)
        if (actual["sha256"], actual["bytes"]) != (record["sha256"], record["bytes"]):
            raise ValueError(f"Packed file identity mismatch: {name}")
        files[name] = actual
    if len(files) != 198 or producer["summary"]["tensors"] != 99:
        raise ValueError("Expected verified 99-tensor native pi05 artifact")
    return {"producer_manifest": identity(producer_file), "manifest": manifest, "files": files,
            "note": "Engine retains BF16 source weights; artifact bytes are not measured VRAM savings."}


def runtime_identity():
    packages = {}
    for name in ("apxinf", "apxinf-py", "apxinf-robo", "numpy", "pillow", "sentencepiece", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    modules, extensions = {}, {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or not name.startswith(("apxinf",)) or not Path(path).is_file():
            continue
        if path.endswith(".py"):
            modules[name] = identity(path)
        elif path.endswith((".so", ".pyd")):
            extensions[name] = identity(path)
    if not extensions:
        raise RuntimeError("Could not identify the actually imported native extension")
    engine = ROOT / "third_party/apxinf-robo/apxinf"
    relative_sources = ["crates/apxinf-py/src/lib.rs", "crates/apxinf-model/src/pi05/model_runner/runner.rs",
                        "crates/apxinf-model/src/pi05/model/model.rs", "crates/apxinf-model/src/pi05/load.rs",
                        "crates/apxinf-model/src/gr00t/vla_runtime.rs", "crates/apxinf-model/src/gr00t/executor.rs",
                        "crates/apxinf-model/src/gr00t/load.rs",
                        "crates/apxinf-cuda/src/transfers.rs"]
    rust = {name: identity(engine / name) for name in relative_sources if (engine / name).is_file()}
    return {"packages": packages, "imported_python_sources": modules, "native_extensions": extensions,
            "workspace_rust_sources": rust, "benchmark_script": identity(__file__),
            "limitation": "Imported SO hash is authoritative. Workspace Rust hashes record inspected source, not a proof that this SO was compiled from those exact files; retain build provenance separately."}


def gpu_identity(device):
    command = ["nvidia-smi", "-i", device,
               "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"]
    try:
        return {"command": command, "stdout": subprocess.check_output(command, text=True, timeout=5).strip(),
                "error": None}
    except Exception as exc:
        return {"command": command, "stdout": None, "error": f"{type(exc).__name__}: {exc}"}


class PowerSampler:
    def __init__(self, device="0", interval=.2):
        self.device, self.interval = device, interval
        self.samples, self.errors = [], []
        self.stop = threading.Event()
    def _run(self):
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output(["nvidia-smi", "-i", self.device,
                    "--query-gpu=power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    text=True, timeout=2).strip()
                values = []
                for value in raw.split(","):
                    try:
                        number = float(value.strip())
                        values.append(number if math.isfinite(number) else None)
                    except ValueError:
                        values.append(None)
                if len(values) != 3:
                    raise ValueError(f"Unexpected telemetry row: {raw}")
                self.samples.append({"monotonic_s": time.perf_counter(), "power_w": values[0],
                                     "utilization_percent": values[1], "memory_mib": values[2]})
            except Exception as exc:
                if len(self.errors) < 5:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop.wait(self.interval)
    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("Telemetry thread did not stop")


def benchmark(policy, args, sampler_factory=PowerSampler):
    md = dict(policy.metadata)
    rng = np.random.default_rng(args.seed)
    if not md.get("image_keys") or not md.get("prompt_key"):
        raise ValueError("Policy metadata lacks required observation keys")
    def make_obs():
        observation = {k: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for k in md["image_keys"]}
        observation[md["prompt_key"]] = "put the moka pot on the stove"
        if md.get("state_key"):
            if not isinstance(md.get("state_dim"), int) or md["state_dim"] <= 0:
                raise ValueError("Active state key requires a positive state_dim")
            observation[md["state_key"]] = rng.uniform(-.3, .3, md["state_dim"]).astype(np.float32)
        return observation
    first_observation = make_obs()
    observation_hash = hashlib.sha256()
    for key, value in sorted(first_observation.items()):
        if isinstance(value, np.ndarray):
            header = {"key": key, "dtype": str(value.dtype), "shape": list(value.shape)}
            payload = np.ascontiguousarray(value).tobytes()
        else:
            header = {"key": key, "type": type(value).__name__}
            payload = str(value).encode("utf-8")
        observation_hash.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        observation_hash.update(len(payload).to_bytes(8, "little"))
        observation_hash.update(payload)
    first = policy.infer(first_observation)
    shape = check_inference(first)
    first_actions = np.asarray(first["actions"]).copy()
    first_tokens = len(first["token_ids"]) if "token_ids" in first else None
    for _ in range(args.warmup):
        check_inference(policy.infer(make_obs()), shape)
    lat_model, lat_total, lat_wrapper, token_counts = [], [], [], []
    with sampler_factory(args.telemetry_device) as telemetry:
        for _ in range(args.samples):
            observation = make_obs()
            started = time.perf_counter()
            result = policy.infer(observation)
            elapsed = (time.perf_counter() - started) * 1e3
            check_inference(result, shape)  # finite validation is outside the timers
            lat_model.append(float(result["timing"]["model_ms"]))
            lat_total.append(float(result["timing"]["total_ms"]))
            lat_wrapper.append(elapsed)
            token_counts.append(len(result["token_ids"]) if "token_ids" in result else None)
    power = [s["power_w"] for s in telemetry.samples if s["power_w"] is not None]
    memory = [s["memory_mib"] for s in telemetry.samples if s["memory_mib"] is not None]
    result = {"n": args.samples, "warmup": args.warmup, "untimed_first_inferences": 1,
              "lat_model_ms": lat_model, "lat_total_ms": lat_total, "lat_wrapper_ms": lat_wrapper,
              "action_shape": shape, "all_outputs_finite": True, "first_token_count": first_tokens,
              "first_observation_sha256": observation_hash.hexdigest(),
              "first_actions": first_actions.tolist(),
              "first_actions_scope": "Untimed synthetic observation for numerical inspection only; not a task-success measurement.",
              "sample_token_counts": token_counts, "quantile_method": "numpy.percentile linear",
              "power_w_mean": float(np.mean(power)) if power else None,
              "power_w_max": max(power) if power else None, "vram_mb_peak": max(memory) if memory else None,
              "telemetry": {"nvidia_smi_device": args.telemetry_device, "samples": telemetry.samples,
                  "errors": telemetry.errors, "scope": "Whole selected GPU during timing phase, including input preparation; not isolated process power/VRAM.",
                  "device_mapping": "nvidia-smi device selection is explicit and may differ from CUDA_VISIBLE_DEVICES remapping."}}
    for name, values in (("model", lat_model), ("total", lat_total), ("wrapper", lat_wrapper)):
        result[f"{name}_ms_p50"] = float(np.percentile(values, 50))
        result[f"{name}_ms_p99"] = float(np.percentile(values, 99))
        result[f"{name}_ms_mean"] = float(np.mean(values))
    result["hz_p50"] = 1000 / result["total_ms_p50"]
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, default="bf16")
    p.add_argument("--extra-kwarg", action="append", default=[])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--telemetry-device", default="0")
    p.add_argument("--seed", type=int, default=7, help="synthetic observations")
    p.add_argument("--model-seed", type=int, default=0, help="model noise, unless seed extra explicitly overrides")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--out", type=Path)
    group.add_argument("--tag")
    args = p.parse_args()
    if args.samples < 1 or args.warmup < 0 or min(args.seed, args.model_seed) < 0:
        p.error("samples > 0, warmup >= 0 and nonnegative seeds required")
    if not re.fullmatch(r"cuda:[0-9]+", args.device):
        p.error("This GPU benchmark requires an explicit cuda:N device")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        p.error("--tag accepts letters, digits, underscore and hyphen")
    args.model_dir = args.model_dir.absolute()
    if args.out is None:
        tag = args.tag or f"{args.model_dir.name}_{args.variant}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
        args.out = ROOT / "results/engine" / f"{tag}.json"
    if args.out.exists() or args.out.is_symlink():
        p.error(f"Refusing existing output: {args.out}")
    return args


def main():
    args = parse_args()
    family, kwargs = prepare_config(args)
    checkpoint = directory_identity(args.model_dir)
    if not any(name.endswith(".safetensors") for name in checkpoint["files"]):
        raise ValueError("No source checkpoint weights")
    assets = {}
    for key in ("backbone", "norm_stats", "tokenizer_path", "calibration", "tactics"):
        if key in kwargs:
            path = Path(kwargs[key])
            assets[key] = directory_identity(path) if path.is_dir() else identity(path)
    packed = packed_identity(args.model_dir, checkpoint) if args.variant == "nvfp4_static" else None
    gpu = gpu_identity(args.telemetry_device)
    from apxinf import AutoPolicy
    policy = None
    started = time.perf_counter()
    try:
        policy = AutoPolicy.from_pretrained(str(args.model_dir), **kwargs)
        load_s = time.perf_counter() - started
        verified = verify_variant(policy, family, args.variant)
        if policy.metadata.get("model_type") != family:
            raise ValueError("Loaded policy model family differs from checkpoint")
        runtime = runtime_identity()
        result = benchmark(policy, args)
        result.update({"schema_version": 2, "model_dir": str(args.model_dir), "model_type": family,
            "variant": args.variant, "variant_requested": args.variant, "variant_verification": verified,
            "load_s": load_s, "policy_metadata": dict(policy.metadata), "constructor_kwargs": kwargs,
            "seed": args.seed, "model_seed": kwargs["seed"], "device": args.device,
            "gpu_identity": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint_identity": checkpoint, "asset_identity": assets, "packed_identity": packed,
            "runtime_identity": runtime,
            "timing_scope": {"entrypoint": "AutoPolicy.infer", "policy_class": type(policy).__name__,
                "model_ms": "Blocking native model call through host float32 action return; includes native input transfers, GPU work, synchronization and D2H.",
                "total_ms": "Policy internal preprocessing/tokenization, blocking native model, postprocessing/decoding; no simulator or network.",
                "wrapper_ms": "External perf_counter around policy.infer; input construction and additional finite checks excluded.",
                "synchronization": "Native binding returns completed host f32 arrays. Inspected pi05 host-return transfer and GR00T executor synchronize CUDA before D2H. These are wall-clock API times, not CUDA-event kernel times.",
                "limitation": "Cross-library input/tokenization/precision paths differ; report separately, not as a matched end-to-end speedup."},
            "input_contract": {"synthetic": True, "image_shape": [256, 256, 3], "image_dtype": "uint8",
                               "prompt": "put the moka pot on the stove", "batch_size": 1}})
    finally:
        if policy is not None:
            policy.close()
    # Only publish after successful inference AND cleanup. Exclusive creation
    # prevents a second invocation from replacing a completed result.
    publish_new(args.out, result)
    print(json.dumps({k: result[k] for k in ("model_type", "variant", "variant_verification", "n", "model_ms_p50", "total_ms_p50", "all_outputs_finite")}, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
