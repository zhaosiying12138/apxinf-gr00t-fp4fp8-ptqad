"""BF16 GR00T get_action latency; use the pinned recovery environment.

Synthetic batch-1 inputs are prepared outside the timed call. The timed entry
includes processor, backbone, action head, CPU transfer and action decoding.
Run --help without CUDA; model loading and timing require exclusive GPU access.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "weights/GR00T-N1.7-LIBERO/libero_10")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backbone", default=os.environ.get("GR00T_BACKBONE_MODEL"),
                   help="local path containing nvidia/Cosmos-Reason2; keep the literal symlink path")
    a = p.parse_args()
    if a.samples < 1 or a.warmup < 0 or not a.device.startswith("cuda"):
        p.error("positive samples, nonnegative warmup and a CUDA device are required")
    if a.out.exists():
        p.error(f"refusing existing output: {a.out}")
    if not a.backbone or not Path(a.backbone).is_dir() or "nvidia/Cosmos-Reason2" not in a.backbone:
        p.error("--backbone/GR00T_BACKBONE_MODEL must name the local Cosmos directory with its factory-compatible name")
    a.checkpoint = a.checkpoint.resolve(strict=True)
    return a


def check_action(result):
    import numpy as np
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Expected Gr00tPolicy (action_dict, info) tuple")
    actions, info = result
    if not isinstance(actions, dict) or not actions or not isinstance(info, dict):
        raise TypeError("Expected nonempty action dictionary and info dictionary")
    shapes = {}
    for name, value in actions.items():
        array = np.asarray(value)
        if not array.size or not np.isfinite(array).all():
            raise ValueError(f"Nonfinite or empty action: {name}")
        shapes[name] = list(array.shape)
    return shapes


def load_policy(args):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from transformers import AutoModel, AutoProcessor
    model_load, processor_load = AutoModel.from_pretrained, AutoProcessor.from_pretrained
    loading = {}

    def checked_model(path, *pos, **kwargs):
        kwargs.update(model_name=args.backbone, local_files_only=True, output_loading_info=True)
        model, info = model_load(path, *pos, **kwargs)
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if info.get(field):
                raise ValueError(f"Checkpoint loading {field}: {info[field]}")
        loading.update(info)
        return model

    def local_processor(path, *pos, **kwargs):
        kwargs.update(model_name=args.backbone, local_files_only=True,
                      transformers_loading_kwargs={"local_files_only": True})
        return processor_load(path, *pos, **kwargs)

    with patch.object(AutoModel, "from_pretrained", side_effect=checked_model), \
         patch.object(AutoProcessor, "from_pretrained", side_effect=local_processor):
        policy = Gr00tPolicy(model_path=str(args.checkpoint),
                            embodiment_tag=EmbodimentTag.LIBERO_PANDA, device=args.device, strict=True)
    sys.path.insert(0, str(ROOT / "rl"))
    from probe_distill import require_full_model
    architecture = require_full_model(policy.model)
    return policy, loading, architecture


def observe_linear_dtypes(model, infer):
    import torch
    signatures = Counter()
    def hook(module, inputs, output):
        if inputs and torch.is_tensor(inputs[0]) and torch.is_tensor(output):
            name = lambda value: str(value.dtype).removeprefix("torch.")
            signatures[f"input={name(inputs[0])};weight={name(module.weight)};output={name(output)}"] += 1
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Linear)]
    try:
        result = infer()
    finally:
        for handle in handles:
            handle.remove()
    return result, dict(signatures)


def main():
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback for a GPU latency result")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_hashes = {p.name: digest(p) for p in sorted(args.checkpoint.iterdir())
                         if p.is_file() and p.suffix in (".json", ".safetensors")}
    torch.cuda.synchronize()
    started = time.perf_counter()
    policy, loading, architecture = load_policy(args)
    policy.model.eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    dtype_elements = Counter()
    for parameter in policy.model.parameters():
        dtype_elements[str(parameter.dtype).removeprefix("torch.")] += parameter.numel()
    if set(dtype_elements) != {"bfloat16"}:
        raise ValueError(f"Expected BF16 model parameters, got {dict(dtype_elements)}")
    rng = np.random.default_rng(args.seed)
    language_key = policy.modality_configs["language"].modality_keys[0]
    def make_obs():
        state = rng.uniform(-.5, .5, 8).astype(np.float32)
        groups = {"x": state[0:1], "y": state[1:2], "z": state[2:3],
                  "roll": state[3:4], "pitch": state[4:5], "yaw": state[5:6], "gripper": state[6:8]}
        return {"video": {key: rng.integers(0, 256, (1, 1, 224, 224, 3), dtype=np.uint8)
                           for key in policy.modality_configs["video"].modality_keys},
                "state": {key: value.reshape(1, 1, -1) for key, value in groups.items()},
                "language": {language_key: [["put the moka pot on the stove"]]}}
    first_obs = make_obs()
    with torch.inference_mode():
        first, signatures = observe_linear_dtypes(policy.model, lambda: policy.get_action(first_obs))
        action_shapes = check_action(first)
        for _ in range(args.warmup):
            check_action(policy.get_action(first_obs))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        for _ in range(args.samples):
            observation = make_obs()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = policy.get_action(observation)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1e3)
            if check_action(result) != action_shapes:
                raise ValueError("Output action shapes changed between timing samples")
    if not signatures:
        raise RuntimeError("No Linear compute dtype observations captured")
    result = {
        "schema_version": 2, "model": "nvidia/GR00T-N1.7-LIBERO", "engine": "pytorch-official",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hashes,
        "script_sha256": digest(__file__), "backbone": args.backbone,
        "policy_source_sha256": digest(sys.modules[type(policy).__module__].__file__),
        "architecture": architecture, "strict_loading_info": loading,
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "gpu": torch.cuda.get_device_name(), "device": args.device,
        "dtype_config": "bfloat16", "dtype_policy": "all_model_parameters_bfloat16",
        "parameter_dtype_elements": dict(dtype_elements),
        "compute_dtype": {"outer_autocast": False, "linear_io_signatures": signatures,
                          "observation_pass": "untimed_first_inference"},
        "timing_scope": {"entrypoint": "Gr00tPolicy.get_action",
            "included": ["processor", "backbone", "action_head", "action_device_to_host", "action_decoding", "policy_validation"],
            "excluded": ["model_loading", "synthetic_input_generation", "finite_output_validation", "simulator", "network"]},
        "input_contract": {"synthetic": True, "image_keys": list(first_obs["video"]),
            "image_shape": [1, 1, 224, 224, 3], "image_dtype": "uint8",
            "state_elements": 8, "prompt": "put the moka pot on the stove"},
        "batch_size": 1, "seed": args.seed, "n": args.samples, "warmup": args.warmup,
        "untimed_first_inferences": 1, "load_s": load_s,
        "latency_ms_mean": float(np.mean(latencies)), "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p99": float(np.percentile(latencies, 99)), "lat_all_ms": latencies,
        "quantile_method": "numpy.percentile linear", "hz_mean": 1000 / float(np.mean(latencies)),
        "vram_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
        "vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "vram_scope": "PyTorch allocator peak after warmup; excludes other processes",
        "action_shapes": action_shapes, "all_outputs_finite": True,
        "tf32": False, "compile_model": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("model", "dtype_config", "parameter_dtype_elements", "timing_scope", "n", "latency_ms_mean")}, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
