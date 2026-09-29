"""LeRobot pi0.5 predict_action_chunk timing in its native BF16 configuration.

Input generation, SentencePiece tokenization and initial H2D are outside timing.
LeRobot retains vision, norms, projector and action/time heads in FP32. This is
an entry-point benchmark with synthetic inputs, not an environment evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import time

from bench_gr00t_pt import digest, observe_linear_dtypes
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "weights/pi05_libero_base")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.samples < 1 or a.warmup < 0 or not a.device.startswith("cuda"):
        p.error("positive samples, nonnegative warmup and a CUDA device are required")
    if a.out.exists():
        p.error(f"refusing existing output: {a.out}")
    a.checkpoint = a.checkpoint.resolve(strict=True)
    return a


def load_policy(args):
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.configs import PreTrainedConfig
    from safetensors.torch import load_file
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise RuntimeError("This strict loader is audited against LeRobot 0.6.1; use its lock file")
    config = PreTrainedConfig.from_pretrained(str(args.checkpoint), local_files_only=True)
    config.device = "cpu"  # Avoid the checkpoint's saved MPS device during construction.
    config.dtype = "bfloat16"
    config.compile_model = False
    policy = PI05Policy(config)
    # LeRobot 0.6.1 from_pretrained catches load errors and can return an
    # uninitialized policy. Use its exact remapper but let strict errors escape.
    original = load_file(str(args.checkpoint / "model.safetensors"), device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(original, config)
    remapped = {(key if key.startswith("model.") else "model." + key): value
                for key, value in fixed.items()}
    if len(remapped) != len(fixed):
        raise ValueError("Key remapping produced collisions")
    policy.load_state_dict(remapped, strict=True)
    loaded = len(remapped)
    del original, fixed, remapped
    policy.eval().to(args.device)
    policy.requires_grad_(False)
    policy.config.device = args.device
    return policy, loaded


def check_action(action):
    import torch
    if not torch.is_tensor(action) or action.ndim != 3 or not action.numel():
        raise TypeError("Expected nonempty batch x horizon x action tensor")
    if not torch.isfinite(action).all().item():
        raise ValueError("Nonfinite action chunk")
    return list(action.shape)


def main():
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import sentencepiece as spm
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback for a GPU latency result")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_hashes = {p.name: digest(p) for p in sorted(args.checkpoint.iterdir())
                         if p.is_file() and p.suffix in (".json", ".safetensors", ".model")}
    torch.cuda.synchronize()
    started = time.perf_counter()
    policy, loaded = load_policy(args)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    dtype_elements = Counter()
    for parameter in policy.parameters():
        dtype_elements[str(parameter.dtype).removeprefix("torch.")] += parameter.numel()
    if set(dtype_elements) != {"bfloat16", "float32"}:
        raise ValueError(f"Expected native LeRobot BF16/FP32 policy, got {dict(dtype_elements)}")
    sp = spm.SentencePieceProcessor(model_file=str(args.checkpoint / "paligemma_tokenizer.model"))
    prompt = "put the moka pot on the stove"
    ids = [2] + sp.encode(prompt) + [1]
    rng = np.random.default_rng(args.seed)
    def image():
        array = rng.integers(0, 256, (1, 3, 256, 256), dtype=np.uint8)
        return torch.from_numpy(array).to(args.device, torch.float32) / 255.0
    def make_batch():
        # Omit empty_camera_0: LeRobot inserts its padded image and false mask.
        return {
            "observation.images.image": image(), "observation.images.image2": image(),
            "observation.state": torch.from_numpy(rng.uniform(-.3, .3, (1, 8)).astype("float32")).to(args.device),
            "observation.language.tokens": torch.tensor([ids], device=args.device),
            "observation.language.attention_mask": torch.ones((1, len(ids)), dtype=torch.bool, device=args.device),
        }
    batch = make_batch()
    with torch.inference_mode():
        first, signatures = observe_linear_dtypes(policy, lambda: policy.predict_action_chunk(batch))
        action_shape = check_action(first)
        for _ in range(args.warmup):
            check_action(policy.predict_action_chunk(batch))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        for _ in range(args.samples):
            batch = make_batch()
            torch.cuda.synchronize()
            started = time.perf_counter()
            action = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1e3)
            if check_action(action) != action_shape:
                raise ValueError("Action shape changed between timing samples")
    if not signatures:
        raise RuntimeError("No Linear compute dtype observations captured")
    result = {
        "schema_version": 2, "model": "lerobot/pi05_libero_base", "engine": "pytorch-lerobot",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hashes,
        "policy_source_sha256": digest(sys.modules[type(policy).__module__].__file__),
        "script_sha256": digest(__file__), "shared_helper_sha256": digest(Path(__file__).with_name("bench_gr00t_pt.py")),
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "lerobot": importlib.metadata.version("lerobot"), "strict_loaded_keys": loaded,
        "gpu": torch.cuda.get_device_name(), "device": args.device,
        "dtype_config": "bfloat16",
        "dtype_policy": "lerobot_bfloat16_with_fp32_vision_norm_projector_and_action_time_heads",
        "parameter_dtype_elements": dict(dtype_elements),
        "compute_dtype": {"outer_autocast": False, "linear_io_signatures": signatures,
                          "observation_pass": "untimed_first_inference"},
        "timing_scope": {"entrypoint": "PI05Policy.predict_action_chunk",
            "included": ["internal_image_resize_and_normalize", "vision_language_backbone", "action_denoising", "action_unpadding"],
            "excluded": ["model_loading", "synthetic_input_generation", "sentencepiece_tokenization", "initial_host_to_device", "finite_output_validation", "environment_state_normalization", "environment_action_denormalization", "simulator", "network"]},
        "input_contract": {"synthetic": True, "prompt": prompt, "token_count": len(ids),
            "token_recipe": "BOS + prompt-only SentencePiece tokens + EOS; no state-to-text preprocessing",
            "images": ["image:256x256", "image2:256x256"], "empty_camera": "omitted; native preprocessing inserts padding and false mask",
            "state": "synthetic 8-vector; predict_action_chunk does not use it directly"},
        "batch_size": 1, "seed": args.seed, "n": args.samples, "warmup": args.warmup,
        "untimed_first_inferences": 1, "load_s": load_s,
        "num_inference_steps": policy.config.num_inference_steps, "chunk_size": policy.config.chunk_size,
        "latency_ms_mean": float(np.mean(latencies)), "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p99": float(np.percentile(latencies, 99)), "lat_all_ms": latencies,
        "quantile_method": "numpy.percentile linear", "hz_mean": 1000 / float(np.mean(latencies)),
        "vram_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
        "vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "vram_scope": "PyTorch allocator peak after warmup; excludes other processes",
        "action_shape": action_shape, "all_outputs_finite": True,
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
