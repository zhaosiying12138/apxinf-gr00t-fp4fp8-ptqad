"""Collect real full-model PTQ Hessians with bounded GPU residency.

Mixed-recipe GPTQ targets include language q/k/v/gate/up and nested DiT FFNs.
One layer's X.T@X is computed in FP32 at a time and accumulated on the CPU.
The full checkpoint is loaded through training.start_from_checkpoint, never
through the unused model_path configuration attribute. New output only.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "rl"))
from bake import alloc, file_hash, inventory, tied_aliases
from probe_distill import replay_context, require_full_model, tensor_tree
from gr00t_runtime import configure_libero_data, verify_libero_statistics
from captured_calibration import CapturedCalibration, load_captured_model


def accumulate(entry, value):
    """Unnormalised exact empirical second moment; FP32 matmul, CPU sum."""
    x = value.detach().reshape(-1, value.shape[-1]).float()
    context = torch.autocast(device_type=x.device.type, enabled=False) if x.device.type in ("cpu", "cuda") else nullcontext()
    with context:
        gram = (x.T @ x).to(device="cpu", dtype=torch.float32)
        absolute = x.abs().sum(0).to(device="cpu", dtype=torch.float32)
    if entry["H"] is None:
        entry["H"], entry["abs"] = gram, absolute
    else:
        entry["H"].add_(gram)
        entry["abs"].add_(absolute)
    entry["n"] += x.shape[0]
    entry["calls"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("PTQ_BASE", str(ROOT / "weights/GR00T-N1.7-LIBERO/libero_10")))
    parser.add_argument("--out", default=os.environ.get("PTQ_CAL_OUT"), required="PTQ_CAL_OUT" not in os.environ)
    parser.add_argument("--dataset", default=os.environ.get("QAD_DATASET", "./demo_data/libero_demo"))
    parser.add_argument("--capture-manifest", help="Frozen audited teacher inputs; requires batch=1 and all windows")
    parser.add_argument("--windows", type=int, default=int(os.environ.get("PTQ_CAL_WINDOWS", "128")))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("PTQ_CAL_BATCH", "1")))
    parser.add_argument("--recipe", choices=["mixed", "calib", "aggr"], default="mixed")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.windows < 1 or args.batch < 1 or args.cpu_threads < 1:
        parser.error("windows, batch and CPU threads must be positive")
    base, output = Path(args.base).resolve(), Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {output}")
    if base == output or base in output.parents:
        raise ValueError("calibration output cannot be inside the checkpoint")
    captured = CapturedCalibration(args.capture_manifest, base, args.windows, args.batch) if args.capture_manifest else None
    output.mkdir(parents=True)
    start = time.time()
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, os.getcwd())
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model import MODEL_REGISTRY

    entries, shards, _ = inventory(base)
    source_weights = {s: {"bytes": (base / s).stat().st_size, "sha256": file_hash(base / s)} for s in shards}
    if captured is not None:
        captured.validate_root_weights(source_weights)
    config = get_default_config().load_dict({"data": {
        "download_cache": False, "override_pretraining_statistics": False,
        "datasets": [{"dataset_paths": [args.dataset], "mix_ratio": 1.0,
                      "embodiment_tag": "libero_sim"}]}})
    config.model = Gr00tN1d7Config.from_pretrained(base, local_files_only=True)
    config.model.model_name = os.environ.get("GR00T_BACKBONE_MODEL", config.model.model_name)
    config.model.use_relative_action = True
    config.training.start_from_checkpoint = str(base)
    config.training.use_fsdp2 = False
    config.training.use_wandb = False
    config.training.transformers_local_files_only = True
    config.load_config_path = None
    configure_libero_data(config, base)
    if captured is not None:
        model = load_captured_model(base, config.model)
    else:
        pipeline = MODEL_REGISTRY.get(type(config.model))(config, output)
        pipeline.setup()
        verify_libero_statistics(pipeline.processor, base)
        model = pipeline.return_model()
    architecture = require_full_model(model)
    model.requires_grad_(False)
    model = model.to(device=args.device, dtype=torch.bfloat16).eval()
    acc, hooks = {}, []
    modules = dict(model.named_modules())
    aliases = tied_aliases(entries)
    nonlinear_targets = {}
    for key, ent in entries.items():
        name = key.removesuffix(".weight")
        if not ent["eligible"] or alloc(name, None, args.recipe) != "nvfp4_gptq" or key in aliases:
            continue
        module = modules.get(name)
        if module is None:
            raise ValueError(f"requested GPTQ target absent from full model: {name}")
        if not isinstance(module, torch.nn.Linear):
            nonlinear_targets[name] = type(module).__name__
    for name, module in modules.items():
        if not isinstance(module, torch.nn.Linear) or module.weight.ndim != 2 or module.weight.shape[1] % 16:
            continue
        if name + ".weight" in aliases:
            continue  # tied output copies the canonical embedding during baking
        if alloc(name, module.weight, args.recipe) != "nvfp4_gptq":
            continue
        key = name + ".weight"
        if key not in entries or list(module.weight.shape) != entries[key]["shape"]:
            raise ValueError(f"loaded module missing/mismatched in checkpoint inventory: {name}")
        acc[name] = {"H": None, "abs": None, "n": 0, "calls": 0}
        def make_hook(layer_name):
            def hook(_module, inputs):
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError(f"unexpected non-tensor input at {layer_name}")
                accumulate(acc[layer_name], inputs[0])
            return hook
        hooks.append(module.register_forward_pre_hook(make_hook(name)))
    if not hooks:
        raise RuntimeError("no GPTQ targets were hooked")
    print(f"[calib] full architecture={architecture}; target_layers={len(hooks)}; "
          f"windows={args.windows} batch={args.batch}; H accumulation=CPU FP32", flush=True)
    if captured is not None:
        iterator = iter(captured)
    else:
        dataset, _ = pipeline.return_dataset()
        collator = pipeline.return_collator()
        iterator = iter(dataset)
    consumed, forwards = 0, 0
    try:
        while consumed < args.windows:
            count = min(args.batch, args.windows - consumed)
            if captured is not None:
                inputs = next(iterator)
            else:
                raw = [next(iterator) for _ in range(count)]
                batch = collator(raw)
                inputs = batch.get("inputs", batch)
            with replay_context(model, args.seed + forwards, "bfloat16"), torch.no_grad():
                model(tensor_tree(inputs, args.device, clone=False))
            consumed += count
            forwards += 1
            print(f"[calib] windows={consumed}/{args.windows} forwards={forwards} "
                  f"elapsed={time.time()-start:.1f}s", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    missing = [name for name, entry in acc.items() if entry["H"] is None or entry["n"] == 0]
    if missing:
        raise RuntimeError(f"hooked layers never executed; refusing incomplete H cache: {missing}")
    output_cache = output / "calib.pt"
    torch.save(acc, output_cache)
    meta = {
        "version": "full-model-cpu-hessian-v2", "status": "complete", "base": str(base),
        "base_weight_files": source_weights, "base_config_sha256": file_hash(base / "config.json"),
        "base_statistics_sha256": file_hash(base / "statistics.json") if (base / "statistics.json").exists() else None,
        "dataset": None if captured is not None else str(Path(args.dataset).resolve()), "architecture": architecture,
        "model_dtype": "bfloat16", "accumulation_dtype": "float32", "accumulation_device": "cpu",
        "tf32_matmul": False, "recipe_targets": args.recipe, "windows_requested": args.windows,
        "windows_consumed": consumed, "windows_are_unique": ("unique capture files; one pass; not independent episodes"
            if captured is not None else "not asserted for iterable dataset"),
        "batch": args.batch, "forwards": forwards, "seed": args.seed,
        "n_layers": len(acc), "H_GB_f32": sum(x["H"].numel() * 4 for x in acc.values()) / 1e9,
        "nonlinear_targets": nonlinear_targets, "tied_weight_aliases": aliases,
        "rows_per_layer": {k: v["n"] for k, v in acc.items()},
        "calls_per_layer": {k: v["calls"] for k, v in acc.items()},
        "cache_sha256": file_hash(output_cache), "implementation_sha256": file_hash(__file__),
        "elapsed_seconds": time.time() - start,
    }
    if captured is not None:
        meta["captured_input_provenance"] = captured.record
    (output / "calib_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[calib] saved {len(acc)} layers, {meta['H_GB_f32']:.3f} GB H; "
          f"actual windows={consumed} -> {output_cache}", flush=True)


if __name__ == "__main__":
    main()
