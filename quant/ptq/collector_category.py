"""Collect category-bank Hessians from the full GR00T model (GPU optional).

Only the seven CategorySpecificLinear matrices are observed. Inputs are split
by embodiment ID before X^T X accumulation, so the resulting cache is directly
usable by ``bake_category.py`` and cannot accidentally mix robots.
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
from category_fp4 import (ARCHITECTURE, ACTIVE_BANK, CACHE_VERSION, EXPECTED_SHAPES,
                          checkpoint_identity, identity, tensor_hash, validate_parent)
from probe_distill import replay_context, require_full_model, tensor_tree
from gr00t_runtime import configure_libero_data, verify_libero_statistics


def accumulate(entry, value):
    x = value.detach().reshape(-1, value.shape[-1]).float()
    with (torch.autocast(device_type=x.device.type, enabled=False)
          if x.device.type in ("cpu", "cuda") else nullcontext()):
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
    parser.add_argument("--parent", required=True, help="Pure PTQ parent checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=os.environ.get("QAD_DATASET", "./demo_data/libero_demo"))
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.windows < 1 or args.batch < 1 or args.cpu_threads < 1:
        parser.error("windows, batch and cpu-threads must be positive")
    parent = Path(args.parent).resolve(strict=True)
    output = Path(args.out).resolve()
    if output.exists() or parent == output or parent in output.parents:
        raise FileExistsError("Refusing to overwrite or nest category calibration output")
    source, entries, _recipe = validate_parent(parent)
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

    config = get_default_config().load_dict({"data": {
        "download_cache": False, "override_pretraining_statistics": False,
        "datasets": [{"dataset_paths": [args.dataset], "mix_ratio": 1.0,
                       "embodiment_tag": "libero_sim"}]}})
    config.model = Gr00tN1d7Config.from_pretrained(parent, local_files_only=True)
    config.model.model_name = os.environ.get("GR00T_BACKBONE_MODEL", config.model.model_name)
    config.model.use_relative_action = True
    config.training.start_from_checkpoint = str(parent)
    config.training.use_fsdp2 = False
    config.training.use_wandb = False
    config.training.transformers_local_files_only = True
    config.load_config_path = None
    configure_libero_data(config, parent)
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, output)
    pipeline.setup()
    verify_libero_statistics(pipeline.processor, parent)
    model = pipeline.return_model()
    architecture = require_full_model(model)
    if architecture != ARCHITECTURE:
        raise ValueError(f"Unexpected model architecture: {architecture}")
    model.requires_grad_(False)
    model = model.to(device=args.device, dtype=torch.bfloat16).eval()
    modules = dict(model.named_modules())
    acc = {key: {str(ACTIVE_BANK): {"H": None, "abs": None, "n": 0, "calls": 0}}
           for key in EXPECTED_SHAPES}
    hooks = []
    for key, shape in EXPECTED_SHAPES.items():
        module_path, tensor_name = key.rsplit(".", 1)
        module = modules.get(module_path)
        if module is None or not hasattr(module, tensor_name) or tuple(getattr(module, tensor_name).shape) != tuple(shape):
            raise ValueError(f"Category module absent or shape mismatch: {key}")

        def make_hook(layer_name):
            def hook(_module, inputs):
                if len(inputs) < 2 or not torch.is_tensor(inputs[0]) or not torch.is_tensor(inputs[1]):
                    raise TypeError(f"Category inputs are not (x, cat_ids): {layer_name}")
                x, cats = inputs[0], inputs[1].reshape(-1)
                if x.ndim != 3 or x.shape[0] != cats.numel() or x.shape[-1] != EXPECTED_SHAPES[layer_name][1]:
                    raise ValueError(f"Category input shape mismatch: {layer_name}")
                active = cats == ACTIVE_BANK
                if bool(active.any()):
                    accumulate(acc[layer_name][str(ACTIVE_BANK)], x[active])
            return hook

        hooks.append(module.register_forward_pre_hook(make_hook(key)))
    print(f"[category-calib] architecture={architecture}; layers={len(hooks)}; "
          f"active_bank={ACTIVE_BANK}; windows={args.windows}; H=CPU FP32", flush=True)
    dataset, _ = pipeline.return_dataset()
    collator = pipeline.return_collator()
    iterator = iter(dataset)
    consumed, forwards = 0, 0
    try:
        while consumed < args.windows:
            count = min(args.batch, args.windows - consumed)
            raw = [next(iterator) for _ in range(count)]
            batch = collator(raw)
            inputs = batch.get("inputs", batch)
            with replay_context(model, args.seed + forwards, "bfloat16"), torch.no_grad():
                model(tensor_tree(inputs, args.device, clone=False))
            consumed += count
            forwards += 1
            print(f"[category-calib] windows={consumed}/{args.windows} forwards={forwards} "
                  f"elapsed={time.time()-start:.1f}s", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    missing = [key for key, banks in acc.items()
               if banks[str(ACTIVE_BANK)]["H"] is None or banks[str(ACTIVE_BANK)]["n"] == 0]
    if missing:
        raise RuntimeError("Active LIBERO category bank was never observed: " + ", ".join(missing))
    cache = output / "calib.pt"
    torch.save(acc, cache)
    provenance = {"parent": source, "parent_recipe_sha256": identity(parent / "ptq_recipe.json")["sha256"],
                  "parent_bake_sha256": identity(parent / "bake_manifest.json")["sha256"]}
    layer_meta = {}
    for key, banks in acc.items():
        entry = banks[str(ACTIVE_BANK)]
        layer_meta[key] = {str(ACTIVE_BANK): {"rows": entry["n"], "calls": entry["calls"],
                       "H_sha256": tensor_hash(entry["H"]), "abs_sha256": tensor_hash(entry["abs"])} }
    metadata = {"version": CACHE_VERSION, "status": "complete", "parent": str(parent),
                "provenance": provenance, "architecture": architecture, "active_bank": ACTIVE_BANK,
                "dataset": str(Path(args.dataset).resolve()), "model_dtype": "bfloat16",
                "accumulation_dtype": "float32", "accumulation_device": "cpu", "tf32_matmul": False,
                "windows_requested": args.windows, "windows_consumed": consumed, "batch": args.batch,
                "forwards": forwards, "seed": args.seed, "n_layers": len(acc), "layers": layer_meta,
                "cache_sha256": identity(cache)["sha256"], "source_category_sha256": source["category_source_sha256"],
                "implementation_sha256": identity(Path(__file__))["sha256"],
                "elapsed_seconds": time.time() - start}
    (output / "calib_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    active_rows = sum(x[str(ACTIVE_BANK)]["n"] for x in acc.values())
    print(f"[category-calib] saved {len(acc)} layers; active bank rows={active_rows}; output={output}", flush=True)


if __name__ == "__main__":
    main()
