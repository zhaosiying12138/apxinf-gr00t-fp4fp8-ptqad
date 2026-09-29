"""Export an additive LoRA checkpoint using its verified frozen PTQ base.

Only A/B are imported from the training checkpoint. The original base supplies
all model keys and dtypes, including keys omitted by the instantiated trainer.
W_export = cast_to_base_dtype(W_base + alpha/r * B@A); no re-quantization.
This is a dense, source-dtype export, not native low-bit storage. Finite-precision
rounding means merged and separate-branch execution must be checked numerically.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant" / "ptq"))
from bake import inventory, file_hash


def json_equal(a, b):
    try:
        return json.loads(a.read_text()) == json.loads(b.read_text())
    except (ValueError, UnicodeError):
        return a.read_bytes() == b.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Exact frozen quantized base used in training")
    parser.add_argument("--ckpt", required=True, help="LoRA training checkpoint directory")
    parser.add_argument("--out", required=True, help="New destination; existing directories are rejected")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--metadata-source", choices=["base", "training"], default="base",
                        help="base preserves the frozen inference normalization; training is an explicit alternative")
    parser.add_argument("--allow-missing-base-key", action="append",
                        default=["backbone.model.lm_head.weight"],
                        help="Unused base key absent from trainer; retained unchanged and recorded")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Verify frozen tensors/A/B without exporting")
    args = parser.parse_args()
    if args.rank <= 0 or args.cpu_threads <= 0:
        parser.error("rank and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    start = time.time()
    base, ckpt, output = map(lambda x: Path(x).resolve(), (args.base, args.ckpt, args.out))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if base == output or ckpt == output or base in output.parents or ckpt in output.parents:
        raise ValueError("output cannot be nested in source checkpoints")
    base_info, base_shards, _ = inventory(base)
    train_info, train_shards, _ = inventory(ckpt)
    train_plain = {k for k in train_info if ".lora_" not in k}
    unknown = train_plain - set(base_info)
    missing = set(base_info) - train_plain
    if unknown or missing - set(args.allow_missing_base_key):
        raise ValueError(f"base key mismatch: unexpected={sorted(unknown)} missing={sorted(missing)}")
    akeys = sorted(k for k in train_info if k.endswith(".lora_A"))
    allowed_lora = {k for a in akeys for k in (a, a.removesuffix(".lora_A") + ".lora_B")}
    if not akeys or {k for k in train_info if ".lora_" in k} != allowed_lora:
        raise ValueError("missing, unmatched or unrecognized LoRA parameters")
    pairs = {}
    for ak in akeys:
        stem = ak.removesuffix(".lora_A")
        bk, wk = stem + ".lora_B", stem + ".weight"
        if wk not in base_info or bk not in train_info:
            raise ValueError(f"missing B/base for {stem}")
        shape = base_info[wk]["shape"]
        if len(shape) != 2 or train_info[ak]["shape"] != [args.rank, shape[1]] or train_info[bk]["shape"] != [shape[0], args.rank]:
            raise ValueError(f"LoRA shape/rank mismatch for {stem}")
        pairs[wk] = (ak, bk)
    if args.metadata_source == "base":
        for name in ("statistics.json", "embodiment_id.json"):
            if (base / name).exists() and (ckpt / name).exists() and not json_equal(base / name, ckpt / name):
                raise ValueError(f"{name} differs between training and base; inspect normalization provenance before exporting")
    recovery_source = next((p for p in (ckpt / "recovery_manifest.json", ckpt.parent / "recovery_manifest.json") if p.exists()), None)
    recovery_manifest = json.loads(recovery_source.read_text()) if recovery_source else None
    if recovery_manifest is None:
        raise ValueError("training recovery_manifest.json required to verify LoRA alpha/rank/base")
    if recovery_manifest["rank"] != args.rank or recovery_manifest["alpha"] != args.alpha:
        raise ValueError("merge rank/alpha differs from the training manifest")
    if Path(recovery_manifest["base"]).resolve() != base:
        raise ValueError("merge base path differs from the training manifest")
    for field, filename in (("base_config_sha256", "config.json"),
                            ("base_statistics_sha256", "statistics.json"),
                            ("base_recipe_sha256", "ptq_recipe.json")):
        actual_hash = file_hash(base / filename) if (base / filename).exists() else None
        if recovery_manifest.get(field) != actual_hash:
            raise ValueError(f"base metadata changed since training: {filename}")
    manifest = {
        "export_version": "base-anchored-additive-v2", "base": str(base), "training_checkpoint": str(ckpt),
        "rank": args.rank, "alpha": args.alpha, "metadata_source": args.metadata_source,
        "base_tensor_count": len(base_info), "training_tensor_count": len(train_info),
        "lora_pairs": len(pairs), "retained_base_only_keys": sorted(missing),
        "storage": "dense original-base dtype; BF16 residual merged without requantization",
        "numerical_note": "Dense casting introduces final rounding; this is not an assertion of bitwise identity to a separate low-rank branch.",
        "recovery_manifest_source": str(recovery_source) if recovery_source else None,
        "recovery_manifest_sha256": file_hash(recovery_source) if recovery_source else None,
        "recovery_manifest": recovery_manifest,
        "base_ptq_recipe_sha256": file_hash(base / "ptq_recipe.json") if (base / "ptq_recipe.json").exists() else None,
        "base_bake_manifest_sha256": file_hash(base / "bake_manifest.json") if (base / "bake_manifest.json").exists() else None,
        "base_weights": {name: {"bytes": (base / name).stat().st_size, "sha256": file_hash(base / name)} for name in base_shards},
        "training_weights": {name: {"bytes": (ckpt / name).stat().st_size, "sha256": file_hash(ckpt / name)} for name in train_shards},
        "implementation_sha256": file_hash(__file__),
    }
    stage = None
    if not args.check_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
    weight_map, payload_bytes, verified = {}, 0, 0
    dtype_counts, residual_stats = Counter(), {}
    with ExitStack() as stack:
        readers = {name: stack.enter_context(safe_open(str(ckpt / name), framework="pt", device="cpu")) for name in train_shards}
        def trained(key):
            return readers[train_info[key]["shard"]].get_tensor(key)
        for shard in base_shards:
            tensors = load_file(str(base / shard), device="cpu")
            for key, weight in list(tensors.items()):
                if key in train_plain:
                    frozen = trained(key)
                    if list(frozen.shape) != list(weight.shape) or not torch.equal(frozen, weight.to(frozen.dtype)):
                        raise ValueError(f"frozen base differs from training tensor: {key}; refusing wrong-base merge")
                    verified += 1
                    del frozen
                if key in pairs:
                    ak, bk = pairs[key]
                    delta = (trained(bk).float() @ trained(ak).float()) * (args.alpha / args.rank)
                    if not torch.isfinite(delta).all():
                        raise ValueError(f"nonfinite LoRA residual: {key}")
                    residual_stats[key] = {"rms": float(delta.square().mean().sqrt()),
                                           "relative_frobenius": float(delta.norm() / weight.float().norm().clamp_min(1e-30))}
                    if stage is not None:
                        tensors[key] = (weight.float() + delta).to(weight.dtype).contiguous()
                    del delta
                weight_map[key] = shard
                payload_bytes += weight.numel() * weight.element_size()
                dtype_counts[str(weight.dtype)] += weight.numel()
            if stage is not None:
                save_file(tensors, str(stage / shard), metadata={"format": "pt"})
            del tensors
            print(f"[merge] checked {shard}; frozen_verified={verified}", flush=True)
    manifest.update(frozen_tensors_verified=verified, output_tensor_bytes=payload_bytes,
                    output_dtype_parameter_counts=dict(dtype_counts), residuals=residual_stats,
                    elapsed_seconds=time.time() - start, status="verified" if args.check_only else "complete")
    if stage is None:
        print(json.dumps({k: manifest[k] for k in ("status", "lora_pairs", "base_tensor_count", "frozen_tensors_verified", "output_tensor_bytes")}, indent=2))
        return
    for path in base.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in ("model.safetensors.index.json", "merge_manifest.json"):
            shutil.copy2(path, stage / path.name)
    if args.metadata_source == "training":
        for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
            if (ckpt / name).exists():
                shutil.copy2(ckpt / name, stage / name)
    # Config comes from the complete base. Keep every source tensor and publish
    # an index describing exactly the physical payload; no LoRA keys remain.
    index = {"metadata": {"total_size": payload_bytes}, "weight_map": weight_map}
    (stage / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    if recovery_source:
        shutil.copy2(recovery_source, stage / "recovery_manifest.json")
    manifest["output_weights"] = {s: {"bytes": (stage / s).stat().st_size, "sha256": file_hash(stage / s)} for s in base_shards}
    (stage / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Header validation checks index membership against physically written keys.
    exported, _, _ = inventory(stage)
    if set(exported) != set(base_info) or any(".lora_" in key for key in exported):
        raise RuntimeError("post-save tensor inventory mismatch")
    if sum(ent["bytes"] for ent in exported.values()) != payload_bytes:
        raise RuntimeError("post-save payload size mismatch")
    stage.rename(output)
    print(f"[merge] {len(pairs)} additive LoRA layers; {len(exported)} base-dtype tensors; "
          f"{payload_bytes / 1e9:.3f} GB logical payload -> {output}", flush=True)


if __name__ == "__main__":
    main()
