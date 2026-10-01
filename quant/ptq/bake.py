"""Bake corrected weight-only PTQ values into original-dtype safetensors.

The saved tensors are dequantized values, not packed native FP4/FP8 storage.
Recipe version 2 corrects nested DiT FFN matching and records the actual
calibration/fallback method per layer. Historical outputs are never replaced.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from quantizers import nvfp4_dequant, fp8_e4m3_dequant, gptq_nvfp4, rtnc_best_clip

CLIP_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)
RECIPE_VERSION = "corrected-allocation-v2"
RECIPES = ["rtn", "calib", "fp8", "mixed", "mixed_fp8_to_fp4", "aggr",
           "head_ffn", "head_lang", "head_lang_vision"]
EMBED = "backbone.model.model.language_model.embed_tokens.weight"
LM_HEAD = "backbone.model.lm_head.weight"


def tied_aliases(entries):
    # GR00T's Qwen3VL backbone ties these Parameter objects at model loading.
    # Physical safetensors may include both. Refuse unequal source values below.
    if EMBED in entries and LM_HEAD in entries:
        if entries[EMBED]["shape"] != entries[LM_HEAD]["shape"]:
            raise ValueError("tied embedding/lm_head shapes differ")
        return {LM_HEAD: EMBED}
    return {}


def is_dit_ffn(name):
    return bool(re.fullmatch(r"action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(?:0\.proj|2)", name))


def alloc_mixed(name, W=None):
    """Project mixed allocation; inspired by, not identical to, upstream recipes."""
    if name.startswith("backbone.model.model.visual.") or name in ("backbone.lm_head", "backbone.model.lm_head"):
        return "fp8"
    if ".language_model.layers." in name:
        return "fp8" if name.endswith(("self_attn.o_proj", "mlp.down_proj")) else "nvfp4_gptq"
    if name.startswith("action_head."):
        if is_dit_ffn(name):
            return "nvfp4_gptq"
        suffix = name.removeprefix("action_head.model.")
        if suffix.startswith("transformer_blocks.") or suffix == "proj_out_1":
            return "fp8"
        return "bf16"
    return "fp8"


def alloc_aggr(name, W=None):
    """Aggressive allocation; no claim that protected sites are unique causes."""
    if name in ("backbone.lm_head", "backbone.model.lm_head"):
        return "nvfp4_gptq"
    if ".language_model.layers." in name:
        return "fp8" if name.endswith(("self_attn.o_proj", "mlp.down_proj")) else "nvfp4_gptq"
    if name.startswith("action_head."):
        if is_dit_ffn(name):
            return "nvfp4_gptq"
        suffix = name.removeprefix("action_head.model.")
        if suffix.startswith("transformer_blocks.") or suffix == "proj_out_1":
            return "nvfp4_rtn"
        return "bf16"
    return "nvfp4_gptq"


def alloc(name, W, recipe):
    if recipe in ("rtn", "calib"):
        return "nvfp4_gptq" if recipe == "calib" else "nvfp4_rtn"
    if recipe == "fp8":
        return "nvfp4_rtn" if name.startswith("action_head.") else "fp8"
    if recipe in ("head_ffn", "head_lang", "head_lang_vision"):
        # Nested ladder: never remove a previous stage's NVFP4 assignment.
        # Names describe additional backbone groups; head linears remain FP4.
        if name.startswith("action_head."):
            return "nvfp4_rtn"
        if ".language_model.layers." in name:
            if name.endswith(("mlp.gate_proj", "mlp.up_proj")):
                return "nvfp4_gptq"
            if recipe != "head_ffn" and name.endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")):
                return "nvfp4_gptq"
        if recipe == "head_lang_vision" and name.startswith("backbone.model.model.visual."):
            return "nvfp4_gptq"
        return "fp8"
    if recipe == "mixed":
        return alloc_mixed(name, W)
    if recipe == "mixed_fp8_to_fp4":
        # Isolate the FP8->FP4 step in the published mixed recipe.  Existing
        # NVFP4 assignments keep their calibrated GPTQ/RTN path; only sites
        # that the mixed recipe protected as FP8 are lowered to NVFP4 RTN.
        mixed = alloc_mixed(name, W)
        return "nvfp4_rtn" if mixed == "fp8" else mixed
    if recipe == "aggr":
        return alloc_aggr(name, W)
    raise ValueError(recipe)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(base):
    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    elif (base / "model.safetensors").exists():
        index, weight_map = None, None
    else:
        raise FileNotFoundError(f"no safetensors index or single file in {base}")
    shards = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    entries = {}
    for shard in shards:
        path = base / shard
        if path.resolve().parent != base.resolve():
            raise ValueError("shard path must be directly inside base")
        with path.open("rb") as stream:
            header_bytes = stream.read(8)
            if len(header_bytes) != 8:
                raise ValueError(f"invalid safetensors header: {path}")
            header = json.loads(stream.read(struct.unpack("<Q", header_bytes)[0]))
        for key, item in header.items():
            if key == "__metadata__":
                continue
            if key in entries:
                raise ValueError(f"duplicate tensor across shards: {key}")
            if weight_map is not None and weight_map.get(key) != shard:
                raise ValueError(f"tensor/index mismatch: {key}")
            shape = item["shape"]
            n = 1
            for dim in shape:
                n *= dim
            eligible = key.endswith(".weight") and len(shape) == 2 and shape[1] % 16 == 0
            reason = None if eligible else ("not_weight" if not key.endswith(".weight") else
                                           "not_2d" if len(shape) != 2 else "input_width_not_multiple_of_16")
            entries[key] = {"shape": shape, "dtype": item["dtype"], "params": n,
                            "bytes": item["data_offsets"][1] - item["data_offsets"][0],
                            "shard": shard, "eligible": eligible, "excluded_reason": reason}
    if weight_map is not None and set(weight_map) != set(entries):
        raise ValueError("index contains missing physical tensors")
    return entries, shards, index


def make_plan(entries, recipe, head_bf16=False):
    layers, excluded = {}, {}
    params = Counter()
    target_bytes = Counter()
    for key, ent in entries.items():
        if not ent["eligible"]:
            excluded[key] = ent
            continue
        name = key.removesuffix(".weight")
        kind = alloc(name, None, recipe)
        if head_bf16 and name.startswith("action_head."):
            kind = "bf16"
        group = "nvfp4" if kind.startswith("nvfp4") else kind
        n = ent["params"]
        params[group] += n
        # Target storage only. Current checkpoint retains its source dtype.
        target_bytes[group] += (n // 2 + n // 16 + 4 if group == "nvfp4" else
                                n + ent["shape"][0] * 4 if group == "fp8" else ent["bytes"])
        layers[name] = {**ent, "requested_method": kind}
    linear = sum(params.values())
    original = sum(ent["bytes"] for ent in entries.values())
    eligible_bytes = sum(ent["bytes"] for ent in entries.values() if ent["eligible"])
    excluded_bytes = original - eligible_bytes
    packed = sum(target_bytes.values())
    memory = {
        "all_checkpoint_params": sum(ent["params"] for ent in entries.values()),
        "linear_params": linear, "eligible_tensor_count": len(layers),
        "excluded_tensor_count": len(excluded),
        "nvfp4_params": params["nvfp4"], "fp8_params": params["fp8"], "bf16_params": params["bf16"],
        "fraction_of_eligible_params": {k: params[k] / linear for k in ("nvfp4", "fp8", "bf16")},
        "fraction_of_all_checkpoint_params": {k: params[k] / sum(e["params"] for e in entries.values())
                                              for k in ("nvfp4", "fp8", "bf16")},
        "source_tensor_bytes": original, "eligible_source_bytes": eligible_bytes,
        "excluded_source_bytes": excluded_bytes, "target_eligible_bytes": packed,
        "target_full_checkpoint_bytes": packed + excluded_bytes,
        "linear_compression_x": eligible_bytes / packed,
        "full_checkpoint_compression_x": original / (packed + excluded_bytes),
        "bf16_GB": eligible_bytes / 1e9, "quantized_GB": packed / 1e9,
        "target_bytes_by_format": dict(target_bytes),
        "storage_note": "Estimate: FP4 payload+one E4M3 scale/16+one FP32 tensor scale; FP8 payload+FP32 row scales; excludes alignment and runtime activations. Saved tensors are dequantized in original dtype.",
    }
    # Physical checkpoint copies and runtime tied Parameters have different
    # denominators. Report both; this does not infer undiscovered tied groups.
    alias_map = tied_aliases(entries)
    distinct_params, distinct_target = params.copy(), target_bytes.copy()
    duplicate_n = duplicate_bytes = duplicate_eligible_bytes = 0
    for alias in alias_map:
        ent = entries[alias]
        duplicate_n += ent["params"]
        duplicate_bytes += ent["bytes"]
        if ent["eligible"]:
            method = layers[alias.removesuffix(".weight")]["requested_method"]
            group = "nvfp4" if method.startswith("nvfp4") else method
            n = ent["params"]
            distinct_params[group] -= n
            distinct_target[group] -= (n // 2 + n // 16 + 4 if group == "nvfp4" else
                                      n + ent["shape"][0] * 4 if group == "fp8" else ent["bytes"])
            duplicate_eligible_bytes += ent["bytes"]
    distinct_all = memory["all_checkpoint_params"] - duplicate_n
    distinct_eligible = sum(distinct_params.values())
    distinct_packed = sum(distinct_target.values())
    distinct_excluded = excluded_bytes - (duplicate_bytes - duplicate_eligible_bytes)
    memory["known_tied_alias_deduplicated"] = {
        "all_tensor_elements": distinct_all, "eligible_tensor_elements": distinct_eligible,
        "omitted_physical_alias_elements": duplicate_n,
        "elements_by_format": dict(distinct_params),
        "fraction_of_eligible": {k: distinct_params[k] / distinct_eligible for k in ("nvfp4", "fp8", "bf16")},
        "fraction_of_all": {k: distinct_params[k] / distinct_all for k in ("nvfp4", "fp8", "bf16")},
        "target_eligible_bytes": distinct_packed,
        "target_full_bytes": distinct_packed + distinct_excluded,
        "eligible_compression_x": (eligible_bytes - duplicate_eligible_bytes) / distinct_packed,
        "full_compression_x": (original - duplicate_bytes) / (distinct_packed + distinct_excluded),
        "note": "Deduplicates only explicitly recorded tied embedding/lm_head aliases; no claim about unrecorded sharing or activation memory.",
    }
    return layers, excluded, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--recipe", choices=RECIPES, default="mixed")
    parser.add_argument("--calib", help="Optional Hessian cache; provenance recorded when loaded")
    parser.add_argument("--calibration-mode", choices=["none", "auto", "required"], default="auto",
                        help="none: deliberate RTN fallback; auto: load supplied cache if it exists; required: require supplied cache")
    parser.add_argument("--gptq-damp", type=float, default=0.01)
    parser.add_argument("--rtn-clip", type=float, default=1.0,
                        help="Shared NVFP4 tensor clip for RTN layers (1.0 is standard RTN)")
    parser.add_argument("--head-bf16", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--plan-only", action="store_true", help="CPU metadata inventory only; creates allocation_plan.json")
    args = parser.parse_args()
    if not (0.0 < args.rtn_clip <= 1.0):
        parser.error("--rtn-clip must be in (0,1]")
    start = time.time()
    base, output = Path(args.base).resolve(), Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if base == output or base in output.parents:
        raise ValueError("output cannot be inside the source checkpoint")
    entries, shards, index = inventory(base)
    aliases = tied_aliases(entries)
    layers, excluded, memory = make_plan(entries, args.recipe, args.head_bf16)
    plan = {"recipe": args.recipe, "recipe_version": RECIPE_VERSION, "base": str(base),
            "storage": "dequantized values in original checkpoint dtype", "layers": layers,
            "excluded_tensors": excluded, "memory": memory,
            "calibration_mode": args.calibration_mode, "calib": args.calib,
            "tied_weight_aliases": aliases,
            "tied_weight_note": "GR00T Qwen3VL embedding is canonical; physically duplicated lm_head receives identical output. Parameter/storage denominators count physical checkpoint tensors, including both copies.",
            "head_bf16": args.head_bf16, "gptq_damp": args.gptq_damp}
    plan["rtn_clip"] = args.rtn_clip
    if args.plan_only:
        output.mkdir(parents=True)
        (output / "allocation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(memory, indent=2), flush=True)
        return
    need_calib = any(x["requested_method"] == "nvfp4_gptq" for x in layers.values())
    calib, calib_provenance, calib_meta = {}, None, None
    if need_calib and args.calibration_mode != "none":
        cache = Path(args.calib).resolve() if args.calib else None
        if cache and cache.is_file():
            calib = torch.load(cache, map_location="cpu", weights_only=True)
            calib_provenance = {"path": str(cache), "sha256": file_hash(cache), "layers": len(calib)}
            meta_path = cache.parent / "calib_meta.json"
            if meta_path.is_file():
                calib_meta = json.loads(meta_path.read_text())
                calib_provenance.update(metadata=calib_meta, metadata_sha256=file_hash(meta_path))
            if args.calibration_mode == "required":
                expected_arch = {"language_layers": 16, "dit_layers": 32, "vl_layers": 4}
                if not calib_meta or calib_meta.get("architecture") != expected_arch:
                    raise ValueError("required calibration must carry full 16/32/4 model metadata")
                if calib_meta.get("cache_sha256") != calib_provenance["sha256"]:
                    raise ValueError("calibration metadata hash does not match cache bytes")
                for field, filename in (("base_config_sha256", "config.json"),
                                        ("base_statistics_sha256", "statistics.json")):
                    expected_hash = file_hash(base / filename) if (base / filename).exists() else None
                    if calib_meta.get(field) != expected_hash:
                        raise ValueError(f"calibration source metadata mismatch: {filename}")
        elif args.calibration_mode == "required":
            raise FileNotFoundError("required calibration cache was not supplied or does not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
    source_files = {shard: {"bytes": (base / shard).stat().st_size, "sha256": file_hash(base / shard)}
                    for shard in shards}
    if calib_meta and calib_meta.get("base_weight_files"):
        expected = {k: v["sha256"] for k, v in calib_meta["base_weight_files"].items()}
        actual = {k: v["sha256"] for k, v in source_files.items()}
        if expected != actual:
            raise ValueError("Hessian cache was collected from different source weights")
    edits = 0
    methods = Counter()
    canonical_outputs = {}
    # Visit canonical sources before aliases even when stored across shards.
    ordered_shards = sorted(shards, key=lambda s: (not any(entries[k]["shard"] == s for k in aliases.values()), s))
    for alias, canonical in aliases.items():
        with safe_open(str(base / entries[canonical]["shard"]), framework="pt", device="cpu") as reader:
            canonical_source = reader.get_tensor(canonical)
        with safe_open(str(base / entries[alias]["shard"]), framework="pt", device="cpu") as reader:
            alias_source = reader.get_tensor(alias)
        if not torch.equal(canonical_source, alias_source):
            raise ValueError(f"tied source weights differ: {canonical} vs {alias}")
        del canonical_source, alias_source
    for shard in ordered_shards:
        tensors = load_file(str(base / shard), device="cpu")
        for key in sorted(tensors, key=lambda k: (k not in aliases.values(), k)):
            name = key.removesuffix(".weight")
            if name not in layers:
                continue
            record = layers[name]
            if key in aliases:
                canonical = aliases[key]
                original_request = record["requested_method"]
                canonical_record = layers[canonical.removesuffix(".weight")]
                record.update({k: canonical_record[k] for k in
                               ("method", "actual_method", "clip", "wmse", "fallback_reason")
                               if k in canonical_record})
                record.update(requested_method=original_request, alias_of=canonical,
                              allocation_note="Tied output copied from canonical embedding; no independent GPTQ")
                tensors[key] = canonical_outputs[canonical].clone().to(tensors[key].dtype)
                edits += 1
                methods[record["actual_method"]] += 1
                continue
            kind = record["requested_method"]
            if kind == "bf16":
                record.update(method="bf16", actual_method="bf16", wmse=0.0)
                methods["bf16"] += 1
                continue
            weight = tensors[key].to(args.device).float()
            clip, actual = 1.0, kind
            if kind == "fp8":
                quantized = fp8_e4m3_dequant(weight)
            elif kind == "nvfp4_rtn":
                quantized = nvfp4_dequant(weight, clip=args.rtn_clip)
            elif name in calib:
                print(f"[bake] GPTQ {name}: shape={tuple(weight.shape)}, calibration_rows={calib[name].get('n')}", flush=True)
                hessian = calib[name]["H"].to(args.device)
                if tuple(hessian.shape) != (weight.shape[1], weight.shape[1]):
                    raise ValueError(f"calibration shape mismatch: {name}")
                clip, _ = rtnc_best_clip(weight, hessian, CLIP_GRID)
                quantized = gptq_nvfp4(weight, hessian, clip=clip, damp=args.gptq_damp)
                record["calibration_rows"] = calib[name].get("n")
                del hessian
            else:
                if args.calibration_mode == "required" and name not in (calib_meta or {}).get("nonlinear_targets", {}):
                    raise ValueError(f"required GPTQ layer missing from H cache and not a documented non-Linear target: {name}")
                actual = "nvfp4_rtn"
                record["fallback_reason"] = ("calibration_explicitly_disabled" if args.calibration_mode == "none"
                                             else "documented_non_linear_target" if name in (calib_meta or {}).get("nonlinear_targets", {})
                                             else "layer_missing_from_cache" if calib else "no_cache_loaded")
                quantized = nvfp4_dequant(weight, clip=args.rtn_clip)
            if not torch.isfinite(quantized).all():
                raise ValueError(f"nonfinite quantized tensor: {name}")
            # Record the exact scale policy used for the output.  Keep this
            # outside the finite-value guard: a finite tensor is the normal
            # path, so provenance must be written for every quantized layer.
            if actual == "nvfp4_rtn":
                clip = args.rtn_clip
            record.update(method=actual, actual_method=actual, clip=clip,
                          wmse=float((quantized - weight).square().mean()))
            tensors[key] = quantized.to(dtype=tensors[key].dtype, device="cpu").contiguous()
            if key in aliases.values():
                canonical_outputs[key] = tensors[key]
            edits += 1
            methods[actual] += 1
            del weight, quantized
        save_file(tensors, str(stage / shard), metadata={"format": "pt"})
        del tensors
        print(f"[bake] saved {shard}; edited={edits}", flush=True)
    for path in base.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in ("ptq_recipe.json", "bake_manifest.json"):
            shutil.copy2(path, stage / path.name)
    if index:
        index["metadata"] = {**index.get("metadata", {}), "total_size": memory["source_tensor_bytes"]}
        (stage / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    plan.update(n_weights_edited=edits, actual_method_tensor_counts=dict(methods),
                calibration_provenance=calib_provenance)
    (stage / "ptq_recipe.json").write_text(json.dumps(plan, indent=2) + "\n")
    manifest = {"status": "complete", "base": str(base), "source_weight_files": source_files,
                "recipe_version": RECIPE_VERSION, "elapsed_seconds": time.time() - start,
                "implementation_sha256": {str(p.relative_to(HERE.parent.parent)): file_hash(p)
                                          for p in [Path(__file__), HERE / "quantizers.py", HERE.parent / "torch_fp4.py"]},
                "output_weight_files": {s: {"bytes": (stage / s).stat().st_size, "sha256": file_hash(stage / s)}
                                        for s in shards}}
    (stage / "bake_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stage.rename(output)
    print(f"[bake] {args.recipe} {RECIPE_VERSION}: {dict(methods)}; "
          f"target eligible compression={memory['linear_compression_x']:.4f}x; "
          f"source-dtype checkpoint -> {output}", flush=True)


if __name__ == "__main__":
    main()
