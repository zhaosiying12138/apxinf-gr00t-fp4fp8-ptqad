"""Auditable NVFP4 extension for GR00T CategorySpecificLinear [C,K,N] weights.

This module never loads a model or selects a GPU. The caller chooses the device.
Each category owns one tensor scale; groups run along the actual input axis K.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from bake import CLIP_GRID, file_hash, inventory
from quantizers import gptq_nvfp4, nvfp4_dequant, layer_mse_tr
from torch_fp4 import _nvfp4_tensor_scale, _nvfp4_block_scales
from fp4_quant import encode_e2m1, decode_e2m1, encode_e4m3, decode_e4m3

VERSION = "category-k-axis-nvfp4-v1"
CACHE_VERSION = "category-bank-hessian-v1"
ACTIVE_BANK = 2
ARCHITECTURE = {"language_layers": 16, "dit_layers": 32, "vl_layers": 4}
META_FILES = ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json",
              "model.safetensors.index.json")
EXPECTED_SHAPES = {
    "action_head.state_encoder.layer1.W": [32, 132, 1024],
    "action_head.state_encoder.layer2.W": [32, 1024, 1536],
    "action_head.action_encoder.W1.W": [32, 132, 1536],
    "action_head.action_encoder.W2.W": [32, 3072, 1536],
    "action_head.action_encoder.W3.W": [32, 1536, 1536],
    "action_head.action_decoder.layer1.W": [32, 1024, 1024],
    "action_head.action_decoder.layer2.W": [32, 1024, 132],
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = file_hash(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "File changed while hashing: " + str(path))
    return {"path": str(path), "bytes": before.st_size, "sha256": digest}


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                                      sort_keys=True).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_identity(folder):
    folder = Path(folder).resolve(strict=True)
    entries, shards, _ = inventory(folder)
    require({p.name for p in folder.glob("*.safetensors")} == set(shards),
            "Unexpected or missing checkpoint shards")
    return {"path": str(folder), "weights": {s: identity(folder / s) for s in shards},
            "metadata": {s: identity(folder / s) for s in META_FILES}}, entries


def category_inventory(entries):
    found = {k: e for k, e in entries.items() if k.endswith(".W")}
    require(set(found) == set(EXPECTED_SHAPES), "Category weight allowlist differs from GR00T N1.7")
    for key, ent in found.items():
        require(ent["shape"] == EXPECTED_SHAPES[key] and ent["dtype"] == "BF16",
                "Category shape/dtype differs: " + key)
    return found


def validate_parent(folder):
    """Verify a pure, original-base PTQ parent and its byte-exact category weights."""
    folder = Path(folder).resolve(strict=True)
    require(not (folder / "merge_manifest.json").exists() and
            not (folder / "recovery_manifest.json").exists(), "Parent must be pure PTQ, without recovery")
    recipe = read_json(folder / "ptq_recipe.json")
    bake = read_json(folder / "bake_manifest.json")
    require(bake.get("status") == "complete" and
            bake.get("recipe_version") == "corrected-allocation-v2" and
            recipe.get("recipe_version") == bake["recipe_version"], "Unsupported/incomplete PTQ parent")
    base = Path(bake["base"]).resolve(strict=True)
    require(base == Path(recipe["base"]).resolve() and base != folder, "Parent base provenance differs")
    parent_id, entries = checkpoint_identity(folder)
    root_id, root_entries = checkpoint_identity(base)
    require(set(entries) == set(root_entries), "Parent changed checkpoint tensor keys")
    require(all(entries[k]["shape"] == root_entries[k]["shape"] and
                entries[k]["dtype"] == root_entries[k]["dtype"] for k in entries),
            "Parent changed checkpoint tensor shapes/dtypes")
    for records, expected, label in ((bake["source_weight_files"], root_id["weights"], "source"),
                                     (bake["output_weight_files"], parent_id["weights"], "output")):
        require(set(records) == set(expected), "Parent " + label + " shard set differs")
        for name, value in expected.items():
            require(all(records[name][field] == value[field] for field in ("bytes", "sha256")),
                    "Parent " + label + " shard identity differs: " + name)
    for filename in META_FILES[:-1]:
        require(parent_id["metadata"][filename]["sha256"] == root_id["metadata"][filename]["sha256"],
                "Parent changed model metadata: " + filename)
    require(read_json(folder / "embodiment_id.json").get("libero_sim") == ACTIVE_BANK,
            "LIBERO must resolve to bank 2")
    categories = category_inventory(entries)
    category_sources = {}
    for key, ent in categories.items():
        require(key in recipe["excluded_tensors"], "Parent already quantized category: " + key)
        with safe_open(str(folder / ent["shard"]), framework="pt", device="cpu") as reader:
            parent_value = reader.get_tensor(key)
        with safe_open(str(base / root_entries[key]["shard"]), framework="pt", device="cpu") as reader:
            source = reader.get_tensor(key)
        require(torch.equal(parent_value, source), "Parent category differs from original BF16: " + key)
        category_sources[key] = tensor_hash(source)
    parent_id["metadata"].update({name: identity(folder / name)
                                  for name in ("ptq_recipe.json", "bake_manifest.json")})
    return {"parent": parent_id, "root_bf16": root_id,
            "category_source_sha256": category_sources}, entries, recipe


def prepare_bank(weight):
    require(weight.ndim == 2 and min(weight.shape) > 0 and bool(torch.isfinite(weight).all()),
            "A category bank must be a finite [K,N] matrix")
    k, n = weight.shape
    padded_k = (k + 15) // 16 * 16
    return F.pad(weight.T.contiguous().float(), (0, padded_k - k)), k, n


def validate_hessian(entry, k, expected_calls=None):
    require(isinstance(entry, dict), "Malformed bank Hessian entry")
    h, absolute = entry["H"], entry["abs"]
    require(isinstance(h, torch.Tensor) and h.dtype == torch.float32 and
            h.device.type == "cpu" and tuple(h.shape) == (k, k), "Bank Hessian shape/dtype/device differs")
    require(isinstance(absolute, torch.Tensor) and absolute.dtype == torch.float32 and
            absolute.device.type == "cpu" and tuple(absolute.shape) == (k,), "Bank absolute statistic differs")
    require(type(entry["n"]) is int and entry["n"] > 0 and type(entry["calls"]) is int and entry["calls"] > 0,
            "Bank calibration row/call counts invalid")
    if expected_calls is not None:
        require(entry["calls"] == expected_calls, "Bank did not execute once per model forward")
    require(bool(torch.isfinite(h).all()) and bool(torch.isfinite(absolute).all()) and
            bool((absolute >= 0).all()) and bool((h.diagonal() >= 0).all()), "Invalid bank statistics")
    require(torch.allclose(h, h.T, atol=1e-5, rtol=1e-5), "Bank Hessian is asymmetric")
    require(bool((h.diagonal() > 0).any()), "Bank calibration contains no nonzero input")
    return h


def _rtn(weight, clip):
    ts = _nvfp4_tensor_scale(weight, clip)
    scales = _nvfp4_block_scales(weight.reshape(weight.shape[0], -1, 16).abs().amax(-1).double() * clip, ts)
    return nvfp4_dequant(weight, clip=clip), {"tscale": ts, "block_scales": scales}


def pack_exact(values, metadata):
    """Serialize the *chosen* GPTQ/RTN scales; never re-estimate output scales."""
    q = values.detach().cpu().numpy()
    scale = metadata["block_scales"].detach().cpu().numpy().astype(np.float32)
    ts = np.float32(metadata["tscale"].detach().cpu().item())
    encoded_scale = encode_e4m3(scale)
    require(np.array_equal(decode_e4m3(encoded_scale), scale), "Nonrepresentable E4M3 scale")
    divisor = (scale * ts)[:, :, None]
    codes = encode_e2m1(q.reshape(*scale.shape, 16) / np.where(divisor != 0, divisor, 1)).astype(np.uint8)
    decoded = decode_e2m1(codes) * divisor
    require(np.array_equal(decoded.reshape(q.shape), q), "NVFP4 packed values do not decode exactly")
    packed = ((codes[..., 0::2] & 15) | ((codes[..., 1::2] & 15) << 4)).reshape(q.shape[0], -1)
    return {"packed": torch.from_numpy(packed.copy()), "scales": torch.from_numpy(encoded_scale.copy()),
            "tscale": torch.tensor(ts, dtype=torch.float32)}


def decode_exact(encoding):
    packed = encoding["packed"].cpu().numpy()
    scale = decode_e4m3(encoding["scales"].cpu().numpy()) * np.float32(encoding["tscale"].item())
    values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    values[:, 0::2] = decode_e2m1(packed & 15)
    values[:, 1::2] = decode_e2m1(packed >> 4)
    return torch.from_numpy((values.reshape(*scale.shape, 16) * scale[..., None]).reshape(values.shape))


@torch.no_grad()
def quantize_bank(weight, entry=None, damp=0.01, rtn_clip=1.0):
    """Compare GPTQ to the best clipped RTN using the same H and saved dtype.

Selection uses calibration only. Ties retain RTN; clipping ties retain the
first (largest) clip. Any failed/nonfinite GPTQ calculation aborts the bake.
"""
    require(math.isfinite(damp) and damp > 0, "GPTQ damping must be finite and positive")
    require(math.isfinite(rtn_clip) and 0 < rtn_clip <= 1, "RTN clip must be in (0,1]")
    matrix, k, n = prepare_bank(weight)
    h = None if entry is None else validate_hessian(entry, k).to(matrix.device)
    if h is not None:
        h = F.pad(h, (0, matrix.shape[1] - k, 0, matrix.shape[1] - k))

    def score(candidate):
        # The deployed checkpoint casts back to BF16; compare those real values.
        delta = candidate.to(weight.dtype).float() - matrix
        value = float(layer_mse_tr(delta, h)) if h is not None else float(delta.square().sum())
        require(math.isfinite(value) and value >= -1e-7, "Invalid calibration objective")
        return max(value, 0.0)

    candidates, best = [], None
    for clip in CLIP_GRID if h is not None else (rtn_clip,):
        values, metadata = _rtn(matrix, clip)
        objective = score(values)
        candidates.append({"method": "nvfp4_rtn", "clip": clip, "objective": objective})
        if best is None or objective < best[0]:
            best = (objective, clip, values, metadata)
    objective, clip, values, metadata = best
    method, gptq_score = "nvfp4_rtn", None
    if h is not None:
        gptq_values, gptq_metadata = gptq_nvfp4(matrix, h, clip=clip, damp=damp, return_metadata=True)
        gptq_score = score(gptq_values)
        candidates.append({"method": "nvfp4_gptq", "clip": clip, "objective": gptq_score})
        if gptq_score < objective:
            objective, values, metadata, method = gptq_score, gptq_values, gptq_metadata, "nvfp4_gptq"
    require(bool(torch.isfinite(values).all()), "Nonfinite category quantization")
    require(not bool(torch.count_nonzero(values[:, k:])), "Padding acquired nonzero weights")
    encoding = pack_exact(values, metadata)
    output = values[:, :k].T.to(dtype=weight.dtype, device="cpu").contiguous()
    require(torch.equal(decode_exact(encoding)[:, :k].T.to(weight.dtype), output), "Encoded/output mismatch")
    record = {
        "method": method, "calibrated": entry is not None, "clip": clip,
        "selection_rule": "min same-H objective after original-dtype cast; RTN wins ties",
        "objective": "tr(delta H delta.T)" if h is not None else "weight squared error",
        "selected_objective": objective, "best_rtn_objective": best[0], "gptq_objective": gptq_score,
        "candidates": candidates, "calibration_rows": entry["n"] if entry else 0,
        "calibration_calls": entry["calls"] if entry else 0,
        "uncalibrated_reason": None if entry else "embodiment bank not observed in LIBERO calibration",
        "source_shape": [k, n], "encoded_shape": [n, matrix.shape[1]], "padding_k": matrix.shape[1] - k,
        "tensor_scale": float(metadata["tscale"]), "source_sha256": tensor_hash(weight),
        "output_sha256": tensor_hash(output),
        "weight_mse_after_cast": float((output.float() - weight.detach().cpu().float()).square().mean()),
    }
    return output, record, encoding


def memory_budget(entries, recipe):
    """Physical and known-tie-deduplicated accounting, including internal K pad."""
    categories = category_inventory(entries)
    aliases = recipe.get("tied_weight_aliases", {})
    formats, costs = {}, {}
    padding = 0
    for key, ent in entries.items():
        n = ent["params"]
        if key in categories:
            c, k, width = ent["shape"]
            padded = c * ((k + 15) // 16 * 16) * width
            padding += padded - n
            formats[key], costs[key] = "nvfp4", padded // 2 + padded // 16 + 4 * c
        elif ent["eligible"]:
            layer_name = key.removesuffix(".weight")
            if layer_name not in recipe["layers"] and key in aliases:
                layer_name = aliases[key].removesuffix(".weight")
            layer = recipe["layers"][layer_name]
            kind = layer.get("actual_method", layer["requested_method"])
            group = "nvfp4" if kind.startswith("nvfp4") else kind
            require(group in ("nvfp4", "fp8", "bf16"), "Unknown parent layer format")
            formats[key] = group
            costs[key] = n // 2 + n // 16 + 4 if group == "nvfp4" else n + ent["shape"][0] * 4 if group == "fp8" else ent["bytes"]
        else:
            formats[key], costs[key] = "excluded", ent["bytes"]

    def summarize(keys):
        counts, target = Counter(), Counter()
        for key in keys:
            counts[formats[key]] += entries[key]["params"]
            target[formats[key]] += costs[key]
        all_n = sum(counts.values())
        eligible = all_n - counts["excluded"]
        encoded = sum(target.values()) - target["excluded"]
        source = sum(entries[k]["bytes"] for k in keys)
        eligible_source = sum(entries[k]["bytes"] for k in keys if formats[k] != "excluded")
        return {"all_tensor_elements": all_n, "eligible_tensor_elements": eligible,
                "elements_by_format": {k: counts[k] for k in ("nvfp4", "fp8", "bf16")},
                "excluded_elements": counts["excluded"],
                "fraction_of_all": {k: counts[k] / all_n for k in ("nvfp4", "fp8", "bf16")},
                "fraction_of_eligible": {k: counts[k] / eligible for k in ("nvfp4", "fp8", "bf16")},
                "target_eligible_bytes": encoded, "target_full_bytes": sum(target.values()),
                "target_bytes_by_format": dict(target), "source_tensor_bytes": source,
                "eligible_source_bytes": eligible_source,
                "eligible_compression_x": eligible_source / encoded,
                "full_compression_x": source / sum(target.values())}

    physical = summarize(entries)
    distinct = summarize([k for k in entries if k not in aliases])
    distinct["omitted_physical_alias_elements"] = sum(entries[k]["params"] for k in aliases)
    distinct["note"] = "Deduplicates only explicitly recorded embedding/lm_head aliases."
    p = physical
    total_category = sum(entries[k]["params"] for k in categories)
    return {
        "all_checkpoint_params": p["all_tensor_elements"], "linear_params": p["eligible_tensor_elements"],
        "eligible_tensor_count": sum(f != "excluded" for f in formats.values()),
        "excluded_tensor_count": sum(f == "excluded" for f in formats.values()),
        "nvfp4_params": p["elements_by_format"]["nvfp4"], "fp8_params": p["elements_by_format"]["fp8"],
        "bf16_params": p["elements_by_format"]["bf16"],
        "fraction_of_eligible_params": p["fraction_of_eligible"],
        "fraction_of_all_checkpoint_params": p["fraction_of_all"],
        "source_tensor_bytes": p["source_tensor_bytes"], "eligible_source_bytes": p["eligible_source_bytes"],
        "excluded_source_bytes": p["source_tensor_bytes"] - p["eligible_source_bytes"],
        "target_eligible_bytes": p["target_eligible_bytes"], "target_full_checkpoint_bytes": p["target_full_bytes"],
        "target_bytes_by_format": {k: v for k, v in p["target_bytes_by_format"].items() if k != "excluded"},
        "linear_compression_x": p["eligible_compression_x"], "full_checkpoint_compression_x": p["full_compression_x"],
        "bf16_GB": p["eligible_source_bytes"] / 1e9, "quantized_GB": p["target_eligible_bytes"] / 1e9,
        "known_tied_alias_deduplicated": distinct,
        "category": {"weights": total_category, "banks_per_layer": 32, "active_libero_bank": ACTIVE_BANK,
                     "active_libero_weight_elements": total_category // 32, "padding_elements": padding,
                     "target_bytes": sum(costs[k] for k in categories), "tensor_scale_count": 32 * len(categories)},
        "storage_note": "Target FP4 payload plus E4M3 scale/16 and FP32 scale per category; includes internal K padding. FP8 uses FP32 row scales. Excludes runtime alignment/activations/LoRA residuals. Checkpoint shards remain original dtype. Inactive categories do not contribute to LIBERO execution.",
    }


def load_calibration(folder, provenance, entries, expected_windows=128):
    folder = Path(folder).resolve(strict=True)
    meta = read_json(folder / "calib_meta.json")
    require(meta.get("version") == CACHE_VERSION and meta.get("status") == "complete", "Invalid category cache version/status")
    require(meta["provenance"] == provenance, "Category cache was collected from another parent")
    require(meta["architecture"] == ARCHITECTURE and meta["active_bank"] == ACTIVE_BANK and
            meta["model_dtype"] == "bfloat16" and meta["accumulation_dtype"] == "float32" and
            meta["accumulation_device"] == "cpu" and meta["tf32_matmul"] is False,
            "Category calibration precision/architecture differs")
    require(type(meta["windows_consumed"]) is int and meta["windows_consumed"] == expected_windows and
            meta["windows_requested"] == expected_windows and type(meta["batch"]) is int and meta["batch"] > 0 and
            meta["forwards"] == math.ceil(expected_windows / meta["batch"]), "Category calibration is incomplete")
    cache_id = identity(folder / "calib.pt")
    require(cache_id["sha256"] == meta["cache_sha256"], "Category cache bytes changed")
    cache = torch.load(folder / "calib.pt", map_location="cpu", weights_only=True)
    require(set(cache) == set(category_inventory(entries)) == set(meta["layers"]), "Category cache coverage differs")
    for key, banks in cache.items():
        require(set(banks) == {str(ACTIVE_BANK)}, "Unexpected/missing calibrated embodiment bank")
        entry = banks[str(ACTIVE_BANK)]
        validate_hessian(entry, entries[key]["shape"][1], meta["forwards"])
        require(meta["layers"][key] == {str(ACTIVE_BANK): {"rows": entry["n"], "calls": entry["calls"],
                    "H_sha256": tensor_hash(entry["H"]), "abs_sha256": tensor_hash(entry["abs"])}},
                "Category calibration statistics differ from metadata")
    return cache, meta, {"cache": cache_id, "metadata": identity(folder / "calib_meta.json")}
