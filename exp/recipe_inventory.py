#!/usr/bin/env python3
"""Create a portable, CPU-only encoding budget from local checkpoint evidence.

Read safetensors headers for shapes and stream bytes for source SHA-256; never
instantiate model/weight tensors or load calibration tensors. CUDA is disabled
before importing the shared allocator. A new output path is always required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quant/ptq"))
sys.path.insert(0, str(ROOT / "rl"))
from bake import RECIPES, inventory, make_plan, tied_aliases
from lora_scope import in_scope


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Input changed while hashing: " + str(path))
    return digest.hexdigest()


def build(base, calibration_meta):
    base = Path(base).resolve(strict=True)
    calibration_meta = Path(calibration_meta).resolve(strict=True)
    protocol_path = ROOT / "exp/recovery_protocol.json"
    protocol = load(protocol_path)
    recovery = protocol["recovery"]
    rank, alpha, scope = recovery["rank"], recovery["alpha"], recovery["scope"]
    require(type(rank) is int and rank > 0 and alpha > 0, "Invalid recovery rank/alpha")
    meta = load(calibration_meta)
    require(meta.get("status") == "complete" and meta.get("recipe_targets") == "calib" and
            meta.get("windows_consumed") == protocol["calibration"]["windows"] and
            meta.get("windows_requested") == protocol["calibration"]["windows"],
            "Inventory requires complete full-scope formal calibration; smoke coverage is insufficient")
    require(meta.get("architecture") == {"language_layers": 16, "dit_layers": 32, "vl_layers": 4},
            "Calibration must use the complete 16/32/4 model")
    entries, shards, index = inventory(base)
    require(index is not None, "A checkpoint shard index is required for freezing provenance")
    require(set(shards) == {p.name for p in base.glob("*.safetensors")}, "Unexpected checkpoint shards")
    base_config_sha = sha(base / "config.json")
    base_statistics_sha = sha(base / "statistics.json")
    require(meta.get("base_config_sha256") == base_config_sha and
            meta.get("base_statistics_sha256") == base_statistics_sha,
            "Calibration config/statistics identity differs from the local base")
    require(set(meta.get("base_weight_files", {})) == set(shards), "Calibration source shard set differs")
    source_weights = {}
    for name in shards:
        row = {"bytes": (base / name).stat().st_size, "sha256": sha(base / name)}
        require(meta["base_weight_files"][name] == row, "Calibration source weight identity differs: " + name)
        source_weights[name] = row
    aliases = tied_aliases(entries)
    require(meta.get("tied_weight_aliases") == aliases, "Calibration alias coverage differs")
    eligible = {key.removesuffix(".weight") for key, row in entries.items()
                if row["eligible"] and key not in aliases}
    nonlinear = meta.get("nonlinear_targets", {})
    require(isinstance(nonlinear, dict) and set(nonlinear) <= eligible and
            all(isinstance(value, str) and value and value != "Linear" for value in nonlinear.values()),
            "Invalid nonlinear exclusions")
    expected_linear = eligible - set(nonlinear)
    rows, calls = meta.get("rows_per_layer", {}), meta.get("calls_per_layer", {})
    require(set(rows) == expected_linear and set(calls) == expected_linear and
            meta.get("n_layers") == len(expected_linear),
            "Full-scope Linear coverage is missing or contains unexpected modules")
    require(expected_linear and all(type(rows[k]) is int and rows[k] > 0 and
            type(calls[k]) is int and calls[k] > 0 for k in expected_linear),
            "Every Linear needs positive observed rows and calls")
    require(type(meta.get("forwards")) is int and meta["forwards"] > 0 and
            all(v == meta["forwards"] for v in calls.values()),
            "Linear call coverage differs from completed calibration forwards")
    selected = sorted(name for name in expected_linear if in_scope(name, scope))
    require(selected, "Recovery scope has no observed Linear modules")
    residual_layers = {}
    for name in selected:
        shape = entries[name + ".weight"]["shape"]
        residual_layers[name] = {"shape": shape, "tensor_elements": rank * sum(shape)}
    residual_elements = sum(row["tensor_elements"] for row in residual_layers.values())
    recipes = {}
    omitted_bytes = sum(entries[alias]["bytes"] for alias in aliases)
    for recipe in RECIPES:
        _, _, memory = make_plan(entries, recipe)
        memory["known_tied_alias_deduplicated"]["source_tensor_bytes"] = memory["source_tensor_bytes"] - omitted_bytes
        recipes[recipe] = memory
    ladder = protocol["quantization_ladder"]
    previous = set()
    for recipe in ladder:
        layers, _, _ = make_plan(entries, recipe)
        current = {key for key, row in layers.items() if row["requested_method"].startswith("nvfp4")}
        require(previous <= current, "FP4 recipe coverage is not nested")
        previous = current
    index_sha = sha(base / "model.safetensors.index.json")
    metadata_sha = sha(calibration_meta)
    source_paths = ["exp/recipe_inventory.py", "quant/ptq/bake.py", "rl/lora_scope.py", "exp/recovery_protocol.json"]
    return {
        "kind": "CPU shape-derived encoding budget, not measured disk size or latency",
        "base": str(base), "bake_sha256": sha(ROOT / "quant/ptq/bake.py"),
        "source_sha256": {p: sha(ROOT / p) for p in source_paths},
        "source_index_sha256": index_sha, "source_config_sha256": base_config_sha,
        "source_statistics_sha256": base_statistics_sha, "base_weight_files": source_weights,
        "tied_aliases": aliases,
        "alias_scope_note": "Only explicit allocator aliases are removed; headers do not prove tensor value equality. Bake/merge separately verify tied values.",
        "eligible_predicate": ".weight, rank 2, input width divisible by 16",
        "recipes": recipes, "ladder": ladder,
        "calibration_provenance": {
            "path": str(calibration_meta), "sha256": metadata_sha,
            "recorded_base_path": meta["base"], "local_base_path": str(base),
            "identity_check": "config/statistics and full serialized shard SHA-256 match; absolute path relocation is allowed",
            "linear_modules": len(expected_linear), "nonlinear_targets": nonlinear,
            "weights_loaded_as_tensors": False, "calibration_tensor_cache_loaded": False,
        },
        "recovery_residual": {
            "kind": "shape_derived_independent_bf16_lora_encoding_budget",
            "scope": scope, "rank": rank, "alpha": alpha, "linear_modules": len(selected),
            "tensor_elements": residual_elements, "dtype": "bfloat16", "bytes_per_element": 2,
            "target_bytes": 2 * residual_elements, "source_index_sha256": index_sha,
            "source_config_sha256": base_config_sha,
            "linear_coverage_source": str(calibration_meta), "linear_coverage_sha256": metadata_sha,
            "modules": residual_layers,
            "note": "Same hypothetical separately stored residual budget for each base; does not assert all recipes underwent recovery or actual packed deployment.",
        },
    }


def generate(base, calibration_meta, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Refusing existing output: " + str(out))
    require(Path(base).resolve() not in out.parents, "Output must not modify the source checkpoint directory")
    report = build(base, calibration_meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--calibration-meta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = generate(args.base, args.calibration_meta, args.out)
    residual = report["recovery_residual"]
    print(json.dumps({"out": str(Path(args.out).resolve()), "base": report["base"],
        "linear_modules": residual["linear_modules"], "lora_elements": residual["tensor_elements"],
        "bf16_residual_bytes": residual["target_bytes"], "cuda_used": False}))


if __name__ == "__main__":
    main()
