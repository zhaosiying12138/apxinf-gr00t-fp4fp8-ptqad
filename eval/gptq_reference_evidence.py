"""CPU-only provenance checks for the single captured-calibration GPTQ arm.

The caller verifies the supplemental plan and frozen source inventory. This
module binds the resulting checkpoints and Hessians back to that plan; it
does not load a policy, rerun quantization, or read evaluation outcomes.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import torch
from safetensors import safe_open

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "quant" / "ptq"))
from bake import CLIP_GRID, RECIPE_VERSION, inventory, make_plan, tied_aliases
from category_fp4 import (
    ACTIVE_BANK, ARCHITECTURE, CACHE_VERSION, VERSION, category_inventory,
    checkpoint_identity, identity, load_calibration, memory_budget, tensor_hash,
    validate_parent,
)

NONLINEAR = (
    "backbone.model.model.language_model.embed_tokens",
    "backbone.model.model.visual.pos_embed",
    "action_head.position_embedding",
)
METADATA = ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def resolve(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def weights_only(records):
    return {name: {field: row[field] for field in ("bytes", "sha256")}
            for name, row in records.items()}


def check_identity(record, path, *, staging=False, actual=None):
    """Staging receipts retain obsolete paths; bind their basename and bytes."""
    actual = identity(path) if actual is None else actual
    recorded = Path(record["path"])
    require((recorded.name == Path(path).name if staging else recorded.resolve() == Path(path).resolve()),
            "Artifact path differs: " + str(path))
    require(all(record.get(key) == actual[key] for key in ("bytes", "sha256")),
            "Artifact identity differs: " + str(path))
    return actual


def source_hash(plan, name):
    return plan["preparation_source_files"][name]["sha256"]


def check_calibration(meta, plan, capture_record, version, collector):
    expected = {
        "version": version, "status": "complete", "dataset": None,
        "architecture": ARCHITECTURE, "model_dtype": "bfloat16",
        "accumulation_dtype": "float32", "accumulation_device": "cpu",
        "tf32_matmul": False, "batch": plan["calibration"]["batch"],
        "seed": plan["calibration"]["seed"],
        "windows_requested": plan["calibration"]["windows"],
        "windows_consumed": plan["calibration"]["windows"],
        "forwards": plan["calibration"]["windows"],
        "implementation_sha256": source_hash(plan, collector),
        "captured_input_provenance": capture_record,
        "windows_are_unique": "unique capture files; one pass; not independent episodes",
    }
    for key, value in expected.items():
        require(key in meta and meta[key] == value and
                (type(value) is not int or type(meta[key]) is int),
                "Captured calibration metadata differs: " + key)
    require(meta["batch"] == 1, "Captured calibration must use batch one")


def check_ordinary(parent, folder, recipe, entries, source, plan, capture_record, root):
    meta_path, cache_path = folder / "calib_meta.json", folder / "calib.pt"
    meta = read(meta_path)
    check_calibration(meta, plan, capture_record, "full-model-cpu-hessian-v2", "quant/ptq/collector.py")
    root_id = source["root_bf16"]
    require(Path(meta["base"]).resolve() == Path(root_id["path"]).resolve(), "Ordinary H base differs")
    require(meta["base_weight_files"] == weights_only(root_id["weights"]), "Ordinary H root weights differ")
    for field, name in (("base_config_sha256", "config.json"), ("base_statistics_sha256", "statistics.json")):
        require(meta[field] == plan["teacher_metadata"][name], "Ordinary H source metadata differs: " + name)
    require(meta.get("recipe_targets") == "calib", "Ordinary H is not full calib coverage")
    for key, value in {"recipe": "calib", "recipe_version": RECIPE_VERSION,
                       "calibration_mode": "required", "head_bf16": False,
                       "gptq_damp": plan["calibration"]["gptq_damp"], "rtn_clip": 1.0}.items():
        require(recipe.get(key) == value, "Ordinary recipe differs: " + key)
    require(list(CLIP_GRID) == plan["calibration"]["clip_grid"], "Frozen clip grid differs")
    require(resolve(root, recipe["calib"]) == cache_path.resolve(), "Ordinary recipe uses another H cache")
    cp = recipe["calibration_provenance"]
    cache_id, meta_id = identity(cache_path), identity(meta_path)
    require(Path(cp["path"]).resolve() == cache_path.resolve(), "Ordinary calibration provenance path differs")
    require(cp["sha256"] == meta["cache_sha256"] == cache_id["sha256"], "Ordinary H cache SHA differs")
    require(cp["metadata"] == meta and cp["metadata_sha256"] == meta_id["sha256"],
            "Ordinary recipe does not bind its calibration metadata")
    planned, excluded, _ = make_plan(entries, "calib", False)
    require(set(recipe["layers"]) == set(planned) and recipe["excluded_tensors"] == excluded,
            "Ordinary recipe tensor coverage differs")
    aliases = tied_aliases(entries)
    require(recipe["tied_weight_aliases"] == meta["tied_weight_aliases"] == aliases, "Ordinary aliases differ")
    nonlinear = {name: "Embedding" for name in NONLINEAR if name + ".weight" in entries}
    require(meta["nonlinear_targets"] == nonlinear, "Ordinary non-Linear fallback allowlist differs")
    h_names = set(planned) - set(nonlinear) - {key.removesuffix(".weight") for key in aliases}
    cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    require(isinstance(cache, dict) and set(cache) == h_names == set(meta["rows_per_layer"]) == set(meta["calls_per_layer"]),
            "Ordinary H layer coverage differs")
    require(cp["layers"] == meta["n_layers"] == len(cache), "Ordinary H layer count differs")
    for name, row in cache.items():
        width = entries[name + ".weight"]["shape"][1]
        require(torch.is_tensor(row["H"]) and row["H"].shape == (width, width) and row["H"].dtype == torch.float32 and
                torch.is_tensor(row["abs"]) and row["abs"].shape == (width,) and row["abs"].dtype == torch.float32,
                "Ordinary H tensor shape/dtype differs: " + name)
        require(type(row["n"]) is int and row["n"] > 0 and row["n"] == meta["rows_per_layer"][name] and
                type(row["calls"]) is int and row["calls"] == meta["calls_per_layer"][name] == meta["forwards"],
                "Ordinary H accounting differs: " + name)
    counts = Counter()
    for name, row in recipe["layers"].items():
        require(all(row.get(key) == value for key, value in planned[name].items()
                    if key not in ("method", "actual_method")), "Ordinary allocation differs: " + name)
        method = row["actual_method"]
        require(row["method"] == method and method in ("nvfp4_gptq", "nvfp4_rtn"), "Ordinary method differs: " + name)
        key = name + ".weight"
        if key in aliases:
            canonical = recipe["layers"][aliases[key].removesuffix(".weight")]
            require(row.get("alias_of") == aliases[key] and method == canonical["actual_method"], "Tied alias method differs")
        elif name in nonlinear:
            require(method == "nvfp4_rtn" and row.get("fallback_reason") == "documented_non_linear_target",
                    "Unexpected ordinary fallback: " + name)
        else:
            require(method == "nvfp4_gptq" and row.get("calibration_rows") == meta["rows_per_layer"][name] and
                    row.get("clip") in plan["calibration"]["clip_grid"], "Ordinary layer did not use frozen GPTQ: " + name)
        counts[method] += 1
    require(dict(counts) == recipe["actual_method_tensor_counts"] and recipe["n_weights_edited"] == len(planned),
            "Ordinary actual method accounting differs")
    bake = read(parent / "bake_manifest.json")
    for name in ("quant/ptq/bake.py", "quant/ptq/quantizers.py", "quant/torch_fp4.py"):
        require(bake["implementation_sha256"][name] == source_hash(plan, name), "Ordinary bake source differs: " + name)
    return meta, dict(counts), {"cache": cache_id, "metadata": meta_id}


def check_final_tensors(parent, checkpoint, entries, final_entries, category_recipe, manifest):
    categories = category_inventory(entries)
    require(set(final_entries) == set(entries) and all(
        all(final_entries[key][field] == row[field] for field in ("shape", "dtype", "shard"))
        for key, row in entries.items()), "Final checkpoint tensor inventory differs")
    require(set(category_recipe["categories"]) == set(categories), "Category recipe coverage differs")
    counts, active_counts, active_hashes = Counter(), Counter(), {}
    changed_shards = {row["shard"] for row in categories.values()}
    for shard in sorted(changed_shards):
        with safe_open(str(parent / shard), framework="pt", device="cpu") as before, safe_open(
                str(checkpoint / shard), framework="pt", device="cpu") as after:
            for key, ent in entries.items():
                if ent["shard"] != shard:
                    continue
                old, new = before.get_tensor(key), after.get_tensor(key)
                if key not in categories:
                    require(torch.equal(old, new), "Category bake changed non-category tensor: " + key)
                    continue
                category = category_recipe["categories"][key]
                require(category["shape"] == ent["shape"], "Category recorded shape differs: " + key)
                banks = category["banks"]
                require(len(banks) == new.shape[0] and [row["bank"] for row in banks] == list(range(new.shape[0])),
                        "Category bank inventory differs: " + key)
                for bank, row in enumerate(banks):
                    method = row["method"]
                    active = bank == ACTIVE_BANK
                    require(row["source_key"] == key and row["source_sha256"] == tensor_hash(old[bank]) and
                            row["output_sha256"] == tensor_hash(new[bank]), "Category bank bytes differ: " + key)
                    require(row["source_dtype"] == str(old.dtype) and row["output_dtype"] == str(new.dtype) and
                            row["inactive_bank"] is (not active) and row["calibrated"] is active,
                            "Category bank provenance differs: " + key)
                    require(method in ("nvfp4_gptq", "nvfp4_rtn") and (active or method == "nvfp4_rtn"),
                            "Invalid category bank method: " + key)
                    if active:
                        h = category_recipe["calibration_metadata"]["layers"][key][str(ACTIVE_BANK)]
                        require(row["calibration_rows"] == h["rows"] and row["calibration_calls"] == h["calls"],
                                "Active category calibration accounting differs: " + key)
                        require(all(type(row.get(field)) in (int, float) and math.isfinite(row[field]) and row[field] >= 0
                                    for field in ("gptq_objective", "best_rtn_objective", "selected_objective")),
                                "Invalid active category calibration objective: " + key)
                        selected = "nvfp4_gptq" if row["gptq_objective"] < row["best_rtn_objective"] else "nvfp4_rtn"
                        require(method == selected and row["selected_objective"] == min(row["gptq_objective"], row["best_rtn_objective"]),
                                "Active category method does not follow calibration selection: " + key)
                        active_counts[method] += 1
                        active_hashes[key] = row["output_sha256"]
                    else:
                        require(row["calibration_rows"] == row["calibration_calls"] == 0,
                                "Inactive category unexpectedly used calibration: " + key)
                    counts[method] += 1
    require(dict(counts) == manifest["methods"] and active_hashes == manifest["output_category_sha256"],
            "Category actual method/output accounting differs")
    return dict(counts), dict(active_counts)


def verify_category_invocation(plan, root, output):
    """The frozen category producer does not serialize its damping argument."""
    receipt_path = output / "category_bake_invocation.json"
    receipt = read(receipt_path)
    require(receipt.get("format") == "gptq_category_bake_invocation_v1" and
            receipt.get("status") == "complete" and type(receipt.get("returncode")) is int and
            receipt["returncode"] == 0, "Category bake invocation is incomplete")
    protocol_path = root / "exp/gptq_reference_protocol_v12.json"
    require(read(protocol_path) == plan and receipt["protocol_sha256"] == identity(protocol_path)["sha256"],
            "Category bake invocation used another supplement protocol")
    producer = (root / "quant/ptq/bake_category.py").resolve()
    require(receipt["producer_sha256"] == source_hash(plan, "quant/ptq/bake_category.py"),
            "Category bake invocation producer differs")
    require(Path(receipt["cwd"]).resolve() == root, "Category bake invocation working directory differs")
    command = receipt["command"]
    require(isinstance(command, list) and bool(command) and all(isinstance(arg, str) for arg in command),
            "Category bake invocation command is invalid")
    main_run = resolve(root, plan["main_run"])
    launcher = read(main_run / "run_manifest.json")["python"]
    # Different virtual environments can resolve to the same uv interpreter.
    # Preserve the launcher path so their site-packages cannot be confused.
    require(Path(command[0]).expanduser().absolute() == Path(launcher).expanduser().absolute(),
            "Category bake interpreter differs from main run")
    wrapper = check_identity(receipt["wrapper"], root / "exp/bake_gptq_reference_category.py")
    final_manifest = check_identity(receipt["final_manifest"], main_run / "final_manifest.json")
    expected_args = [str(producer), "--parent", str((output / "ordinary_parent").resolve()),
                     "--calib", str((output / "category_h").resolve()),
                     "--out", str((output / "w4a4_category").resolve()),
                     "--expected-windows", str(plan["calibration"]["windows"]),
                     "--method", plan["calibration"]["category_method"],
                     "--gptq-damp", str(plan["calibration"]["gptq_damp"])]
    require(command[1:] == expected_args, "Category bake invocation arguments differ from frozen recipe")
    log_path = Path(receipt["log"]["path"]).resolve()
    require(log_path.is_relative_to(output) and log_path != receipt_path,
            "Category bake log must be a separate supplemental artifact")
    log = check_identity(receipt["log"], log_path)
    return {"receipt": identity(receipt_path), "command": command, "log": log,
            "producer_sha256": receipt["producer_sha256"], "wrapper": wrapper,
            "final_manifest": final_manifest}


def verify_gptq(plan, root: Path) -> dict:
    """Verify the fixed output layout under plan.execution.output_root."""
    root = Path(root).resolve()
    output = resolve(root, plan["execution"]["output_root"])
    invocation = verify_category_invocation(plan, root, output)
    parent, checkpoint = output / "ordinary_parent", output / "w4a4_category"
    ordinary_h, category_h = output / "ordinary_h", output / "category_h"
    require(not any((checkpoint / name).exists() for name in ("merge_manifest.json", "recovery_manifest.json")),
            "GPTQ reference must not contain recovery/LoRA")
    source, entries, recipe = validate_parent(parent)
    root_id = source["root_bf16"]
    require(weights_only(root_id["weights"]) == plan["teacher_weights"], "GPTQ root differs from frozen teacher weights")
    for name, digest in plan["teacher_metadata"].items():
        require(root_id["metadata"][name]["sha256"] == digest, "GPTQ root metadata differs: " + name)
    capture_path = resolve(root, plan["capture_manifest"]["path"])
    capture_id = check_identity({**plan["capture_manifest"], "path": str(capture_path)}, capture_path)
    capture = read(capture_path)
    require(Path(capture["source_audit"]["teacher"]).resolve() == Path(root_id["path"]).resolve() and
            capture["source_audit"]["teacher_weights"] == plan["teacher_weights"] and
            capture["teacher_metadata"] == plan["teacher_metadata"], "Frozen capture teacher differs from GPTQ root")
    capture_record = {**capture_id, **{key: capture[key] for key in (
        "source_audit", "protocol_file", "order", "calibration_forward", "implementation_sha256")}}
    ordinary_meta, ordinary_counts, ordinary_ids = check_ordinary(
        parent, ordinary_h, recipe, entries, source, plan, capture_record, root)
    expected_provenance = {"parent": source,
                           "parent_recipe_sha256": identity(parent / "ptq_recipe.json")["sha256"],
                           "parent_bake_sha256": identity(parent / "bake_manifest.json")["sha256"]}
    _, category_meta, category_ids = load_calibration(
        category_h, expected_provenance, entries, plan["calibration"]["windows"])
    check_calibration(category_meta, plan, capture_record, CACHE_VERSION, "quant/ptq/collector_category.py")
    require(Path(category_meta["parent"]).resolve() == parent.resolve() and
            category_meta["source_category_sha256"] == source["category_source_sha256"], "Category H source differs")
    manifest, cat_recipe = read(checkpoint / "category_bake_manifest.json"), read(checkpoint / "category_ptq_recipe.json")
    require(manifest["status"] == "complete" and manifest["version"] == cat_recipe["version"] == VERSION and
            cat_recipe["recipe"] == "category_nvfp4_extension", "Incomplete/unsupported category bake")
    require(manifest["parent"] == source and manifest["root_bf16"] == cat_recipe["source_base"] == root_id and
            Path(cat_recipe["parent"]).resolve() == parent.resolve(), "Category bake parent/root chain differs")
    for name, field in (("ptq_recipe.json", "parent_recipe"), ("bake_manifest.json", "parent_bake_manifest")):
        parent_file = identity(parent / name)
        check_identity(manifest[field], parent / name, actual=parent_file)
        recipe_field = "parent_recipe_sha256" if field == "parent_recipe" else "parent_bake_manifest_sha256"
        require(cat_recipe[recipe_field] == parent_file["sha256"] and identity(checkpoint / name)["sha256"] == parent_file["sha256"],
                "Category copied parent metadata differs: " + name)
    check_identity(manifest["category_recipe"], checkpoint / "category_ptq_recipe.json", staging=True)
    require(cat_recipe["calibration"] == category_ids and cat_recipe["calibration_metadata"] == category_meta,
            "Category recipe uses another calibration cache")
    for document in (manifest, cat_recipe):
        require(document["active_bank_method"] == plan["calibration"]["category_method"] == "gptq_active" and
                document["active_libero_bank"] == ACTIVE_BANK and document["rtn_clip"] == 1.0 and
                document["source_category_sha256"] == source["category_source_sha256"] and
                document["implementation_sha256"] == source_hash(plan, "quant/ptq/bake_category.py"),
                "Category bake method/source differs")
    final_id, final_entries = checkpoint_identity(checkpoint)
    for name in METADATA:
        require(final_id["metadata"][name]["sha256"] == root_id["metadata"][name]["sha256"], "Final metadata differs: " + name)
    changed_shards = {row["shard"] for row in category_inventory(entries).values()}
    require(set(manifest["output_weights"]) == changed_shards and set(final_id["weights"]) == set(source["parent"]["weights"]),
            "Category output shard coverage differs")
    for name, actual in final_id["weights"].items():
        if name in changed_shards:
            check_identity(manifest["output_weights"][name], checkpoint / name, staging=True, actual=actual)
        else:
            require(weights_only({name: actual}) == weights_only({name: source["parent"]["weights"][name]}),
                    "Category bake changed an ordinary shard: " + name)
    category_counts, active_counts = check_final_tensors(parent, checkpoint, entries, final_entries, cat_recipe, manifest)
    memory = memory_budget(entries, recipe)
    require(cat_recipe["memory"] == manifest["memory"] == memory and
            memory["eligible_tensor_count"] == plan["quantization"]["eligible_weight_tensors"] and
            memory["fp8_params"] == memory["bf16_params"] == 0, "Final NVFP4 coverage/memory differs")
    return {"checkpoint": str(checkpoint.resolve()), "weight_identity": final_id,
            "root_identity": root_id, "parent_identity": source["parent"],
            "category_bake_invocation": invocation,
            "calibration_provenance": {"capture": capture_record,
                "ordinary": {**ordinary_ids, "metadata_content": ordinary_meta},
                "category": {**category_ids, "metadata_content": category_meta}},
            "actual_method_counts": {"ordinary": ordinary_counts, "category_banks": category_counts,
                                     "active_category_banks": active_counts}, "memory": memory,
            "verification": "CPU file, tensor, calibration and source identity checks; no model execution"}
