"""Bake the seven CategorySpecificLinear weights into a new PTQ checkpoint.

The checkpoint remains source-dtype/dequantized for compatibility with the
existing model loader. ``category_bake_manifest.json`` records the native
NVFP4 encoding metadata and every parent/output identity; no parent directory
is modified or overwritten.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from bake import file_hash, inventory
from category_fp4 import (VERSION, ACTIVE_BANK, EXPECTED_SHAPES, checkpoint_identity,
                           category_inventory, identity, load_calibration, memory_budget,
                           quantize_bank, tensor_hash, validate_parent)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, help="Pure PTQ parent (head_lang_vision or calib)")
    parser.add_argument("--calib", help="Category calibration directory; required for bank 2 GPTQ comparison")
    parser.add_argument("--out", required=True, help="New checkpoint directory")
    parser.add_argument("--expected-windows", type=int, default=128)
    parser.add_argument("--gptq-damp", type=float, default=0.01)
    args = parser.parse_args()
    parent = Path(args.parent).resolve(strict=True)
    output = Path(args.out).resolve()
    if output.exists() or output == parent or parent in output.parents:
        raise FileExistsError("Refusing to overwrite or nest category PTQ output")
    source, entries, parent_recipe = validate_parent(parent)
    categories = category_inventory(entries)
    cache = cache_records = cache_meta = None
    if args.calib:
        # Validate the full parent provenance before accepting any H matrix.
        calib_dir = Path(args.calib).resolve(strict=True)
        expected_provenance = {"parent": source,
                               "parent_recipe_sha256": identity(parent / "ptq_recipe.json")["sha256"],
                               "parent_bake_sha256": identity(parent / "bake_manifest.json")["sha256"]}
        cache, cache_meta, cache_records = load_calibration(calib_dir, expected_provenance,
                                                            entries, args.expected_windows)
    elif args.calib is None:
        raise ValueError("A complete category calibration cache is required for the active LIBERO bank")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
    records, methods = {}, Counter()
    category_sources = {}
    category_shards = {key: ent["shard"] for key, ent in categories.items()}
    try:
        all_shards = sorted(set(ent["shard"] for ent in entries.values()))
        category_shard_names = set(category_shards.values())
        for shard in all_shards:
            if shard not in category_shard_names:
                # Category banks occupy only one or two physical shards.  The
                # remaining parent shards still belong in the output index.
                shutil.copy2(parent / shard, stage / shard)
                continue
            tensors = load_file(str(parent / shard), device="cpu")
            for key in sorted(categories):
                if category_shards[key] != shard:
                    continue
                value = tensors[key]
                category_sources[key] = tensor_hash(value)
                bank_records = cache[key] if cache is not None else {}
                # Inactive embodiment banks have no observed H. Quantize each
                # bank with RTN, while bank 2 may select GPTQ against its H.
                output_banks = []
                for bank in range(value.shape[0]):
                    bank_tensor = value[bank]
                    entry = bank_records.get(str(ACTIVE_BANK)) if bank == ACTIVE_BANK else None
                    quantized, record, encoding = quantize_bank(bank_tensor, entry, args.gptq_damp)
                    output_banks.append(quantized)
                    output_record = dict(record)
                    output_record.update({"bank": bank, "source_key": key,
                                          "source_dtype": str(value.dtype),
                                          "output_dtype": str(quantized.dtype),
                                          "inactive_bank": bank != ACTIVE_BANK})
                    output_record.pop("source_sha256", None)
                    output_record["source_sha256"] = tensor_hash(bank_tensor)
                    output_record["output_sha256"] = tensor_hash(quantized)
                    records.setdefault(key, {"shape": list(value.shape), "banks": []})["banks"].append(output_record)
                    methods[record["method"]] += 1
                replacement = torch.stack(output_banks).to(dtype=value.dtype).contiguous()
                require(list(replacement.shape) == list(value.shape) and replacement.dtype == value.dtype,
                        "Category output shape/dtype changed: " + key)
                tensors[key] = replacement
            save_file(tensors, str(stage / shard), metadata={"format": "pt"})
        # Preserve every non-weight artifact from the parent. The source PTQ
        # recipe/bake manifests remain available for chain verification.
        for path in parent.iterdir():
            if path.is_file() and not path.name.endswith(".safetensors"):
                shutil.copy2(path, stage / path.name)
        index = read_json(parent / "model.safetensors.index.json")
        (stage / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        budget = memory_budget(entries, parent_recipe)
        category_recipe = {
            "version": VERSION, "recipe": "category_nvfp4_extension", "parent": str(parent),
            "parent_recipe_sha256": identity(parent / "ptq_recipe.json")["sha256"],
            "parent_bake_manifest_sha256": identity(parent / "bake_manifest.json")["sha256"],
            "source_base": source["root_bf16"], "active_libero_bank": ACTIVE_BANK,
            "categories": records, "memory": budget,
            "source_category_sha256": category_sources,
            "calibration": cache_records, "calibration_metadata": cache_meta,
            "implementation_sha256": identity(Path(__file__))["sha256"],
        }
        (stage / "category_ptq_recipe.json").write_text(json.dumps(category_recipe, indent=2) + "\n")
        manifest = {
            "status": "complete", "version": VERSION, "parent": source,
            "parent_recipe": identity(parent / "ptq_recipe.json"),
            "parent_bake_manifest": identity(parent / "bake_manifest.json"),
            "root_bf16": source["root_bf16"], "source_category_sha256": category_sources,
            "output_category_sha256": {},
            "category_recipe": identity(stage / "category_ptq_recipe.json"),
            "methods": dict(methods), "active_libero_bank": ACTIVE_BANK,
            "memory": budget, "implementation_sha256": identity(Path(__file__))["sha256"],
        }
        # Populate output identities only after all replacement tensors exist.
        manifest["output_weights"] = {name: identity(stage / name) for name in sorted(set(category_shards.values()))}
        manifest["output_category_sha256"] = {key: records[key]["banks"][ACTIVE_BANK]["output_sha256"] for key in records}
        (stage / "category_bake_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Verify headers, index membership and unchanged non-category tensor
        # values before publishing the new directory atomically.
        final_entries, final_shards, _ = inventory(stage)
        require(set(final_entries) == set(entries) and final_shards == sorted(set(entries[key]["shard"] for key in entries)),
                "Output tensor inventory differs from parent")
        for key, ent in final_entries.items():
            require(ent["shape"] == entries[key]["shape"] and ent["dtype"] == entries[key]["dtype"],
                    "Output tensor header differs: " + key)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"status": "complete", "out": str(output), "methods": dict(methods),
                      "fp4_fraction_deduplicated": budget["known_tied_alias_deduplicated"]["fraction_of_all"]["nvfp4"],
                      "active_bank": ACTIVE_BANK}, indent=2), flush=True)


if __name__ == "__main__":
    main()
