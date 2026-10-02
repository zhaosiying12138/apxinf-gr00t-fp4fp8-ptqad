#!/usr/bin/env python3
"""Build the CPU-only encoding budget for the frozen v11 W4A4 recipe.

The category baker already records the exact element and metadata budgets.  We
reuse that signed memory record and add the independently stored BF16 LoRA
residual budget from the audited QAD manifest.  No model tensors are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def build(checkpoint: Path, recovery_manifest: Path, out: Path) -> dict:
    checkpoint = checkpoint.resolve(strict=True)
    recovery_manifest = recovery_manifest.resolve(strict=True)
    recipe_path = checkpoint / "category_ptq_recipe.json"
    bake_path = checkpoint / "category_bake_manifest.json"
    parent_recipe = checkpoint / "ptq_recipe.json"
    parent_bake = checkpoint / "bake_manifest.json"
    for path in (recipe_path, bake_path, parent_recipe, parent_bake):
        if not path.is_file():
            raise FileNotFoundError(path)
    recipe = json.loads(recipe_path.read_text())
    memory = recipe.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("category recipe has no memory accounting")
    if memory.get("eligible_tensor_count") != 479:
        raise ValueError("v11 requires 479 eligible weight tensors")
    if memory.get("nvfp4_params") != memory.get("linear_params") or memory.get("fp8_params") != 0 or memory.get("bf16_params") != 0:
        raise ValueError("selected v11 recipe is not all-NVFP4")
    recovery = json.loads(recovery_manifest.read_text())
    if recovery.get("scope") != "all_ordinary_linear" or recovery.get("rank") != 32 or recovery.get("alpha") != 64.0:
        raise ValueError("recovery manifest does not match v11 all_ordinary_linear rank-32 contract")
    if recovery.get("lora_linear_modules") != 468:
        raise ValueError("recovery manifest does not cover the audited 468 Linear adapters")
    tensor_elements = int(recovery["trainable_parameters"])
    if tensor_elements <= 0:
        raise ValueError("invalid trainable parameter count")
    aliases = {"backbone.model.lm_head.weight": "backbone.model.model.language_model.embed_tokens.weight"}
    entry = dict(memory)
    base_path = Path(recipe.get("source_base", {}).get("path", recipe.get("parent"))).resolve(strict=True)
    inventory = {
        "schema_version": "v11-selected-all-nvfp4-category",
        "kind": "CPU shape-derived encoding budget for the frozen W4A4 all-NVFP4 recipe; not measured disk size or latency",
        "base": recipe.get("source_base", {}).get("path", recipe.get("parent")),
        "recipe": "all_nvfp4_gptq_category",
        "selected_recipe_artifacts": {
            "memory": "paper/evidence/selected_recipe/category_memory.json",
            "ptq_recipe": "paper/evidence/selected_recipe/category_ptq_recipe.json",
            "bake_manifest": "paper/evidence/selected_recipe/category_bake_manifest.json",
        },
        "source_recipe_sha256": sha(recipe_path),
        "source_bake_manifest_sha256": sha(bake_path),
        "tied_aliases": aliases,
        "eligible_predicate": "479 eligible weight tensors: 472 rank-2 tensors and seven rank-3 category tensors; 469 ordinary and seven category Linear operators use W4A4, while three embedding/position tensors are weight-only",
        "recipes": {"all_nvfp4_gptq_category": entry},
        "ladder": ["all_nvfp4_gptq_category"],
        "recovery_residual": {
            "kind": "shape_derived_independent_bf16_lora_encoding_budget",
            "scope": "all_ordinary_linear",
            "rank": 32,
            "alpha": 64,
            "linear_modules": 468,
            "tensor_elements": tensor_elements,
            "dtype": "bfloat16",
            "bytes_per_element": 2,
            "target_bytes": tensor_elements * 2,
            "source_index_sha256": sha(base_path / "model.safetensors.index.json"),
            "source_config_sha256": sha(base_path / "config.json"),
            "linear_coverage_source": str(recovery_manifest),
            "linear_coverage_sha256": sha(recovery_manifest),
            "note": "独立保存的 BF16 LoRA A/B 旁路预算；不把它误写成 packed W4A4 checkpoint 文件大小。",
        },
        "source_identities": {
            "category_recipe": identity(recipe_path),
            "category_bake_manifest": identity(bake_path),
            "parent_recipe": identity(parent_recipe),
            "parent_bake_manifest": identity(parent_bake),
            "recovery_manifest": identity(recovery_manifest),
        },
        "publication_note": "v11 只公开全 NVFP4 W4A4 PTQ 基座及其 QAD/OPD BF16 residual 旁路；FP8 是同一分配器支持的回退格式，本次选择中为 0 元素。",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    return inventory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--recovery-manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = build(Path(args.checkpoint), Path(args.recovery_manifest), Path(args.out))
    print(json.dumps({"status": "complete", "out": str(Path(args.out).resolve()),
                      "recipe": result["recipe"],
                      "nvfp4_params": result["recipes"][result["recipe"]]["nvfp4_params"],
                      "residual_bytes": result["recovery_residual"]["target_bytes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
