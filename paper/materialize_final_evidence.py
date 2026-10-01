#!/usr/bin/env python3
"""Materialize a portable evidence package from one verified final manifest.

The command is deliberately downstream of the experiment.  It refuses an
incomplete run, copies the paired heldout JSON byte-for-byte through
``collect_pairing_evidence``, records the selected category/PTQ provenance,
and optionally builds or copies the CPU-only training-cost evidence.  It never
edits the run directory, ``paper/evidence`` or the raw JSON paths embedded in
the copied files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
sys.path.insert(0, str(PAPER))
import collect_pairing_evidence
import extract_final_evidence


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def category_memory_kind(memory: dict[str, Any], protocol: dict[str, Any]) -> str:
    """Validate actual format counts for the frozen W4A4 contract."""
    counts = [memory.get(name + "_params") for name in ("nvfp4", "fp8", "bf16")]
    total = memory.get("linear_params")
    require(type(total) is int and total > 0 and
            all(type(n) is int and n >= 0 for n in counts) and sum(counts) == total,
            "selected category format counts do not sum to eligible elements")
    fractions = memory.get("fraction_of_eligible_params", {})
    for name, count in zip(("nvfp4", "fp8", "bf16"), counts):
        fraction = fractions.get(name)
        require(type(fraction) in (float, int) and math.isfinite(fraction) and
                math.isclose(fraction, count / total, rel_tol=0, abs_tol=1e-12),
                f"selected category {name} fraction disagrees with element count")
    mixed = counts[0] > 0 and counts[1] > 0
    if protocol.get("quantization_scope", {}).get("recipe") == "mixed":
        require(mixed, "mixed protocol requires both NVFP4 and FP8 eligible elements")
    return "mixed_nvfp4_fp8" if mixed else "nvfp4_only" if counts[0] == total else "other"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def _copy_file(source: Path, target: Path, mapping: list[dict[str, Any]], role: str) -> None:
    source = Path(source).resolve(strict=True)
    require(source.is_file() and not source.is_symlink(), f"source is not a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    source_identity = identity(source)
    require(identity(target)["sha256"] == source_identity["sha256"],
            f"source changed during copy: {source}")
    mapping.append({"source": source_identity, "published_path": str(target), "role": role})


def _copy_tree(source: Path, target: Path, mapping: list[dict[str, Any]], role: str) -> None:
    source = Path(source).resolve(strict=True)
    require(source.is_dir() and not source.is_symlink(), f"evidence directory is missing: {source}")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    require(files, f"evidence directory is empty: {source}")
    for file in files:
        relative = file.relative_to(source)
        _copy_file(file, target / relative, mapping, role)


def _selected_recipe_sources(final: dict[str, Any]) -> list[tuple[Path, str]]:
    checkpoint_value = final.get("selected_ptq_checkpoint", final.get("selected_pressure_checkpoint"))
    require(checkpoint_value, "final manifest has no selected PTQ checkpoint")
    checkpoint = Path(checkpoint_value).resolve(strict=True)
    require(checkpoint.is_dir(), f"selected PTQ checkpoint is missing: {checkpoint}")
    sources: list[tuple[Path, str]] = []
    category_recipe = checkpoint / "category_ptq_recipe.json"
    category_bake = checkpoint / "category_bake_manifest.json"
    if category_recipe.is_file() or category_bake.is_file():
        require(category_recipe.is_file() and category_bake.is_file(),
                "category PTQ checkpoint has incomplete provenance")
        sources.extend(((category_recipe, "selected_category_recipe"),
                        (category_bake, "selected_category_bake_manifest")))
        recipe = json.loads(category_recipe.read_text(encoding="utf-8"))
        # The original parent checkpoint may be intentionally cleaned after
        # the category checkpoint was baked.  Resolve its recorded path
        # without requiring the directory to exist; the byte-identical
        # provenance copies below are the fallback in that case.
        parent = Path(recipe.get("parent", "")).resolve()
    else:
        recipe_file = checkpoint / "ptq_recipe.json"
        bake_file = checkpoint / "bake_manifest.json"
        require(recipe_file.is_file() and bake_file.is_file(),
                "selected PTQ checkpoint lacks PTQ provenance")
        sources.extend(((recipe_file, "selected_ptq_recipe"),
                        (bake_file, "selected_bake_manifest")))
        recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
        parent = Path(recipe.get("base", "")).resolve()
    # Category checkpoints retain byte-identical copies of the parent recipe
    # and bake manifest.  The large parent weight shards are intentionally not
    # bundled; if the original directory was cleaned, preserve those copies
    # and mark exact parent reconstruction as an external prerequisite.
    for name in ("ptq_recipe.json", "bake_manifest.json"):
        path = (parent / name) if parent.is_dir() else (checkpoint / name)
        require(path.is_file(), f"selected PTQ parent provenance is missing: {path}")
        role = "selected_parent_" + name.removesuffix(".json")
        sources.append((path, role + ("_copy" if path.parent == checkpoint else "")))
    return sources


def _materialize_training(stage: Path, final_path: Path, protocol_path: Path,
                          orchestrator_run: Path | None, training_evidence: Path | None,
                          mapping: list[dict[str, Any]]) -> dict[str, Any]:
    """Build or copy the existing CPU-only cost package into ``stage``."""
    target = stage / "evidence" / "training"
    if orchestrator_run is not None:
        import collect_training_costs
        require(orchestrator_run.resolve() == final_path.parent.resolve(),
                "--orchestrator-run must be the final manifest's run directory")
        collect_training_costs.collect(None, None, None, target, protocol_path,
                                      orchestrator_run=orchestrator_run)
        role = "verified_training_costs"
    else:
        require(training_evidence is not None, "training evidence requires --orchestrator-run or --training-evidence")
        import collect_training_costs
        collect_training_costs.verify_published(training_evidence)
        _copy_tree(training_evidence, target, mapping, "verified_training_costs")
        role = "verified_training_costs_copy"
    # The collector writes its own manifest after all files are checked. Add
    # those files to the top-level mapping as well; no content is rewritten.
    for file in sorted(target.rglob("*")):
        if file.is_file() and not any(row["published_path"] == str(file) for row in mapping):
            mapping.append({"source": identity(file), "published_path": str(file), "role": role})
    return json.loads((target / "costs.json").read_text(encoding="utf-8"))


def materialize(final_manifest: str | Path, out: str | Path,
                recipe_inventory: str | Path | None = None,
                orchestrator_run: str | Path | None = None,
                training_evidence: str | Path | None = None) -> dict[str, Any]:
    final_path = Path(final_manifest).resolve(strict=True)
    require(final_path.name == "final_manifest.json", "--final-manifest must name final_manifest.json")
    run_dir = final_path.parent
    target = Path(out).absolute()
    require(not target.exists() and not target.is_symlink(), f"refusing existing output: {target}")
    require(bool(orchestrator_run) ^ bool(training_evidence),
            "provide exactly one of --orchestrator-run or --training-evidence")
    # This is the single gate for result numbers. It reads only the final
    # manifest, its declared protocol, and its declared heldout comparison.
    final_results = extract_final_evidence.extract(run_dir)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    protocol_path = Path(final["protocol_file"]).resolve(strict=True)
    recipe_inventory = Path(recipe_inventory or PAPER / "evidence/recipe_inventory.json").resolve(strict=True)
    require(recipe_inventory.is_file(), f"recipe inventory is missing: {recipe_inventory}")
    if orchestrator_run is not None:
        orchestrator_run = Path(orchestrator_run).resolve(strict=True)
    if training_evidence is not None:
        training_evidence = Path(training_evidence).resolve(strict=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    mapping: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="materialize-final-", dir=target.parent) as temporary:
        stage = Path(temporary) / target.name
        stage.mkdir()
        (stage / "final_results.json").write_text(json.dumps(final_results, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
        _copy_file(final_path, stage / "final_manifest.json", mapping, "verified_final_manifest")
        _copy_file(protocol_path, stage / "protocol" / protocol_path.name, mapping, "verified_protocol")
        _copy_file(recipe_inventory, stage / "evidence" / "recipe_inventory.json", mapping,
                   "frozen_recipe_inventory")
        for source, role in _selected_recipe_sources(final):
            _copy_file(source, stage / "evidence" / "selected_recipe" / source.name, mapping, role)
        selected_value = final.get("selected_ptq_checkpoint", final.get("selected_pressure_checkpoint"))
        require(selected_value, "final manifest has no selected PTQ checkpoint")
        selected_checkpoint = Path(selected_value).resolve(strict=True)
        selected_recipe_file = selected_checkpoint / "category_ptq_recipe.json"
        if selected_recipe_file.is_file():
            category_recipe = json.loads(selected_recipe_file.read_text(encoding="utf-8"))
            category_memory = category_recipe.get("memory")
            require(isinstance(category_memory, dict), "selected category recipe lacks memory accounting")
            kind = category_memory_kind(
                category_memory, json.loads(protocol_path.read_text(encoding="utf-8")))
            summary = {
                "format": "selected_category_recipe_memory_v2",
                "checkpoint": str(selected_checkpoint),
                "recipe": category_recipe.get("recipe"),
                "memory": category_memory,
                "source_recipe": "category_ptq_recipe.json",
                "source_recipe_sha256": digest(selected_recipe_file),
                "weight_allocation": kind,
                "note": "Format fractions use the recorded eligible-element denominator; runtime storage and LoRA residuals are separate.",
            }
            (stage / "evidence" / "selected_recipe" / "category_memory.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        round_dir = Path(final["heldout_round"]).resolve(strict=True)
        # Keep the published layout compatible with the paper validator and
        # compare_recovery: paired_comparison.json and heldout_<arm>/ live
        # directly under evidence/.  The source run remains nested below its
        # heldout_round directory; only the verified JSON mirrors are copied.
        pair_dir = stage / "evidence"
        collect_pairing_evidence.collect(round_dir, pair_dir, str(protocol_path))
        for file in sorted(pair_dir.rglob("*")):
            if file.is_file():
                mapping.append({"source": identity(file), "published_path": str(file),
                                "role": "verified_heldout_pairing_copy"})

        costs = _materialize_training(stage, final_path, protocol_path, orchestrator_run,
                                      training_evidence, mapping)
        # Replace the temporary absolute paths in the mapping with paths
        # relative to the final package; JSON evidence itself is untouched.
        for row in mapping:
            path_value = Path(row["published_path"])
            if path_value.is_absolute():
                row["published_path"] = str(path_value.relative_to(stage))
        manifest = {"version": 1, "status": "complete",
                    "scope": "complete final manifest; heldout pairing; selected recipe metadata; verified training costs",
                    "source_final_manifest": final_results["source"],
                    "selected_recipe": final_results["selected_recipe"],
                    "files": mapping,
                    "raw_absolute_paths_preserved": True,
                    "development_scores_included": False}
        (stage / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
        (stage / "final_costs_summary.json").write_text(json.dumps(costs, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
        # Validate every copied file before installing the directory. The
        # output remains atomic and an existing publication package is never
        # replaced.
        for row in mapping:
            path = stage / row["published_path"]
            require(path.is_file() and digest(path) == row["source"]["sha256"],
                    f"materialized file changed: {path}")
        stage.rename(target)
    return {"status": "complete", "out": str(target), "selected_recipe": final_results["selected_recipe"],
            "public_arms": list(final_results["public_arms"]), "development_scores_included": False,
            "files": len(mapping) + 2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--recipe-inventory")
    parser.add_argument("--orchestrator-run")
    parser.add_argument("--training-evidence")
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(args.final_manifest, args.out, args.recipe_inventory,
                                     args.orchestrator_run, args.training_evidence),
                           ensure_ascii=False, indent=2))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[materialize-final-evidence] ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
