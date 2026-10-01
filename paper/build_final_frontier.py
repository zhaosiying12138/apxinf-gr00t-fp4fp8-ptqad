#!/usr/bin/env python3
"""Build the v11 W4A4 frontier from verified final evidence.

This is the only entry point used to create ``frontier_comparison.json`` for
the v11 paper.  It consumes the heldout-only output of
``extract_final_evidence.py`` and the paired comparison named by that output;
it never scans development candidates or an old ``references`` directory.
The command is intentionally fail-closed while the recovery run is still in
progress, and refuses to overwrite an existing publication file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")
RECOVERY = {"qad", "continued_qad", "qad_opd"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def finite(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"{label} is not finite")
    return float(value)


def validate_arm(name: str, row: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(row, dict), f"{name}: arm is not an object")
    successes, count = row.get("successes"), row.get("count")
    require(type(successes) is int and type(count) is int and count == 160 and
            0 <= successes <= count, f"{name}: requires a complete 160-episode arm")
    tasks = row.get("per_task")
    require(isinstance(tasks, dict) and len(tasks) == 10, f"{name}: requires ten task totals")
    episodes = 0
    task_successes = 0
    for task, value in tasks.items():
        require(isinstance(value, dict), f"{name}/{task}: malformed task row")
        n, k = value.get("episodes"), value.get("successes")
        require(type(n) is int and n == 16 and type(k) is int and 0 <= k <= n,
                f"{name}/{task}: requires sixteen episodes")
        rate = finite(value.get("success_rate"), f"{name}/{task}.success_rate")
        require(math.isclose(rate, k / n, rel_tol=0, abs_tol=1e-12),
                f"{name}/{task}: success rate disagrees")
        episodes += n
        task_successes += k
    require((episodes, task_successes) == (count, successes),
            f"{name}: per-task totals disagree")
    macro = finite(row.get("macro_success_rate"), f"{name}.macro_success_rate")
    expected_macro = sum(value["success_rate"] for value in tasks.values()) / 10
    require(math.isclose(macro, expected_macro, rel_tol=0, abs_tol=1e-12),
            f"{name}: macro rate disagrees")
    return {"successes": successes, "count": count, "macro_success_rate": macro,
            "per_task": tasks}


def final_arm(final: dict[str, Any], name: str) -> dict[str, Any]:
    if name in final.get("public_arms", {}):
        return final["public_arms"][name]
    require(name == "continued_qad", "final_results omits continued_qad control")
    return final.get("control", {}).get("continued_qad", {})


def validate_final_and_pair(final_path: Path, final: dict[str, Any], pair_path: Path,
                            pair: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(final.get("format") == "publication_final_results_v1" and
            final.get("status") == "complete", "final_results is not complete v11 evidence")
    require(final.get("selected_recipe") == "all_nvfp4_gptq_category",
            "v11 frontier requires the selected all-NVFP4 category recipe")
    source = final.get("source", {}).get("heldout_comparison", {})
    require(source.get("bytes") == pair_path.stat().st_size and source.get("sha256") == digest(pair_path),
            "final_results heldout comparison identity differs")
    require(pair.get("environment_pairing_verified") is True and
            pair.get("protocol_consistency_verified") is True and
            pair.get("source_accounting_verified") is True,
            "paired comparison has not passed all v11 integrity gates")
    arms = pair.get("arms")
    require(isinstance(arms, dict) and set(arms) == set(ARMS),
            "paired comparison must contain exactly the five v11 arms")
    validated = {}
    for name in ARMS:
        current = validate_arm(name, arms[name])
        exposed = final_arm(final, name)
        # extract_final_evidence intentionally keeps only heldout aggregates;
        # bind each aggregate back to the raw paired comparison before making
        # a chart point.
        require(exposed.get("successes") == current["successes"] and
                exposed.get("episodes") == current["count"] and
                math.isclose(finite(exposed.get("macro_success_rate"), f"final/{name}"),
                             current["macro_success_rate"], rel_tol=0, abs_tol=1e-12) and
                exposed.get("per_task") == current["per_task"],
                f"final_results/{name} disagrees with paired comparison")
        validated[name] = current
    return validated


def load_inventory(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    inventory = read(path)
    recipe_name = "all_nvfp4_gptq_category"
    require(inventory.get("schema_version") == "v11-selected-all-nvfp4-category" and
            inventory.get("ladder") == [recipe_name] and set(inventory.get("recipes", {})) == {recipe_name},
            "recipe inventory is not the v11 all-NVFP4 category snapshot")
    mixed = inventory["recipes"][recipe_name]
    counts = [mixed.get(name + "_params") for name in ("nvfp4", "fp8", "bf16")]
    require(type(mixed.get("linear_params")) is int and mixed["linear_params"] > 0 and
            all(type(value) is int and value >= 0 for value in counts) and
            sum(counts) == mixed["linear_params"],
            "selected format counts do not sum to eligible elements")
    fractions = mixed.get("fraction_of_eligible_params", {})
    for name, count in zip(("nvfp4", "fp8", "bf16"), counts):
            require(math.isclose(finite(fractions.get(name), f"selected fraction {name}"),
                             count / mixed["linear_params"], rel_tol=0, abs_tol=1e-12),
                f"mixed fraction {name} disagrees with format count")
    # v11 freezes full-coverage W4A4: every eligible element is NVFP4.
    # FP8 is a supported fallback format in the allocator, but it is
    # intentionally absent from the selected main recipe.
    require(counts[0] > 0 and counts[1] == 0 and counts[2] == 0,
            "selected recipe is not full NVFP4")
    require(type(mixed.get("source_tensor_bytes")) is int and mixed["source_tensor_bytes"] > 0 and
            type(mixed.get("target_full_checkpoint_bytes")) is int and mixed["target_full_checkpoint_bytes"] > 0 and
            math.isclose(mixed["full_checkpoint_compression_x"],
                         mixed["source_tensor_bytes"] / mixed["target_full_checkpoint_bytes"],
                         rel_tol=1e-12),
            "selected physical encoding budget is inconsistent")
    alias = mixed.get("known_tied_alias_deduplicated", {})
    require(type(alias.get("source_tensor_bytes")) is int and alias["source_tensor_bytes"] > 0 and
            type(alias.get("target_full_bytes")) is int and alias["target_full_bytes"] > 0 and
            math.isclose(alias["full_compression_x"], alias["source_tensor_bytes"] / alias["target_full_bytes"],
                         rel_tol=1e-12),
            "selected deduplicated encoding budget is inconsistent")
    residual = inventory.get("recovery_residual")
    require(isinstance(residual, dict) and residual.get("dtype") == "bfloat16" and
            residual.get("bytes_per_element") == 2 and type(residual.get("target_bytes")) is int and
            residual["target_bytes"] == residual["tensor_elements"] * 2,
            "recipe inventory has an invalid BF16 recovery residual")
    memory_path = ROOT / "paper/evidence/selected_recipe/category_memory.json"
    recipe_path = ROOT / "paper/evidence/selected_recipe/category_ptq_recipe.json"
    bake_path = ROOT / "paper/evidence/selected_recipe/category_bake_manifest.json"
    artifact_paths = inventory.get("selected_recipe_artifacts", {})
    require(artifact_paths.get("memory") == "paper/evidence/selected_recipe/category_memory.json" and
            artifact_paths.get("ptq_recipe") == "paper/evidence/selected_recipe/category_ptq_recipe.json" and
            artifact_paths.get("bake_manifest") == "paper/evidence/selected_recipe/category_bake_manifest.json",
            "inventory selected recipe artifact paths are not the v11 published files")
    require(memory_path.is_file() and recipe_path.is_file() and bake_path.is_file(),
            "selected category recipe artifacts are missing; materialize the completed run first")
    memory_artifact = read(memory_path)
    recipe_artifact = read(recipe_path)
    bake_artifact = read(bake_path)
    require(memory_artifact.get("format") in {"selected_category_recipe_memory_v1", "selected_category_recipe_memory_v2"},
            "unsupported selected category memory format")
    require(memory_artifact.get("recipe") == "category_nvfp4_extension" and
            recipe_artifact.get("recipe") == memory_artifact.get("recipe") and
            memory_artifact.get("memory") == recipe_artifact.get("memory") == mixed,
            "selected category memory and inventory disagree")
    require(bake_artifact.get("status") == "complete" and
            bake_artifact.get("version") == recipe_artifact.get("version") and
            bake_artifact.get("memory") == mixed,
            "selected category bake manifest and inventory disagree")
    source_digest = memory_artifact.get("source_recipe_sha256",
                                        memory_artifact.get("source_category_recipe_sha256"))
    require(source_digest == digest(recipe_path), "selected category recipe SHA-256 disagrees")
    return inventory, mixed, residual, {"memory": memory_path, "recipe": recipe_path, "bake": bake_path}


def budget(memory: dict[str, Any], residual: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    scopes = {
        "physical": (memory["source_tensor_bytes"], memory["target_full_checkpoint_bytes"]),
        "known_alias_deduplicated": (memory["known_tied_alias_deduplicated"]["source_tensor_bytes"],
                                      memory["known_tied_alias_deduplicated"]["target_full_bytes"]),
    }
    result = {}
    for scope, (source, base) in scopes.items():
        require(type(source) is int and source > 0 and type(base) is int and base > 0,
                f"invalid {scope} mixed budget")
        extra = 0 if residual is None else residual["target_bytes"]
        total = base + extra
        result[scope] = {"source_bytes": source, "base_bytes": base,
                         "residual_bytes": extra, "total_bytes": total,
                         "compression_x": source / total}
    return result


def make_point(name: str, row: dict[str, Any], costs: dict[str, dict[str, Any]],
               role: str, residual: dict[str, Any] | None) -> dict[str, Any]:
    return {"name": name, "role": role, "recipe": "all_nvfp4_gptq_category", "count": row["count"],
            "successes": row["successes"], "macro_success_rate": row["macro_success_rate"],
            "per_task": row["per_task"],
            "encoding_budget": budget_from_base(costs, residual)}


def budget_from_base(base_costs: dict[str, dict[str, Any]], residual: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    extra = 0 if residual is None else residual["target_bytes"]
    result = {}
    for scope, cost in base_costs.items():
        total = cost["base_bytes"] + extra
        result[scope] = {"source_bytes": cost["source_bytes"], "base_bytes": cost["base_bytes"],
                         "residual_bytes": extra, "total_bytes": total,
                         "compression_x": cost["source_bytes"] / total}
    return result


def build(final_results: str | Path, paired_comparison: str | Path,
          inventory_path: str | Path, out: str | Path) -> dict[str, Any]:
    final_path = Path(final_results).resolve()
    pair_path = Path(paired_comparison).resolve()
    out_path = Path(out).absolute()
    require(final_path.is_file(),
            f"final_results is missing: {final_path}; finish v11 heldout evaluation, then run "
            "paper/extract_final_evidence.py --run-dir <run> --out <final_results.json>")
    require(pair_path.is_file(),
            f"paired comparison is missing: {pair_path}; point --paired-comparison at the completed v11 heldout_round/paired_comparison.json")
    require(not out_path.exists() and not out_path.is_symlink(), f"refusing existing output: {out_path}")
    final = read(final_path)
    pair = read(pair_path)
    arms = validate_final_and_pair(final_path, final, pair_path, pair)
    inventory, mixed, residual, artifacts = load_inventory(Path(inventory_path).resolve(strict=True))
    pure_costs = budget(mixed, None)
    recovery_costs = budget(mixed, residual)
    points = [make_point("bf16", arms["bf16"],
                         {scope: {**cost, "base_bytes": cost["source_bytes"]} for scope, cost in pure_costs.items()},
                         "bf16", None),
              make_point("ptq", arms["ptq"], pure_costs, "ptq", None)]
    points.extend(make_point(name, arms[name], recovery_costs, "recovery", residual)
                  for name in ("qad", "continued_qad", "qad_opd"))
    observed = {}
    for scope in ("physical", "known_alias_deduplicated"):
        nondominated = []
        for point in points:
            dominated = any(
                other["encoding_budget"][scope]["total_bytes"] <= point["encoding_budget"][scope]["total_bytes"] and
                other["successes"] >= point["successes"] and
                (other["encoding_budget"][scope]["total_bytes"] < point["encoding_budget"][scope]["total_bytes"] or
                 other["successes"] > point["successes"])
                for other in points)
            if not dominated:
                nondominated.append(point["name"])
        observed[scope] = nondominated
    result = {
        "version": 1, "status": "complete", "environment_pairing_verified": True,
        "protocol_consistency_verified": True, "source_accounting_verified": True,
        "selected_recipe": "all_nvfp4_gptq_category", "reference_order": [], "references": {},
        "points": points, "observed_nondominated_points": observed,
        "source": {"final_results": identity(final_path), "paired_comparison": identity(pair_path),
                    "recipe_inventory": identity(Path(inventory_path).resolve(strict=True)),
                    "selected_category_memory": identity(artifacts["memory"]),
                    "selected_category_recipe": identity(artifacts["recipe"]),
                    "selected_category_bake_manifest": identity(artifacts["bake"])},
        "scope": "v11 heldout-only BF16, selected W4A4 all-NVFP4 PTQ, QAD, continued-QAD control and QAD+OPD",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-results", required=True)
    parser.add_argument("--paired-comparison", required=True)
    parser.add_argument("--inventory", default=str(PAPER / "evidence/recipe_inventory.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        result = build(args.final_results, args.paired_comparison, args.inventory, args.out)
        print(json.dumps({"status": result["status"], "out": str(Path(args.out).resolve()),
                          "points": [p["name"] for p in result["points"]]}, ensure_ascii=False))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[build-final-frontier] ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
