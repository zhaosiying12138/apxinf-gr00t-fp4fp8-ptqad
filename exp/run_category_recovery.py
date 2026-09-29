#!/usr/bin/env python3
"""Category-aware recovery orchestrator built on the audited v3 runner.

The recovery stages (QAD, continued-QAD, OPD and heldout) are identical to
the v3 protocol and retain its resumable/fail-closed implementation. This
entrypoint changes only the frozen pressure selection contract: its PTQ base
must be one of the two category-aware checkpoints produced by
``run_category_development.py``. It never aliases a category bake as a plain
PTQ arm and refuses a development selection whose raw evidence is incomplete.

Use ``--validate-only`` first. A normal invocation delegates all mutations to
the existing serial driver after this category-specific validation; therefore
the command is safe to resume and never overwrites a completed stage.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_high_fp4_v3 as v3

CANDIDATES = ("head_lang_vision_category", "calib_category")
PARENT_RECIPES = {"head_lang_vision_category": "head_lang_vision", "calib_category": "calib"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category_protocol(path):
    path = Path(path).resolve(strict=True)
    data = read(path)
    if data.get("version") != 4 or data.get("id") != "category-fp4-stress-v4":
        raise v3.OrchestrationError("category recovery requires the frozen v4 protocol")
    parts = data.get("partitions", {})
    normalized = {}
    for name, episodes in (("development", 5), ("collection", 4), ("heldout", 10)):
        row = parts.get(name, {})
        indices = row.get("init_state_indices")
        if (type(row.get("seed")) is not int or row.get("episodes_per_task") != episodes or
                not isinstance(indices, list) or len(indices) != episodes or
                len(set(indices)) != episodes or any(type(i) is not int or not 0 <= i < 50 for i in indices)):
            raise v3.OrchestrationError(f"invalid category {name} partition")
        normalized[name] = {"seed": row["seed"], "episodes_per_task": episodes,
                            "init_state_indices": indices}
    selection = data.get("selection", {})
    rule = selection.get("pressure_rule", {})
    if (selection.get("pressure_candidates") != list(CANDIDATES) or
            rule.get("choose") != "highest_fp4" or
            float(rule.get("min_drop_from_bf16", -1)) != .20 or
            float(rule.get("min_absolute_success", -1)) != .30):
        raise v3.OrchestrationError("category pressure rule differs from the frozen v4 protocol")
    category = data.get("category_quantization", {})
    if (category.get("parent_recipes") != ["head_lang_vision", "calib"] or
            category.get("active_category") != 2 or category.get("categories") != 32):
        raise v3.OrchestrationError("category quantization contract differs")
    return {"data": data, "partitions": normalized, "selection": selection,
            "sha256": sha(path), "path": str(path)}


def validate_category_selection(path, protocol):
    """Audit category development arms and return v3-compatible selection data."""
    path = Path(path).resolve(strict=True)
    selection = read(path)
    if (selection.get("protocol_sha256") != protocol["sha256"] or
            selection.get("selection_uses_heldout") is not False or
            selection.get("selected_recipe") not in CANDIDATES):
        raise v3.OrchestrationError("category selection is not bound to v4 development evidence")
    arms = selection.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"bf16", *CANDIDATES}:
        raise v3.OrchestrationError("category selection arm set differs from v4 protocol")
    development = path.parent / "development"
    audits = {}
    for arm in ("bf16", *CANDIDATES):
        row = arms[arm]
        if type(row.get("successes")) is not int or type(row.get("episodes")) is not int:
            raise v3.OrchestrationError(f"category selection score is not integer: {arm}")
        checkpoint = Path(row.get("checkpoint", "")).resolve(strict=True)
        if arm != "bf16":
            recipe = checkpoint / "category_ptq_recipe.json"
            manifest = checkpoint / "category_bake_manifest.json"
            if not recipe.is_file() or not manifest.is_file():
                raise v3.OrchestrationError(f"category arm lacks category provenance: {arm}")
            recipe_data = read(recipe)
            if recipe_data.get("recipe") != "category_nvfp4_extension":
                raise v3.OrchestrationError(f"category arm is not a category bake: {arm}")
        audited = v3.eval_audit(development / arm, protocol, "development", checkpoint)
        if (audited["successes"] != row["successes"] or audited["episodes"] != row["episodes"]):
            raise v3.OrchestrationError(f"category score disagrees with raw evidence: {arm}")
        audits[arm] = audited
    v3.require_pairing(audits)
    baseline = Fraction(audits["bf16"]["successes"], audits["bf16"]["episodes"])
    rule = protocol["selection"]["pressure_rule"]
    drop = Fraction(str(rule["min_drop_from_bf16"]))
    floor = Fraction(str(rule["min_absolute_success"]))
    qualifying = [arm for arm in CANDIDATES
                  if baseline - Fraction(audits[arm]["successes"], audits[arm]["episodes"]) >= drop and
                  Fraction(audits[arm]["successes"], audits[arm]["episodes"]) >= floor]
    if qualifying != selection.get("qualifying_candidates") or selection["selected_recipe"] != (qualifying[-1] if qualifying else None):
        raise v3.OrchestrationError("category pressure selection is not the declared highest-FP4 choice")
    selected = arms[selection["selected_recipe"]]
    base = Path(selected["checkpoint"]).resolve(strict=True)
    parent_recipe = read(base / "category_ptq_recipe.json")
    return {**selection, "selection_file": str(path), "selection_sha256": sha(path),
            "selected_ptq_checkpoint": str(base),
            "selected_ptq_recipe_sha256": sha(base / "category_ptq_recipe.json"),
            "selected_parent_recipe_sha256": parent_recipe["parent_recipe_sha256"],
            "verified_qualifying_candidates": qualifying}


# Patch only the protocol/selection boundary. Driver stage code remains the
# audited v3 implementation and is intentionally not duplicated here.
v3.load_protocol = category_protocol
v3.validate_ptq = validate_category_selection


if __name__ == "__main__":
    # v3's parser supplies all recovery options and --validate-only; require a
    # category protocol by default while allowing an explicit override for
    # reproducible audit commands.
    argv = sys.argv[1:]
    if "--protocol-file" not in argv:
        argv += ["--protocol-file", str(ROOT / "exp/recovery_protocol_v4_category_fp4.json")]
    raise SystemExit(v3.main(argv))
