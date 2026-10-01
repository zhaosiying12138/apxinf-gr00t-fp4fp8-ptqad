#!/usr/bin/env python3
"""Run the audited QAD/OPD recovery driver on the frozen mixed FP4/FP8 arm.

This entrypoint keeps the serial, resumable stages in ``run_high_fp4_v3.py``
and changes only the protocol boundary.  The pressure arm is the existing
``head_lang_vision_category`` checkpoint, whose category provenance records a
mixed NVFP4/FP8 allocation.  The wrapper verifies that provenance before the
shared driver can train or evaluate any recovery arm.

The first invocation for a new evidence directory should use ``--validate-only``
with a completed development selection.  A normal invocation delegates all
stage mutations to the shared driver and remains fail-closed/resumable.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_high_fp4_v3 as v3

CANDIDATE = "mixed"
# ``run_high_fp4_v3.main`` uses the imported module docstring for ``--help``;
# expose this wrapper's mixed-arm contract to users of the entrypoint.
v3.__doc__ = __doc__


def load_protocol(path: Path) -> dict:
    """Load and validate the v7 mixed pressure contract."""
    data = v3.jread(path)
    if data.get("id") != "mixed-pressure-recovery-v7" or data.get("version") != 7:
        raise v3.OrchestrationError("mixed recovery entrypoint requires protocol id/version v7")
    partitions = {}
    expected_partitions = {
        "development": (440000, [4, 5, 6, 7, 8]),
        "collection": (550000, [20, 21, 22, 23]),
        "heldout": (670000, list(range(40, 50))),
    }
    for name, episodes in (("development", 5), ("collection", 4), ("heldout", 10)):
        row = data["partitions"][name]
        indices = row["init_state_indices"]
        if (type(row.get("seed")) is not int or row.get("episodes_per_task") != episodes or
                not isinstance(indices, list) or len(indices) != episodes or
                len(set(indices)) != episodes or
                any(type(x) is not int or not 0 <= x < 50 for x in indices) or
                (row["seed"], indices) != expected_partitions[name]):
            raise v3.OrchestrationError(f"invalid v7 {name} partition")
        partitions[name] = {"seed": row["seed"], "episodes_per_task": episodes,
                            "init_state_indices": indices}

    selection = data.get("selection", {})
    rule = selection.get("pressure_rule", {})
    if (selection.get("pressure_candidates") != [CANDIDATE] or
            rule.get("choose") != "highest_fp4" or
            float(rule.get("min_drop_from_bf16", -1)) != 0.05 or
            float(rule.get("min_absolute_success", -1)) != 0.30):
        raise v3.OrchestrationError("protocol does not match the frozen mixed pressure rule")
    if data.get("quantization_scope", {}).get("recipe") != CANDIDATE:
        raise v3.OrchestrationError("mixed protocol must declare recipe=mixed")
    expected = selection.get("pressure_candidate_checkpoints", {}).get(CANDIDATE)
    scope_expected = data.get("quantization_scope", {}).get("candidate_checkpoint")
    if not expected or expected != scope_expected:
        raise v3.OrchestrationError("mixed candidate checkpoint is missing or duplicated inconsistently")
    return {"data": data, "partitions": partitions, "selection": selection,
            "sha256": v3.sha(path), "path": str(path)}


def _validate_mixed_checkpoint(path: Path, protocol: dict) -> None:
    """Require the declared category bake and both payload formats."""
    expected = Path(protocol["selection"]["pressure_candidate_checkpoints"][CANDIDATE]).resolve()
    actual = path.resolve()
    if actual != expected:
        raise v3.OrchestrationError(
            f"mixed arm checkpoint differs from the frozen candidate: {actual} != {expected}")
    category_recipe_path = actual / "category_ptq_recipe.json"
    category_manifest_path = actual / "category_bake_manifest.json"
    if not category_recipe_path.is_file() or not category_manifest_path.is_file():
        raise v3.OrchestrationError("mixed arm lacks category bake provenance")
    category_recipe = v3.jread(category_recipe_path)
    category_manifest = v3.jread(category_manifest_path)
    if category_recipe.get("recipe") != "category_nvfp4_extension":
        raise v3.OrchestrationError("mixed arm is not a category NVFP4 extension")
    if category_manifest.get("status") != "complete":
        raise v3.OrchestrationError("mixed category bake is not complete")
    memory = category_recipe.get("memory", {})
    fractions = memory.get("fraction_of_eligible_params", {})
    if not (float(fractions.get("nvfp4", 0.0)) > 0.0 and
            float(fractions.get("fp8", 0.0)) > 0.0):
        raise v3.OrchestrationError("mixed arm does not record both NVFP4 and FP8 eligible weights")


# Save the shared implementation before replacing its protocol-boundary hooks.
_shared_validate_ptq = v3.validate_ptq


def validate_ptq(path: Path, protocol: dict) -> dict:
    """Run the standard development evidence audit, then lock the mixed arm."""
    result = _shared_validate_ptq(path, protocol)
    if result.get("selected_recipe") != CANDIDATE:
        raise v3.OrchestrationError("mixed selection did not select the mixed pressure arm")
    arms = result.get("arms", {})
    mixed = arms.get(CANDIDATE, {})
    checkpoint = Path(mixed.get("checkpoint", ""))
    _validate_mixed_checkpoint(checkpoint, protocol)
    return result


v3.load_protocol = load_protocol
v3.validate_ptq = validate_ptq


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--protocol-file" not in argv:
        argv += ["--protocol-file", str(ROOT / "exp/recovery_protocol_v7_mixed_pressure.json")]
    raise SystemExit(v3.main(argv))
