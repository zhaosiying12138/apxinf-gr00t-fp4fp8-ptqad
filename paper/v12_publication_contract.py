"""Cheap release-entry checks; raw-log audits remain in validate_publication.py."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

PROTOCOL_SHA256 = "31addae911f92db65dda3c1c3145e8b798dca46c49a5ae12bbe10d232b5a4521"
PROTOCOL_NAME = "recovery_protocol_v12_rtn_w4a4.json"
ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")


def validate_v12_results(final: dict, *, root: Path) -> None:
    """Reject stale/incomplete evidence before a generator changes public files."""
    if final.get("format") != "publication_final_results_v1" or final.get("status") != "complete":
        raise ValueError("Publication requires complete extracted final results")
    protocol = Path(root) / "exp" / PROTOCOL_NAME
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("The frozen v12 protocol changed")
    source = final.get("source", {}).get("protocol", {})
    if source.get("sha256") != PROTOCOL_SHA256 or Path(source.get("path", "")).name != PROTOCOL_NAME:
        raise ValueError("Final evidence is not bound to the frozen v12 protocol")
    if final.get("selected_recipe") != "rtn_w4a4_category":
        raise ValueError("Final evidence is not the v12 RTN W4A4 selection")
    if final.get("selection", {}).get("selection_uses_heldout") is not False:
        raise ValueError("Selection must explicitly exclude held-out results")
    if set(final.get("public_arms", {})) != {"bf16", "ptq", "qad", "qad_opd"} or set(final.get("control", {})) != {"continued_qad"}:
        raise ValueError("Publication requires exactly the five declared arms")
    task_names = None
    for arm in ARMS:
        row = (final["control"] if arm == "continued_qad" else final["public_arms"])[arm]
        tasks = row.get("per_task", {})
        if len(tasks) != 10 or row.get("episodes") != 160:
            raise ValueError(f"{arm}: expected ten tasks and 160 scored episodes")
        if task_names is None:
            task_names = set(tasks)
        elif set(tasks) != task_names:
            raise ValueError(f"{arm}: tasks differ across arms")
        successes = 0
        for name, task in tasks.items():
            count = task.get("successes")
            if task.get("episodes") != 16 or type(count) is not int or not 0 <= count <= 16:
                raise ValueError(f"{arm}/{name}: incomplete or invalid episode counts")
            if not math.isclose(task.get("success_rate", -1), count / 16, abs_tol=1e-12):
                raise ValueError(f"{arm}/{name}: success rate disagrees with counts")
            successes += count
        if row.get("successes") != successes or not math.isclose(row.get("success_rate", -1), successes / 160, abs_tol=1e-12):
            raise ValueError(f"{arm}: aggregate does not match task evidence")
        if not math.isclose(row.get("macro_success_rate", -1), successes / 160, abs_tol=1e-12):
            raise ValueError(f"{arm}: macro score differs from equal-size task average")
