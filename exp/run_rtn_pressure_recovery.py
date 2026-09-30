#!/usr/bin/env python3
"""Run the audited QAD/OPD recovery driver on the frozen RTN pressure arm.

The implementation is shared with ``run_high_fp4_v3.py``.  This wrapper only
changes the protocol boundary: the single pressure arm is ``rtn`` and the
minimum development drop is five percentage points.  All evidence still uses
the protocol hash recorded by the evaluator, so the selection and every
subsequent stage remain fail-closed and resumable.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_high_fp4_v3 as v3


def load_protocol(path: Path) -> dict:
    # Reimplement the small loader so the historical v3 entrypoint remains
    # unchanged and the v6 contract is explicit in this new file.
    data = v3.jread(path)
    partitions = {}
    for name, episodes in (("development", 5), ("collection", 4), ("heldout", 10)):
        row = data["partitions"][name]
        indices = row["init_state_indices"]
        if (type(row.get("seed")) is not int or row.get("episodes_per_task") != episodes or
                not isinstance(indices, list) or len(indices) != episodes or
                len(set(indices)) != episodes or any(type(x) is not int or not 0 <= x < 50 for x in indices)):
            raise v3.OrchestrationError(f"invalid v6 {name} partition")
        partitions[name] = {"seed": row["seed"], "episodes_per_task": episodes,
                            "init_state_indices": indices}
    selection = data.get("selection", {})
    rule = selection.get("pressure_rule", {})
    if (selection.get("pressure_candidates") != ["rtn"] or
            rule.get("choose") != "highest_fp4" or
            float(rule.get("min_drop_from_bf16", -1)) != 0.05 or
            float(rule.get("min_absolute_success", -1)) != 0.30):
        raise v3.OrchestrationError("protocol does not match the frozen RTN pressure rule")
    if data.get("quantization_scope", {}).get("recipe") != "rtn":
        raise v3.OrchestrationError("v6 protocol must declare recipe=rtn")
    return {"data": data, "partitions": partitions, "selection": selection,
            "sha256": v3.sha(path), "path": str(path)}


v3.load_protocol = load_protocol


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--protocol-file" not in argv:
        argv += ["--protocol-file", str(ROOT / "exp/recovery_protocol_v6_rtn_pressure.json")]
    raise SystemExit(v3.main(argv))
