#!/usr/bin/env python3
"""Create a fail-closed development selection for the W4A4 recovery driver.

The selector binds one frozen protocol to raw development manifests and logs.
It creates relative symlinks beside ``selection.json`` so the existing audited
driver can resolve the evidence without copying or rewriting absolute-path
manifests.  No held-out file is read.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from run_high_fp4_v3 import (
    _candidate_fp4_fraction,
    eval_audit,
    load_protocol,
    model_id,
    sha,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-file", required=True)
    p.add_argument("--bf16", required=True, help="BF16 development output directory")
    p.add_argument("--candidate", required=True, help="candidate name registered in protocol")
    p.add_argument("--candidate-output", required=True, help="candidate development output directory")
    p.add_argument("--out", required=True, help="new directory containing selection.json")
    p.add_argument("--force", action="store_true", help="remove an existing output directory")
    return p.parse_args()


def link(out: Path, name: str, target: Path) -> Path:
    dst = out / name
    if dst.exists() or dst.is_symlink():
        raise SystemExit(f"refusing to replace existing evidence link: {dst}")
    dst.symlink_to(target)
    return dst


def main() -> int:
    a = parse_args()
    protocol_file = Path(a.protocol_file).expanduser().resolve()
    protocol = load_protocol(protocol_file)
    candidates = protocol["selection"]["pressure_candidates"]
    if candidates != [a.candidate]:
        raise SystemExit(f"protocol candidates must be exactly [{a.candidate!r}], got {candidates!r}")
    out = Path(a.out).expanduser().resolve()
    if out.exists():
        if not a.force:
            raise SystemExit(f"output exists; choose a new directory or use --force: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    bf16 = Path(a.bf16).expanduser().resolve()
    candidate = Path(a.candidate_output).expanduser().resolve()
    links = {"bf16": link(out, "bf16", bf16), a.candidate: link(out, a.candidate, candidate)}

    audited = {}
    for name, path in (("bf16", links["bf16"]), (a.candidate, links[a.candidate])):
        manifest = json.loads((path / "eval_manifest.json").read_text())
        audited[name] = eval_audit(path, protocol, "development",
                                   Path(manifest["checkpoint"]).resolve())
    # ``eval_audit`` above checks every raw manifest.  Bind the actual model
    # checkpoint from each manifest after the audit, then record its identity.
    arms = {}
    source_sha256 = {}
    for name, path in (("bf16", links["bf16"]), (a.candidate, links[a.candidate])):
        manifest = json.loads((path / "eval_manifest.json").read_text())
        checkpoint = Path(manifest["checkpoint"]).resolve()
        row = audited[name]
        record = {
            "checkpoint": str(checkpoint),
            "successes": row["successes"],
            "episodes": row["episodes"],
            "macro_success_rate": row["macro_success_rate"],
            "micro_success_rate": row["micro_success_rate"],
            "pairing_sha256": row["pairing_sha256"],
            "environment_pairing_verified": True,
            "model_identity": model_id(checkpoint),
        }
        if row.get("activation_installation") is not None:
            record["activation_installation"] = row["activation_installation"]
        if name != "bf16":
            record["fp4_fraction"] = _candidate_fp4_fraction(checkpoint)
        arms[name] = record
        for filename in ("eval_manifest.json", "task_results.json", "summary.json"):
            source_sha256[f"{name}/{filename}"] = sha(path / filename)

    pairing = {v["pairing_sha256"] for v in arms.values()}
    if len(pairing) != 1:
        raise SystemExit(f"development evidence is not paired: {pairing}")
    pressure = protocol["selection"]["pressure_rule"]
    base = arms["bf16"]["micro_success_rate"]
    minimum = float(pressure["min_drop_from_bf16"])
    maximum = float(pressure.get("target_drop_from_bf16", [0.0, 1.0])[1])
    candidate_rate = arms[a.candidate]["micro_success_rate"]
    drop = base - candidate_rate
    qualifies = minimum <= drop <= maximum and candidate_rate >= float(pressure["min_absolute_success"])
    if not qualifies:
        raise SystemExit(f"candidate does not satisfy pressure gate: BF16={base:.4f}, candidate={candidate_rate:.4f}, drop={drop:.4f}")
    selection = {
        "status": "complete",
        "protocol_file": str(protocol_file),
        "protocol_sha256": protocol["sha256"],
        "selection_uses_heldout": False,
        "selection_rule": "highest-FP4 qualifying full-coverage W4A4 candidate",
        "selected_recipe": a.candidate,
        "qualifying_candidates": [a.candidate],
        "pressure_candidates": candidates,
        "pairing_sha256": next(iter(pairing)),
        "source_sha256": source_sha256,
        "candidate_fp4_fraction": {a.candidate: arms[a.candidate]["fp4_fraction"]},
        "arms": arms,
    }
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selection": str(out / "selection.json"), "selected_recipe": a.candidate,
                      "bf16": [arms["bf16"]["successes"], arms["bf16"]["episodes"]],
                      "ptq": [arms[a.candidate]["successes"], arms[a.candidate]["episodes"]],
                      "drop": drop, "fp4_fraction": arms[a.candidate]["fp4_fraction"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
