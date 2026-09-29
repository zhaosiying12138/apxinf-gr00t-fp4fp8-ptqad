#!/usr/bin/env python3
"""Export CPU metadata-only allocation budgets; never claim measured storage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quant/ptq"))
from bake import inventory, make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    entries, _, _ = inventory(Path(args.base).resolve())
    ladder = ["fp8", "head_ffn", "head_lang", "head_lang_vision", "calib"]
    rows, previous = {}, set()
    for recipe in ["mixed", *ladder]:
        layers, _, budget = make_plan(entries, recipe)
        fp4 = {k for k, v in layers.items() if v["requested_method"].startswith("nvfp4")}
        if recipe in ladder:
            assert previous <= fp4, "FP4 coverage must be nested"
            previous = fp4
        rows[recipe] = budget | {"fp4_tensor_count": len(fp4)}
    evidence = ROOT / "paper/evidence"
    evidence.mkdir(exist_ok=True)
    implementation = ROOT / "quant/ptq/bake.py"
    result = {
        "kind": "format_encoding_budget_not_measured_file_or_vram",
        "base": str(Path(args.base).resolve()),
        "allocation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "ladder": ladder,
        "rows": rows,
    }
    (evidence / "compression_budget.json").write_text(json.dumps(result, indent=2) + "\n")
    for recipe in ladder:
        row = rows[recipe]
        print(f"{recipe:18s} FP4/eligible={row['fraction_of_eligible_params']['nvfp4']:.4%} "
              f"eligible={row['linear_compression_x']:.5f}x full={row['full_checkpoint_compression_x']:.5f}x")


if __name__ == "__main__":
    main()
