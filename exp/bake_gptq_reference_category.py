"""Record the exact frozen category-bake command without changing its producer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.compare_gptq_reference import identity, preflight, read, require, verify_record
from eval.compare_recovery import compare_round


def command_for(plan, python, root=ROOT):
    output = root / plan["execution"]["output_root"]
    return [str(python), str(root / "quant/ptq/bake_category.py"),
            "--parent", str(output / "ordinary_parent"),
            "--calib", str(output / "category_h"),
            "--out", str(output / "w4a4_category"),
            "--expected-windows", str(plan["calibration"]["windows"]),
            "--method", plan["calibration"]["category_method"],
            "--gptq-damp", str(plan["calibration"]["gptq_damp"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-command", action="store_true", help="CPU preparation only; launches nothing")
    args = parser.parse_args()
    plan_path = ROOT / "exp/gptq_reference_protocol_v12.json"
    plan, _ = preflight(plan_path)
    run = ROOT / plan["main_run"]
    state = read(run / "run_manifest.json")
    python = Path(state["python"]).absolute()
    command = command_for(plan, python)
    if args.print_command:
        print(json.dumps({"status": "command_only_no_experiment", "command": command}))
        return
    require(state.get("status") == "complete", "Main five-arm run is incomplete")
    final = read(run / "final_manifest.json")
    require(final.get("format") == "w4a4_recovery_v12_final_manifest" and
            final["protocol_sha256"] == plan["main_protocol"]["sha256"] and
            final.get("selection_uses_heldout") is False, "Invalid main final manifest")
    verify_record(final["heldout_comparison"])
    require(read(final["heldout_comparison"]["path"]) == compare_round(final["heldout_round"]),
            "Main comparison no longer reproduces")
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True).stdout.strip()
    require(not active, "A GPU compute process is still running")
    output = ROOT / plan["execution"]["output_root"]
    receipt, log = output / "category_bake_invocation.json", output / "category_bake.log"
    require(not receipt.exists() and not log.exists() and not (output / "w4a4_category").exists(),
            "Refusing to overwrite a bake attempt")
    require((output / "category_h/calib_meta.json").is_file(), "Category calibration is not complete")
    record = {"format": "gptq_category_bake_invocation_v1", "status": "running",
              "command": command, "cwd": str(ROOT),
              "protocol_sha256": identity(plan_path)["sha256"],
              "producer_sha256": identity(ROOT / "quant/ptq/bake_category.py")["sha256"],
              "wrapper": identity(__file__), "started_utc": datetime.now(timezone.utc).isoformat(),
              "final_manifest": identity(run / "final_manifest.json")}
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # Existing category producer is CPU-only.
    env["OMP_NUM_THREADS"] = "4"
    try:
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, status="complete" if result.returncode == 0 else "failed",
                      finished_utc=datetime.now(timezone.utc).isoformat(), log=identity(log))
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(receipt)
    require(record["status"] == "complete", "Category bake failed; inspect the preserved invocation and log")
    print(json.dumps({"status": "complete", "receipt": str(receipt), "checkpoint": str(output / "w4a4_category")}))


if __name__ == "__main__":
    main()
