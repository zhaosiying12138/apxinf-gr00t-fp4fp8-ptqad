"""CPU-only supplemental GPTQ comparison; never select or run a policy.

The original five-arm protocol, selection and result files are read-only.
Only an independently named supplementary report is written, without overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from exp.run_high_fp4_v3 import eval_audit, load_protocol, require_pairing
from compare_recovery import ARMS, PROTOCOL_FIELDS, compare_round
from run_recovery_eval import TASKS, validate_resets
from paper.paired_uncertainty import holm_adjust, paired_effect

PLAN_SHA256 = "ba1f3972893fcc2bf2ab10ec9b0dd75569f6f5a10b79e594cd4d273837035a5b"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_record(record, root=ROOT):
    actual = identity(root / record["path"])
    require(all(actual[k] == record[k] for k in ("bytes", "sha256")),
            "Frozen input changed: " + record["path"])
    return actual


def preflight(plan_path, root=ROOT):
    require(identity(plan_path)["sha256"] == PLAN_SHA256, "Supplement protocol changed")
    plan = read(plan_path)
    require(plan["format"] == "w4a4_gptq_supplement_protocol_v1", "Unknown supplement protocol")
    records = [plan["main_protocol"], plan["capture_manifest"],
               *plan["preparation_source_files"].values()]
    verified = {row["path"]: verify_record(row, root) for row in records}
    previous = identity(root / plan["amendment"]["previous_protocol_path"])
    require(previous["sha256"] == plan["amendment"]["previous_protocol_sha256"],
            "Supplement amendment source changed")
    main = read(root / plan["main_protocol"]["path"])
    require(plan["execution"]["eval_partition"] == main["partitions"]["heldout"],
            "Supplement heldout differs from main")
    require(plan["execution"]["eval_contract"] == main["evaluation_contract"],
            "Supplement execution differs from main")
    return plan, verified


def read_episodes(folder):
    """Called after eval_audit verifies these outcomes against the raw logs."""
    manifest, rows = read(folder / "eval_manifest.json"), read(folder / "task_results.json")
    result = []
    for ti, task in enumerate(TASKS):
        resets = validate_resets(rows[task], manifest["seed"] + 1000 * ti,
                                 manifest["init_state_indices"])
        for reset, success in zip(resets, rows[task]["results"]):
            result.append({"task": task, **{key: reset[key] for key in (
                "episode_index", "seed", "init_state_index", "settle_steps",
                "initial_state_sha256", "restored_state_sha256", "init_state_bank_sha256")},
                "success": success})
    return result


def verify_bf16_identity(checkpoint, files, plan):
    """The main reference and the calibration teacher must be the same model."""
    checkpoint = Path(checkpoint).resolve()
    weights = {Path(name).name: row for name, row in files.items()
               if Path(name).parent == checkpoint and name.endswith(".safetensors")}
    require(weights == plan["teacher_weights"], "Main BF16 differs from GPTQ calibration teacher")
    for name, expected in plan["teacher_metadata"].items():
        require(files[str(checkpoint / name)]["sha256"] == expected,
                "Main BF16 teacher metadata differs: " + name)


def summarize(arms, plan):
    """Use the frozen contrasts and return descriptive-only secondary effects."""
    comparison = plan["comparison"]
    expected = set(comparison["reference_arms"]) | {comparison["additional_arm"]}
    require(set(arms) == expected, "All six prespecified arms are required")
    settings = comparison["confidence_interval"]
    counts = {}
    n = plan["execution"]["eval_partition"]["episodes_per_task"]
    reference = arms["bf16"]
    for name, rows in arms.items():
        require(len(rows) == len(TASKS) * n, "Incomplete arm: " + name)
        require(all(type(row.get("success")) is bool for row in rows), "Outcomes must be Boolean")
        require([{k: v for k, v in r.items() if k != "success"} for r in rows] ==
                [{k: v for k, v in r.items() if k != "success"} for r in reference],
                "Episode identities do not pair: " + name)
        per_task = {}
        for task in TASKS:
            chosen = [row for row in rows if row["task"] == task]
            require(len(chosen) == n and {r["episode_index"] for r in chosen} == set(range(n)),
                    "Missing/duplicate task episodes: " + name + "/" + task)
            successes = sum(row["success"] for row in chosen)
            per_task[task] = {"successes": successes, "episodes": n, "success_rate": successes / n}
        successes = sum(row["success"] for row in rows)
        counts[name] = {"successes": successes, "episodes": len(rows),
                        "macro_success_rate": sum(x["success_rate"] for x in per_task.values()) / len(TASKS),
                        "micro_success_rate": successes / len(rows), "per_task": per_task}
    contrasts = {}
    for name, (baseline, treatment) in comparison["contrasts"].items():
        contrasts[name] = {"baseline": baseline, "treatment": treatment, **paired_effect(
            arms[baseline], arms[treatment], replicates=settings["replicates"],
            seed=settings["seed"], level=settings["level"])}
    adjusted = holm_adjust({name: row["exact_mcnemar_two_sided_p"] for name, row in contrasts.items()})
    for name, row in contrasts.items():
        row["holm_adjusted_p"] = adjusted[name]
    descriptive = []
    for baseline, treatment in comparison["additional_descriptive_comparisons"]:
        left, right = arms[baseline], arms[treatment]
        descriptive.append({"baseline": baseline, "treatment": treatment,
            "difference_pp": 100 * (counts[treatment]["macro_success_rate"] - counts[baseline]["macro_success_rate"]),
            "only_treatment_success": sum(not a["success"] and b["success"] for a, b in zip(left, right)),
            "only_baseline_success": sum(a["success"] and not b["success"] for a, b in zip(left, right)),
            "interpretation": "Descriptive only; no hypothesis test or equivalence claim."})
    return {"arms": counts, "contrasts": contrasts, "descriptive_comparisons": descriptive,
            "confidence_interval": settings, "tests": comparison["tests"],
            "interpretation": comparison["interpretation"]}


def compare(plan_path, evaluation=None):
    plan, sources = preflight(plan_path)
    run = ROOT / plan["main_run"]
    final_path = run / "final_manifest.json"
    require(final_path.is_file(), "Main five-arm final manifest is not available")
    final, state = read(final_path), read(run / "run_manifest.json")
    require(state.get("status") == "complete", "Main five-arm run is incomplete")
    require(final.get("selection_uses_heldout") is False, "Main selection is not development-only")
    require(final.get("required_arms") == list(ARMS), "Main arm set differs")
    require(state["protocol_sha256"] == final["protocol_sha256"] == plan["main_protocol"]["sha256"],
            "Main protocol differs from supplement")
    # These CPU readers do not construct a model or use a CUDA device.
    from exp.action_chunk_diagnostics import bind_final_checkpoint
    from gptq_reference_evidence import verify_gptq
    gptq = verify_gptq(plan, ROOT)
    round_dir = run / "artifacts/heldout_round"
    require(Path(final["heldout_round"]).resolve() == round_dir.resolve(), "Main heldout directory differs")
    comparison_path = round_dir / "paired_comparison.json"
    verify_record(final["heldout_comparison"])
    require(Path(final["heldout_comparison"]["path"]).resolve() == comparison_path.resolve(),
            "Main comparison path differs")
    require(read(comparison_path) == compare_round(round_dir), "Main comparison no longer reproduces")
    protocol = load_protocol(ROOT / plan["main_protocol"]["path"])
    audits, episodes, checkpoint_sources = {}, {}, {}
    main_manifest = None
    collection_path = ROOT / plan["execution"]["collection_manifest"]
    for arm in ARMS:
        checkpoint, files = bind_final_checkpoint(final_path, arm,
                                                   {"protocol_sha256": protocol["sha256"]})
        if arm == "bf16":
            verify_bf16_identity(checkpoint, files, plan)
        folder = round_dir / ("heldout_" + arm)
        audits[arm] = eval_audit(folder, protocol, "heldout", checkpoint)
        episodes[arm] = read_episodes(folder)
        checkpoint_sources[arm] = files
        manifest = read(folder / "eval_manifest.json")
        require(Path(manifest["collection_manifest"]).resolve() == collection_path.resolve(),
                "Main collection reference differs")
        if main_manifest is None:
            main_manifest = manifest
    # Verify the collection itself, not merely an existing manifest path.
    collection_manifest = read(collection_path)
    collection_checkpoint = Path(collection_manifest["checkpoint"]).resolve()
    require(collection_checkpoint == Path(final["selected_qad_model_identity"]["path"]).resolve(),
            "Collection used another QAD checkpoint")
    collection_audit = eval_audit(collection_path.parent, protocol, "collection", collection_checkpoint)
    folder = Path(evaluation).resolve() if evaluation else ROOT / plan["execution"]["output_root"] / "heldout_gptq"
    audits["gptq"] = eval_audit(folder, protocol, "heldout", Path(gptq["checkpoint"]))
    manifest = read(folder / "eval_manifest.json")
    require(all(manifest.get(key) == main_manifest.get(key) for key in PROTOCOL_FIELDS),
            "GPTQ evaluation execution contract differs from main")
    require(Path(manifest["collection_manifest"]).resolve() == collection_path.resolve(),
            "GPTQ collection reference differs")
    require_pairing(audits)
    episodes["gptq"] = read_episodes(folder)
    return {"format": "w4a4_gptq_supplement_comparison_v1", "status": "verified",
            "protocol": identity(plan_path), "final_manifest": identity(final_path),
            "main_run_manifest": identity(run / "run_manifest.json"),
            "main_comparison": identity(comparison_path), "frozen_sources": sources,
            "analysis_sources": {str(path.relative_to(ROOT)): identity(path) for path in (
                Path(__file__), ROOT / "eval/gptq_reference_evidence.py",
                ROOT / "exp/action_chunk_diagnostics.py", ROOT / "exp/bake_gptq_reference_category.py")},
            "raw_evaluation_audits": audits, "collection_audit": collection_audit,
            "selected_checkpoint_sources": checkpoint_sources, "gptq_evidence": gptq,
            "paired_scope": "Official environment resets; policy noise is seeded per task, not paired per episode.",
            "episodes": episodes, **summarize(episodes, plan)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "exp/gptq_reference_protocol_v12.json"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evaluation", help="Explicit new output directory after an audited infrastructure retry")
    parser.add_argument("--out")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.preflight_only:
        plan, checked = preflight(Path(args.plan))
        print(json.dumps({"status": "frozen_inputs_verified_no_experiment_run", "sources": len(checked),
                          "protocol_sha256": identity(args.plan)["sha256"]}))
        return
    if not args.out:
        parser.error("--out is required for a comparison")
    output = Path(args.out)
    require(not output.exists(), "Refusing to overwrite supplementary report")
    result = compare(Path(args.plan), args.evaluation)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "arms": result["arms"], "out": str(output)}))


if __name__ == "__main__":
    main()
