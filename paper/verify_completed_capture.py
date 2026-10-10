#!/usr/bin/env python3
"""CPU-only verification shown in screenshots of completed v12 evidence.

This program reads archived JSON and original log bytes. It never imports torch,
loads a model, starts a process, creates a tensor/cache, or writes any file. The
reported values are derived from the final evidence; no training is replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

MODE = "verify-completed"
FINAL_FORMAT = "w4a4_recovery_v12_final_manifest"
FIGURES = ("shot_bake", "shot_collect", "shot_qad", "shot_rollout",
           "shot_opdcache", "shot_opd", "shot_evalserver")
CAPTIONS = {
    "shot_bake": "已完成 RTN 配方与编码预算核验",
    "shot_collect": "已完成教师演示来源核验",
    "shot_qad": "已完成 QAD 训练证据核验",
    "shot_rollout": "已完成学生闭环采集证据核验",
    "shot_opdcache": "已完成教师缓存证据核验",
    "shot_opd": "已完成 OPD 续训证据核验",
    "shot_evalserver": "已完成正式评测服务日志核验",
}


def need(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def local(root, relative):
    need(isinstance(relative, str) and "\\" not in relative, "Invalid evidence path")
    name = PurePosixPath(relative)
    need(relative and not name.is_absolute() and name.as_posix() == relative and
         all(part not in (".", "..") for part in name.parts), "Unsafe evidence path: " + relative)
    path = root
    for part in name.parts:
        path = path / part
        need(not path.is_symlink(), "Evidence symlink: " + relative)
    need(path.is_file() and path.resolve().is_relative_to(root.resolve()),
         "Missing regular evidence: " + relative)
    return path


def identity(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": value.hexdigest()}


def checked(root, relative, record):
    path = local(root, relative)
    need(identity(path) == {key: record.get(key) for key in ("bytes", "sha256")},
         "Evidence identity differs: " + relative)
    return path


def installed_sources(root, final_sha):
    """Bind the installed source map to this final run, without private weights."""
    root = Path(root).resolve(strict=True)
    mapping = load(local(root, "evidence_manifest.json"))
    need(mapping.get("format") == "installed_final_evidence_v1" and
         mapping.get("status") == "complete", "A completed installed evidence bundle is required")
    final = load(local(root, "final_manifest.json"))
    need(final.get("format") == FINAL_FORMAT and final.get("selection_uses_heldout") is False,
         "Completed v12 final manifest is required")
    need(identity(root / "final_manifest.json")["sha256"] == final_sha,
         "Installed final manifest differs from capture release")
    result = load(local(root, "final_results.json"))
    need(result.get("format") == "publication_final_results_v1" and result.get("status") == "complete",
         "Completed final results are required")
    checked(root, "final_manifest.json", result["source"]["final_manifest"])
    checked(root, "paired_comparison.json", result["source"]["heldout_comparison"])
    original = load(checked(root, "source_bundle_manifest.json", mapping["source_bundle_manifest"]))
    need(result["source"] == original.get("source_final_manifest"), "Installed source release differs")
    rows = mapping.get("files", [])
    need(isinstance(rows, list) and rows, "Empty installed source map")
    records = {}
    for row in rows:
        relative = row["published_path"]
        need(relative not in records, "Duplicate installed source: " + relative)
        checked(root, relative, row["source"])
        records[relative] = {key: row["source"][key] for key in ("bytes", "sha256")}
    need("final_manifest.json" in records and "final_results.json" in records,
         "Installed map lacks final evidence")
    return root, final, records


def source_contract(root, final_sha):
    root, final, records = installed_sources(root, final_sha)
    common = ("final_manifest.json", "final_results.json", "paired_comparison.json",
              "training/costs.json", "training/orchestration/final_manifest.json",
              "training/orchestration/run_manifest.json")
    groups = {
        "shot_bake": ["recipe_inventory.json", "selected_recipe/category_ptq_recipe.json",
                      "selected_recipe/category_bake_manifest.json"],
        "shot_collect": ["training/stages/qad/orchestrator_training_request.json",
                         "training/stages/qad/recovery_manifest.json"],
        "shot_qad": ["training/stages/qad/" + name for name in
                     ("train.log", "recovery_manifest.json", "runtime_metrics.json", "checkpoint/trainer_state.json")],
        "shot_rollout": [name for name in records if name.startswith("training/collection/") and
                         (name.endswith(".log") or name.endswith("eval_manifest.json") or
                          name.endswith("capture_manifest.json"))],
        "shot_opdcache": ["training/teacher/teacher_probes.json", "training/stages/qad_opd/recovery_manifest.json"],
        "shot_opd": ["training/stages/" + arm + "/" + name
                     for arm in ("qad_opd", "continued_qad") for name in
                     ("train.log", "recovery_manifest.json", "runtime_metrics.json", "checkpoint/trainer_state.json")],
        "shot_evalserver": ["heldout_raw_logs.json", *[name for name in records
                            if name.startswith("heldout_") and
                            (name.endswith(".server.log") or name.endswith("eval_manifest.json"))]],
    }
    need(groups["shot_rollout"] and any(name.endswith(".server.log") for name in groups["shot_evalserver"]),
         "Original collection/server logs are missing")
    for figure, names in groups.items():
        need(set(common) | set(names) <= records.keys(), "Missing completed evidence for " + figure)
        groups[figure] = {name: records[name] for name in sorted(set(common) | set(names))}
    return {"evidence_root": str(root), "evidence_files": groups,
            "evidence_manifest_sha256": identity(root / "evidence_manifest.json")["sha256"],
            "protocol_sha256": final["protocol_sha256"]}


def verify_plan(plan, figure, evidence_root=None):
    """Recheck every displayed source byte and the completed-run relationships."""
    need(plan.get("capture_mode") == MODE and figure in FIGURES,
         "A verify-completed capture plan and known figure are required")
    root = Path(evidence_root or plan["evidence_root"]).resolve(strict=True)
    records = plan.get("evidence_files", {}).get(figure)
    need(isinstance(records, dict) and records, "Capture has no bound original evidence")
    for relative, record in records.items():
        checked(root, relative, record)
    final = load(root / "final_manifest.json")
    need(final.get("format") == FINAL_FORMAT and final.get("selection_uses_heldout") is False and
         identity(root / "final_manifest.json")["sha256"] == plan.get("final_manifest_sha256") and
         final.get("protocol_sha256") == plan.get("protocol_sha256"), "Capture final release differs")
    need(load(root / "training/orchestration/final_manifest.json") == final,
         "Training archive belongs to another final run")
    run = load(root / "training/orchestration/run_manifest.json")
    need(run.get("status") == "complete" and run.get("protocol_sha256") == final["protocol_sha256"],
         "Archived run is incomplete or uses another protocol")
    costs = load(root / "training/costs.json")
    need(costs.get("status") == "complete" and costs.get("protocol_sha256") == final["protocol_sha256"],
         "Training costs belong to another protocol")
    comparison = load(root / "paired_comparison.json")
    need(comparison.get("environment_pairing_verified") is True and
         comparison.get("protocol_consistency_verified") is True and
         set(comparison.get("arms", {})) == {"bf16", "ptq", "qad", "continued_qad", "qad_opd"},
         "Final five-arm pairing is incomplete")
    checked(root, "paired_comparison.json", final["heldout_comparison"])
    return root, final, costs, records


def summary_for(root, final, costs, figure, records):
    read = lambda relative: load(root / relative)
    if figure == "shot_bake":
        inv = read("recipe_inventory.json"); recipe = inv["recipes"][inv["recipe"]]
        need(recipe["nvfp4_params"] == recipe["linear_params"] and
             recipe.get("fp8_params", 0) == recipe.get("bf16_params", 0) == 0,
             "Recipe is not all-eligible NVFP4")
        fields = ("all_checkpoint_params", "linear_params", "eligible_tensor_count", "nvfp4_params",
                  "fraction_of_all_checkpoint_params", "target_full_checkpoint_bytes", "full_checkpoint_compression_x")
        residual_fields = ("scope", "rank", "alpha", "linear_modules", "tensor_elements", "dtype", "target_bytes")
        return {"recipe": inv["recipe"], "encoding_budget": {key: recipe[key] for key in fields if key in recipe},
                "recovery_residual": {key: inv["recovery_residual"][key] for key in residual_fields
                                      if key in inv["recovery_residual"]}}
    if figure == "shot_collect":
        request = read("training/stages/qad/orchestrator_training_request.json")
        rec = read("training/stages/qad/recovery_manifest.json")
        dataset = request["capture_dataset_identity"]
        need(hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest() ==
             rec["capture_dataset_sha256"] and dataset["sample_count"] == rec["capture_dataset_samples"],
             "Demonstration archive identity differs from completed training")
        return {"capture_dataset": dataset["root"], "sample_count": dataset["sample_count"],
                "recorded_source_files": len(dataset["files"]),
                "capture_dataset_sha256": rec["capture_dataset_sha256"],
                "scope": "Recorded demonstration identities used by completed QAD; private observation tensors are not reopened"}
    if figure in ("shot_qad", "shot_opd"):
        arms = ("qad",) if figure == "shot_qad" else ("qad_opd", "continued_qad")
        data = {}
        for arm in arms:
            prefix = "training/stages/" + arm + "/"
            rec = read(prefix + "recovery_manifest.json")
            metrics = read(prefix + "runtime_metrics.json")
            state = read(prefix + "checkpoint/trainer_state.json")
            need(metrics.get("status") == "completed" and metrics.get("error") is None and
                 metrics.get("global_steps") == state.get("global_step") == state.get("max_steps") ==
                 costs["training"][arm]["optimizer_steps"] and rec.get("w4a4_enabled") is True,
                 "Completed training receipts disagree: " + arm)
            data[arm] = {"completed_optimizer_steps": state["global_step"],
                         "scope": rec["scope"], "rank": rec["rank"], "alpha": rec["alpha"],
                         "probe_weight": rec["probe_weight"], "trainable_parameters": rec["trainable_parameters"],
                         "execution_mode": rec["execution_mode"], "wall_seconds": metrics["wall_seconds"],
                         "raw_training_log": {"path": prefix + "train.log", **records[prefix + "train.log"]}}
        return data
    if figure == "shot_opdcache":
        metadata = read("training/teacher/teacher_probes.json")
        rec = read("training/stages/qad_opd/recovery_manifest.json")
        cache = costs["teacher_labeling"]["cache_identity"]
        need(cache["sha256"] == rec["probe_cache_sha256"] and metadata.get("source_kind") == "student_rollout" and
             metadata["count"] == len(metadata["source_observation_files"]) and
             metadata["action_mask"] == rec["probe_action_mask"], "Completed teacher cache provenance differs")
        return {key: metadata[key] for key in ("objective", "count", "action_mask", "source_kind", "student_checkpoint")} | {
            "recorded_cache_identity": cache, "scope": "Completed cache metadata and training binding; cache tensors are not regenerated or reopened"}
    if figure == "shot_rollout":
        manifest = read("training/collection/eval_manifest.json")
        need(manifest.get("purpose") == "collection" and manifest.get("protocol_sha256") == final["protocol_sha256"],
             "Collection uses another partition/protocol")
        return {"completed_collection": costs["collection"], "collection_checkpoint": manifest["checkpoint"],
                "original_log_files_verified": sum(name.endswith(".log") for name in records)}
    manifests = {name: read(name) for name in records if name.startswith("heldout_") and name.endswith("eval_manifest.json")}
    need(len(manifests) == 5 and all(value.get("purpose") == "heldout" and
         value.get("protocol_sha256") == final["protocol_sha256"] for value in manifests.values()),
         "Server logs do not have five final heldout manifests")
    logs = [name for name in records if name.endswith(".server.log")]
    need(len(logs) == 50 and all((root / name).stat().st_size > 0 for name in logs),
         "Expected original server logs for five arms and ten tasks")
    return {"verified_original_server_logs": len(logs), "heldout_arms": sorted(Path(name).parent.name for name in manifests),
            "scope": "Completed heldout server log bytes and evaluation manifests; no server launched and no health RPC replayed"}


def audit(plan, figure, evidence_root=None):
    root, final, costs, records = verify_plan(plan, figure, evidence_root)
    details = summary_for(root, final, costs, figure, records)
    return {"capture_audit": "completed_evidence_verified", "capture_mode": MODE, "figure": figure,
            "caption": CAPTIONS[figure], "final_manifest_sha256": plan["final_manifest_sha256"],
            "protocol_sha256": plan["protocol_sha256"], "source_files_verified": len(records),
            "source_contract_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "execution_scope": "CPU read-only evidence verification; no training, quantization, rollout, cache generation or GPU execution",
            "details": details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--figure", choices=FIGURES, required=True)
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args(argv)
    plan = load(args.plan)
    self_record = plan.get("script_files", {}).get(Path(__file__).name)
    need(self_record == identity(Path(__file__)), "Audit helper differs from prepared plan")
    result = audit(plan, args.figure, args.evidence_root)
    # The original logs are checked in full. This is the actual result of this
    # read-only command, never a staged or filtered transcript of a training run.
    print(json.dumps(result, ensure_ascii=False, indent=2))
    receipt = {key: value for key, value in result.items() if key not in ("details", "caption", "execution_scope")}
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("Completed evidence verification failed: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
