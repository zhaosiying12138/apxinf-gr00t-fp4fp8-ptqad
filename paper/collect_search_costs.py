#!/usr/bin/env python3
"""Archive completed W4A4 recovery search costs, without running an experiment.

Reads metadata/logs and hashes private tensors on CPU; never imports torch,
loads a model, launches subprocesses, or reads held-out results. Existing output
directories are refused. Public --verify needs only the lightweight archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

try:
    from . import collect_training_costs as costs
except ImportError:  # direct CLI invocation
    import collect_training_costs as costs

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "w4a4_completed_search_costs_v1"
EXCLUDED = ["calibration", "PTQ baking", "installation", "smoke", "failed retries",
            "downtime", "screenshots", "heldout evaluation"]


def need(value, message):
    costs.need(value, message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_identity(record):
    path = Path(record["path"]).resolve(strict=True)
    need(costs.identity(path) == {k: record[k] for k in ("bytes", "sha256")},
         "Recorded input changed: " + str(path))
    return path


class ScopedEvidence:
    def __init__(self, evidence, prefix):
        self.evidence, self.prefix = evidence, prefix

    def add(self, source, target, role):
        return self.evidence.add(source, self.prefix + "/" + target, role)

    def json(self, source, target, role="raw_metadata"):
        return json.loads(self.add(source, target, role))


def copy_evaluation(evidence, folder, target, protocol, purpose, evaluation, protocol_sha):
    manifest = evidence.json(folder / "eval_manifest.json", target + "/eval_manifest.json")
    tasks = evidence.json(folder / "task_results.json", target + "/task_results.json")
    summary = evidence.json(folder / "summary.json", target + "/summary.json")
    part = protocol["partitions"][purpose]
    need(manifest["protocol_sha256"] == protocol_sha and
         manifest["purpose"] == summary["purpose"] == purpose and
         manifest["seed"] == summary["seed"] == part["seed"] and
         manifest["init_state_indices"] == part["init_state_indices"] and
         manifest["episodes"] == part["episodes_per_task"] and
         set(tasks) == set(evaluation.TASKS) and summary["tasks_complete"] == 10,
         "Evaluation protocol/count differs: " + str(folder))
    successes = episodes = 0
    for i, name in enumerate(evaluation.TASKS):
        row = tasks[name]
        raw = folder / (name + ".log")
        parsed = evaluation.parse_log(raw)
        need(row["returncode"] == 0 and all(row[k] == v for k, v in parsed.items()) and
             row["episodes"] == part["episodes_per_task"], "Incomplete task: " + name)
        evaluation.validate_resets(row, part["seed"] + 1000 * i, part["init_state_indices"])
        evidence.add(raw, target + "/" + raw.name, "raw_task_log")
        server = folder / (name + ".server.log")
        if server.is_file():
            evidence.add(server, target + "/" + server.name, "raw_server_log")
        successes += row["successes"]
        episodes += row["episodes"]
        if purpose in ("collection", "teacher_supervision"):
            for filename in ("capture_manifest.json", "capture_counts.json"):
                evidence.add(folder / "observations" / name / filename,
                             target + "/observations/" + name + "/" + filename, "capture_metadata")
    need(summary["total_successes"] == successes and summary["total_episodes"] == episodes,
         "Evaluation summary disagrees with raw logs")
    costs.positive(summary["wall_seconds_including_server_loads"], "evaluation elapsed")


def summarize(root):
    """Recompute every displayed cost from copied raw files, without private paths."""
    index = read(root / "inputs.json")
    protocol_path = root / "protocol/recovery_protocol.json"
    protocol = read(protocol_path)
    protocol_sha = costs.identity(protocol_path)["sha256"]
    need(index.get("protocol_sha256") == protocol_sha and
         costs.frozen_w4a4_protocol_matches(protocol, protocol_sha),
         "Search costs do not match a supported frozen W4A4 protocol")
    rows = {}
    for name, spec in index["candidates"].items():
        base = root / "candidates" / name
        stage = base / "stages" / spec["role"]
        metrics = read(stage / "runtime_metrics.json")
        manifest = read(stage / "recovery_manifest.json")
        state = read(stage / "checkpoint/trainer_state.json")
        receipt = read(base / "receipt.json")
        steps = metrics["global_steps"]
        need(metrics["status"] == "completed" and metrics["error"] is None and
             receipt["status"] == "complete" and
             steps == metrics["requested_optimizer_steps"] == state["global_step"] ==
             state["max_steps"] == receipt["optimizer_steps"], "Incomplete training: " + name)
        draws = sum(costs.microbatches_per_update(manifest, steps)) * manifest["micro_batch"]
        need(abs(draws / manifest["capture_dataset_samples"] - state["epoch"]) < 1e-8,
             "Training epoch/tail-batch accounting differs")
        probe = costs.probe_cost_fields((base / "train.log").read_text(), spec["role"], steps, manifest)
        exported = read(base / "exports" / spec["role"] / "merge_manifest.json")
        dev = read(root / "development" / name / "summary.json")
        need(exported["status"] == "complete" and dev["tasks_complete"] == 10,
             "Incomplete export or development evaluation")
        rows[name] = {
            "role": spec["role"], "selected": name in index["selected"].values(),
            "learning_rate": receipt["learning_rate"], "opd_weight": receipt["opd_weight"],
            "optimizer_steps": steps, "demonstration_window_draws": draws, **probe,
            "wall_seconds": costs.positive(metrics["wall_seconds"], "training elapsed"),
            "timing_scope": metrics["timing_scope"],
            "cuda_peak_memory": costs.validate_peaks(metrics["cuda_peak_memory"]),
            "parameter_storage_by_dtype": costs.validate_storage(metrics["parameter_storage_by_dtype"]),
            "development_wall_seconds": costs.positive(dev["wall_seconds_including_server_loads"], "dev elapsed"),
            "development_episodes": dev["total_episodes"],
            "export_reported_seconds": costs.positive(exported["elapsed_seconds"], "export elapsed"),
            "export_tensor_bytes": exported["output_tensor_bytes"],
            "export_tensor_count": exported["base_tensor_count"],
        }
    selected = {arm: rows[name] for arm, name in index["selected"].items()}
    supervision = read(root / "teacher_supervision/summary.json")
    collection = read(root / "collection/summary.json")
    teacher = read(root / "teacher/teacher_probes.json")
    per_task = Counter(Path(item["path"]).parent.name for item in teacher["source_observation_files"])
    forward_records = re.findall(r"\[probe-cache\] (\d+)/(\d+) pred=\(1, 40, 132\)",
                                 (root / "teacher/labeling.log").read_text())
    need(forward_records == [(str(i), str(teacher["count"])) for i in range(1, teacher["count"] + 1)],
         "Teacher forward log does not match cache count")
    capture_count = lambda folder: sum(read(p)["saved"] for p in
                                      (root / folder / "observations").glob("*/capture_counts.json"))
    qed = read(root / "candidates" / index["selected"]["qad"] / "stages/qad/recovery_manifest.json")
    totals = {key: sum(row[key] for row in rows.values()) for key in (
        "optimizer_steps", "demonstration_window_draws", "scheduled_teacher_backward_passes",
        "wall_seconds", "development_wall_seconds", "development_episodes", "export_reported_seconds")}
    return {
        "format": FORMAT, "scope": "Completed recovery search stages; no heldout results or end-to-end wall clock",
        "protocol_sha256": index["protocol_sha256"], "selected_stages": index["selected"],
        "candidates": rows, "completed_search_totals_by_timing_scope": totals,
        "selected_qad_opd_training": {
            key: selected["qad"][key] + selected["qad_opd"][key] for key in
            ("optimizer_steps", "demonstration_window_draws", "scheduled_teacher_backward_passes", "wall_seconds")},
        "opd_over_continued_training_wall_ratio": selected["qad_opd"]["wall_seconds"] / selected["continued_qad"]["wall_seconds"],
        "teacher_supervision": {
            "episodes": supervision["total_episodes"], "successful_trajectories": supervision["total_successes"],
            "raw_captured_windows": capture_count("teacher_supervision"),
            "training_windows": qed["capture_dataset_samples"],
            "wall_seconds": supervision["wall_seconds_including_server_loads"], "cuda_peak_memory": None},
        "student_collection": {"episodes": collection["total_episodes"],
            "captured_observations": capture_count("collection"),
            "wall_seconds": collection["wall_seconds_including_server_loads"], "cuda_peak_memory": None},
        "teacher_labeling": {"elapsed_seconds": teacher["elapsed_seconds"], "timing_scope": teacher["timing_scope"],
            "model_dtype": teacher["model_dtype"], "autocast_dtype": teacher["autocast_dtype"],
            "forward_calls": len(forward_records), "backward_calls": 0,
            "source_observations_per_task": dict(per_task),
            "cuda_peak_memory": costs.validate_peaks(teacher["cuda_peak_memory"])},
        "pressure_development": {name: {
            "episodes": read(root / "pressure" / name / "summary.json")["total_episodes"],
            "wall_seconds": read(root / "pressure" / name / "summary.json")["wall_seconds_including_server_loads"]}
            for name in index["pressure_arms"]},
        "excluded": EXCLUDED,
        "accounting_notes": [costs.ACCOUNTING_NOTES[i] for i in (0, 1, 2, 3, 5, 7, 8)] + [
            "Search totals sum only completed stages of each timing scope; selected QAD training is counted once.",
            "Teacher supervision is training-data collection, not a heldout success-rate estimate.",
            "Model weight identities are retained from producer receipts, not rehashed by this collector. Observation and teacher-cache hashes are checked; tensor bytes are omitted.",
            "Missing collection/export allocator peaks, recovery power/energy, separate adapter preparation and end-to-end elapsed are not imputed."],
    }


def verify(folder):
    folder = Path(folder).resolve(strict=True)
    manifest = read(folder / "evidence_manifest.json")
    need(manifest.get("format") == FORMAT and manifest.get("status") == "complete", "Incomplete archive")
    listed = set()
    for item in manifest["files"]:
        relative = Path(item["published_path"])
        need(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() not in listed,
             "Unsafe or duplicate archive path")
        path = folder / relative
        need(not path.is_symlink() and path.resolve().is_relative_to(folder), "Archive path escapes root")
        need(costs.identity(path) == {k: item[k] for k in ("bytes", "sha256")}, "Archive hash differs: " + str(relative))
        listed.add(relative.as_posix())
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    need(actual == listed | {"evidence_manifest.json"}, "Archive has missing or unexpected files")
    result = summarize(folder)
    need(read(folder / "summary.json") == result, "Derived cost summary differs from raw evidence")
    return {"status": "verified", "files": len(listed), "protocol_sha256": result["protocol_sha256"],
            "completed_search_totals_by_timing_scope": result["completed_search_totals_by_timing_scope"]}


def collect(run_dir, out):
    run, out = Path(run_dir).resolve(strict=True), Path(out).resolve()
    need(not out.exists(), "Refusing existing output: " + str(out))
    evidence = costs.Evidence()
    # The mutable orchestrator state locates inputs. Do not archive its unrelated
    # in-progress heldout entries; the individual completed receipts below are
    # the authoritative inputs for this cost package.
    state_bytes = (run / "run_manifest.json").read_bytes()
    state = json.loads(state_bytes)
    protocol = evidence.json(state["protocol_file"], "protocol/recovery_protocol.json")
    protocol_sha = costs.identity(state["protocol_file"])["sha256"]
    need(state["protocol_sha256"] == protocol_sha and
         costs.frozen_w4a4_protocol_matches(protocol, protocol_sha),
         "Search costs require the archived v11 or frozen v12 W4A4 protocol")
    if protocol.get("version") == 12:
        need(state.get("selected_recipe") == "rtn_w4a4_category",
             "v12 search costs require the selected RTN W4A4 recipe")
    evaluation = costs.evaluation_helpers(ROOT / "eval/run_recovery_eval.py")
    for relative in ("paper/collect_training_costs.py", "paper/collect_search_costs.py", "eval/run_recovery_eval.py"):
        evidence.add(ROOT / relative, "source/" + relative, "collector_time_source")

    def receipt(name, target):
        row = evidence.json(run / "stages" / (name + ".json"), target)
        need(row.get("status") == "complete" and row.get("protocol_sha256") == protocol_sha,
             "Missing completed stage receipt: " + name)
        return row

    selections = {}
    for kind, name in (("qad", "select_qad_lr"), ("opd", "select_opd_weight")):
        rec = receipt(name, "selection/" + name + "_receipt.json")
        path = checked_identity(rec["selection_identity"])
        selections[kind] = evidence.json(path, "selection/" + name + ".json")

    def resolve(record):
        path = checked_identity(record)
        target = "selection/inputs/" + hashlib.sha256(str(path).encode()).hexdigest()[:20] + "-" + path.name
        evidence.add(path, target, "selection_input")
        return path

    selected_lr = costs.audit_recovery_selection(selections["qad"], "qad", protocol, protocol_sha, resolve, evaluation)
    selected_weight = costs.audit_recovery_selection(selections["opd"], "opd", protocol, protocol_sha, resolve, evaluation)
    qad_name = "train_qad_lr_" + str(float(selected_lr))
    initial = run / "artifacts" / qad_name / ("checkpoint-" + str(protocol["selection"]["qad_optimizer_steps"]))
    names = [("train_qad_lr_" + str(float(lr)), "qad", lr, 0.) for lr in protocol["selection"]["qad_learning_rates"]]
    names += [("train_continued_qad", "continued_qad", selected_lr, 0.)]
    need(protocol["selection"]["opd_weights"] == [.25, 1.], "Unsupported OPD stage naming")
    names += [("train_opd_025", "qad_opd", selected_lr, .25), ("train_opd_100", "qad_opd", selected_lr, 1.)]
    inputs = {"protocol_sha256": protocol_sha,
        "run_manifest_locator": {"path": str(run / "run_manifest.json"), "bytes": len(state_bytes),
                                 "sha256": hashlib.sha256(state_bytes).hexdigest()},
        "candidates": {}, "selected": {
        "qad": qad_name, "continued_qad": "train_continued_qad",
        "qad_opd": "train_opd_025" if selected_weight == .25 else "train_opd_100"}}
    manifests = {}
    for name, role, lr, weight in names:
        scoped = ScopedEvidence(evidence, "candidates/" + name)
        rec = receipt(name, "candidates/" + name + "/receipt.json")
        need(rec["learning_rate"] == lr and rec["opd_weight"] == weight, "Training stage configuration differs")
        for key in ("runtime_identity", "training_request_identity"):
            checked_identity(rec[key])
        derived = costs.normalized_protocol(protocol, {"selected_qad_learning_rate": lr,
                                                     "selected_opd_weight": weight or selected_weight})
        folder = run / "artifacts" / name
        expected_steps = protocol["selection"]["qad_optimizer_steps" if role == "qad" else "continuation_optimizer_steps"]
        need(rec["optimizer_steps"] == expected_steps, "Stage update count differs from frozen budget")
        stage, meta, checkpoint = costs.audit_stage(folder, role, expected_steps, derived,
            protocol_sha, scoped, None if role == "qad" else initial,
            source_snapshots=run / "operations/opd-tail-fix/source_snapshots")
        need(costs.identity(folder / "recovery_manifest.json")["sha256"] == rec["recovery_manifest_sha256"],
             "Training recovery metadata differs from receipt")
        evidence.add(run / "logs" / (name + ".log"), "candidates/" + name + "/train.log", "raw_training_log")
        merge_name, dev_name = name.replace("train_", "merge_", 1), name.replace("train_", "dev_", 1)
        merged = costs.audit_merge(run / "artifacts" / merge_name, checkpoint, meta, scoped, role)
        merge_rec = receipt(merge_name, "candidates/" + name + "/export_receipt.json")
        need(merge_rec["merge_manifest_sha256"] == costs.identity(run / "artifacts" / merge_name / "merge_manifest.json")["sha256"],
             "Export differs from completed receipt")
        dev_rec = receipt(dev_name, "development/" + name + "/receipt.json")
        for item in dev_rec["evaluation_identity"].values():
            checked_identity(item)
        copy_evaluation(evidence, run / "artifacts" / dev_name, "development/" + name, protocol, "development", evaluation, protocol_sha)
        inputs["candidates"][name] = {"role": role}
        manifests[name] = meta

    pressure_path = Path(state["selection_file"])
    need(costs.identity(pressure_path)["sha256"] == state["selection_sha256"], "PTQ selection changed")
    pressure = evidence.json(pressure_path, "pressure/selection.json")
    need(pressure["status"] == "complete" and pressure["protocol_sha256"] == protocol_sha and
         pressure["selection_uses_heldout"] is False, "Invalid PTQ selection")
    inputs["pressure_arms"] = list(pressure["arms"])
    for rel, sha in pressure["source_sha256"].items():
        need(costs.identity(pressure_path.parent / rel)["sha256"] == sha, "PTQ selection source changed")
    for name in inputs["pressure_arms"]:
        copy_evaluation(evidence, pressure_path.parent / name, "pressure/" + name, protocol, "development", evaluation, protocol_sha)

    supervision = Path(manifests[qad_name]["capture_dataset"])
    copy_evaluation(evidence, supervision, "teacher_supervision", protocol, "teacher_supervision", evaluation, protocol_sha)
    # The bound training request identifies all usable success-window inputs.
    capture = state["capture_dataset_identity"]
    need(capture["root"] == str(supervision) and capture["sample_count"] == manifests[qad_name]["capture_dataset_samples"],
         "Training demonstration identity differs")
    # The frozen identity includes rejected_*.pt as well as usable sample_*.pt.
    # Match the producer's capture_dataset_id and replay loader convention.
    need(sum(Path(item["path"]).name.startswith("sample_") and
             Path(item["path"]).suffix == ".pt" for item in capture["files"]) == capture["sample_count"],
         "Bound demonstration tensor count differs")
    for item in capture["files"]:
        path = supervision / item["path"]
        need(costs.identity(path) == {k: item[k] for k in ("bytes", "sha256")}, "Demonstration source changed")
    collection_rec = receipt("collection_qad", "collection/receipt.json")
    for item in collection_rec["evaluation_identity"].values():
        checked_identity(item)
    copy_evaluation(evidence, run / "artifacts/collection_qad", "collection", protocol, "collection", evaluation, protocol_sha)
    evidence.add(run / "logs/collection_qad.log", "collection/driver.log", "raw_collection_driver")

    teacher_rec = receipt("teacher_cache", "teacher/receipt.json")
    checked_identity(teacher_rec["cache_identity"])
    teacher = evidence.json(checked_identity(teacher_rec["metadata_identity"]), "teacher/teacher_probes.json")
    need(teacher["source_kind"] == "student_rollout" and teacher["count"] == teacher["requested_count"] == 160 and
         teacher["timing_scope"] == costs.TEACHER_SCOPE and teacher["eval_mode"] is True,
         "Teacher labeling scope/count differs")
    need(len({item["path"] for item in teacher["source_observation_files"]}) == 160, "Duplicate teacher source")
    for item in teacher["source_observation_files"]:
        checked_identity(item)
    for field, relative in (("labeling_implementation_sha256", "rl/opd_probe_cache.py"),
                            ("replay_implementation_sha256", "rl/probe_distill.py")):
        source = costs.source_file(relative, teacher[field], ROOT, run / "operations/opd-tail-fix/source_snapshots")
        evidence.add(source, "teacher/source/" + relative, "verified_teacher_producer_bytes")
    evidence.add(run / "logs/teacher-cache.log", "teacher/labeling.log", "raw_teacher_labeling_log")
    cache_sha = teacher_rec["cache_identity"]["sha256"]
    need(all(meta["probe_cache_sha256"] == cache_sha for meta in manifests.values() if meta["probe_weight"]),
         "OPD candidates do not share the same teacher cache")
    # Recheck in-memory source bytes before exposing a package; collectors may
    # run while documentation/code elsewhere in the checkout is being edited.
    for row, data in evidence.files.values():
        need(costs.identity(row["original_absolute_path"]) == {k: row[k] for k in ("bytes", "sha256")},
             "Evidence changed during collection")
    out.mkdir(parents=True, exist_ok=False)
    for row, data in evidence.files.values():
        target = out / row["published_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (out / "inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n")
    result = summarize(out)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text(
        "# 已完成恢复搜索的成本证据\n\n"
        "`summary.json` 从本目录原始副本复算选定训练路径和全部候选成本。"
        "`evidence_manifest.json` 保留原始路径、文件长度、SHA-256与复制用途。\n\n"
        "验证命令：`python3 paper/collect_search_costs.py --verify paper/evidence/search_costs`。"
        "验证只需公开副本，不读取模型、教师缓存或私有原目录。\n\n"
        "本包不包含 held-out 结果；各计时范围分别相加，不能作为统一端到端耗时。"
        "模型、观测张量和教师缓存未复制；保留生产者记录的权重身份，采集时核对观测和缓存哈希。\n", encoding="utf-8")
    files = [row for row, _ in evidence.files.values()]
    files += [{"original_absolute_path": None, "published_path": name, "role": "derived_cost_evidence",
               **costs.identity(out / name)} for name in ("inputs.json", "summary.json", "README.md")]
    manifest = {"format": FORMAT, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "private_tensor_verification": "Observation/cache hashes checked; model identities retained from producer receipts; tensor bytes omitted",
                "files": sorted(files, key=lambda row: row["published_path"])}
    (out / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return verify(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        if args.run_dir or args.out:
            parser.error("--verify is exclusive with --run-dir/--out")
        result = verify(args.verify)
    else:
        if args.run_dir is None or args.out is None:
            parser.error("provide --run-dir and a new --out")
        result = collect(args.run_dir, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
