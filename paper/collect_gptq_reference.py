#!/usr/bin/env python3
"""Archive one verified GPTQ reference; replay public logs/statistics on CPU.

Collection calls the original comparison, including its real weight/H checks.
Public verification never opens an original absolute path, imports torch, or
claims to repeat tensor verification. Original JSON and log bytes are preserved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import ModuleType, SimpleNamespace

try:
    from .collect_training_costs import Evidence, identity
except ImportError:
    # Direct execution, including Python -I from a relocated source snapshot.
    _helper_spec = importlib.util.spec_from_file_location(
        "_gptq_archive_cost_helpers", Path(__file__).with_name("collect_training_costs.py"))
    _helper = importlib.util.module_from_spec(_helper_spec)
    _helper_spec.loader.exec_module(_helper)
    Evidence, identity = _helper.Evidence, _helper.identity

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "w4a4_gptq_reference_archive_v1"
ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")
JSON_NAMES = ("eval_manifest.json", "task_results.json", "summary.json")
REPLAY_SOURCES = (
    "eval/run_recovery_eval.py", "eval/compare_recovery.py",
    "exp/run_high_fp4_v3.py", "paper/paired_uncertainty.py",
    "eval/compare_gptq_reference.py", "paper/publication_guard.py",
    "paper/activation_evidence.py",
)
SCOPE = {
    "raw_logs_replayed": True,
    "paired_statistics_recomputed": True,
    "source_time_tensor_verification_receipt_preserved": True,
    "tensor_bytes_included": False,
    "tensor_contents_reverified_offline": False,
}


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local(root, relative):
    need(isinstance(relative, str) and relative and "\\" not in relative,
         "Invalid archive path")
    name = PurePosixPath(relative)
    need(not name.is_absolute() and ".." not in name.parts and name.as_posix() == relative,
         "Unsafe archive path: " + relative)
    path = Path(root) / relative
    need(not any(p.is_symlink() for p in (path, *path.parents)) and
         path.resolve().is_relative_to(Path(root).resolve()), "Archive path escapes root")
    return path


def matches(path, record):
    actual = identity(path)
    need(actual["sha256"] == record.get("sha256") and
         ("bytes" not in record or actual["bytes"] == record["bytes"]),
         "Evidence identity differs: " + str(path))
    return actual


def record(path):
    return {"path": str(Path(path).resolve()), **identity(path)}


def publish_new(stage, out):
    """The same no-replace publication primitive as the action archive."""
    if sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(stage), -100, os.fsencode(out), 1):
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number), str(out))
    elif os.name == "nt":
        stage.rename(out)
    else:
        raise ValueError("Atomic no-replace publication supports Linux/Windows; verification is portable")


@contextmanager
def replay_sources(folder):
    """Execute the byte-verified source copies, without cached checkout modules.

All seven modules have standard-library-only top-level imports. None of the
model/weight audit entry points is called here. Restore import state even when
a malformed archive fails, so the surrounding publication process is unchanged.
"""
    snapshot = Path(folder) / "source"
    names = ("eval", "exp", "paper", "run_recovery_eval", "compare_recovery",
             "exp.run_high_fp4_v3", "paper.paired_uncertainty", "publication_guard",
             "activation_evidence", "_archived_gptq_comparison")
    previous = {name: sys.modules.get(name) for name in names}
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        for package in ("eval", "exp", "paper"):
            module = ModuleType(package)
            module.__path__ = [str(snapshot / package)]
            sys.modules[package] = module

        def load(name, relative):
            path = local(snapshot, relative)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            need(Path(module.__file__).resolve() == path.resolve(), "Replay loaded another source")
            return module

        parser = load("run_recovery_eval", "eval/run_recovery_eval.py")
        pairing = load("compare_recovery", "eval/compare_recovery.py")
        load("exp.run_high_fp4_v3", "exp/run_high_fp4_v3.py")
        load("paper.paired_uncertainty", "paper/paired_uncertainty.py")
        comparison = load("_archived_gptq_comparison", "eval/compare_gptq_reference.py")
        load("publication_guard", "paper/publication_guard.py")
        activation = load("activation_evidence", "paper/activation_evidence.py")
        yield SimpleNamespace(parser=parser, pairing=pairing, comparison=comparison,
                              activation=activation)
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def audit_files(audit):
    """Map the original audit's three JSON files and task log pairs by basename."""
    rows = {}
    for name, value in audit["evaluation_identity"].items():
        need(name in JSON_NAMES and Path(value["path"]).name == name, "Unexpected evaluation identity")
        rows[name] = value
    need(set(rows) == set(JSON_NAMES), "Incomplete evaluation identities")
    for task, value in audit["raw_log_identities"].items():
        name = task + ".log"
        need(Path(value["path"]).name == name and name not in rows, "Unexpected raw-log identity")
        rows[name] = value
    servers = audit.get("activation_installation", {}).get("server_logs", [])
    for value in servers:
        name = Path(value["path"]).name
        need(name.endswith(".server.log") and name not in rows, "Duplicate/invalid server identity")
        rows[name] = value
    return rows


def replay_evaluation(folder, audit, protocol, protocol_path, purpose, quantized, modules):
    parser, pairing = modules.parser, modules.pairing
    manifest, results, summary = (read(folder / name) for name in JSON_NAMES)
    expected_names = set(JSON_NAMES) | {task + suffix for task in parser.TASKS
                                      for suffix in (".log", ".server.log")}
    records = audit_files(audit)
    need(set(records) == expected_names, "Evaluation audit must contain ten complete log pairs")
    for name, receipt in records.items():
        matches(folder / name, receipt)
    part = protocol["partitions"][purpose]
    expected = {"purpose": purpose, "seed": part["seed"], "episodes": part["episodes_per_task"],
                "tasks": parser.TASKS, "init_state_indices": part["init_state_indices"],
                "protocol_sha256": identity(protocol_path)["sha256"]}
    for key in ("task_count", "n_envs", "task_seed_stride", "episode_seed_stride",
                "server_seed_offset", "settle_steps", "n_action_steps", "max_episode_steps",
                "initial_state_protocol"):
        expected[key] = protocol["evaluation_contract"][key]
    need(all(manifest.get(k) == v and (type(v) is not int or type(manifest[k]) is int)
             for k, v in expected.items()), "Archived evaluation protocol differs")
    adapter = purpose == "collection" or folder.name in ("heldout_qad", "heldout_continued_qad", "heldout_qad_opd")
    variables = manifest.get("environment_summary", {}).get("variables", {})
    need(all(variables.get(key) == value for key, value in {
        "FP4VLA_QUANT": "0", "FP4VLA_W4A4": str(int(quantized)),
        "FP4VLA_W4A4_ADAPTER": str(int(adapter)),
        "FP4VLA_SATURATE_F16_ACTIVATIONS": str(int(quantized)),
    }.items()), "Archived evaluation numerical environment differs")
    need(set(results) == set(parser.TASKS), "Incomplete task result set")
    if purpose == "heldout":
        need(pairing._validate_manifest("gptq archive", manifest,
             {manifest["protocol_file"]: protocol_path}), "Versioned evaluation required")
    successes, episodes, reset_pairs = 0, 0, []
    for ti, task in enumerate(parser.TASKS):
        row = results[task]
        need(type(row.get("returncode")) is int and row["returncode"] == 0,
             "Incomplete task return code")
        parsed = parser.parse_log(folder / (task + ".log"))
        need(all(key in row and row[key] == value for key, value in parsed.items()),
             "Raw log differs from task results: " + task)
        outcomes = row["results"]
        need(len(outcomes) == part["episodes_per_task"] and
             all(type(value) is bool for value in outcomes), "Incomplete Boolean outcomes")
        seed = part["seed"] + 1000 * ti
        pairing._validate_declared_accounting("archive", task, row, outcomes, seed, True)
        resets = parser.validate_resets(row, seed, part["init_state_indices"])
        for reset in resets:
            reset_pairs.append({"task": task, **{key: reset[key] for key in (
                "episode_index", "seed", "init_state_index", "settle_steps",
                "initial_state_sha256", "restored_state_sha256", "init_state_bank_sha256")}})
        modules.activation.validate_activation_log(folder / (task + ".server.log"), protocol,
                                                   quantized=quantized)
        successes += sum(outcomes)
        episodes += len(outcomes)
    macro = sum(results[t]["successes"] / results[t]["episodes"] for t in parser.TASKS) / len(parser.TASKS)
    need(audit["pairing_sha256"] == hashlib.sha256(json.dumps(reset_pairs, sort_keys=True).encode()).hexdigest(),
         "Recorded pairing digest differs from raw resets")
    for key, value in {"successes": successes, "episodes": episodes,
                       "macro_success_rate": macro, "micro_success_rate": successes / episodes}.items():
        need(audit.get(key) == value, "Recorded evaluation audit accounting differs: " + key)
    if purpose == "heldout":
        pairing._validate_summary("archive", summary, manifest, successes, episodes, macro, True)
    else:
        need(all(summary.get(k) == v for k, v in {"purpose": purpose, "seed": part["seed"],
             "tasks_complete": len(parser.TASKS), "total_successes": successes,
             "total_episodes": episodes, "macro_success_rate": macro}.items()),
             "Collection summary differs from logs")
    need(summary.get("micro_success_rate") == successes / episodes, "Micro success accounting differs")
    return manifest


def calibration_costs(folder):
    def seconds(path):
        value = read(path).get("elapsed_seconds")
        need(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
             "Invalid recorded calibration elapsed_seconds")
        return value

    receipt = read(folder / "artifacts/category_bake_invocation.json")
    start, finish = receipt.get("started_utc"), receipt.get("finished_utc")
    wall = None
    if start is not None and finish is not None:
        a, b = datetime.fromisoformat(start), datetime.fromisoformat(finish)
        need(a.tzinfo is not None and b.tzinfo is not None, "Invocation timestamps must carry time zones")
        wall = (b - a).total_seconds()
        need(math.isfinite(wall) and wall >= 0, "Invalid category invocation duration")
    return {
        "ordinary_collection_seconds": seconds(folder / "calibration/ordinary/calib_meta.json"),
        "ordinary_bake_seconds": seconds(folder / "artifacts/ordinary_parent/bake_manifest.json"),
        "category_collection_seconds": seconds(folder / "calibration/category/calib_meta.json"),
        "category_bake_wall_seconds": wall,
        "scopes": {
            "ordinary_collection_seconds": "Collector-recorded elapsed_seconds; producer-defined collection scope.",
            "ordinary_bake_seconds": "Ordinary bake producer-recorded elapsed_seconds.",
            "category_collection_seconds": "Collector-recorded elapsed_seconds; producer-defined collection scope.",
            "category_bake_wall_seconds": "Category wrapper finished_utc minus started_utc; includes process startup and logging.",
        },
    }


def verify(folder, main_evidence=None):
    folder = Path(folder).resolve(strict=True)
    main_evidence = Path(main_evidence).resolve(strict=True) if main_evidence is not None else folder.parent
    manifest_path = local(folder, "evidence_manifest.json")
    manifest = read(manifest_path)
    need(manifest.get("format") == FORMAT and manifest.get("status") == "complete" and
         manifest.get("scope") == SCOPE, "Incomplete or unsupported GPTQ archive")
    files, indexed = {manifest_path}, {}
    for row in manifest["files"]:
        relative = row["published_path"]
        need(relative not in indexed, "Duplicate archive path")
        path = local(folder, relative)
        need(path.suffix.lower() not in (".pt", ".pth", ".safetensors", ".npy", ".npz"),
             "Tensor payloads must not enter the GPTQ archive")
        matches(path, row)
        indexed[relative] = row
        files.add(path)
    actual = set()
    for path in folder.rglob("*"):
        need(not path.is_symlink(), "Archive contains a symbolic link")
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    need(actual == set(indexed) | {"evidence_manifest.json"}, "Missing or unlisted archive files")
    external = {}
    for row in manifest["main_evidence_files"]:
        relative = row["published_path"]
        need(relative not in external, "Duplicate main evidence binding")
        path = local(main_evidence, relative)
        matches(path, row)
        external[relative] = row
        files.add(path)

    report = read(folder / "comparison.json")
    need(report.get("format") == "w4a4_gptq_supplement_comparison_v1" and report.get("status") == "verified",
         "A verified supplementary comparison is required")
    matches(folder / "comparison.json", manifest["source_report"])
    matches(folder / "protocol/supplement.json", report["protocol"])
    matches(main_evidence / "final_manifest.json", report["final_manifest"])
    matches(main_evidence / "paired_comparison.json", report["main_comparison"])
    matches(folder / "main/run_manifest.json", report["main_run_manifest"])
    plan, protocol = read(folder / "protocol/supplement.json"), read(folder / "protocol/main.json")
    matches(folder / "protocol/main.json", plan["main_protocol"])
    matches(folder / "protocol/previous_supplement.json",
            {"sha256": plan["amendment"]["previous_protocol_sha256"]})
    matches(folder / "calibration/inputs.json", plan["capture_manifest"])
    protocol_sha = identity(folder / "protocol/main.json")["sha256"]
    final = read(main_evidence / "final_manifest.json")
    need(final["protocol_sha256"] == manifest["protocol_sha256"] == protocol_sha and
         final.get("selection_uses_heldout") is False and final.get("required_arms") == list(ARMS),
         "Final manifest differs from the fixed five-arm experiment")
    need(read(folder / "main/run_manifest.json").get("status") == "complete", "Main run was incomplete")
    matches(folder / "main/selection.json", {"sha256": final["selection_sha256"]})
    need(plan["execution"]["eval_partition"] == protocol["partitions"]["heldout"] and
         plan["execution"]["eval_contract"] == protocol["evaluation_contract"], "Supplement protocol differs from main")
    for name, row in {**plan["preparation_source_files"], **report["analysis_sources"]}.items():
        matches(local(folder, "source/" + name), row)
    expected_frozen = {row["path"]: row for row in (plan["main_protocol"], plan["capture_manifest"],
                                                    *plan["preparation_source_files"].values())}
    need(set(report["frozen_sources"]) == set(expected_frozen), "Frozen source set differs")
    for name, row in expected_frozen.items():
        need(all(report["frozen_sources"][name][key] == row[key] for key in ("bytes", "sha256")),
             "Frozen source receipt differs: " + name)
    for name in (*REPLAY_SOURCES, "paper/collect_gptq_reference.py", "paper/collect_training_costs.py"):
        need("source/" + name in indexed, "Replay source is missing: " + name)

    gptq = report["gptq_evidence"]
    for key, target in (("root_identity", "root_bf16"), ("parent_identity", "ordinary_parent"),
                        ("weight_identity", "w4a4_category")):
        for name, row in gptq[key]["metadata"].items():
            matches(local(folder, f"artifacts/{target}/{name}"), row)
    for kind in ("ordinary", "category"):
        receipt = gptq["calibration_provenance"][kind]
        path = folder / f"calibration/{kind}/calib_meta.json"
        matches(path, receipt["metadata"])
        need(read(path) == receipt["metadata_content"], "Calibration metadata content differs")
    for arm, records in report["selected_checkpoint_sources"].items():
        for original, row in records.items():
            if Path(original).suffix == ".json":
                prefix = hashlib.sha256(str(Path(original).parent).encode()).hexdigest()[:16]
                path = local(folder, f"selected_metadata/{arm}/{prefix}/{Path(original).name}")
                matches(path, row)
    invocation = gptq["category_bake_invocation"]
    matches(folder / "artifacts/category_bake_invocation.json", invocation["receipt"])
    matches(folder / "artifacts/category_bake.log", invocation["log"])
    call = read(folder / "artifacts/category_bake_invocation.json")
    need(call.get("status") == "complete" and type(call.get("returncode")) is int and call["returncode"] == 0 and
         call.get("command") == invocation["command"], "Category invocation was incomplete or differs")
    for name in ("category_ptq_recipe.json", "category_bake_manifest.json"):
        need("artifacts/w4a4_category/" + name in indexed, "Missing category provenance")
    category_recipe = read(folder / "artifacts/w4a4_category/category_ptq_recipe.json")
    category_bake = read(folder / "artifacts/w4a4_category/category_bake_manifest.json")
    for name, field, digest_field in (
        ("ptq_recipe.json", "parent_recipe", "parent_recipe_sha256"),
        ("bake_manifest.json", "parent_bake_manifest", "parent_bake_manifest_sha256"),
    ):
        need("artifacts/ordinary_parent/" + name in indexed, "Missing ordinary GPTQ provenance")
        parent_file = folder / "artifacts/ordinary_parent" / name
        matches(parent_file, category_bake[field])
        need(category_recipe[digest_field] == identity(parent_file)["sha256"] and
             Path(category_bake[field]["path"]) == Path(gptq["parent_identity"]["path"]) / name,
             "Category recipe is not bound to the ordinary parent provenance")
    teacher = read(folder / "calibration/inputs.json")["source_audit"]
    for name in JSON_NAMES:
        need("calibration/teacher/" + name in indexed, "Missing calibration teacher receipt")
    for task, row in teacher["tasks"].items():
        matches(folder / ("calibration/teacher/" + task + ".log"), {"sha256": row["raw_log_sha256"]})
        capture_folder = folder / "calibration/teacher/observations" / task
        matches(capture_folder / "capture_manifest.json", {"sha256": row["capture_manifest_sha256"]})
        matches(capture_folder / "reset_events.jsonl",
                {"sha256": read(capture_folder / "capture_manifest.json")["reset_event_file_sha256"]})

    with replay_sources(folder) as modules:
        need(identity(folder / "protocol/supplement.json")["sha256"] == modules.comparison.PLAN_SHA256,
             "Supplement protocol is not the frozen analysis plan")
        tasks = modules.parser.TASKS
        expected_main = {"final_manifest.json", "paired_comparison.json"} | {
            f"heldout_{arm}/{name}" for arm in ARMS for name in
            (*JSON_NAMES, *(task + suffix for task in tasks for suffix in (".log", ".server.log")))}
        need(set(external) == expected_main, "Main evidence binding set is incomplete or unexpected")
        expected_arms = set(ARMS) | {"gptq"}
        need(set(report["raw_evaluation_audits"]) == expected_arms, "Six evaluation audits are required")
        protocol_path = folder / "protocol/main.json"
        episodes, manifests = {}, {}
        for arm in (*ARMS, "gptq"):
            arm_folder = (folder if arm == "gptq" else main_evidence) / ("heldout_" + arm)
            manifests[arm] = replay_evaluation(arm_folder, report["raw_evaluation_audits"][arm],
                protocol, protocol_path, "heldout", arm != "bf16", modules)
            episodes[arm] = modules.comparison.read_episodes(arm_folder)
        first = manifests["bf16"]
        need(all(all(m.get(key) == first.get(key) for key in modules.pairing.PROTOCOL_FIELDS)
                 for m in manifests.values()), "Heldout execution contracts differ")
        need(all(m.get("collection_manifest") == first.get("collection_manifest") for m in manifests.values()),
             "Heldout collection references differ")
        collection = replay_evaluation(folder / "collection", report["collection_audit"],
            protocol, protocol_path, "collection", True, modules)
        need(collection["checkpoint"] == final["selected_qad_model_identity"]["path"] and
             first["collection_manifest"] == report["collection_audit"]["evaluation_identity"]["eval_manifest.json"]["path"],
             "Collection was not from the selected QAD checkpoint")
        paired = modules.pairing.compare_round(main_evidence, {first["protocol_file"]: protocol_path})
        need(paired == read(main_evidence / "paired_comparison.json"), "Main paired comparison no longer reproduces")
        need(episodes == report["episodes"], "Supplement episode table differs from raw logs")
        computed = modules.comparison.summarize(episodes, plan)
        need(all(report.get(key) == value for key, value in computed.items()),
             "Supplement statistics do not reproduce from paired raw logs")

    return {"status": "verified", "comparison": report, "manifest": manifest,
            "files": sorted(files), "protocol_sha256": protocol_sha,
            "calibration_costs": calibration_costs(folder), **SCOPE}


def collect(report_path, out, main_evidence=None):
    report_path, out = Path(report_path).resolve(strict=True), Path(out).absolute()
    main_evidence = Path(main_evidence or ROOT / "paper/evidence").resolve(strict=True)
    need(not out.exists() and not out.is_symlink(), "Refusing existing GPTQ archive: " + str(out))
    report = read(report_path)
    need(report.get("format") == "w4a4_gptq_supplement_comparison_v1" and report.get("status") == "verified",
         "A verified supplementary comparison is required")
    evidence, main_rows = Evidence(), {}

    def add(path, target, role, expected=None):
        path = Path(path)
        need(path.is_file() and not path.is_symlink(), "Missing regular source: " + str(path))
        if expected is not None:
            matches(path, expected)
        return evidence.add(path, target, role)

    def bind_main(source, target):
        path = local(main_evidence, target)
        actual = matches(path, source)
        main_rows[target] = {"original_absolute_path": source["path"], "published_path": target,
                             **actual, "role": "existing_main_experiment_evidence"}

    add(report_path, "comparison.json", "verified_original_supplement_comparison")
    plan = json.loads(add(report["protocol"]["path"], "protocol/supplement.json", "frozen_supplement_protocol", report["protocol"]))
    add(ROOT / plan["main_protocol"]["path"], "protocol/main.json", "frozen_main_protocol", plan["main_protocol"])
    add(ROOT / plan["amendment"]["previous_protocol_path"], "protocol/previous_supplement.json",
        "previous_supplement_protocol", {"sha256": plan["amendment"]["previous_protocol_sha256"]})
    capture = json.loads(add(ROOT / plan["capture_manifest"]["path"], "calibration/inputs.json",
                            "frozen_calibration_capture", plan["capture_manifest"]))
    for name, row in plan["preparation_source_files"].items():
        add(ROOT / row["path"], "source/" + name, "frozen_preparation_source", row)
    for name, row in report["analysis_sources"].items():
        add(row["path"], "source/" + name, "original_analysis_source", row)
    for name in REPLAY_SOURCES:
        if "source/" + name not in evidence.files:
            add(ROOT / name, "source/" + name, "publication_replay_source")
    add(__file__, "source/paper/collect_gptq_reference.py", "archive_implementation")
    add(ROOT / "paper/collect_training_costs.py", "source/paper/collect_training_costs.py", "archive_dependency")
    bind_main(report["final_manifest"], "final_manifest.json")
    bind_main(report["main_comparison"], "paired_comparison.json")
    final = read(main_evidence / "final_manifest.json")
    add(report["main_run_manifest"]["path"], "main/run_manifest.json", "completed_main_run", report["main_run_manifest"])
    add(final["selection_file"], "main/selection.json", "original_development_selection",
        {"sha256": final["selection_sha256"]})
    for arm, audit in report["raw_evaluation_audits"].items():
        need(arm in (*ARMS, "gptq"), "Unexpected comparison arm")
        for name, row in audit_files(audit).items():
            target = "heldout_" + arm + "/" + name
            if arm in ARMS:
                bind_main(row, target)
            else:
                add(row["path"], target, "supplement_raw_evaluation", row)
    for name, row in audit_files(report["collection_audit"]).items():
        add(row["path"], "collection/" + name, "selected_qad_collection", row)
    for arm, records in report["selected_checkpoint_sources"].items():
        for path, row in records.items():
            if Path(path).suffix == ".json":
                prefix = hashlib.sha256(str(Path(path).parent).encode()).hexdigest()[:16]
                add(path, f"selected_metadata/{arm}/{prefix}/{Path(path).name}", "selected_checkpoint_metadata", row)
    gptq = report["gptq_evidence"]
    for key, target in (("root_identity", "root_bf16"), ("parent_identity", "ordinary_parent"),
                        ("weight_identity", "w4a4_category")):
        for name, row in gptq[key]["metadata"].items():
            add(row["path"], f"artifacts/{target}/{name}", "gptq_checkpoint_metadata", row)
    for name in ("category_ptq_recipe.json", "category_bake_manifest.json"):
        add(Path(gptq["checkpoint"]) / name, "artifacts/w4a4_category/" + name, "gptq_category_provenance")
    # checkpoint_identity has only model metadata. validate_parent currently
    # adds these receipts, but never rely on that incidental augmentation.
    category_bake = read(Path(gptq["checkpoint"]) / "category_bake_manifest.json")
    for name, field in (("ptq_recipe.json", "parent_recipe"), ("bake_manifest.json", "parent_bake_manifest")):
        add(Path(gptq["parent_identity"]["path"]) / name, "artifacts/ordinary_parent/" + name,
            "ordinary_gptq_provenance", category_bake[field])
    for kind in ("ordinary", "category"):
        row = gptq["calibration_provenance"][kind]["metadata"]
        add(row["path"], f"calibration/{kind}/calib_meta.json", "gptq_calibration_metadata", row)
    invocation = gptq["category_bake_invocation"]
    add(invocation["receipt"]["path"], "artifacts/category_bake_invocation.json", "category_command_receipt", invocation["receipt"])
    add(invocation["log"]["path"], "artifacts/category_bake.log", "category_bake_log", invocation["log"])
    audit = capture["source_audit"]
    for name in JSON_NAMES:
        add(Path(audit["evaluation"]) / name, "calibration/teacher/" + name, "calibration_teacher_evaluation")
    for task, row in audit["tasks"].items():
        add(Path(audit["evaluation"]) / (task + ".log"), "calibration/teacher/" + task + ".log",
            "calibration_teacher_log", {"sha256": row["raw_log_sha256"]})
        capture_folder = Path(audit["observations"]) / task
        capture_row = json.loads(add(capture_folder / "capture_manifest.json",
            "calibration/teacher/observations/" + task + "/capture_manifest.json", "calibration_capture_manifest",
            {"sha256": row["capture_manifest_sha256"]}))
        add(capture_folder / "reset_events.jsonl", "calibration/teacher/observations/" + task + "/reset_events.jsonl",
            "calibration_reset_event_log", {"sha256": capture_row["reset_event_file_sha256"]})

    # This is the sole expensive source-side gate. No publish-time verifier
    # depends on these live tensors or their private absolute paths.
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(ROOT))
        from eval.compare_gptq_reference import compare
        evaluation = Path(report["raw_evaluation_audits"]["gptq"]["evaluation_identity"]["eval_manifest.json"]["path"]).parent
        need(compare(Path(report["protocol"]["path"]), evaluation) == report,
             "Original comparison does not reproduce with live GPTQ tensor verification")
    finally:
        sys.path[:] = original_path
    for row, _ in evidence.files.values():
        matches(row["original_absolute_path"], row)
    for row in main_rows.values():
        matches(local(main_evidence, row["published_path"]), row)
    manifest = {"format": FORMAT, "status": "complete", "protocol_sha256": final["protocol_sha256"],
                "scope": SCOPE, "source_report": record(report_path),
                "files": sorted((row for row, _ in evidence.files.values()), key=lambda row: row["published_path"]),
                "main_evidence_files": sorted(main_rows.values(), key=lambda row: row["published_path"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="collect-gptq-", dir=out.parent) as temporary:
        stage = Path(temporary) / "archive"
        stage.mkdir()
        for row, data in evidence.files.values():
            target = local(stage, row["published_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (stage / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = verify(stage, main_evidence)
        need(not out.exists(), "GPTQ archive appeared during collection")
        publish_new(stage, out)
        result["files"] = sorted(out / path.relative_to(stage) if path.is_relative_to(stage) else path
                                 for path in result["files"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report")
    parser.add_argument("--out")
    parser.add_argument("--verify")
    parser.add_argument("--main-evidence")
    args = parser.parse_args()
    if args.verify:
        need(not args.report and not args.out, "--verify cannot be combined with --report/--out")
        result = verify(args.verify, args.main_evidence)
    else:
        need(args.report and args.out, "--report and --out are required for collection")
        result = collect(args.report, args.out, args.main_evidence)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("comparison", "manifest", "files")} | {"files": len(result["files"])}, indent=2))


if __name__ == "__main__":
    main()
