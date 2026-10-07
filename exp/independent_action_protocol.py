"""Freeze an independent action diagnostic against a completed v12 five-arm run.

This module is CPU-only. It references original receipts without rewriting them;
the original model protocol and the diagnostic protocol have different identities.
Preparing a protocol never collects data, trains, selects models or runs inference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from rl.checkpoint_identity import file_sha256
from eval.run_recovery_eval import TASKS, validate_recovery_checkpoint

ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")
FORMAT = "independent_action_diagnostic_protocol_v1"
DRAFT = Path(__file__).with_name("independent_action_protocol_draft.json")
CONTRACT_FIELDS = ("task_count", "n_envs", "task_seed_stride", "episode_seed_stride",
                   "server_seed_offset", "settle_steps", "n_action_steps",
                   "max_episode_steps", "initial_state_protocol")
SETTINGS = {
    "reference_arm": "bf16", "source_kind": "diagnostic_rollout",
    "checkpoint_role": "diagnostic_reference", "capture_every": 4,
    "capture_max_per_episode": 0, "success_only": False,
    "windows_per_episode": 4, "sampling": "stratified_time",
    "noise_seeds": [2026100700, 2026100701],
    "training_allowed": False, "model_selection_allowed": False,
    "allowed_operations": ["capture", "action_inference", "compare"],
    "interpretation": "independent_observation_action_diagnostic_not_success_rate_or_new_task_generalization",
}
PARTITIONS = {"diagnostics": {
    "seed": 990000, "init_state_indices": [29], "episodes_per_task": 1,
    "purpose": "independent diagnostic observations; never training or model selection",
    "capture_sampling": {"mode": "full_trajectory_candidates", "every_server_calls": 4,
                         "safety_candidates_per_episode": 192},
}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(path):
    path = Path(path).expanduser().resolve(strict=True)
    require(path.is_file(), f"Missing regular file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _declaration(data, status):
    require(isinstance(data, dict) and data.get("format") == FORMAT,
            "Unknown independent diagnostic protocol")
    require(data.get("status") == status, f"Diagnostic protocol must be {status}")
    require(data.get("diagnostics") == SETTINGS, "Diagnostic sampling or allowed use differs")
    require(data.get("partitions") == PARTITIONS, "Diagnostic partition must be bank 29 only")
    require(data.get("tasks") == TASKS, "Diagnostics require the ten official tasks in order")
    require(set(data) == {"format", "status", "diagnostics", "partitions", "tasks",
                          "reference", "evaluation_contract", "quantization_scope"},
            "Undeclared diagnostic protocol fields are forbidden")


def audit_final_reference(final_manifest):
    """Reuse final/raw-log/weight validators; do not reinterpret scores or select."""
    from paper.prepare_w4a4_captures import validate_final
    from paper.extract_final_evidence import extract
    from paper.materialize_final_evidence import audit_heldout_raw_logs
    from eval.compare_recovery import compare_round
    from exp.action_chunk_diagnostics import bind_final_checkpoint

    final_path = Path(final_manifest).expanduser().resolve(strict=True)
    before = identity(final_path)
    bundle = validate_final(final_path)
    final, original = bundle["final"], bundle["protocol"]
    # Existing publication adapter enforces the exact v12 bytes and 160 scored
    # episodes per arm. Results are checked, never used to choose an arm here.
    extract(final_path.parent)
    round_dir = Path(final["heldout_round"]).resolve(strict=True)
    require(bundle["comparison_path"] == round_dir / "paired_comparison.json",
            "Final comparison must belong to its declared heldout round")
    require(bundle["comparison"] == compare_round(round_dir),
            "Final five-arm comparison does not reproduce")
    raw_files = audit_heldout_raw_logs(round_dir)
    selection_path = Path(final["selection_file"]).resolve(strict=True)
    selection = read(selection_path)
    require(selection.get("selection_uses_heldout") is False,
            "Original PTQ selection must exclude heldout")
    require(selection.get("selected_recipe") == final["selected_pressure_recipe"],
            "Final recipe differs from original PTQ selection")
    sources = {
        "source_final": before,
        "source_run": identity(final_path.parent / "run_manifest.json"),
        "source_selection": identity(selection_path),
        "source_comparison": identity(bundle["comparison_path"]),
        "original_model_protocol": identity(bundle["protocol_path"]),
    }
    require(sources["source_selection"]["sha256"] == final["selection_sha256"],
            "Original PTQ selection changed")
    arms = {}
    for arm in ARMS:
        # The legacy branch verifies weights against the ORIGINAL protocol.
        # No diagnostic-purpose input is passed, so this cannot recurse into us.
        checkpoint, files = bind_final_checkpoint(
            final_path, arm, {"protocol_sha256": final["protocol_sha256"]})
        validate_recovery_checkpoint(checkpoint)
        quantized = any((checkpoint / n).is_file() for n in
                        ("ptq_recipe.json", "category_ptq_recipe.json", "merge_manifest.json"))
        require(quantized == (arm != "bf16"), f"Final arm has wrong quantization identity: {arm}")
        require((checkpoint / "merge_manifest.json").is_file() ==
                (arm in ("qad", "continued_qad", "qad_opd")),
                f"Final arm has wrong adapter identity: {arm}")
        evaluation = read(round_dir / ("heldout_" + arm) / "eval_manifest.json")
        require(Path(evaluation["checkpoint"]).resolve() == checkpoint,
                f"Final heldout checkpoint differs from selected {arm}")
        if arm == "bf16":
            model = selection["arms"]["bf16"]["model_identity"]
        elif arm == "ptq":
            model = selection["arms"][final["selected_pressure_recipe"]]["model_identity"]
            require(checkpoint == Path(final["selected_ptq_checkpoint"]).resolve(),
                    "Final selected PTQ checkpoint differs")
        else:
            key = {"qad": "selected_qad_model_identity", "continued_qad": "selected_continued_model_identity",
                   "qad_opd": "selected_opd_model_identity"}[arm]
            model = final[key]
        arms[arm] = {"checkpoint": str(checkpoint), "model_identity": model,
                     "checkpoint_files": files}
    # Finalization is not valid if source JSON changed while weights were read.
    require(all(identity(record["path"]) == record for record in sources.values()),
            "Final reference changed during audit")
    require(bundle["comparison"] == compare_round(round_dir) and
            raw_files == audit_heldout_raw_logs(round_dir), "Final raw evidence changed during audit")
    used = {i for part in original["partitions"].values() for i in part["init_state_indices"]}
    ledger = original.get("ledger", {})
    used.update(ledger.get("excluded_historical_heldout", []))
    used.update(ledger.get("excluded_historical_smoke", []))
    require(29 not in used, "Diagnostic bank overlaps original or excluded historical partitions")
    raw_sha = hashlib.sha256(json.dumps(raw_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**sources, "arms": arms, "raw_heldout_identity": {"file_count": len(raw_files), "sha256": raw_sha},
            "evaluation_contract": {key: original["evaluation_contract"][key] for key in CONTRACT_FIELDS},
            "quantization_scope": original["quantization_scope"]}


def prepare(draft_file, final_manifest, out):
    """Write a new frozen protocol only after every final reference passes."""
    draft_path, output = Path(draft_file).resolve(strict=True), Path(out).absolute()
    require(not output.exists(), f"Refusing to overwrite diagnostic protocol: {output}")
    draft_identity = identity(draft_path)
    data = read(draft_path)
    _declaration(data, "draft_not_executable")
    require(all(data[key] is None for key in ("reference", "evaluation_contract", "quantization_scope")),
            "A diagnostic draft must not pretend to contain frozen source identities")
    reference = audit_final_reference(final_manifest)
    frozen = copy.deepcopy(data)
    frozen.update(status="frozen", reference=reference,
                  evaluation_contract=reference["evaluation_contract"],
                  quantization_scope=reference["quantization_scope"])
    require(identity(draft_path) == draft_identity, "Diagnostic draft changed during preparation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"status": "prepared_not_executed", "diagnostic_protocol": identity(output),
            "original_model_protocol": reference["original_model_protocol"],
            "source_final": reference["source_final"]}


def validate_diagnostic_protocol(protocol_file, final_manifest=None, checkpoint=None):
    """Audit frozen references; optional checkpoint admits only the BF16 collector.

    Action-output collection for the other four arms consumes ``report['arms']``;
    the checkpoint argument is specifically the independent observation collector.
    """
    protocol_path = Path(protocol_file).resolve(strict=True)
    before = identity(protocol_path)
    data = read(protocol_path)
    _declaration(data, "frozen")
    reference = data.get("reference")
    require(isinstance(reference, dict), "Frozen final reference is missing")
    source_final = reference.get("source_final", {})
    require(isinstance(source_final, dict) and isinstance(source_final.get("path"), str),
            "Final reference path is missing")
    final_path = Path(source_final["path"]).resolve(strict=True)
    if final_manifest is not None:
        require(Path(final_manifest).resolve(strict=True) == final_path,
                "Requested final manifest differs from frozen diagnostic reference")
    require(identity(final_path) == source_final, "Frozen final manifest changed")
    actual = audit_final_reference(final_path)
    require(actual == reference, "Frozen final reference or model weights changed")
    require(data["evaluation_contract"] == actual["evaluation_contract"] and
            data["quantization_scope"] == actual["quantization_scope"],
            "Diagnostic inference contract differs from the original models")
    teacher = actual["arms"]["bf16"]
    if checkpoint is not None:
        require(Path(checkpoint).resolve(strict=True) == Path(teacher["checkpoint"]),
                "Independent observations require the frozen BF16 diagnostic reference")
    require(identity(protocol_path) == before, "Diagnostic protocol changed during audit")
    require(before["sha256"] != actual["original_model_protocol"]["sha256"],
            "Original model and independent diagnostic protocols must remain distinct")
    return {"status": "verified", "data": data, "settings": data["diagnostics"],
            "diagnostic_protocol": before, "diagnostic_protocol_sha256": before["sha256"],
            "original_model_protocol_sha256": actual["original_model_protocol"]["sha256"],
            **actual, "allowed_teacher": teacher}


validate = validate_diagnostic_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare", help="CPU audit and freeze; launch nothing")
    freeze.add_argument("--draft", default=str(DRAFT))
    freeze.add_argument("--final-manifest", required=True)
    freeze.add_argument("--out", required=True)
    check = commands.add_parser("validate", help="CPU reference audit only")
    check.add_argument("--protocol-file", required=True)
    check.add_argument("--final-manifest")
    check.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.draft, args.final_manifest, args.out)
    else:
        result = validate_diagnostic_protocol(args.protocol_file, args.final_manifest, args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
