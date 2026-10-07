"""Read-only reuse of an audited PTQ base for a bounded sampling ablation.

Source development scores retain their original protocol identity. A reference
is never a new evaluation and cannot authorize continuation beyond QAD dev.
Relative declaration paths resolve against the target protocol's directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re


class PTQReferenceError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise PTQReferenceError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _source_identity(record, relative_to, label):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"},
             f"Invalid {label} reference identity")
    _require(isinstance(record["path"], str) and bool(record["path"]), f"Missing {label} path")
    digest = record["sha256"]
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
             f"Invalid {label} SHA-256")
    path = Path(record["path"]).expanduser()
    path = (path if path.is_absolute() else relative_to / path).resolve()
    _require(path.is_file(), f"Missing {label} file: {path}")
    _require(_sha(path) == digest, f"{label} file SHA differs from reference")
    return path, {"path": str(path), "sha256": digest}


def validate_reference(path, normalized_protocol):
    """Audit source evidence under its own frozen protocol, then attach ancestry.

The caller must enforce the returned allowed_until='qad_dev' run boundary.
Neither the original selection nor its source files are modified.
"""
    # Keep the module importable by the driver without a circular import.
    if __package__:
        from . import run_high_fp4_v3 as driver
    else:
        import run_high_fp4_v3 as driver

    _require(isinstance(normalized_protocol, dict), "Expected a normalized target protocol")
    target_path_value = normalized_protocol.get("path")
    _require(isinstance(target_path_value, str) and bool(target_path_value), "Target protocol lacks a path")
    target_path = Path(target_path_value).expanduser().resolve()
    _require(target_path.is_file(), "Target protocol file is missing")
    actual_target = driver.load_protocol(target_path)
    for field in ("data", "partitions", "selection", "sha256"):
        _require(_same(actual_target.get(field), normalized_protocol.get(field)),
                 f"Normalized target protocol differs from frozen file: {field}")
    target = actual_target["data"]
    declaration = target.get("ptq_reference")
    _require(isinstance(declaration, dict) and set(declaration) == {
        "scope", "allowed_until", "source_protocol", "source_selection"}, "Invalid ptq_reference declaration")
    _require(declaration["scope"] == "fixed_base_for_sampling_ablation", "PTQ reference has unsupported scope")
    _require(declaration["allowed_until"] == "qad_dev", "PTQ reference is restricted to qad_dev")
    views = target.get("capture_views")
    _require(isinstance(views, dict) and set(views) == {"modes", "windows_per_episode"}
             and views.get("modes") == ["head", "stratified"]
             and type(views.get("windows_per_episode")) is int and views["windows_per_episode"] > 0,
             "Sampling reference requires frozen head/stratified capture_views")
    learning_rates = target["selection"].get("qad_learning_rates")
    _require(isinstance(learning_rates, list) and len(learning_rates) == 1
             and type(learning_rates[0]) in (int, float)
             and math.isfinite(learning_rates[0]) and learning_rates[0] > 0,
             "Sampling reference requires one finite positive QAD learning rate")

    source_path, source_identity = _source_identity(
        declaration["source_protocol"], target_path.parent, "Source protocol")
    selection_path, selection_identity = _source_identity(
        declaration["source_selection"], target_path.parent, "Source selection")
    _require(Path(path).expanduser().resolve() == selection_path,
             "Requested PTQ selection path differs from declared source selection")
    source_protocol = driver.load_protocol(source_path)
    source = source_protocol["data"]
    _require("ptq_reference" not in source, "Nested PTQ references are not allowed")
    _require(source_protocol["sha256"] == source_identity["sha256"], "Source protocol changed while reading")
    _require(target.get("w4a4") is True and source.get("w4a4") is True,
             "Sampling reference requires matching W4A4 protocols")
    _require(_same(target["partitions"]["development"], source["partitions"]["development"]),
             "Target development partition differs from source")
    for field in ("quantization_scope", "evaluation_contract"):
        _require(isinstance(target.get(field), dict) and bool(target[field])
                 and _same(target[field], source.get(field)), f"Target {field} differs from source")
    for field in ("pressure_candidates", "pressure_rule", "pressure_candidate_checkpoints"):
        _require(_same(target["selection"].get(field), source["selection"].get(field)),
                 f"Target {field} differs from source")

    # The existing full audit binds raw development logs, reset pairing,
    # scores, checkpoint bytes, recipe bytes and the original selection rule.
    validated = driver.validate_ptq(selection_path, source_protocol)
    _require(validated.get("protocol_sha256") == source_identity["sha256"],
             "Validated source selection lost its original protocol identity")
    _require("reference_provenance" not in validated, "Source selection is already a PTQ reference")
    for frozen_path, frozen_sha in ((source_path, source_identity["sha256"]),
                                    (selection_path, selection_identity["sha256"]),
                                    (target_path, actual_target["sha256"])):
        _require(_sha(frozen_path) == frozen_sha, f"Reference file changed during audit: {frozen_path}")
    result = copy.deepcopy(validated)
    result["reference_provenance"] = {
        "scope": declaration["scope"], "allowed_until": "qad_dev",
        "source_protocol": source_identity, "source_selection": selection_identity,
        "target_protocol": {"path": str(target_path), "sha256": actual_target["sha256"]},
        "target_protocol_sha256": actual_target["sha256"],
        "source_score_is_new_evaluation": False,
        "score_scope": "Original source-protocol development evidence reused for a fixed PTQ base; no new evaluation",
    }
    return result
