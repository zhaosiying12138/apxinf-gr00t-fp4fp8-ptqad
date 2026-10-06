"""Bind verified search costs to the selected training and final publication.

The caller must first run the search and training collectors' full verifiers.
This module checks their relationship using portable archived metadata only;
it never opens original private paths or the mutable orchestrator run state.
"""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

try:
    from .publication_guard import check_record, require
except ImportError:  # direct publication-validator invocation
    from publication_guard import check_record, require


MODEL_KEYS = {
    "qad": "selected_qad_model_identity",
    "continued_qad": "selected_continued_model_identity",
    "qad_opd": "selected_opd_model_identity",
}


class _Archive:
    def __init__(self, root, inputs):
        self.root = Path(root).resolve(strict=True)
        self.inputs = inputs
        manifest_path = self.root / "evidence_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs.add(manifest_path)
        require(manifest.get("status") == "complete", "Incomplete binding archive")
        records = manifest.get("files")
        require(isinstance(records, list) and records, "Missing binding evidence records")
        self.records = {}
        for row in records:
            require(isinstance(row, dict), "Invalid binding evidence record")
            relative = row.get("published_path")
            require(isinstance(relative, str) and relative and "\\" not in relative and
                    not PurePosixPath(relative).is_absolute() and
                    not PureWindowsPath(relative).drive and
                    ".." not in PurePosixPath(relative).parts and
                    PurePosixPath(relative).as_posix() == relative,
                    "Unsafe binding evidence path")
            require(relative not in self.records, "Duplicate binding evidence path: " + relative)
            self.records[relative] = row

    def checked(self, relative):
        require(relative in self.records, "Missing binding evidence record: " + relative)
        row = self.records[relative]
        require(type(row.get("bytes")) is int and row["bytes"] > 0 and
                isinstance(row.get("sha256"), str) and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None,
                "Incomplete binding file identity: " + relative)
        original = row.get("original_absolute_path")
        derived = (relative in ("inputs.json", "summary.json") and
                   row.get("role") == "derived_cost_evidence" and original is None)
        require(derived or (isinstance(original, str) and
                (PurePosixPath(original).is_absolute() or PureWindowsPath(original).is_absolute())),
                "Invalid binding source path: " + relative)
        path = check_record({"path": relative, "bytes": row["bytes"], "sha256": row["sha256"]}, self.root)
        self.inputs.add(path)
        return path, row

    def read(self, relative):
        path, _ = self.checked(relative)
        return json.loads(path.read_text(encoding="utf-8"))


def _number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid binding number: " + label)
    return value


def _receipt_identity(receipt, key, row, label):
    record = receipt.get(key)
    require(isinstance(record, dict) and
            record.get("path") == row["original_absolute_path"] and
            type(record.get("bytes")) is int and record["bytes"] == row["bytes"] and
            record.get("sha256") == row["sha256"],
            "Search receipt identity differs: " + label)


def verify_search_cost_binding(search_dir, training_dir, final_manifest_path) -> set[Path]:
    """Return all checked inputs after binding already-verified cost archives.

The final manifest must be the exact file archived with the selected training
costs. Each selected search candidate must carry that training run's recovery,
runtime, request and merge metadata, with matching producer receipts and final
selected model identities. Original paths are compared, never dereferenced.
"""
    inputs = {Path(__file__).resolve()}
    search = _Archive(search_dir, inputs)
    training = _Archive(training_dir, inputs)
    archived_final_path, final_record = training.checked("orchestration/final_manifest.json")
    final_path = Path(final_manifest_path).resolve(strict=True)
    # Re-use the verified archive identity for the external final manifest.
    check_record({"path": final_path.name, "bytes": final_record["bytes"],
                  "sha256": final_record["sha256"]}, final_path.parent)
    inputs.add(final_path)
    final = json.loads(archived_final_path.read_text(encoding="utf-8"))
    index = search.read("inputs.json")
    summary = search.read("summary.json")
    protocol = final.get("protocol_sha256")
    require(isinstance(protocol, str) and protocol and
            index.get("protocol_sha256") == summary.get("protocol_sha256") == protocol,
            "Search/training final protocol differs")
    selected = index.get("selected")
    require(isinstance(selected, dict) and set(selected) == set(MODEL_KEYS),
            "Search selected arms differ from final training arms")
    require(all(isinstance(name, str) and name not in (".", "..") and
                re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in selected.values()) and
            len(set(selected.values())) == len(MODEL_KEYS),
            "Search selected stages are unsafe or reused")
    require(summary.get("selected_stages") == selected, "Search selected summary differs")
    candidates, rows = index.get("candidates"), summary.get("candidates")
    require(isinstance(candidates, dict) and isinstance(rows, dict), "Missing search candidate index")
    learning_rate = _number(final.get("selected_qad_learning_rate"), "final learning rate")
    opd_weight = _number(final.get("selected_opd_weight"), "final OPD weight")
    for arm, model_key in MODEL_KEYS.items():
        name = selected[arm]
        spec, row = candidates.get(name), rows.get(name)
        require(isinstance(spec, dict) and spec.get("role") == arm and
                isinstance(row, dict) and row.get("role") == arm and row.get("selected") is True,
                "Search selected candidate role/flag differs: " + arm)
        base = "candidates/" + name + "/"
        receipt = search.read(base + "receipt.json")
        exported = search.read(base + "export_receipt.json")
        require(receipt.get("status") == exported.get("status") == "complete" and
                receipt.get("stage") == name and
                receipt.get("protocol_sha256") == exported.get("protocol_sha256") == protocol,
                "Search selected receipt stage/protocol differs: " + arm)
        weight = opd_weight if arm == "qad_opd" else 0.0
        for settings in (receipt, row):
            require(_number(settings.get("learning_rate"), arm + " learning rate") == learning_rate and
                    _number(settings.get("opd_weight"), arm + " OPD weight") == weight,
                    "Search selected configuration differs from final manifest: " + arm)
        require(isinstance(final.get(model_key), dict) and final[model_key] and
                exported.get("model_identity") == final[model_key],
                "Search selected model identity differs from final manifest: " + arm)
        if arm == "qad":
            require(isinstance(final.get("selected_qad_checkpoint_identity"), dict) and
                    final["selected_qad_checkpoint_identity"] and
                    receipt.get("checkpoint_identity") == final["selected_qad_checkpoint_identity"],
                    "Search selected QAD checkpoint differs from final manifest")
        bound = {}
        for relative in ("stages/" + arm + "/recovery_manifest.json",
                         "stages/" + arm + "/runtime_metrics.json",
                         "stages/" + arm + "/orchestrator_training_request.json",
                         "exports/" + arm + "/merge_manifest.json"):
            _, search_record = search.checked(base + relative)
            _, training_record = training.checked(relative)
            require(all(search_record[key] == training_record[key]
                        for key in ("bytes", "sha256", "original_absolute_path")),
                    "Search selected metadata differs from final training archive: " + relative)
            bound[PurePosixPath(relative).name] = search_record
        require(receipt.get("recovery_manifest_sha256") == bound["recovery_manifest.json"]["sha256"],
                "Search recovery receipt hash differs: " + arm)
        _receipt_identity(receipt, "runtime_identity", bound["runtime_metrics.json"], arm + " runtime")
        _receipt_identity(receipt, "training_request_identity",
                          bound["orchestrator_training_request.json"], arm + " request")
        require(exported.get("merge_manifest_sha256") == bound["merge_manifest.json"]["sha256"],
                "Search merge receipt hash differs: " + arm)
    return inputs
