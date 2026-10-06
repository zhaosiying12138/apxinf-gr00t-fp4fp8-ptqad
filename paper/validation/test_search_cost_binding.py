#!/usr/bin/env python3
"""CPU-only cross-archive binding regressions; all records are synthetic.

The collectors own arithmetic and full archive validation. These small fixtures
represent their already-verified output and exercise only publication binding.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
import search_cost_binding as binding


def identity(path):
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class Fixture:
    def __init__(self, root):
        self.search, self.training = root / "relocated-search", root / "relocated-training"
        self.final = root / "published-final.json"
        self.private = root / "unavailable-original-run"
        self.maps = {self.search: {}, self.training: {}}
        self.protocol = "a" * 64
        self.selected = {arm: "chosen_" + arm for arm in binding.MODEL_KEYS}
        self.final_data = {"protocol_sha256": self.protocol,
                           "selected_qad_learning_rate": 5e-5, "selected_opd_weight": 0.25}
        candidates, rows = {}, {}
        for arm, key in binding.MODEL_KEYS.items():
            name = self.selected[arm]
            base = "candidates/" + name + "/"
            original = self.private / "artifacts" / name
            checkpoint = {"path": str(original / "checkpoint-2000"),
                          "metadata": {"recovery_manifest.json": arm},
                          "shards": [{"name": "adapter.safetensors", "bytes": 256, "sha256": "b" * 64}]}
            model = {"path": str(original / "export"), "metadata": {"config.json": arm},
                     "shards": [{"name": "model.safetensors", "bytes": 512, "sha256": "c" * 64}]}
            self.final_data[key] = model
            if arm == "qad":
                self.final_data["selected_qad_checkpoint_identity"] = checkpoint
            metadata = {}
            for filename in ("recovery_manifest.json", "runtime_metrics.json",
                             "orchestrator_training_request.json", "merge_manifest.json"):
                relative = ("exports/" if filename == "merge_manifest.json" else "stages/") + arm + "/" + filename
                value = {"arm": arm, "protocol_sha256": self.protocol, "fixture_kind": filename}
                source = original / ("export" if filename == "merge_manifest.json" else "training") / filename
                self.write(self.training, relative, value, source)
                self.write(self.search, base + relative, value, source)
                metadata[filename] = {"path": str(source), **identity(self.search / base / relative)}
            weight = 0.25 if arm == "qad_opd" else 0.0
            receipt = {"stage": name, "status": "complete", "protocol_sha256": self.protocol,
                       "learning_rate": 5e-5, "opd_weight": weight, "checkpoint_identity": checkpoint,
                       "runtime_identity": metadata["runtime_metrics.json"],
                       "training_request_identity": metadata["orchestrator_training_request.json"],
                       "recovery_manifest_sha256": metadata["recovery_manifest.json"]["sha256"]}
            exported = {"stage": "export_" + name, "status": "complete", "protocol_sha256": self.protocol,
                        "model_identity": model, "training_checkpoint": checkpoint["path"],
                        "merge_manifest_sha256": metadata["merge_manifest.json"]["sha256"]}
            self.write(self.search, base + "receipt.json", receipt, self.private / "stages" / (name + ".json"))
            self.write(self.search, base + "export_receipt.json", exported,
                       self.private / "stages" / ("export_" + name + ".json"))
            candidates[name] = {"role": arm}
            rows[name] = {"role": arm, "selected": True, "learning_rate": 5e-5, "opd_weight": weight}
        self.write(self.training, "orchestration/final_manifest.json", self.final_data,
                   self.private / "final_manifest.json")
        self.final.write_bytes((self.training / "orchestration/final_manifest.json").read_bytes())
        self.write(self.search, "inputs.json", {"protocol_sha256": self.protocol, "candidates": candidates,
                   "selected": self.selected, "run_manifest_locator": {"path": str(self.private / "run_manifest.json"),
                   "bytes": 12345, "sha256": "d" * 64}}, self.private / "inputs.json")
        self.write(self.search, "summary.json", {"protocol_sha256": self.protocol,
                   "selected_stages": self.selected, "candidates": rows}, self.private / "summary.json")
        for relative in ("inputs.json", "summary.json"):
            self.maps[self.search][relative].update(original_absolute_path=None, role="derived_cost_evidence")
        self.save_map(self.search)

    def write(self, folder, relative, value, source=None):
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        if source is None:
            source = self.maps[folder][relative]["original_absolute_path"]
        self.maps[folder][relative] = {"published_path": relative,
                                      "original_absolute_path": None if source is None else str(source),
                                      **identity(path), "role": "derived_cost_evidence" if source is None else "synthetic binding fixture"}
        self.save_map(folder)

    def save_map(self, folder):
        (folder / "evidence_manifest.json").write_text(json.dumps({"status": "complete",
            "files": list(self.maps[folder].values())}), encoding="utf-8")

    def change(self, folder, relative, mutate):
        value = json.loads((folder / relative).read_text(encoding="utf-8"))
        mutate(value)
        self.write(folder, relative, value)

    def candidate(self, arm, relative):
        return "candidates/" + self.selected[arm] + "/" + relative

    def verify(self):
        return binding.verify_search_cost_binding(self.search, self.training, self.final)


class SearchCostBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.f = Fixture(self.root)

    def test_relocated_archives_pass_without_private_files_and_return_all_inputs(self):
        self.assertFalse(self.f.private.exists())
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.f.verify(), set(before) | {Path(binding.__file__).resolve()})
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_same_protocol_different_final_rejected(self):
        value = copy.deepcopy(self.f.final_data)
        value["another_run"] = True
        self.f.final.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "Changed evidence"):
            self.f.verify()

    def test_json_equivalent_final_with_different_bytes_rejected(self):
        self.f.final.write_text(json.dumps(self.f.final_data, indent=2))
        with self.assertRaisesRegex(RuntimeError, "Changed evidence"):
            self.f.verify()

    def test_other_run_metadata_rejected_even_with_valid_search_hashes(self):
        for arm in binding.MODEL_KEYS:
            for filename in ("recovery_manifest.json", "runtime_metrics.json",
                             "orchestrator_training_request.json", "merge_manifest.json"):
                with self.subTest(arm=arm, filename=filename):
                    f = Fixture(self.root / (arm + "-" + filename))
                    relative = ("exports/" if filename == "merge_manifest.json" else "stages/") + arm + "/" + filename
                    f.change(f.search, f.candidate(arm, relative), lambda value: value.update(another_run=True))
                    with self.assertRaisesRegex(RuntimeError, "metadata differs from final training"):
                        f.verify()

    def test_archived_file_change_after_collector_verification_rejected(self):
        for folder, relative in ((self.f.search, "inputs.json"),
                                 (self.f.training, "stages/qad/runtime_metrics.json")):
            with self.subTest(folder=folder):
                path = folder / relative
                before = path.read_bytes()
                path.write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "Changed evidence"):
                    self.f.verify()
                path.write_bytes(before)

    def test_missing_duplicate_or_incomplete_required_record_rejected(self):
        for error in ("missing", "duplicate", "bytes", "sha256"):
            with self.subTest(error=error):
                f = Fixture(self.root / error)
                path = f.search / "evidence_manifest.json"
                manifest = json.loads(path.read_text())
                relative = f.candidate("qad", "stages/qad/runtime_metrics.json")
                row = next(row for row in manifest["files"] if row["published_path"] == relative)
                if error == "missing":
                    manifest["files"].remove(row)
                elif error == "duplicate":
                    manifest["files"].append(copy.deepcopy(row))
                else:
                    row.pop(error)
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "binding (evidence|file)"):
                    f.verify()

    def test_unsafe_or_noncanonical_archive_path_rejected(self):
        for relative in ("../escaped.json", "/absolute.json", "C:\\escaped.json", "a/../b.json", "a//b.json"):
            with self.subTest(relative=relative):
                row = {"published_path": relative}
                self.f.maps[self.f.search]["unsafe"] = row
                self.f.save_map(self.f.search)
                with self.assertRaisesRegex(RuntimeError, "Unsafe binding evidence path"):
                    self.f.verify()

    def test_symlink_escape_rejected(self):
        path = self.f.training / "stages/qad/runtime_metrics.json"
        outside = self.root / "outside-runtime.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "escapes its root"):
            self.f.verify()

    def test_selected_map_must_cover_unique_safe_three_arms(self):
        changes = {"missing": lambda selected: selected.pop("qad"),
                   "extra": lambda selected: selected.update(extra="extra"),
                   "reused": lambda selected: selected.update(qad=selected["qad_opd"]),
                   "escape": lambda selected: selected.update(qad="../other"),
                   "unknown": lambda selected: selected.update(qad="other")}
        for label, mutate in changes.items():
            with self.subTest(label=label):
                f = Fixture(self.root / label)
                f.change(f.search, "inputs.json", lambda value: mutate(value["selected"]))
                with self.assertRaisesRegex(RuntimeError, "Search selected"):
                    f.verify()

    def test_selected_role_and_flag_must_agree(self):
        for relative, key, value in (("inputs.json", "role", "continued_qad"),
                                     ("summary.json", "role", "continued_qad"),
                                     ("summary.json", "selected", 1)):
            with self.subTest(relative=relative, key=key):
                f = Fixture(self.root / (relative + key))
                f.change(f.search, relative, lambda data: data["candidates"][f.selected["qad"]].update({key: value}))
                with self.assertRaisesRegex(RuntimeError, "role/flag differs"):
                    f.verify()

    def test_selected_configuration_must_match_final(self):
        for arm in binding.MODEL_KEYS:
            for key, value in (("learning_rate", 1e-4), ("opd_weight", 1.0),
                               ("opd_weight", False), ("learning_rate", float("nan"))):
                with self.subTest(arm=arm, key=key, value=value):
                    f = Fixture(self.root / (arm + key + str(value)))
                    f.change(f.search, f.candidate(arm, "receipt.json"), lambda data: data.update({key: value}))
                    with self.assertRaisesRegex(RuntimeError, "configuration differs|Invalid binding number"):
                        f.verify()

    def test_summary_configuration_must_match_receipt_and_final(self):
        self.f.change(self.f.search, "summary.json", lambda value:
                      value["candidates"][self.f.selected["qad"]].update(learning_rate=1e-4))
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            self.f.verify()

    def test_each_final_model_identity_must_match_export_receipt(self):
        for arm in binding.MODEL_KEYS:
            with self.subTest(arm=arm):
                f = Fixture(self.root / arm)
                f.change(f.search, f.candidate(arm, "export_receipt.json"), lambda value:
                         value["model_identity"]["shards"][0].update(sha256="f" * 64))
                with self.assertRaisesRegex(RuntimeError, "model identity differs"):
                    f.verify()

    def test_selected_qad_checkpoint_identity_must_match_final(self):
        self.f.change(self.f.search, self.f.candidate("qad", "receipt.json"), lambda value:
                      value["checkpoint_identity"].update(path="/another/checkpoint"))
        with self.assertRaisesRegex(RuntimeError, "QAD checkpoint differs"):
            self.f.verify()

    def test_receipt_hashes_must_identify_checked_metadata(self):
        fields = (("receipt.json", "recovery_manifest_sha256"),
                  ("receipt.json", "runtime_identity"),
                  ("receipt.json", "training_request_identity"),
                  ("export_receipt.json", "merge_manifest_sha256"))
        for filename, key in fields:
            with self.subTest(key=key):
                f = Fixture(self.root / key)
                def mutate(value):
                    if isinstance(value[key], dict):
                        value[key]["sha256"] = "f" * 64
                    else:
                        value[key] = "f" * 64
                f.change(f.search, f.candidate("qad", filename), mutate)
                with self.assertRaisesRegex(RuntimeError, "receipt (identity|hash) differs"):
                    f.verify()

    def test_same_bytes_from_different_original_run_rejected(self):
        relative = self.f.candidate("qad", "stages/qad/runtime_metrics.json")
        self.f.maps[self.f.search][relative]["original_absolute_path"] = "/another-run/runtime_metrics.json"
        self.f.save_map(self.f.search)
        with self.assertRaisesRegex(RuntimeError, "metadata differs from final training"):
            self.f.verify()

    def test_raw_metadata_requires_an_absolute_original_path(self):
        relative = self.f.candidate("qad", "stages/qad/runtime_metrics.json")
        for source in (None, "relative/runtime_metrics.json"):
            with self.subTest(source=source):
                self.f.maps[self.f.search][relative].update(original_absolute_path=source, role="derived_cost_evidence")
                self.f.save_map(self.f.search)
                with self.assertRaisesRegex(RuntimeError, "Invalid binding source path"):
                    self.f.verify()

    def test_protocol_and_selected_receipt_stage_must_match(self):
        cases = (("inputs.json", "protocol_sha256", "e" * 64),
                 ("summary.json", "protocol_sha256", "e" * 64),
                 (self.f.candidate("qad", "receipt.json"), "stage", "different_stage"),
                 (self.f.candidate("qad", "export_receipt.json"), "protocol_sha256", "e" * 64))
        for i, (relative, key, value) in enumerate(cases):
            with self.subTest(relative=relative, key=key):
                f = Fixture(self.root / str(i))
                f.change(f.search, relative, lambda data: data.update({key: value}))
                with self.assertRaisesRegex(RuntimeError, "protocol differs|stage/protocol differs"):
                    f.verify()


if __name__ == "__main__":
    unittest.main()
