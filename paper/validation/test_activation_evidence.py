"""Synthetic server receipts; never modify experiment evidence or run inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activation_evidence import validate_activation_log
import validate_publication as validation

PROTOCOL = {"quantization_scope": {"ordinary_linear_operators": 2,
    "category_linear_layers": 1, "activation_linear_count": 3, "category_w4a4_required": True}}
REPORT = {"mode": "all", "ste": False, "ordinary_total": 2, "ordinary_w4a4": 2,
    "ordinary_bf16": 0, "category_total": 1, "category_w4a4": 1, "category_bf16": 0,
    "records": [{"name": name, "format": "W4A4"} for name in ("visual", "head", "category")]}


class ActivationEvidenceTests(unittest.TestCase):
    def check_log(self, text, quantized=True, protocol=None):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "task.server.log"
            path.write_text(text, encoding="utf-8")
            result = validate_activation_log(path, protocol or PROTOCOL, quantized=quantized)
            self.assertEqual(path.read_text(encoding="utf-8"), text)
            return result

    def text(self, report):
        return "startup\n[fp4vla] activation-only " + json.dumps(report) + "\nready\n"

    def test_complete_installation_and_unquantized_baseline(self):
        self.assertEqual(self.check_log(self.text(REPORT))["module_records"], 3)
        self.assertIsNone(self.check_log("BF16 ready\n", quantized=False))

    def test_missing_duplicate_or_invalid_report(self):
        for text in ("ready\n", self.text(REPORT) * 2,
                     "[fp4vla] activation-only invalid\n", "[fp4vla] activation-only []\n"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                self.check_log(text)

    def test_request_cannot_substitute_for_actual_full_coverage(self):
        for changes in ({"ordinary_w4a4": 1, "ordinary_bf16": 1},
                        {"ordinary_total": 1, "ordinary_w4a4": 1},
                        {"category_w4a4": 0, "category_bf16": 1},
                        {"category_total": True}, {"ordinary_bf16": False},
                        {"mode": "selected"}, {"ste": True}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                self.check_log(self.text({**REPORT, **changes}))

    def test_per_module_receipts_must_support_aggregate(self):
        for mutation in (lambda r: r.pop(), lambda r: r[1].update(name="visual"),
                         lambda r: r[0].update(format="BF16")):
            report = copy.deepcopy(REPORT)
            mutation(report["records"])
            with self.assertRaises(RuntimeError): self.check_log(self.text(report))

    def test_bf16_cannot_carry_quantized_server_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "BF16"):
            self.check_log(self.text(REPORT), quantized=False)

    def test_protocol_scope_cannot_be_inferred_from_defaults(self):
        for key in PROTOCOL["quantization_scope"]:
            protocol = copy.deepcopy(PROTOCOL)
            protocol["quantization_scope"].pop(key)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.check_log(self.text(REPORT), protocol=protocol)

    def publication_fixture(self, changed_arm=None):
        # Isolate the new server gate from the separately tested rollout parser.
        # All identity checks below use real files; the bad server is rehashed,
        # so a checksum-only validator would incorrectly accept it.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); paper = root / "paper"; evidence = paper / "evidence"
            evidence.mkdir(parents=True)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({**PROTOCOL, "version": 12}))
            parser = root / "eval/run_recovery_eval.py"
            parser.parent.mkdir(); parser.write_text("# synthetic parser source\n")
            def identity(path):
                return {"bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            records = {}
            for arm in ("bf16", "ptq", "qad", "continued_qad", "qad_opd"):
                folder = evidence / ("heldout_" + arm); folder.mkdir()
                for suffix in (".log", ".server.log"):
                    path = folder / ("task" + suffix)
                    text = "synthetic completed log\n"
                    if suffix == ".server.log" and arm != "bf16" and arm != changed_arm:
                        text = self.text(REPORT)
                    path.write_text(text)
                    records[str(path.relative_to(evidence))] = identity(path)
            manifest = {"version": 1, "status": "complete", "protocol_sha256": validation.sha(protocol),
                        "parser": {"path": "eval/run_recovery_eval.py", **identity(parser)},
                        "parser_functions": ["parse_log", "validate_resets"], "files": records}
            (evidence / "heldout_raw_logs.json").write_text(json.dumps(manifest))
            fake_audit = SimpleNamespace(audit_heldout_raw_logs=lambda _: records)
            with patch.object(validation, "P", paper), patch.dict(sys.modules, {
                "run_recovery_eval": SimpleNamespace(TASKS=["task"]),
                "materialize_final_evidence": fake_audit}):
                inputs = set()
                validation.check_heldout_raw_logs(inputs, protocol)
                self.assertIn(paper / "activation_evidence.py", inputs)

    def test_publication_checks_each_task_server_after_hash_verification(self):
        self.publication_fixture()
        for arm in ("ptq", "qad", "continued_qad", "qad_opd"):
            with self.subTest(arm=arm), self.assertRaisesRegex(RuntimeError, "one activation"):
                self.publication_fixture(changed_arm=arm)


if __name__ == "__main__": unittest.main()
