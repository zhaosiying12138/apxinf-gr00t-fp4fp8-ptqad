"""CPU regression checks for truthful OPD comparison receipts under override."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("opd_receipt_driver", ROOT / "exp/run_high_fp4_v3.py")
DRIVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DRIVER)


class OPDSelectionReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.driver = driver = DRIVER.Driver.__new__(DRIVER.Driver)
        driver.art = root / "artifacts"
        driver.art.mkdir()
        driver.base = root / "bf16_model"
        driver.protocol_path = ROOT / "exp/recovery_protocol_v12_rtn_w4a4.json"
        driver.protocol = DRIVER.load_protocol(driver.protocol_path)
        driver.selection_path = root / "pressure" / "selection.json"
        driver.selection_path.parent.mkdir()
        driver.selection = {"selected_recipe": "rtn_w4a4_category",
                            "selected_ptq_checkpoint": str(root / "ptq_model")}
        driver.selection_path.write_text(json.dumps(driver.selection))
        driver.allow_opd_fallback = True
        self.scores = {}
        self.folder(driver.selection_path.parent / "bf16", 46)
        self.folder(driver.selection_path.parent / "rtn_w4a4_category", 41)
        self.qad_rows = {lr: self.folder(root / f"qad_{lr}", count)
                         for lr, count in zip(driver.protocol["selection"]["qad_learning_rates"], (44, 43))}
        self.opd_rows = {weight: self.folder(root / f"opd_{weight}", count)
                         for weight, count in zip(driver.protocol["selection"]["opd_weights"], (42, 40))}
        self.control = self.folder(root / "continued", 43)
        self.qad_dir = driver.art / "select_qad_lr"
        self.qad_dir.mkdir()
        self.opd_dir = driver.art / "select_opd"
        self.opd_dir.mkdir()
        # Only expensive/raw-input auditing is replaced. The production
        # selection, exact scores, tie-break, persisted-baseline reread, and
        # full report recomputation all execute in these tests.
        audit = patch.object(DRIVER, "eval_audit", side_effect=self.audit)
        models = patch.object(DRIVER, "model_id", side_effect=lambda p: {"path": str(p)})
        audit.start()
        models.start()
        self.addCleanup(audit.stop)
        self.addCleanup(models.stop)
        driver.select(self.qad_dir, "qad", self.qad_rows)

    def folder(self, path, count):
        path.mkdir(parents=True)
        self.scores[str(path)] = {"successes": count, "episodes": 50,
                                  "pairing_sha256": "same-reset-identities"}
        return path

    def audit(self, path, protocol, purpose, model):
        self.assertEqual(purpose, "development")
        self.assertEqual(protocol["sha256"], self.driver.protocol["sha256"])
        return deepcopy(self.scores[str(path)])

    def report(self):
        return self.driver.selection_report("opd", self.opd_rows, control=self.control)

    def persist_and_verify(self, report):
        (self.opd_dir / "selection.json").write_text(json.dumps(report))
        return self.driver.select_verify(self.opd_dir, "opd")

    def test_override_records_negative_result_and_retains_qad_reference(self):
        report = self.report()
        self.assertIs(report["opd_not_below_control"], False)
        self.assertIs(report["opd_beats_continued_qad"], False)
        self.assertIs(report["opd_beats_qad"], False)
        self.assertEqual(report["selected_opd_score"], 42 / 50)
        self.assertEqual(report["qad_reference_score"], 44 / 50)
        self.assertEqual(report["qad_reference_evaluation"]["successes"], 44)
        self.assertIn("qad_selection_identity", report)
        self.assertEqual(report["selected_qad_learning_rate"], min(self.qad_rows))
        before = (self.qad_dir / "selection.json").read_bytes()
        self.persist_and_verify(report)
        self.assertEqual((self.qad_dir / "selection.json").read_bytes(), before)

    def test_tie_and_positive_outcomes_keep_original_lower_weight_tie_break(self):
        for count in (43, 45):
            with self.subTest(successes=count):
                for row in self.opd_rows.values():
                    self.scores[str(row)]["successes"] = count
                report = self.report()
                self.assertEqual(report["selected_opd_weight"], min(self.opd_rows))
                self.assertIs(report["opd_not_below_control"], True)
                self.assertIs(report["opd_beats_continued_qad"], count > 43)
                self.assertIs(report["opd_beats_qad"], count > 44)
                self.persist_and_verify(report)

    def test_strict_gates_still_reject_losses_and_equal_qad(self):
        self.driver.allow_opd_fallback = False
        for count, error in ((42, "below continued-QAD"), (43, "below continued-QAD"),
                             (44, "not strictly above.*QAD")):
            with self.subTest(successes=count):
                for row in self.opd_rows.values():
                    self.scores[str(row)]["successes"] = count
                with self.assertRaisesRegex(DRIVER.OrchestrationError, error):
                    self.report()
        for row in self.opd_rows.values():
            self.scores[str(row)]["successes"] = 45
        report = self.report()
        self.assertIs(report["opd_gate_override"], False)
        self.assertIs(report["opd_selection_gate"]["require_opd_strictly_above_qad"], True)
        self.persist_and_verify(report)

    def test_verifier_rejects_tampered_flags_scores_missing_reference_and_override(self):
        original = self.report()
        mutations = {
            "false_gain": lambda r: r.update(opd_not_below_control=True),
            "false_qad_gain": lambda r: r.update(opd_beats_qad=True),
            "wrong_score": lambda r: r.update(qad_reference_score=0.1),
            "missing_reference": lambda r: r.pop("qad_reference_evaluation"),
            "unapproved_override": lambda r: r.update(opd_gate_override=False),
            "nonboolean_override": lambda r: r.update(opd_gate_override=1),
            "nonboolean_flag": lambda r: r.update(opd_not_below_control=0),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                report = deepcopy(original)
                mutate(report)
                with self.assertRaises(DRIVER.OrchestrationError):
                    self.persist_and_verify(report)

    def test_changed_development_outcomes_invalidate_persisted_receipt(self):
        report = self.report()
        self.scores[str(self.opd_rows[min(self.opd_rows)])]["successes"] += 1
        with self.assertRaisesRegex(DRIVER.OrchestrationError, "recorded selection differs"):
            self.persist_and_verify(report)


if __name__ == "__main__":
    unittest.main()
