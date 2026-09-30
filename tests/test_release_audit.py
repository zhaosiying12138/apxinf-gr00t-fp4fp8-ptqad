"""CPU tests for the read-only release evidence auditor."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import audit_recovery_release as audit


class ReleaseAuditTest(unittest.TestCase):
    def test_equal_and_missing_have_distinct_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "evidence.json"
            path.write_text("{}\n")
            checker = audit.Audit(root)
            ident = checker.digest(path)
            self.assertTrue(checker.file(path, ident, "present"))
            self.assertFalse(checker.file(root / "absent", ident, "absent"))
            self.assertEqual(checker.checks[-1]["status"], "unverifiable")

    def test_hash_contradiction_is_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "evidence.json"
            path.write_text("{}\n")
            checker = audit.Audit(root)
            self.assertFalse(checker.file(path, {"sha256": "0" * 64}, "changed"))
            self.assertEqual(checker.checks[-1]["status"], "failed")

    def test_protocol_prose_mismatch_is_failed(self):
        checker = audit.Audit(ROOT, payload=False)
        protocol = {
            "partitions": {
                "development": {"init_state_indices": [4]},
                "collection": {"init_state_indices": [5]},
                "heldout": {"init_state_indices": [6]},
            },
            "selection": {
                "pressure_rule": {"min_drop_from_bf16": 0.05},
                "rule": "at least 0.20 below BF16",
                "train_seed": 1, "qad_optimizer_steps": 2,
                "continuation_optimizer_steps": 1, "recovery_scope": "head",
                "rank": 1, "alpha": 2, "effective_demo_batch": 1,
                "opd_every": 1, "qad_learning_rates": [1e-4], "opd_weights": [1.0],
            },
        }
        run = {"train_seed": 1, "qad_steps": 2, "continuation_steps": 1,
               "lora_scope": "head", "rank": 1, "alpha": 2,
               "effective_demo_batch": 1, "opd_every": 1,
               "qad_learning_rates": [1e-4], "opd_weights": [1.0]}
        audit.protocol_checks(checker, protocol, run)
        pressure = next(x for x in checker.checks if x["id"] == "protocol.pressure_text")
        self.assertEqual(pressure["status"], "failed")

    def test_audit_run_does_not_create_or_mutate_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "run_manifest.json").write_text(json.dumps({"status": "initialized"}) + "\n")
            before = sorted(p.relative_to(root) for p in root.rglob("*"))
            report = audit.audit_run(run, root, payload=False)
            after = sorted(p.relative_to(root) for p in root.rglob("*"))
            self.assertEqual(before, after)
            self.assertEqual(report["status"], "unverifiable")
            self.assertTrue(report["read_only"])


if __name__ == "__main__":
    unittest.main()
