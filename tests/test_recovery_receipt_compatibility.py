"""CPU checks for immutable v12 receipts and audited source migrations."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("receipt_driver", ROOT / "exp/run_high_fp4_v3.py")
DRIVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DRIVER)


class TrainingReceiptCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.env = {key: "1" for key in DRIVER.ENV_SUMMARY_KEYS
                    if key not in DRIVER.OPTIONAL_LAUNCH_ENV_KEYS}
        self.env.update(QAD_MAX_GRAD_NORM=None, QAD_LR="5e-05", QAD_STEPS="2000",
                        FP4VLA_SATURATE_F16_ACTIVATIONS="0")
        # Actual v12 initialization schema: request has two optional nulls,
        # recovery has only numerical keys, and neither records the timeout.
        request_values = {**self.env, "HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None}
        self.request = {"environment": deepcopy(self.env),
                        "environment_summary": {"variables": request_values}}
        self.recovery = {"environment_summary": {"variables": deepcopy(self.env)}}

    def test_legacy_optional_absence_matches_null_without_mutating_receipts(self):
        original = deepcopy((self.request, self.recovery))
        normalized = DRIVER.validate_training_environment(self.request, self.recovery)
        self.assertEqual(normalized, self.env)
        self.assertEqual((self.request, self.recovery), original)
        self.request["environment_summary"]["variables"]["PTQAD_ZMQ_TIMEOUT_MS"] = None
        self.assertEqual(DRIVER.validate_training_environment(self.request, self.recovery), self.env)

    def test_explicit_optional_value_is_not_discarded(self):
        for key in DRIVER.OPTIONAL_LAUNCH_ENV_KEYS:
            with self.subTest(key=key):
                request = deepcopy(self.request)
                request["environment_summary"]["variables"][key] = "120000" if "TIMEOUT" in key else "1"
                with self.assertRaisesRegex(DRIVER.OrchestrationError, "summary is unstable"):
                    DRIVER.validate_training_environment(request, self.recovery)

    def test_all_numerical_keys_are_required_in_each_receipt(self):
        for key in self.env:
            for target in ("request", "recovery"):
                with self.subTest(key=key, target=target):
                    request, recovery = deepcopy((self.request, self.recovery))
                    row = request if target == "request" else recovery
                    del row["environment_summary"]["variables"][key]
                    with self.assertRaisesRegex(DRIVER.OrchestrationError, "lacks numerical keys"):
                        DRIVER.validate_training_environment(request, recovery)

    def test_numerical_changes_and_unknown_receipt_fields_are_rejected(self):
        for key in (*self.env, "UNKNOWN_CONTRACT"):
            with self.subTest(key=key):
                recovery = deepcopy(self.recovery)
                recovery["environment_summary"]["variables"][key] = "changed"
                with self.assertRaisesRegex(DRIVER.OrchestrationError, "differs from training request"):
                    DRIVER.validate_training_environment(self.request, recovery)

    def test_missing_or_malformed_environment_is_rejected(self):
        for value in (None, {}, {"variables": []}):
            with self.subTest(value=value):
                recovery = {"environment_summary": value}
                with self.assertRaises(DRIVER.OrchestrationError):
                    DRIVER.validate_training_environment(self.request, recovery)

    def test_protocol_identity_is_still_a_hard_training_contract(self):
        driver = DRIVER.Driver.__new__(DRIVER.Driver)
        driver.selection = {"selected_ptq_checkpoint": "/base"}
        driver.rank, driver.alpha, driver.scope = 32, 64.0, "all_ordinary_linear"
        driver.seed, driver.every, driver.batch = 20261006, 1, 16
        driver.protocol = {"sha256": "frozen-protocol"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runtime_metrics.json").write_text(json.dumps({
                "status": "completed", "global_steps": 2000, "requested_optimizer_steps": 2000}))
            rec = {"base": "/base", "rank": 32, "alpha": 64.0, "scope": driver.scope,
                   "train_seed": driver.seed, "protocol_sha256": "changed-protocol",
                   "optimizer_resumed": False, "probe_weight": 0.0, "probe_every": 1,
                   "micro_batch": 1, "effective_global_batch": 16, "gradient_accumulation_steps": 16}
            (root / "recovery_manifest.json").write_text(json.dumps(rec))
            with self.assertRaisesRegex(DRIVER.OrchestrationError, "frozen recovery settings"):
                driver.train_verify(root, 2000, lr=5e-5)


class ResumeSourceCompatibilityTest(unittest.TestCase):
    def test_only_exact_reviewed_prior_hashes_can_migrate(self):
        for old in DRIVER.AUDITED_RESUME_SOURCE_SHA256:
            DRIVER.verify_resume_source({"implementation_sha256": old}, "current")
            DRIVER.verify_resume_source({"orchestrator_source_sha256": old}, "current")
        DRIVER.verify_resume_source({"orchestrator_source_sha256": "current"}, "current")
        for state in ({}, {"implementation_sha256": "f" * 64},
                      {"orchestrator_source_sha256": "f" * 64}):
            with self.subTest(state=state), self.assertRaisesRegex(DRIVER.OrchestrationError, "not been audited"):
                DRIVER.verify_resume_source(state, "current")

    def test_migration_preserves_original_identity_and_stage_receipt_bytes(self):
        old = next(iter(DRIVER.AUDITED_RESUME_SOURCE_SHA256))
        with tempfile.TemporaryDirectory() as tmp:
            driver = DRIVER.Driver.__new__(DRIVER.Driver)
            driver.run_dir = Path(tmp) / "run"
            driver.run_dir.mkdir()
            for attr, name in (("art", "artifacts"), ("work", "work"), ("logs", "logs"),
                               ("stages", "stages"), ("receipts", "cleanup_receipts")):
                setattr(driver, attr, driver.run_dir / name)
                getattr(driver, attr).mkdir()
            driver.protocol, driver.selection = {"sha256": "protocol"}, {"selection_sha256": "selection"}
            driver.base = Path("/base")
            driver.seed, driver.qsteps, driver.csteps = 1, 2000, 2000
            driver.scope, driver.rank, driver.alpha = "all_ordinary_linear", 32, 64.0
            driver.capture_dataset_identity, driver.teacher_capture_identity = {"sha": "capture"}, {"sha": "teacher"}
            driver.w4a4 = True
            receipt = driver.stages / "train.json"
            original_receipt = b'{"status":"complete","producer":"immutable"}\n'
            receipt.write_bytes(original_receipt)
            state = {"output_layout": "stable_paths_v2", "protocol_sha256": "protocol",
                     "selection_sha256": "selection", "base": str(driver.base),
                     "train_seed": 1, "qad_steps": 2000, "continuation_steps": 2000,
                     "lora_scope": driver.scope, "rank": 32, "alpha": 64.0,
                     "capture_dataset_identity": driver.capture_dataset_identity,
                     "teacher_capture_identity": driver.teacher_capture_identity,
                     "w4a4": True, "implementation_sha256": old,
                     "orchestrator_source_sha256": old, "stages": {}}
            (driver.run_dir / "run_manifest.json").write_text(json.dumps(state))
            with patch.object(DRIVER, "sha", return_value="current"):
                migrated = driver.load_state()
            self.assertEqual(migrated["implementation_sha256"], old)
            self.assertEqual(migrated["legacy_orchestrator_source_sha256"], old)
            self.assertEqual(migrated["orchestrator_source_sha256"], "current")
            self.assertEqual(receipt.read_bytes(), original_receipt)


if __name__ == "__main__":
    unittest.main()
