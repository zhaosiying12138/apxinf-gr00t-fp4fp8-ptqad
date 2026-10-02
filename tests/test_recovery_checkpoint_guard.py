"""CPU checks that raw recovery checkpoints cannot silently lose adapters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from eval.run_recovery_eval import validate_recovery_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class RecoveryCheckpointGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "checkpoint-100"
        self.checkpoint.mkdir()

    def write(self, filename, value):
        (self.checkpoint / filename).write_text(json.dumps(value))

    def test_clean_bf16_and_ptq_remain_accepted(self):
        validate_recovery_checkpoint(self.checkpoint)
        self.write("ptq_recipe.json", {})
        validate_recovery_checkpoint(self.checkpoint)

    def test_raw_recovery_is_rejected_even_with_copied_ptq_recipe(self):
        self.write("recovery_manifest.json", {"w4a4_enabled": True})
        for recipe in (False, True):
            with self.subTest(ptq_recipe=recipe):
                if recipe:
                    self.write("ptq_recipe.json", {})
                with self.assertRaisesRegex(ValueError, "Raw recovery checkpoint"):
                    validate_recovery_checkpoint(self.checkpoint)

    def test_completed_formal_deployment_remains_accepted(self):
        self.write("recovery_manifest.json", {"w4a4_enabled": True})
        self.write("merge_manifest.json", {
            "status": "complete", "base": "/frozen-ptq-base",
            "training_checkpoint": "/training/checkpoint-2000",
        })
        validate_recovery_checkpoint(self.checkpoint)

    def test_incomplete_or_invalid_deployment_cannot_bypass_guard(self):
        self.write("recovery_manifest.json", {"w4a4_enabled": True})
        for manifest in (None, {}, {"status": "complete"},
                         {"status": "verified", "base": "/base", "training_checkpoint": "/train"},
                         {"status": "complete", "base": "", "training_checkpoint": "/train"}):
            with self.subTest(manifest=manifest):
                self.write("merge_manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "complete merge_manifest"):
                    validate_recovery_checkpoint(self.checkpoint)
        (self.checkpoint / "merge_manifest.json").write_text("{")
        with self.assertRaisesRegex(ValueError, "Invalid recovery deployment manifest"):
            validate_recovery_checkpoint(self.checkpoint)

    def test_batch_entry_rejects_before_output_or_server_launch(self):
        self.write("recovery_manifest.json", {"w4a4_enabled": True})
        output = self.root / "evaluation"
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "eval/run_recovery_eval.py"),
             "--checkpoint", str(self.checkpoint), "--out", str(output),
             "--seed", "1", "--purpose", "smoke", "--task-count", "1"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Raw recovery checkpoint", result.stderr)
        self.assertFalse(output.exists())

    def test_direct_server_rejects_before_gr00t_import_under_both_env_modes(self):
        self.write("recovery_manifest.json", {"w4a4_enabled": True})
        for mode in ("0", "1"):
            with self.subTest(w4a4=mode):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "eval/serve_recovery.py"),
                     "--model-path=" + str(self.checkpoint)],
                    capture_output=True, text=True, timeout=10,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                         "FP4VLA_W4A4": mode, "FP4VLA_W4A4_ADAPTER": mode})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Raw recovery checkpoint", result.stderr)
                self.assertNotIn("No module named", result.stderr)


if __name__ == "__main__":
    unittest.main()
