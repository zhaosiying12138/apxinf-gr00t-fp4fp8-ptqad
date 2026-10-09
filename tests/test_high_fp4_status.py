"""Regression checks for optimizer progress in mixed loading/training logs."""
import json
from pathlib import Path
import tempfile
import unittest

from exp.high_fp4_status import status


class TrainingProgressTests(unittest.TestCase):
    def stage(self, log, steps=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recovery = root / "recovery"
            folder = recovery / "artifacts" / "train_example"
            folder.mkdir(parents=True)
            (recovery / "logs").mkdir()
            (recovery / "run_manifest.json").write_text("{}")
            (recovery / "logs" / "train_example.log").write_text(log)
            if steps is not None:
                (folder / "orchestrator_training_request.json").write_text(
                    json.dumps({"environment": {"QAD_STEPS": steps}})
                )
            return status(root / "development", recovery)["recovery"]["stages"]["train_example"]

    def test_startup_loading_is_not_training(self):
        log = ("\rLoading checkpoint shards: 100%|##########| 2/2 [00:10<00:00]\n"
               "Initializing datasets: 100%|##########| 1/1 [00:00<00:00]\n")
        result = self.stage(log, "2000")
        self.assertNotIn("logged_optimizer_steps", result)
        self.assertEqual(result["requested_optimizer_steps"], 2000)
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("logged_optimizer_steps", self.stage(log))

    def test_real_training_survives_later_loading_bars(self):
        log = ("\r  0%|          | 0/2000 [00:00<?, ?it/s]\r"
               "  1%|#         | 21/2000 [00:45<10:00]\r"
               "Loading checkpoint shards: 100%|##########| 2/2 [00:01<00:00]\n")
        self.assertEqual(self.stage(log, "2000")["logged_optimizer_steps"], 21)

    def test_short_smoke_training_is_preserved(self):
        log = "\r 50%|#####     | 1/2 [00:05<00:05]\r"
        result = self.stage(log, "2")
        self.assertEqual(result["logged_optimizer_steps"], 1)
        self.assertEqual(result["requested_optimizer_steps"], 2)

    def test_different_budget_does_not_replace_requested_steps(self):
        log = "\r 50%|#####     | 50/100 [00:05<00:05]\r"
        result = self.stage(log, "2000")
        self.assertNotIn("logged_optimizer_steps", result)
        self.assertEqual(result["requested_optimizer_steps"], 2000)

    def test_full_bar_alone_does_not_prove_completion(self):
        result = self.stage("\r100%|##########| 2000/2000 [00:05<00:00]\r", "2000")
        self.assertEqual(result["logged_optimizer_steps"], 2000)
        self.assertEqual(result["status"], "incomplete")

    def test_invalid_request_does_not_invent_requested_budget(self):
        for invalid in (True, False, 2.5, 0, -2, "bad", "2.5"):
            with self.subTest(steps=invalid):
                result = self.stage("Loading checkpoint shards: 100%|#| 2/2 [done]\\n", invalid)
                self.assertNotIn("logged_optimizer_steps", result)
                self.assertNotIn("requested_optimizer_steps", result)

    def test_tail_cut_cannot_strip_loader_label(self):
        # The 64-KiB tail starts immediately after the loader label.
        suffix = "100%|##########| 2/2 [00:01<00:00]\n"
        suffix += "x" * (65536 - len(suffix))
        result = self.stage("Loading checkpoint shards: " + suffix)
        self.assertNotIn("logged_optimizer_steps", result)


if __name__ == "__main__":
    unittest.main()
