"""CPU-only checks for recovery rollout capture environment setup."""
from pathlib import Path
import tempfile
import unittest

from eval.run_recovery_eval import configure_capture_environment


class RecoveryEvalCaptureEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "eval"
        self.manifest = {"protocol_sha256": "p" * 64}
        self.indices = [20, 21, 22, 23]

    def test_fresh_event_file_path_is_exported_for_teacher_and_collection(self):
        for purpose in ("teacher_supervision", "collection"):
            with self.subTest(purpose=purpose):
                env = {"OPD_CAPTURE_DIR": "stale", "UNRELATED": "kept"}
                task = "task_" + purpose
                capture_dir = configure_capture_environment(
                    env, self.output, task, purpose, self.manifest, 850000, self.indices)
                event_file = self.output / "observations" / task / "reset_events.jsonl"
                self.assertEqual(capture_dir, event_file.parent)
                self.assertFalse(event_file.exists())
                self.assertEqual(env["OPD_CAPTURE_DIR"], str(capture_dir))
                self.assertEqual(env["FP4VLA_CAPTURE_EVENT_FILE"], str(event_file))
                self.assertEqual(env["FP4VLA_CAPTURE_PURPOSE"], purpose)
                self.assertEqual(env["FP4VLA_CAPTURE_INIT_STATE_INDICES"], "20,21,22,23")
                self.assertEqual(env["FP4VLA_CAPTURE_PROTOCOL_SHA256"], "p" * 64)
                self.assertEqual(env["OPD_CAPTURE_PER_EPISODE"], "4")
                self.assertEqual(env["UNRELATED"], "kept")

    def test_existing_event_file_is_replaced_and_capture_remains_enabled(self):
        env = {}
        task = "task"
        capture_dir = self.output / "observations" / task
        capture_dir.mkdir(parents=True)
        event_file = capture_dir / "reset_events.jsonl"
        event_file.write_text("stale event\n")

        returned = configure_capture_environment(
            env, self.output, task, "collection", self.manifest, 960000, self.indices)

        self.assertEqual(returned, capture_dir)
        self.assertFalse(event_file.exists())
        self.assertEqual(env["FP4VLA_CAPTURE_EVENT_FILE"], str(event_file))

    def test_non_capture_purpose_removes_inherited_capture_directory(self):
        env = {
            "OPD_CAPTURE_DIR": "stale",
            "FP4VLA_CAPTURE_EVENT_FILE": "stale-events",
            "OPD_CAPTURE_LIMIT": "160",
        }
        returned = configure_capture_environment(
            env, self.output, "task", "heldout", self.manifest, 670000, self.indices)

        self.assertIsNone(returned)
        self.assertNotIn("OPD_CAPTURE_DIR", env)
        self.assertEqual(env["FP4VLA_CAPTURE_EVENT_FILE"], "stale-events")
        self.assertEqual(env["OPD_CAPTURE_LIMIT"], "160")


if __name__ == "__main__":
    unittest.main()
