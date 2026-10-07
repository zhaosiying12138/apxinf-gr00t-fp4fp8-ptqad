"""CPU-only checks for recovery rollout capture environment setup."""
from pathlib import Path
import json
import tempfile
import unittest

from eval.run_recovery_eval import (capture_sampling_config, configure_capture_environment,
                                   validate_capture_partition_separation)


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
            "FP4VLA_CAPTURE_SAMPLING_MODE": "full_trajectory_candidates",
        }
        returned = configure_capture_environment(
            env, self.output, "task", "heldout", self.manifest, 670000, self.indices)

        self.assertIsNone(returned)
        self.assertNotIn("OPD_CAPTURE_DIR", env)
        self.assertNotIn("FP4VLA_CAPTURE_SAMPLING_MODE", env)
        self.assertEqual(env["FP4VLA_CAPTURE_EVENT_FILE"], "stale-events")
        self.assertEqual(env["OPD_CAPTURE_LIMIT"], "160")

    def test_full_mode_is_explicit_and_uses_safety_bounds(self):
        config = {"mode": "full_trajectory_candidates", "every_server_calls": 4,
                  "safety_candidates_per_episode": 192}
        manifest = dict(self.manifest, capture_sampling=capture_sampling_config(
            {"capture_sampling": config}, "collection", 4))
        env = {}
        configure_capture_environment(env, self.output, "task", "collection", manifest, 10, self.indices)
        self.assertEqual(env["FP4VLA_CAPTURE_SAMPLING_MODE"], "full_trajectory_candidates")
        self.assertEqual(env["OPD_CAPTURE_PER_EPISODE"], "192")
        self.assertEqual(env["OPD_CAPTURE_PER_TASK"], "768")
        self.assertEqual(env["OPD_CAPTURE_LIMIT"], "768")
        # Old protocols must overwrite any stale full-capture mode.
        configure_capture_environment(env, self.output, "legacy", "collection", self.manifest, 10, self.indices)
        self.assertEqual(env["FP4VLA_CAPTURE_SAMPLING_MODE"], "prefix")
        self.assertEqual(env["OPD_CAPTURE_PER_EPISODE"], "4")
        self.assertIsNone(capture_sampling_config({}, "heldout", 16))

    def test_capture_config_rejects_heldout_unknown_keys_and_truncating_cap(self):
        config = {"mode": "full_trajectory_candidates", "every_server_calls": 4,
                  "safety_candidates_per_episode": 192}
        for purpose in ("heldout", "development", "smoke"):
            with self.subTest(purpose=purpose), self.assertRaisesRegex(ValueError, "training capture"):
                capture_sampling_config({"capture_sampling": config}, purpose, 4)
        for bad in ({**config, "safety_candidates_per_episode": 4},
                    {**config, "every_server_calls": True},
                    {**config, "mode": "prefix"}, {**config, "unknown": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                capture_sampling_config({"capture_sampling": bad}, "collection", 4)

    def test_capture_partition_isolation_includes_terminal_reset(self):
        protocol = self.output.parent / "protocol.json"
        partitions = {"collection": {"seed": 200000, "init_state_indices": [20, 21]},
                      "teacher_supervision": {"seed": 210000, "init_state_indices": [20, 21]},
                      "development": {"seed": 300000, "init_state_indices": [4, 5]}}
        protocol.write_text(json.dumps({"partitions": partitions}))
        validate_capture_partition_separation(protocol, "collection")
        partitions["development"]["init_state_indices"] = [21, 22]
        protocol.write_text(json.dumps({"partitions": partitions}))
        with self.assertRaisesRegex(ValueError, "bank indices overlap"):
            validate_capture_partition_separation(protocol, "collection")
        partitions["development"] = {"seed": 209002, "init_state_indices": [4, 5]}
        protocol.write_text(json.dumps({"partitions": partitions}))
        with self.assertRaisesRegex(ValueError, "reset seeds overlap"):
            validate_capture_partition_separation(protocol, "collection")


if __name__ == "__main__":
    unittest.main()
