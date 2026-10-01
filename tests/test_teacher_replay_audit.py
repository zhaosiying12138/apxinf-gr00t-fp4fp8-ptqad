"""Check that training cannot accept failed, mislabeled or uncovered captures."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from exp.verify_teacher_replay import TASKS, audit_replay, digest
from eval.run_recovery_eval import parse_log


class TeacherReplayAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.teacher = self.root / "teacher"
        self.teacher.mkdir()
        for name in ("config.json", "statistics.json"):
            (self.teacher / name).write_text("{}")
        (self.teacher / "model.safetensors").write_bytes(b"identity-only-fixture")
        self.protocol = self.root / "protocol.json"
        self.write(self.protocol, {"partitions": {
            "teacher_supervision": {"seed": 850000, "episodes_per_task": 3, "init_state_indices": [20, 21, 22]},
            "heldout": {"init_state_indices": [24, 25]}}})
        self.evaluation = self.root / "teacher_rollouts"
        self.evaluation.mkdir()
        self.observations = self.evaluation / "observations"
        self.write(self.evaluation / "eval_manifest.json", {
            "purpose": "teacher_supervision", "checkpoint": str(self.teacher),
            "protocol_sha256": digest(self.protocol), "tasks": TASKS,
            "seed": 850000, "episodes": 3, "init_state_indices": [20, 21, 22]})
        results = {}
        for index, task in enumerate(TASKS):
            folder = self.observations / task
            folder.mkdir(parents=True)
            resets = [{"episode_index": e, "seed": 850000 + index * 1000 + e,
                       "init_state_index": 20 + e, "settle_steps": 10,
                       "initial_state_sha256": "a" * 64, "restored_state_sha256": "b" * 64,
                       "init_state_bank_sha256": "c" * 64} for e in range(3)]
            log = self.evaluation / (task + ".log")
            log.write_text("".join("FP4VLA_EPISODE_RESET " + json.dumps(r) + "\n" for r in resets)
                           + f"results: ('{task}', [True, True, True])\n")
            results[task] = {**parse_log(log), "returncode": 0}
            events = folder / "reset_events.jsonl"
            events.write_text("".join(json.dumps(r) + "\n" for r in resets))
            self.write(folder / "capture_manifest.json", {
                "source_kind": "teacher_rollout", "checkpoint_role": "teacher",
                "finalized": True, "finalized_purpose": "teacher_supervision",
                "student_checkpoint": str(self.teacher), "task_name": task,
                "protocol_sha256": digest(self.protocol),
                "student_statistics_sha256": digest(self.teacher / "statistics.json"),
                "student_config_sha256": digest(self.teacher / "config.json"),
                "reset_event_file_sha256": digest(events), "accepted_samples": 3,
                "successful_episode_indices": [0, 1, 2]})
            for episode, reset in enumerate(resets):
                torch.save({"source_kind": "teacher_rollout", "checkpoint_role": "teacher",
                            "task_name": task, "student_checkpoint": str(self.teacher),
                            "student_statistics_sha256": digest(self.teacher / "statistics.json"),
                            "episode_index": episode, "episode_seed": reset["seed"],
                            "init_state_index": reset["init_state_index"], "reset_identity": reset,
                            "episode_success": True, "inputs": {"action": torch.zeros(1, 16, 32),
                            "action_mask": torch.cat((torch.ones(1, 16, 7), torch.zeros(1, 16, 25)), dim=-1)}},
                           folder / f"sample_{episode:06d}.pt")
        self.write(self.evaluation / "task_results.json", results)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def audit(self):
        return audit_replay(self.observations, self.protocol, self.teacher)

    def sample_change(self, **changes):
        path = self.observations / TASKS[0] / "sample_000000.pt"
        sample = torch.load(path, weights_only=True)
        sample.update(changes)
        torch.save(sample, path)

    def test_verified_ten_task_teacher(self):
        report = self.audit()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["task_count"], 10)
        self.assertEqual(report["sample_count"], 30)

    def test_student_or_unfinalized_sample_rejected(self):
        for changed in ({"source_kind": "student_rollout"}, {"episode_success": False}):
            with self.subTest(changed=changed):
                self.sample_change(source_kind="teacher_rollout", episode_success=True)
                self.sample_change(**changed)
                with self.assertRaises(ValueError):
                    self.audit()

    def test_manifest_success_count_is_not_payload_coverage(self):
        folder = self.observations / TASKS[0]
        # Three snapshots from one successful episode are not three episodes.
        first = torch.load(folder / "sample_000000.pt", weights_only=True)
        for episode in (1, 2):
            torch.save(first, folder / f"sample_{episode:06d}.pt")
        with self.assertRaisesRegex(ValueError, "Too few successful episodes"):
            self.audit()

    def test_reset_mismatch_rejected(self):
        self.sample_change(episode_seed=999)
        with self.assertRaisesRegex(ValueError, "episode_seed"):
            self.audit()

    def test_unknown_training_sample_rejected(self):
        torch.save({}, self.observations / "sample_surprise.pt")
        with self.assertRaisesRegex(ValueError, "Unexpected samples"):
            self.audit()

    def test_changed_raw_results_rejected(self):
        path = self.evaluation / "task_results.json"
        rows = json.loads(path.read_text())
        rows[TASKS[0]]["results"][0] = False
        self.write(path, rows)
        with self.assertRaisesRegex(ValueError, "raw log mismatch"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
