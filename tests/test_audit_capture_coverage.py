"""CPU provenance checks for temporal capture coverage, without model loading."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from exp.audit_capture_coverage import audit, identity


class CaptureCoverageTests(unittest.TestCase):
    def fixture(self, root, purpose="teacher_supervision"):
        folder = root / "observations/task_a"
        folder.mkdir(parents=True)
        source_kind = "teacher_rollout" if purpose == "teacher_supervision" else "student_rollout"
        resets = [{"event": "reset", "task_name": "task_a", "episode_index": i,
                   "seed": 100 + i, "init_state_index": 20 + i,
                   "initial_state_sha256": "a" * 64, "restored_state_sha256": "b" * 64,
                   "init_state_bank_sha256": "c" * 64} for i in range(2)]
        events = folder / "reset_events.jsonl"
        events.write_text("".join(json.dumps(r) + "\n" for r in resets))
        log = root / "task_a.log"
        log.write_text("results: " + repr(("libero_sim/task_a", [True, False],
                                          {"episode_lengths": [164, 720]})) + "\n")
        (root / "eval_manifest.json").write_text(json.dumps({"purpose": purpose,
            "tasks": ["task_a"], "episodes": 2, "protocol_sha256": "protocol"}))
        (root / "task_results.json").write_text(json.dumps({"task_a": {
            "results": [True, False], "resets": resets, "log_sha256": identity(log, root)["sha256"]}}))
        (folder / "capture_manifest.json").write_text(json.dumps({
            "finalized": True, "purpose": purpose, "finalized_purpose": purpose,
            "task_name": "task_a", "source_kind": source_kind, "protocol_sha256": "protocol",
            "total_samples": 4, "accepted_samples": 2, "unsuccessful_samples": 2,
            "rejected_samples": 2 if purpose == "teacher_supervision" else 0,
            "scored_episode_count": 2, "successful_episode_count": 1,
            "reset_event_file_sha256": identity(events, root)["sha256"]}))
        (folder / "capture_counts.json").write_text(json.dumps({"saved": 4}))
        for index, call in enumerate((1, 5, 25, 29)):
            episode = index // 2
            prefix = "rejected" if episode == 1 and purpose == "teacher_supervision" else "sample"
            torch.save({"task_name": "task_a", "episode_index": episode,
                "capture_index": index, "server_call": call, "episode_success": episode == 0,
                "reset_identity": resets[episode], "episode_seed": 100 + episode,
                "init_state_index": 20 + episode, "episode_result_index": episode,
                "source_kind": source_kind, "inputs": {"state": torch.zeros(1)}},
                folder / f"{prefix}_{index:06d}.pt")
        return folder

    def mutate(self, path, **updates):
        sample = torch.load(path, map_location="cpu", weights_only=True)
        sample.update(updates)
        torch.save(sample, path)

    def test_teacher_report_binds_files_and_filters_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            report = audit(root)
            self.assertEqual(report["summary"]["accepted_windows"], 2)
            self.assertEqual(report["summary"]["rejected_windows"], 2)
            self.assertEqual(report["summary"]["successful_episode_length_range"], [164, 164])
            sample = report["tasks"][0]["episodes"][0]["accepted_samples"][0]
            self.assertEqual(sample["file"], identity(folder / "sample_000000.pt", root))
            self.assertEqual(report["tasks"][0]["episode_lengths_log_line"], 1)
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_wrong_success_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self.fixture(root)
            self.mutate(folder / "sample_000000.pt", episode_success=False)
            with self.assertRaisesRegex(ValueError, "success label"):
                audit(root)

    def test_missing_or_duplicate_capture_index_is_rejected(self):
        for problem in ("missing", "duplicate"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = self.fixture(root)
                if problem == "missing":
                    (folder / "sample_000001.pt").unlink()
                else:
                    self.mutate(folder / "sample_000001.pt", capture_index=0)
                with self.assertRaisesRegex(ValueError, "capture_index"):
                    audit(root)

    def test_collection_correctly_retains_failed_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, purpose="collection")
            report = audit(root)
            self.assertEqual(report["summary"]["accepted_windows"], 4)
            self.assertEqual(report["summary"]["successful_windows"], 2)
            self.assertEqual(report["summary"]["rejected_windows"], 0)
            failed = report["tasks"][0]["episodes"][1]
            self.assertFalse(failed["success"])
            self.assertEqual(failed["accepted_server_calls"], [25, 29])
            self.assertEqual(report["summary"]["accepted_source_episode_length_range"], [164, 720])

    def test_sample_reset_identity_and_original_log_are_checked(self):
        for problem in ("reset", "log"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = self.fixture(root)
                if problem == "reset":
                    self.mutate(folder / "sample_000000.pt", episode_seed=999)
                else:
                    (root / "task_a.log").write_text("results: ('altered', [], {})\n")
                with self.assertRaisesRegex(ValueError, "identity|hash"):
                    audit(root)


if __name__ == "__main__":
    unittest.main()
