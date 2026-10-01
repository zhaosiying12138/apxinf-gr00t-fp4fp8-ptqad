"""CPU-only tests for reset-linked teacher-supervision captures."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "rl"))
sys.path.insert(0, str(PROJECT / "eval"))
from capture_onpolicy import install_capture
from run_recovery_eval import finalize_capture_samples


class CaptureMetadataTests(unittest.TestCase):
    def _checkpoint(self, root):
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "statistics.json").write_text(json.dumps({
            "libero_sim": {"action": {"action": {"mean": [0.0], "std": [1.0]}}}}))
        (root / "processor_config.json").write_text(json.dumps({
            "processor_kwargs": {"modality_configs": {
                "libero_sim": {"action": {"delta_indices": [0, 1],
                                             "modality_keys": ["action"]}}}}}))

    def test_capture_consumes_reset_identity_event(self):
        class FakeCollator:
            def __call__(self, features):
                return {"inputs": {"state": torch.ones(len(features), 1, 2)}}

        class FakeModel:
            def get_action(self, inputs):
                return {"action_pred": torch.zeros(1, 2, 7)}

        model_module = ModuleType("gr00t.model.gr00t_n1d7.gr00t_n1d7")
        model_module.Gr00tN1d7 = FakeModel
        processor_module = ModuleType("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
        processor_module.Gr00tN1d7DataCollator = FakeCollator
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self._checkpoint(checkpoint)
            event_file = root / "reset_events.jsonl"
            reset = {
                "event": "reset", "task_name": "task", "episode_index": 1,
                "seed": 101, "init_state_index": 3,
                "initial_state_sha256": "a" * 64,
                "restored_state_sha256": "b" * 64,
                "init_state_bank_sha256": "c" * 64,
            }
            event_file.write_text(json.dumps(reset) + "\n")
            out = root / "capture"
            with patch.dict(sys.modules, {model_module.__name__: model_module,
                                          processor_module.__name__: processor_module}):
                install_capture(out, checkpoint, every=1, event_file=event_file,
                                capture_metadata={"task_name": "task"})
                features = [{"vlm_content": {"text": "instruction"}}]
                # The patched model calls the collator hook exactly as GR00T does.
                with torch.inference_mode():
                    FakeCollator()(features)
                    FakeModel().get_action({"state": torch.ones(1, 1, 2)})
                sample = torch.load(out / "sample_000000.pt", weights_only=True)
            self.assertEqual(sample["task_name"], "task")
            self.assertEqual(sample["episode_index"], 1)
            self.assertEqual(sample["episode_seed"], 101)
            self.assertEqual(sample["init_state_index"], 3)
            self.assertIsNone(sample["episode_success"])
            self.assertEqual(sample["reset_identity"], reset)

    def test_finalize_marks_success_and_removes_failed_from_training_glob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._checkpoint(root / "checkpoint")
            (root / "capture_manifest.json").write_text("{}")
            reset = {
                "seed": 100, "init_state_index": 2,
                "initial_state_sha256": "a" * 64,
                "restored_state_sha256": "b" * 64,
                "init_state_bank_sha256": "c" * 64,
            }
            for index in (0, 1):
                sample = {
                    "task_name": "task", "episode_index": index,
                    "episode_seed": 100 + index, "init_state_index": 2 + index,
                    "reset_identity": {**reset, "task_name": "task",
                        "episode_index": index, "seed": 100 + index,
                        "init_state_index": 2 + index},
                    "inputs": {"state": torch.zeros(1)},
                }
                (root / f"sample_{index:06d}.pt").unlink(missing_ok=True)
                torch.save(sample, root / f"sample_{index:06d}.pt")
            result = {"resets": [reset, {**reset, "seed": 101, "init_state_index": 3}],
                      "results": [True, False]}
            summary = finalize_capture_samples(root, "task", result, "teacher_supervision")
            self.assertEqual(summary["accepted_samples"], 1)
            self.assertEqual(summary["rejected_samples"], 1)
            self.assertTrue((root / "sample_000000.pt").is_file())
            self.assertTrue((root / "rejected_000001.pt").is_file())
            self.assertFalse((root / "sample_000001.pt").exists())
            accepted = torch.load(root / "sample_000000.pt", weights_only=True)
            rejected = torch.load(root / "rejected_000001.pt", weights_only=True)
            self.assertTrue(accepted["episode_success"])
            self.assertFalse(rejected["episode_success"])
            manifest = json.loads((root / "capture_manifest.json").read_text())
            self.assertEqual(manifest["accepted_samples"], 1)
            self.assertEqual(manifest["rejected_samples"], 1)

    def test_episode_quota_reserves_capacity_for_later_episodes(self):
        class FakeCollator:
            def __call__(self, features):
                return {"inputs": {"state": torch.ones(len(features), 1, 2)}}

        class FakeModel:
            def get_action(self, inputs):
                return {"action_pred": torch.zeros(1, 2, 7)}

        model_module = ModuleType("gr00t.model.gr00t_n1d7.gr00t_n1d7")
        model_module.Gr00tN1d7 = FakeModel
        processor_module = ModuleType("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
        processor_module.Gr00tN1d7DataCollator = FakeCollator
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._checkpoint(root / "checkpoint")
            events = root / "events.jsonl"
            capture = root / "capture"
            with patch.dict(sys.modules, {model_module.__name__: model_module,
                                          processor_module.__name__: processor_module}):
                install_capture(capture, root / "checkpoint", every=1,
                                per_task=6, limit=6, per_episode=2, event_file=events,
                                capture_metadata={"task_name": "task"})
                for episode in range(3):
                    reset = {"event": "reset", "task_name": "task", "episode_index": episode,
                             "seed": 100 + episode, "init_state_index": 20 + episode,
                             "initial_state_sha256": "a" * 64, "restored_state_sha256": "b" * 64,
                             "init_state_bank_sha256": "c" * 64}
                    with events.open("a") as stream:
                        stream.write(json.dumps(reset) + "\n")
                    for _ in range(5):
                        FakeCollator()([{"vlm_content": {"text": "instruction"}}])
                        FakeModel().get_action({})
                samples = [torch.load(p, weights_only=True) for p in sorted(capture.glob("sample_*.pt"))]
            self.assertEqual([s["episode_index"] for s in samples], [0, 0, 1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
