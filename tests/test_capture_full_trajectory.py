"""CPU-only checks for opt-in complete-trajectory candidate capture."""
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "rl"))
from capture_onpolicy import install_capture


class FullTrajectoryCaptureTests(unittest.TestCase):
    def checkpoint(self, root):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "statistics.json").write_text(json.dumps({
            "libero_sim": {"action": {"action": {"mean": [0.0], "std": [1.0]}}}}))
        (checkpoint / "processor_config.json").write_text(json.dumps({
            "processor_kwargs": {"modality_configs": {
                "libero_sim": {"action": {"delta_indices": [0, 1],
                                             "modality_keys": ["action"]}}}}}))
        return checkpoint

    def reset(self, path, episode):
        record = {"event": "reset", "task_name": "task", "episode_index": episode,
                  "seed": 100 + episode, "init_state_index": 20 + episode,
                  "initial_state_sha256": "a" * 64, "restored_state_sha256": "b" * 64,
                  "init_state_bank_sha256": "c" * 64}
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    @contextmanager
    def fake_runtime(self):
        class FakeCollator:
            def __init__(self):
                self.calls = 0

            def __call__(self, features):
                self.calls += 1
                # Capture must not consume a second stochastic collation.
                state = torch.rand(len(features), 1, 2) + random.random() + np.random.random()
                return {"inputs": {"state": state}}

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def get_action(self, inputs):
                self.calls += 1
                batch = inputs["state"].shape[0]
                return {"action_pred": torch.randn(batch, 2, 7) + inputs["state"].mean(),
                        "untouched": "model output"}

        model_module = ModuleType("gr00t.model.gr00t_n1d7.gr00t_n1d7")
        model_module.Gr00tN1d7 = FakeModel
        collator_module = ModuleType("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
        collator_module.Gr00tN1d7DataCollator = FakeCollator
        with patch.dict(sys.modules, {model_module.__name__: model_module,
                                      collator_module.__name__: collator_module}):
            yield FakeModel(), FakeCollator()

    def query(self, model, collator, batch=1):
        with torch.inference_mode():
            inputs = collator([{"vlm_content": {"text": "instruction"}}] * batch)["inputs"]
            return model.get_action(inputs)

    def install(self, root, **overrides):
        settings = {"every": 4, "per_task": 100, "limit": 100,
                    "per_episode": 100, "event_file": root / "resets.jsonl",
                    "capture_metadata": {"task_name": "task"},
                    "sampling_mode": "full_trajectory_candidates"}
        settings.update(overrides)
        install_capture(root / "capture", self.checkpoint(root), **settings)

    def candidates(self, root):
        return [torch.load(p, map_location="cpu", weights_only=True)
                for p in sorted((root / "capture").glob("candidate_*.pt"))]

    def test_late_candidates_and_all_query_counts_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp, self.fake_runtime() as (model, collator):
            root = Path(tmp)
            self.install(root)
            self.reset(root / "resets.jsonl", 0)
            for _ in range(40):
                self.query(model, collator)
            samples = self.candidates(root)
            self.assertEqual([s["episode_call"] for s in samples], list(range(1, 41, 4)))
            self.assertEqual([s["capture_index"] for s in samples], list(range(10)))
            self.assertTrue(all(s["sampling_mode"] == "full_trajectory_candidates" for s in samples))
            counts = json.loads((root / "capture/capture_counts.json").read_text())
            self.assertEqual(counts["calls"], 40)
            self.assertEqual(counts["episode_query_counts"], {"0": 40})
            self.assertEqual(counts["candidates_saved"], counts["saved"])
            self.assertIs(counts["safety_limit_hit"], False)
            self.assertEqual((model.calls, collator.calls), (40, 40))
            self.assertFalse(list((root / "capture").glob("sample_*.pt")))
            manifest = json.loads((root / "capture/capture_manifest.json").read_text())
            self.assertEqual(manifest["candidate_prefix"], "candidate_")

    def test_sampling_restarts_on_non_global_interval_reset(self):
        with tempfile.TemporaryDirectory() as tmp, self.fake_runtime() as (model, collator):
            root = Path(tmp)
            self.install(root)
            for episode, length in enumerate((6, 8)):
                self.reset(root / "resets.jsonl", episode)
                for _ in range(length):
                    self.query(model, collator)
            samples = self.candidates(root)
            self.assertEqual([(s["episode_index"], s["episode_call"], s["server_call"])
                              for s in samples], [(0, 1, 1), (0, 5, 5), (1, 1, 7), (1, 5, 11)])
            counts = json.loads((root / "capture/capture_counts.json").read_text())
            self.assertEqual(counts["episode_query_counts"], {"0": 6, "1": 8})
            self.assertEqual(counts["calls"], 14)

    def test_actions_rng_and_inference_count_match_uncaptured_run(self):
        def run(capture):
            torch.manual_seed(12)
            random.seed(34)
            np.random.seed(56)
            with tempfile.TemporaryDirectory() as tmp, self.fake_runtime() as (model, collator):
                root = Path(tmp)
                if capture:
                    self.install(root)
                    self.reset(root / "resets.jsonl", 0)
                outputs = [self.query(model, collator) for _ in range(10)]
                return (outputs, torch.random.get_rng_state(), random.getstate(),
                        np.random.get_state(), model.calls, collator.calls)
        baseline, captured = run(False), run(True)
        for expected, actual in zip(baseline[0], captured[0]):
            self.assertEqual(expected["untouched"], actual["untouched"])
            self.assertTrue(torch.equal(expected["action_pred"], actual["action_pred"]))
        self.assertTrue(torch.equal(baseline[1], captured[1]))
        self.assertEqual(baseline[2], captured[2])
        self.assertEqual(baseline[3][0], captured[3][0])
        np.testing.assert_array_equal(baseline[3][1], captured[3][1])
        self.assertEqual(baseline[3][2:], captured[3][2:])
        self.assertEqual(baseline[4:], captured[4:])

    def test_each_safety_budget_raises_instead_of_truncating(self):
        for bound in ("per_episode", "per_task", "limit"):
            with self.subTest(bound=bound), tempfile.TemporaryDirectory() as tmp, self.fake_runtime() as runtime:
                root = Path(tmp)
                model, collator = runtime
                self.install(root, **{bound: 1})
                self.reset(root / "resets.jsonl", 0)
                for _ in range(4):
                    self.query(model, collator)
                with self.assertRaisesRegex(RuntimeError, "safety bound exceeded"):
                    self.query(model, collator)
                self.assertEqual(len(self.candidates(root)), 1)
                counts = json.loads((root / "capture/capture_counts.json").read_text())
                self.assertEqual(counts["calls"], 5)
                self.assertTrue(counts["safety_limit_hit"])

    def test_single_environment_and_reset_contract_required(self):
        with tempfile.TemporaryDirectory() as tmp, self.fake_runtime() as (model, collator):
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "requires reset events"):
                self.install(root, event_file=None, per_episode=None)
        with tempfile.TemporaryDirectory() as tmp, self.fake_runtime() as (model, collator):
            root = Path(tmp)
            self.install(root)
            self.reset(root / "resets.jsonl", 0)
            with self.assertRaisesRegex(ValueError, "exactly one environment"):
                self.query(model, collator, batch=2)
        with tempfile.TemporaryDirectory() as tmp, self.fake_runtime() as (model, collator):
            root = Path(tmp)
            self.install(root)
            with self.assertRaisesRegex(RuntimeError, "reset identity"):
                self.query(model, collator)

    def test_full_mode_requires_explicit_integer_episode_safety_bound(self):
        for invalid in (None, 0, -1, True, 1.5):
            with self.subTest(bound=invalid), tempfile.TemporaryDirectory() as tmp, self.fake_runtime():
                root = Path(tmp)
                with self.assertRaisesRegex(ValueError, "positive integer per_episode safety bound"):
                    self.install(root, per_episode=invalid)
                self.assertFalse((root / "capture/capture_manifest.json").exists())

    def test_legacy_default_still_silently_stops_at_existing_cap(self):
        with tempfile.TemporaryDirectory() as tmp, self.fake_runtime() as (model, collator):
            root = Path(tmp)
            # Deliberately omit the new argument to check legacy compatibility.
            install_capture(root / "capture", self.checkpoint(root), every=4,
                            per_task=2, limit=2, event_file=root / "resets.jsonl",
                            per_episode=2, capture_metadata={"task_name": "task"})
            self.reset(root / "resets.jsonl", 0)
            for _ in range(20):
                self.query(model, collator)
            files = sorted((root / "capture").glob("sample_*.pt"))
            self.assertEqual(len(files), 2)
            self.assertEqual([torch.load(p, map_location="cpu", weights_only=True)["server_call"]
                              for p in files], [1, 5])
            self.assertFalse(self.candidates(root))


if __name__ == "__main__":
    unittest.main()
