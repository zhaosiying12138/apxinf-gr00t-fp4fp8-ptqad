"""CPU-only evidence and data-isolation checks for full-query capture views."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from capture_sampling import (CANDIDATE_SCHEMA, VIEW_SCHEMA, finalize_candidates,
                              materialize_capture_view, plan_capture_view, select_indices)


class CaptureSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def fixture(self, queries=(33, 9), success=(True, False), purpose="teacher_supervision"):
        directory = self.root / "raw"
        directory.mkdir()
        kind = "teacher_rollout" if purpose == "teacher_supervision" else "student_rollout"
        manifest = {"sampling_mode": "full_trajectory_candidates", "every_server_calls": 4,
                    "per_episode_limit": 24, "task_name": "task", "purpose": purpose,
                    "n_envs": 1, "sampling_interval_basis": "episode_call", "candidate_prefix": "candidate_",
                    "source_kind": kind, "student_checkpoint": "/immutable/checkpoint",
                    "protocol_sha256": "d" * 64}
        self.write(directory / "capture_manifest.json", manifest)
        resets = []
        saved = 0
        prefix = 0
        for episode, count in enumerate(queries):
            reset = {"task_name": "task", "episode_index": episode, "seed": 100 + episode,
                     "init_state_index": 20 + episode, "initial_state_sha256": "a" * 64,
                     "restored_state_sha256": "b" * 64, "init_state_bank_sha256": "c" * 64}
            resets.append(reset)
            for call in range(1, count + 1, 4):
                sample = {"task_name": "task", "source_kind": kind, "episode_index": episode,
                          "sampling_mode": "full_trajectory_candidates",
                          "episode_seed": 100 + episode, "init_state_index": 20 + episode,
                          "capture_index": saved, "episode_call": call, "server_call": prefix + call,
                          "reset_identity": reset, "episode_success": None,
                          "inputs": {"state": torch.tensor([[saved]], dtype=torch.bfloat16),
                                     "input_ids": torch.tensor([[1, saved]], dtype=torch.int64)},
                          "student_checkpoint": "/immutable/checkpoint", "task_text": "instruction"}
                torch.save(sample, directory / f"candidate_{saved:06d}.pt")
                saved += 1
            prefix += count
        self.write(directory / "capture_counts.json", {
            "calls": sum(queries), "saved": saved, "candidates_saved": saved,
            "episode_query_counts": {str(i): count for i, count in enumerate(queries)},
            "safety_limit_hit": False})
        return directory, {"results": list(success), "resets": resets}, purpose

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def hashes(directory):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}

    def finalize(self, **kwargs):
        directory, result, purpose = self.fixture(**kwargs)
        summary = finalize_candidates(directory, "task", result, purpose)
        return directory, result, purpose, summary

    def test_selection_floor_partitions_and_lower_medians(self):
        self.assertEqual(select_indices(12, 4, "stratified"), [1, 4, 7, 10])
        self.assertEqual(select_indices(10, 4, "stratified"), [0, 3, 5, 8])
        self.assertEqual(select_indices(8, 4, "stratified"), [0, 2, 4, 6])
        self.assertEqual(select_indices(9, 4, "head"), [0, 1, 2, 3])
        self.assertEqual(select_indices(3, 4, "stratified"), [0, 1, 2])
        self.assertEqual(select_indices(0), [])
        for arguments in [(True, 4, "head"), (-1, 4, "head"), (2, 0, "head"), (3, 2, "random")]:
            with self.assertRaises(ValueError):
                select_indices(*arguments)

    def test_full_finalize_keeps_failures_and_is_idempotent(self):
        directory, result, purpose, summary = self.finalize()
        self.assertEqual(summary["total_candidates"], 12)
        self.assertEqual(summary["failed_candidates"], 3)
        self.assertEqual(list(directory.glob("sample_*.pt")), [])
        self.assertEqual(list(directory.glob("rejected_*.pt")), [])
        manifest = json.loads((directory / "capture_manifest.json").read_text())
        self.assertEqual(manifest["schema"], CANDIDATE_SCHEMA)
        self.assertEqual(len(manifest["candidates"]), 12)
        before = self.hashes(directory)
        self.assertEqual(finalize_candidates(directory, "task", result, purpose), summary)
        self.assertEqual(before, self.hashes(directory))

    def test_two_views_preserve_tensor_and_identity_content_and_source(self):
        directory, _, _, _ = self.finalize()
        before = self.hashes(directory)
        for mode in ("head", "stratified"):
            plan = plan_capture_view(directory, "task", mode=mode)
            viewdir = self.root / mode
            view = materialize_capture_view(plan, viewdir)
            self.assertEqual(view["schema"], VIEW_SCHEMA)
            self.assertFalse(view["legacy_audit_compatible"])
            self.assertEqual(view["accepted_samples"], 4)
            self.assertEqual(view["rejected_samples"], 3)
            self.assertEqual(len(list(viewdir.glob("sample_*.pt"))), 4)
            self.assertEqual(len(list(viewdir.glob("rejected_*.pt"))), 3)
            for index, entry in enumerate(view["selected_sources"]):
                old = torch.load(directory / entry["source_filename"], weights_only=True)
                new = torch.load(viewdir / entry["output_filename"], weights_only=True)
                self.assertEqual(new["capture_index"], index)
                self.assertEqual(new["source_capture_index"], old["capture_index"])
                self.assertEqual(new["source_server_call"], old["server_call"])
                for key in ("reset_identity", "episode_index", "episode_call", "episode_success",
                            "source_kind", "student_checkpoint", "task_text", "server_call"):
                    self.assertEqual(new[key], old[key])
                for key, tensor in old["inputs"].items():
                    self.assertEqual(new["inputs"][key].dtype, tensor.dtype)
                    self.assertTrue(torch.equal(new["inputs"][key], tensor))
        self.assertEqual(before, self.hashes(directory))

    def test_collection_keeps_failed_selected_inputs_for_training(self):
        directory, _, _, _ = self.finalize(purpose="collection")
        view = materialize_capture_view(plan_capture_view(directory, "task", mode="stratified"),
                                        self.root / "collection_view")
        self.assertEqual(view["accepted_samples"], 7)
        self.assertEqual(view["rejected_samples"], 0)
        self.assertEqual(view["unsuccessful_samples"], 3)

    def test_incomplete_last_interval_fails_before_modifying_candidates(self):
        directory, result, purpose = self.fixture()
        (directory / "candidate_000011.pt").unlink()
        counts = json.loads((directory / "capture_counts.json").read_text())
        counts.update(saved=11, candidates_saved=11)
        self.write(directory / "capture_counts.json", counts)
        before = self.hashes(directory)
        with self.assertRaisesRegex(ValueError, "Incomplete query-interval"):
            finalize_candidates(directory, "task", result, purpose)
        self.assertEqual(before, self.hashes(directory))

    def test_identity_or_call_mismatch_fails_before_any_success_label_write(self):
        directory, result, purpose = self.fixture()
        path = directory / "candidate_000011.pt"
        sample = torch.load(path, weights_only=True)
        sample["server_call"] += 1
        torch.save(sample, path)
        before = self.hashes(directory)
        with self.assertRaisesRegex(ValueError, "server_call"):
            finalize_candidates(directory, "task", result, purpose)
        self.assertEqual(before, self.hashes(directory))

    def test_unobserved_or_extra_unscored_episode_is_rejected(self):
        directory, result, purpose = self.fixture()
        path = directory / "capture_counts.json"
        original = json.loads(path.read_text())
        for query_counts in ({"0": 33, "1": 0}, {"0": 33}, {"0": 33, "1": 9, "2": 1}):
            self.write(path, {**original, "episode_query_counts": query_counts})
            with self.assertRaises(ValueError):
                finalize_candidates(directory, "task", result, purpose)

    def test_safety_flag_missing_or_hit_is_rejected(self):
        directory, result, purpose = self.fixture()
        path = directory / "capture_counts.json"
        counts = json.loads(path.read_text())
        for value in (True, None, 0):
            self.write(path, {**counts, "safety_limit_hit": value})
            with self.assertRaisesRegex(ValueError, "safety status"):
                finalize_candidates(directory, "task", result, purpose)

    def test_checkpoint_mix_or_partial_write_fails_without_changes(self):
        directory, result, purpose = self.fixture()
        path = directory / "candidate_000011.pt"
        original = torch.load(path, weights_only=True)
        torch.save({**original, "student_checkpoint": "/different/checkpoint"}, path)
        before = self.hashes(directory)
        with self.assertRaisesRegex(ValueError, "student_checkpoint"):
            finalize_candidates(directory, "task", result, purpose)
        self.assertEqual(before, self.hashes(directory))
        torch.save(original, path)
        (directory / "candidate_000011.pt.tmp").write_bytes(b"interrupted")
        before = self.hashes(directory)
        with self.assertRaisesRegex(ValueError, "Unfinished capture"):
            finalize_candidates(directory, "task", result, purpose)
        self.assertEqual(before, self.hashes(directory))

    def test_source_mutation_and_forged_plan_rejected(self):
        directory, _, _, _ = self.finalize()
        plan = plan_capture_view(directory, "task", mode="head")
        forged = copy.deepcopy(plan)
        forged["selected"][0]["output_filename"] = "../escaped.pt"
        with self.assertRaisesRegex(ValueError, "plan or source"):
            materialize_capture_view(forged, self.root / "bad_view")
        self.assertFalse((self.root / "bad_view").exists())
        path = directory / "candidate_000000.pt"
        sample = torch.load(path, weights_only=True)
        sample["inputs"]["state"] += 1
        torch.save(sample, path)
        with self.assertRaisesRegex(ValueError, "changed after finalization"):
            materialize_capture_view(plan, self.root / "changed_source")

    def test_refuses_output_overwrite_or_source_child(self):
        directory, _, _, _ = self.finalize()
        plan = plan_capture_view(directory, "task", mode="head")
        with self.assertRaises(ValueError):
            materialize_capture_view(plan, directory / "view")
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            materialize_capture_view(plan, output)

    def test_reset_log_mutation_blocks_view_and_refinalization(self):
        directory, result, purpose = self.fixture()
        events = directory / "reset_events.jsonl"
        events.write_text("original reset log\n")
        path = directory / "capture_manifest.json"
        self.write(path, {**json.loads(path.read_text()), "reset_event_file": str(events)})
        finalize_candidates(directory, "task", result, purpose)
        events.write_text("changed reset log\n")
        with self.assertRaisesRegex(ValueError, "Reset event file changed"):
            plan_capture_view(directory, "task", mode="head")
        with self.assertRaisesRegex(ValueError, "Reset event file changed"):
            finalize_candidates(directory, "task", result, purpose)

    def test_capture_contract_fields_are_required(self):
        directory, result, purpose = self.fixture()
        path = directory / "capture_manifest.json"
        original = json.loads(path.read_text())
        for key, value in (("n_envs", 2), ("sampling_interval_basis", "server_call"),
                           ("candidate_prefix", "sample_"), ("protocol_sha256", "bad")):
            self.write(path, {**original, key: value})
            with self.assertRaises(ValueError):
                finalize_candidates(directory, "task", result, purpose)
        self.write(path, original)
        candidate = directory / "candidate_000000.pt"
        sample = torch.load(candidate, weights_only=True)
        sample.pop("sampling_mode")
        torch.save(sample, candidate)
        with self.assertRaisesRegex(ValueError, "sampling_mode"):
            finalize_candidates(directory, "task", result, purpose)

    def test_runtime_to_finalized_views_integration_without_gr00t_or_gpu(self):
        from capture_onpolicy import install_capture

        class FakeCollator:
            def __call__(self, features):
                return {"inputs": {"state": torch.ones(1, 1, 2)}}

        class FakeModel:
            calls = 0

            def get_action(self, inputs):
                self.calls += 1
                return {"action_pred": torch.full((1, 2, 7), float(self.calls))}

        model_module = ModuleType("gr00t.model.gr00t_n1d7.gr00t_n1d7")
        model_module.Gr00tN1d7 = FakeModel
        collate_module = ModuleType("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
        collate_module.Gr00tN1d7DataCollator = FakeCollator
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        for filename in ("config.json", "statistics.json"):
            (checkpoint / filename).write_text("{}")
        events = self.root / "events.jsonl"
        directory = self.root / "runtime"
        resets = []
        with patch.dict(sys.modules, {model_module.__name__: model_module, collate_module.__name__: collate_module}), \
             patch("capture_onpolicy.libero_action_spec", return_value={}), \
             patch("capture_onpolicy.action_mask", side_effect=lambda action, spec: torch.ones_like(action)):
            install_capture(directory, checkpoint, every=4, per_task=20, limit=20, per_episode=10,
                            event_file=events, sampling_mode="full_trajectory_candidates",
                            capture_metadata={"task_name": "task", "purpose": "collection",
                                              "source_kind": "student_rollout", "protocol_sha256": "d" * 64})
            model = FakeModel()
            for episode, queries in enumerate((18, 6)):
                reset = {"event": "reset", "task_name": "task", "episode_index": episode,
                         "seed": 100 + episode, "init_state_index": 20 + episode,
                         "initial_state_sha256": "a" * 64, "restored_state_sha256": "b" * 64,
                         "init_state_bank_sha256": "c" * 64}
                resets.append(reset)
                with events.open("a") as stream:
                    stream.write(json.dumps(reset) + "\n")
                for _ in range(queries):
                    inputs = FakeCollator()([{"vlm_content": {"text": "instruction"}}])["inputs"]
                    model.get_action(inputs)
            self.assertEqual(model.calls, 24)
        summary = finalize_candidates(directory, "task", {"results": [True, False], "resets": resets}, "collection")
        self.assertEqual(summary["total_candidates"], 7)
        plan = plan_capture_view(directory, "task", mode="stratified")
        view = materialize_capture_view(plan, self.root / "runtime_view")
        self.assertEqual(view["total_samples"], 6)
        self.assertEqual(view["unsuccessful_samples"], 2)


if __name__ == "__main__":
    unittest.main()
