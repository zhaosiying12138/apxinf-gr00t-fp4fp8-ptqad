"""CPU integration checks for complete-run provenance and paired capture views."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
from eval.run_recovery_eval import (TASKS, capture_sampling_config,
                                    finalize_capture_samples, parse_log)
from exp.derive_capture_views import derive_views

MODE = "full_trajectory_candidates"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


class DeriveCaptureViewsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture_count = 0

    def fixture(self, purpose="teacher_supervision", finalized=True, formal_teacher=False,
                success=(True, False)):
        """Make ten synthetic task logs and call the real runner finalizer route."""
        self.fixture_count += 1
        directory = self.root / f"source_{self.fixture_count}"
        directory.mkdir()
        protocol = self.root / f"protocol_{self.fixture_count}.json"
        partition = {"seed": 950000, "episodes_per_task": 2, "init_state_indices": [20, 21],
                     "capture_sampling": {"mode": MODE, "every_server_calls": 4,
                                          "safety_candidates_per_episode": 181}}
        protocol_data = {"partitions": {purpose: partition}}
        if formal_teacher:
            protocol_data["capture_views"] = {"modes": ["head", "stratified"], "windows_per_episode": 4}
            protocol_data["partitions"]["heldout"] = {
                "seed": 970000, "episodes_per_task": 2, "init_state_indices": [24, 25]}
        write_json(protocol, protocol_data)
        config = capture_sampling_config(partition, purpose, 2)
        checkpoint_path = self.root / (f"frozen_checkpoint_{self.fixture_count}" if formal_teacher else "frozen_checkpoint")
        checkpoint = str(checkpoint_path)
        if formal_teacher:
            checkpoint_path.mkdir()
            write_json(checkpoint_path / "config.json", {"fixture": "identity-only"})
            write_json(checkpoint_path / "statistics.json", {"fixture": "normalization identity"})
            (checkpoint_path / "model.safetensors").write_bytes(b"hash-only fixture; never model-loaded")
        kind = "teacher_rollout" if purpose == "teacher_supervision" else "student_rollout"
        manifest = {"purpose": purpose, "tasks": TASKS, "n_envs": 1, "episodes": 2,
                    "protocol_file": str(protocol), "protocol_sha256": digest(protocol),
                    "init_state_indices": [20, 21], "seed": 950000, "max_episode_steps": 720,
                    "checkpoint": checkpoint, "capture_sampling": config}
        if formal_teacher:
            manifest["environment_summary"] = {"variables": {key: "0" for key in (
                "FP4VLA_QUANT", "FP4VLA_W4A4", "FP4VLA_W4A4_ADAPTER",
                "FP4VLA_SATURATE_F16_ACTIVATIONS")}}
        write_json(directory / "eval_manifest.json", manifest)
        task_results = {}
        for task_index, task in enumerate(TASKS):
            seed = 950000 + 1000 * task_index
            capture_dir = directory / "observations" / task
            capture_dir.mkdir(parents=True)
            capture = {"task_name": task, "purpose": purpose, "source_kind": kind,
                       "student_checkpoint": checkpoint, "protocol_sha256": digest(protocol),
                       "seed": seed, "init_state_indices": "20,21", "sampling_mode": MODE,
                       "n_envs": 1, "every_server_calls": 4, "per_episode_limit": 181,
                       "per_task_limit": 362, "total_limit": 362,
                       "sampling_interval_basis": "episode_call", "candidate_prefix": "candidate_"}
            if formal_teacher:
                capture.update(checkpoint_role="teacher" if purpose == "teacher_supervision" else "student",
                               student_config_sha256=digest(checkpoint_path / "config.json"),
                               student_statistics_sha256=digest(checkpoint_path / "statistics.json"))
            write_json(capture_dir / "capture_manifest.json", capture)
            resets = [{"task_name": task, "episode_index": episode, "seed": seed + episode,
                       "init_state_index": 20 + episode, "settle_steps": 10,
                       "initial_state_sha256": "a" * 64, "restored_state_sha256": "b" * 64,
                       "init_state_bank_sha256": "c" * 64} for episode in range(2)]
            log = directory / f"{task}.log"
            log.write_text("".join("FP4VLA_EPISODE_RESET " + json.dumps(reset) + "\n" for reset in resets)
                           + f"results: ('libero_sim/{task}', {list(success)!r})\n", encoding="utf-8")
            result = {**parse_log(log), "seed": seed, "returncode": 0}
            saved = 0
            prefix = 0
            for episode, queries in enumerate((33, 10)):
                for episode_call in range(1, queries + 1, 4):
                    sample = {"source_kind": kind, "sampling_mode": MODE, "task_name": task,
                              "student_checkpoint": checkpoint, "episode_index": episode,
                              "episode_seed": seed + episode, "init_state_index": 20 + episode,
                              "reset_identity": resets[episode], "episode_success": None,
                              "capture_index": saved, "episode_call": episode_call,
                              "server_call": prefix + episode_call,
                              "inputs": {"state": torch.tensor([[task_index, saved]], dtype=torch.bfloat16),
                                         "input_ids": torch.tensor([[task_index, saved]], dtype=torch.int64)}}
                    if formal_teacher:
                        sample.update(checkpoint_role=capture["checkpoint_role"],
                                      student_statistics_sha256=capture["student_statistics_sha256"])
                        sample["inputs"].update({
                            "embodiment_id": torch.zeros(1, dtype=torch.int64),
                            "attention_mask": torch.ones(1, 2, dtype=torch.int64),
                            "pixel_values": torch.zeros(4, 3, dtype=torch.bfloat16),
                            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
                            "action": torch.zeros(1, 16, 32, dtype=torch.bfloat16),
                            "action_mask": torch.cat((torch.ones(1, 16, 7), torch.zeros(1, 16, 25)), dim=-1),
                        })
                    torch.save(sample, capture_dir / f"candidate_{saved:06d}.pt")
                    saved += 1
                prefix += queries
            write_json(capture_dir / "capture_counts.json", {
                "calls": 43, "saved": saved, "candidates_saved": saved,
                "episode_query_counts": {"0": 33, "1": 10}, "safety_limit_hit": False})
            if finalized:
                result["capture"] = finalize_capture_samples(capture_dir, task, result, purpose, sampling_mode=MODE)
            task_results[task] = result
        write_json(directory / "task_results.json", task_results)
        write_json(directory / "summary.json", {"tasks_complete": 10, "total_episodes": 20,
                                                "total_successes": 10 * sum(success), "purpose": purpose})
        return directory

    def test_runner_route_and_matched_teacher_views_preserve_source_and_tensors(self):
        source = self.fixture()
        before = tree_hashes(source)
        for task in TASKS:
            folder = source / "observations" / task
            self.assertFalse(list(folder.glob("sample_*.pt")))
            self.assertEqual(json.loads((folder / "capture_manifest.json").read_text())["schema"],
                             "fp4vla_capture_candidates_v1")
        output = self.root / "paired_views"
        receipt = derive_views(source, output)
        self.assertEqual(receipt["status"], "complete")
        self.assertFalse(receipt["legacy_audit_compatible"])
        self.assertEqual(receipt["views"]["head"], receipt["views"]["stratified"])
        self.assertEqual(receipt["views"]["head"], {"tasks": 10, "accepted_samples": 40, "rejected_samples": 30})
        self.assertEqual(before, tree_hashes(source))
        for mode in ("head", "stratified"):
            self.assertFalse((output / mode / "eval_manifest.json").exists())
            self.assertFalse((output / mode / "summary.json").exists())
            for task in TASKS:
                view = output / mode / "observations" / task
                manifest = json.loads((view / "capture_manifest.json").read_text())
                self.assertEqual(len(list(view.glob("sample_*.pt"))), 4)
                self.assertEqual(len(list(view.glob("rejected_*.pt"))), 3)
                self.assertIn("not a new evaluation run", manifest["data_view"])
                for selected in manifest["selected_sources"]:
                    old = torch.load(source / "observations" / task / selected["source_filename"], weights_only=True)
                    new = torch.load(view / selected["output_filename"], weights_only=True)
                    self.assertEqual(old["reset_identity"], new["reset_identity"])
                    self.assertEqual(old["source_kind"], new["source_kind"])
                    self.assertEqual(old["episode_success"], new["episode_success"])
                    for key, tensor in old["inputs"].items():
                        self.assertEqual(tensor.dtype, new["inputs"][key].dtype)
                        self.assertTrue(torch.equal(tensor, new["inputs"][key]))
        plan = json.loads((output / "paired_selection_plan.json").read_text())
        self.assertNotEqual([x["source_capture_index"] for x in plan["plans"]["head"][0]["selected"]],
                            [x["source_capture_index"] for x in plan["plans"]["stratified"][0]["selected"]])

    def test_collection_failed_states_are_kept_in_both_views(self):
        source = self.fixture(purpose="collection")
        before = tree_hashes(source)
        receipt = derive_views(source, self.root / "collection_views")
        self.assertEqual(receipt["views"]["head"], {"tasks": 10, "accepted_samples": 70, "rejected_samples": 0})
        self.assertEqual(receipt["views"]["head"], receipt["views"]["stratified"])
        self.assertEqual(before, tree_hashes(source))

    def test_raw_log_tampering_rejected_before_output_creation(self):
        source = self.fixture()
        log = source / f"{TASKS[3]}.log"
        log.write_text(log.read_text() + "tampered raw bytes\n")
        output = self.root / "rejected_log"
        with self.assertRaisesRegex(ValueError, "Raw rollout disagrees"):
            derive_views(source, output)
        self.assertFalse(output.exists())

    def test_protocol_bytes_or_declared_sha_tampering_is_rejected(self):
        for change in ("protocol_bytes", "manifest_sha", "capture_sha"):
            with self.subTest(change=change):
                source = self.fixture()
                manifest_path = source / "eval_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if change == "protocol_bytes":
                    with Path(manifest["protocol_file"]).open("a") as stream:
                        stream.write("\n")
                elif change == "manifest_sha":
                    write_json(manifest_path, {**manifest, "protocol_sha256": "e" * 64})
                else:
                    path = source / "observations" / TASKS[0] / "capture_manifest.json"
                    write_json(path, {**json.loads(path.read_text()), "protocol_sha256": "e" * 64})
                output = self.root / f"rejected_{change}"
                with self.assertRaises(ValueError):
                    derive_views(source, output)
                self.assertFalse(output.exists())

    def test_runner_refuses_mode_mismatch_without_mutating_raw_candidates(self):
        source = self.fixture(finalized=False)
        folder = source / "observations" / TASKS[0]
        result = json.loads((source / "task_results.json").read_text())[TASKS[0]]
        before = tree_hashes(source)
        with self.assertRaisesRegex(ValueError, "mode"):
            finalize_capture_samples(folder, TASKS[0], result, "teacher_supervision", sampling_mode="prefix")
        self.assertEqual(before, tree_hashes(source))
        output = self.root / "unfinalized_view"
        with self.assertRaises(ValueError):
            derive_views(source, output)
        self.assertFalse(output.exists())

    def test_derivation_refuses_mode_mismatch(self):
        source = self.fixture()
        path = source / "eval_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["capture_sampling"]["mode"] = "prefix"
        write_json(path, manifest)
        output = self.root / "rejected_mode"
        with self.assertRaisesRegex(ValueError, "configuration"):
            derive_views(source, output)
        self.assertFalse(output.exists())

    def test_partial_tasks_and_partial_episode_logs_are_rejected(self):
        for change in ("missing_task", "task_failed", "partial_log", "partial_summary"):
            with self.subTest(change=change):
                source = self.fixture()
                results_path = source / "task_results.json"
                results = json.loads(results_path.read_text())
                if change == "missing_task":
                    results.pop(TASKS[-1])
                    write_json(results_path, results)
                elif change == "task_failed":
                    results[TASKS[-1]]["returncode"] = 1
                    write_json(results_path, results)
                elif change == "partial_log":
                    path = source / f"{TASKS[-1]}.log"
                    path.write_text(path.read_text().replace("[True, False]", "[True]"))
                    # Even a rewritten result receipt must not certify fewer episodes.
                    results[TASKS[-1]].update(parse_log(path))
                    write_json(results_path, results)
                else:
                    path = source / "summary.json"
                    write_json(path, {**json.loads(path.read_text()), "tasks_complete": 9})
                output = self.root / f"rejected_{change}"
                with self.assertRaises(ValueError):
                    derive_views(source, output)
                self.assertFalse(output.exists())

    def test_existing_output_and_source_child_are_rejected(self):
        source = self.fixture()
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            derive_views(source, existing)
        with self.assertRaises(ValueError):
            derive_views(source, source / "nested")


if __name__ == "__main__":
    unittest.main()
