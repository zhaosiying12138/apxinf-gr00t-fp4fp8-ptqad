"""Real CPU chain: completed final -> protocol -> capture hooks -> frozen inputs.

Only the GR00T model/collator and their small action shape are substituted. Final
audits, raw log parsing, reset checks, candidate capture/finalization and input
selection use production implementations.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from eval import run_recovery_eval as evaluation
from exp import independent_action_protocol as protocol
from exp.independent_action_inputs import freeze_diagnostic_inputs
from rl.capture_sampling import plan_capture_view, select_indices
from rl.checkpoint_identity import checkpoint_files
from tests import test_capture_full_trajectory as runtime_tests
from tests import test_independent_action_protocol as protocol_tests
from capture_onpolicy import install_capture


def capture_fixture(root):
    final = protocol_tests.fixture(root)
    protocol_file = root / "diagnostic_protocol.json"
    protocol.prepare(protocol.DRAFT, final, protocol_file)
    teacher = root / "models/bf16"
    reference = evaluation.validate_diagnostic_capture(protocol_file, teacher)
    data = reference["data"]
    part = data["partitions"]["diagnostics"]
    sampling = evaluation.capture_sampling_config(part, "diagnostics", 1, 720)
    source = root / "diagnostic_rollouts"
    source.mkdir()
    manifest = {**data["evaluation_contract"], "purpose": "diagnostics", "tasks": evaluation.TASKS,
                "checkpoint": str(teacher), "checkpoint_files": checkpoint_files(teacher),
                "protocol_file": str(protocol_file), "protocol_sha256": protocol.file_sha256(protocol_file),
                "diagnostic_reference": reference, "capture_sampling": sampling,
                "episodes": 1, "seed": part["seed"], "init_state_indices": [29],
                "training_eligible": False}
    protocol_tests.write(source / "eval_manifest.json", manifest)
    runtime = runtime_tests.FullTrajectoryCaptureTests()
    rows = {}
    # One short episode (three windows) plus nine long episodes (nine
    # candidates each). Alternating failures prove no outcome-based filtering.
    query_counts = [9] + [33] * 9
    for index, task in enumerate(evaluation.TASKS):
        seed = part["seed"] + index * 1000
        env = {}
        folder = evaluation.configure_capture_environment(env, source, task, "diagnostics", manifest, seed, [29])
        reset = {"event": "reset", "task_name": task, "episode_index": 0, "seed": seed,
                 "init_state_index": 29, "settle_steps": 10,
                 "initial_state_sha256": f"{index + 1:064x}",
                 "restored_state_sha256": f"{index + 100:064x}",
                 "init_state_bank_sha256": f"{index + 1000:064x}"}
        Path(env["FP4VLA_CAPTURE_EVENT_FILE"]).write_text(json.dumps(reset) + "\n")
        metadata = {"task_name": task, "purpose": "diagnostics", "source_kind": "diagnostic_rollout",
                    "checkpoint_role": "diagnostic_reference", "training_eligible": False,
                    "seed": seed, "init_state_indices": "29", "protocol_file": str(protocol_file),
                    "protocol_sha256": manifest["protocol_sha256"], "diagnostic_reference": reference}
        with runtime.fake_runtime() as (model, collator), \
                patch("capture_onpolicy.libero_action_spec", return_value={}), \
                patch("capture_onpolicy.action_mask", side_effect=lambda action, spec: torch.ones_like(action)), \
                contextlib.redirect_stdout(io.StringIO()):
            install_capture(folder, teacher, every=4, per_task=192, limit=192, per_episode=192,
                            event_file=env["FP4VLA_CAPTURE_EVENT_FILE"], capture_metadata=metadata,
                            sampling_mode="full_trajectory_candidates")
            for _ in range(query_counts[index]):
                runtime.query(model, collator)
            assert model.calls == collator.calls == query_counts[index]
        log = source / (task + ".log")
        log.write_text("FP4VLA_EPISODE_RESET " + json.dumps(reset) + "\nresults: ('fixture', " +
                       repr([bool(index % 2)]) + ")\n")
        (source / (task + ".server.log")).write_text("CPU capture fixture; no production model\n")
        raw = evaluation.parse_log(log)
        captured = evaluation.finalize_capture_samples(folder, task, raw, "diagnostics", "full_trajectory_candidates")
        rows[task] = {**raw, "seed": seed, "returncode": 0, "capture": captured}
    protocol_tests.write(source / "task_results.json", rows)
    protocol_tests.write(source / "summary.json", {"tasks_complete": 10, "total_episodes": 10,
        "total_successes": 5, "macro_success_rate": .5, "purpose": "diagnostics", "seed": part["seed"],
        "checkpoint_files_verified_unchanged": True, "diagnostic_reference_verified_unchanged": True,
        "training_eligible": False})
    return source / "observations", protocol_file, final


class IndependentActionInputsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.capture, self.protocol, self.final = capture_fixture(self.root)
        self.task = evaluation.TASKS[0]
        self.folder = self.capture / self.task

    def freeze(self):
        return freeze_diagnostic_inputs(self.capture, self.protocol)

    def test_complete_real_chain_preserves_failures_short_episodes_and_two_noise_seeds(self):
        before = {str(path): protocol.file_sha256(path) for path in self.capture.parent.rglob("*") if path.is_file()}
        result = self.freeze()
        self.assertEqual(result["format"], "gr00t_action_inputs_v2")
        self.assertEqual(result["purpose"], "diagnostics")
        self.assertEqual(result["unique_observations"], 39)
        self.assertEqual(result["sample_count"], 78)
        self.assertEqual(len({sample["path"] for sample in result["samples"]}), 39)
        self.assertEqual(sum(not row["episode_success"] for row in result["samples"]), 38)
        self.assertEqual(result["temporal_coverage"][self.task][0]["selected_ranks"], [0, 1, 2])
        self.assertEqual(result["temporal_coverage"][evaluation.TASKS[1]][0]["selected_ranks"],
                         select_indices(9, 4, "stratified"))
        for index in range(39):
            rows = result["samples"][index * 2:index * 2 + 2]
            self.assertEqual([row["noise_index"] for row in rows], [0, 1])
            self.assertEqual([row["seed"] for row in rows], protocol.SETTINGS["noise_seeds"])
            self.assertEqual(rows[0]["path"], rows[1]["path"])
            self.assertEqual(rows[0]["observation_index"], rows[1]["observation_index"])
        self.assertNotEqual(result["protocol_sha256"], result["original_model_protocol_sha256"])
        self.assertEqual(result["final_reference"], protocol.identity(self.final))
        self.assertEqual(result, self.freeze())
        self.assertEqual(before, {str(path): protocol.file_sha256(path)
                                 for path in self.capture.parent.rglob("*") if path.is_file()})
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_training_views_are_created_and_view_builder_refuses_diagnostics(self):
        self.freeze()
        self.assertEqual(list(self.capture.rglob("sample_*.pt")), [])
        self.assertEqual(list(self.capture.rglob("rejected_*.pt")), [])
        self.assertEqual(list(self.capture.rglob("selection_plan.json")), [])
        with self.assertRaisesRegex(ValueError, "cannot be materialized as training"):
            plan_capture_view(self.folder, self.task, mode="stratified")

    def test_protocol_identity_tamper_rejected(self):
        protocol_tests.mutate(self.capture.parent / "eval_manifest.json",
                             lambda data: data.update(protocol_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "protocol identity changed"):
            self.freeze()

    def test_changed_actual_teacher_weights_rejected(self):
        (self.root / "models/bf16/model.safetensors").write_bytes(b"different model")
        with self.assertRaisesRegex(ValueError, "weights changed"):
            self.freeze()

    def test_capture_model_substitution_rejected(self):
        protocol_tests.mutate(self.capture.parent / "eval_manifest.json",
                             lambda data: data.update(checkpoint=str(self.root / "models/qad")))
        with self.assertRaisesRegex(ValueError, "different final BF16"):
            self.freeze()

    def test_candidate_tensor_tamper_rejected(self):
        path = self.folder / "candidate_000000.pt"
        sample = torch.load(path, map_location="cpu", weights_only=True)
        sample["inputs"]["state"].add_(100)
        torch.save(sample, path)
        with self.assertRaisesRegex(ValueError, "Candidate changed after finalization"):
            self.freeze()

    def test_unselected_candidate_is_also_audited(self):
        folder = self.capture / evaluation.TASKS[1]
        path = folder / "candidate_000001.pt"  # ranks for nine candidates are 0, 2, 4, 7
        self.assertNotIn(1, select_indices(9, 4, "stratified"))
        sample = torch.load(path, map_location="cpu", weights_only=True)
        sample["inputs"]["state"].add_(100)
        torch.save(sample, path)
        with self.assertRaisesRegex(ValueError, "Candidate changed after finalization"):
            self.freeze()

    def test_missing_late_candidate_rejected(self):
        (self.capture / evaluation.TASKS[1] / "candidate_000008.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Candidate file count"):
            self.freeze()

    def test_failed_episode_cannot_be_relabelled_successful(self):
        path = self.folder / "candidate_000000.pt"
        sample = torch.load(path, map_location="cpu", weights_only=True)
        self.assertIs(sample["episode_success"], False)
        sample["episode_success"] = True
        torch.save(sample, path)
        with self.assertRaisesRegex(ValueError, "conflicting success metadata"):
            self.freeze()

    def test_raw_log_tamper_rejected_even_with_unmodified_results(self):
        path = self.capture.parent / (self.task + ".log")
        path.write_text(path.read_text() + "changed log\n")
        with self.assertRaisesRegex(ValueError, "changed diagnostic rollout"):
            self.freeze()

    def test_reset_bank_substitution_rejected_even_with_reparsed_raw_result(self):
        path = self.capture.parent / (self.task + ".log")
        path.write_text(path.read_text().replace('"init_state_index": 29', '"init_state_index": 28'))
        raw = evaluation.parse_log(path)
        protocol_tests.mutate(self.capture.parent / "task_results.json",
                             lambda data: data[self.task].update(raw))
        with self.assertRaisesRegex(ValueError, "reset differs"):
            self.freeze()

    def test_training_view_file_contamination_rejected(self):
        shutil.copyfile(self.folder / "candidate_000000.pt", self.folder / "sample_000000.pt")
        with self.assertRaisesRegex(ValueError, "training-view files"):
            self.freeze()

    def test_runtime_reference_receipt_substitution_rejected(self):
        protocol_tests.mutate(self.capture.parent / "eval_manifest.json",
            lambda data: data["diagnostic_reference"]["source_final"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "reference|Reference"):
            self.freeze()


if __name__ == "__main__":
    unittest.main()
