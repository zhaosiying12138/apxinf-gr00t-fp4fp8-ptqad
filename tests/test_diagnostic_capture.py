"""CPU-only diagnostic collection integration and exclusion from training."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

from eval import run_recovery_eval as evaluation
from eval import serve_recovery as server
from exp import independent_action_protocol as protocol
from rl.capture_sampling import _json_sha, _verify_finalized, plan_capture_view
from tests import test_capture_full_trajectory as runtime_tests
from tests import test_independent_action_protocol as protocol_tests
from tests.test_derived_replay_training_guard import dataset_types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from capture_onpolicy import install_capture
from rl.opd_probe_cache import prepare_observation_source


class DiagnosticCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.final = protocol_tests.fixture(self.root)
        self.protocol = self.root / "diagnostic_protocol.json"
        protocol.prepare(protocol.DRAFT, self.final, self.protocol)
        self.teacher = self.root / "models/bf16"
        self.runtime = runtime_tests.FullTrajectoryCaptureTests()

    def audited_manifest(self):
        reference = evaluation.validate_diagnostic_capture(self.protocol, self.teacher)
        config = evaluation.capture_sampling_config(
            evaluation.protocol_entry(self.protocol, "diagnostics"), "diagnostics", 1)
        return {"protocol_file": str(self.protocol), "protocol_sha256": protocol.file_sha256(self.protocol),
                "diagnostic_reference": reference, "capture_sampling": config,
                "checkpoint": str(self.teacher), "max_episode_steps": 720}

    def collect(self, success=False):
        manifest = self.audited_manifest()
        env = {}
        output = self.root / "capture"
        task = evaluation.TASKS[0]
        directory = evaluation.configure_capture_environment(
            env, output, task, "diagnostics", manifest, 990000, [29])
        metadata = {"task_name": task, "purpose": "diagnostics", "source_kind": "diagnostic_rollout",
            "checkpoint_role": "diagnostic_reference", "training_eligible": False,
            "protocol_file": str(self.protocol), "protocol_sha256": manifest["protocol_sha256"],
            "init_state_indices": "29", "diagnostic_reference": manifest["diagnostic_reference"]}
        # The fixture GR00T model only creates small CPU tensors. The actual
        # capture hooks, count accounting, finalization and reader stay real.
        with self.runtime.fake_runtime() as (model, collator), \
                patch("capture_onpolicy.libero_action_spec", return_value={}), \
                patch("capture_onpolicy.action_mask", side_effect=lambda action, spec: torch.ones_like(action)):
            install_capture(directory, self.teacher, every=4, per_task=192, limit=192, per_episode=192,
                event_file=env["FP4VLA_CAPTURE_EVENT_FILE"], capture_metadata=metadata,
                sampling_mode="full_trajectory_candidates")
            reset = {"event": "reset", "task_name": task, "episode_index": 0, "seed": 990000,
                "init_state_index": 29, "initial_state_sha256": "a" * 64,
                "restored_state_sha256": "b" * 64, "init_state_bank_sha256": "c" * 64}
            Path(env["FP4VLA_CAPTURE_EVENT_FILE"]).write_text(json.dumps(reset) + "\n")
            for _ in range(18):
                self.runtime.query(model, collator)
            self.assertEqual((model.calls, collator.calls), (18, 18))
        summary = evaluation.finalize_capture_samples(directory, task,
            {"results": [success], "resets": [reset]}, "diagnostics", "full_trajectory_candidates")
        return directory, task, summary

    def test_explicit_protocol_and_complete_final_are_required_before_process_or_output(self):
        output = self.root / "must_not_exist"
        argv = ["run_recovery_eval", "--checkpoint", str(self.teacher), "--out", str(output),
                "--seed", "990000", "--purpose", "diagnostics"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), \
                patch.object(evaluation.subprocess, "Popen") as launch, \
                patch.object(evaluation.socket, "socket") as socket:
            with self.assertRaises(SystemExit):
                evaluation.main()
            launch.assert_not_called()
            socket.assert_not_called()
        protocol_tests.mutate(self.root / "run_manifest.json", lambda data: data.update(status="running"))
        with patch.object(sys, "argv", argv + ["--protocol-file", str(self.protocol)]), \
                patch.object(evaluation.subprocess, "Popen") as launch, \
                patch.object(evaluation.socket, "socket") as socket:
            with self.assertRaises(ValueError):
                evaluation.main()
            launch.assert_not_called()
            socket.assert_not_called()
        self.assertFalse(output.exists())

    def test_only_audited_bf16_reference_is_allowed(self):
        report = evaluation.validate_diagnostic_capture(self.protocol, self.teacher)
        self.assertEqual(report["allowed_teacher"]["checkpoint"], str(self.teacher))
        for arm in ("ptq", "qad", "continued_qad", "qad_opd"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                evaluation.validate_diagnostic_capture(self.protocol, self.root / "models" / arm)

    def test_explicit_full_sampling_and_bank29_environment_required(self):
        config = protocol.PARTITIONS["diagnostics"]["capture_sampling"]
        for partition in ({}, {"capture_sampling": {**config, "mode": "prefix"}},
                          {"capture_sampling": {**config, "every_server_calls": 8}}):
            with self.subTest(partition=partition), self.assertRaises(ValueError):
                evaluation.capture_sampling_config(partition, "diagnostics", 1)
        manifest = self.audited_manifest()
        env = {"FP4VLA_DIAGNOSTIC_REFERENCE_JSON": "stale"}
        folder = evaluation.configure_capture_environment(env, self.root / "out", "task", "diagnostics",
                                                         manifest, 990000, [29])
        self.assertEqual(env["OPD_CAPTURE_PER_EPISODE"], "192")
        self.assertEqual(env["OPD_CAPTURE_EVERY"], "4")
        self.assertEqual(json.loads(env["FP4VLA_DIAGNOSTIC_REFERENCE_JSON"]), manifest["diagnostic_reference"])
        self.assertEqual(folder.name, "task")
        for changed, indices in (({**manifest, "diagnostic_reference": None}, [29]), (manifest, [20])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                evaluation.configure_capture_environment({}, self.root / "invalid", "task", "diagnostics",
                                                         changed, 990000, indices)
        self.assertFalse((self.root / "invalid").exists())
        evaluation.configure_capture_environment(env, self.root / "out", "other", "heldout", manifest, 1, [29])
        self.assertNotIn("FP4VLA_DIAGNOSTIC_REFERENCE_JSON", env)
        self.assertNotIn("FP4VLA_CAPTURE_PROTOCOL_FILE", env)

    def test_failed_episode_keeps_all_query_candidates_with_training_exclusion(self):
        directory, task, summary = self.collect(success=False)
        self.assertEqual((summary["total_candidates"], summary["failed_candidates"]), (5, 5))
        _, manifest, counts, loaded = _verify_finalized(directory, task)
        self.assertEqual(counts["episode_query_counts"], {"0": 18})
        self.assertEqual([sample["episode_call"] for _, sample in loaded], [1, 5, 9, 13, 17])
        for _, sample in loaded:
            self.assertEqual(sample["source_kind"], "diagnostic_rollout")
            self.assertEqual(sample["checkpoint_role"], "diagnostic_reference")
            self.assertIs(sample["episode_success"], False)
            self.assertIs(sample["training_eligible"], False)
            self.assertEqual(sample["diagnostic_reference_sha256"], _json_sha(manifest["diagnostic_reference"]))
        self.assertEqual(list(directory.glob("sample_*.pt")), [])
        self.assertEqual(list(directory.glob("rejected_*.pt")), [])
        with self.assertRaisesRegex(ValueError, "cannot be materialized as training"):
            plan_capture_view(directory, task, mode="stratified")
        self.assertFalse(torch.cuda.is_initialized())

    def test_successful_diagnostic_is_still_forbidden_from_training_and_opd(self):
        directory, task, summary = self.collect(success=True)
        self.assertEqual(summary["successful_candidates"], 5)
        with self.assertRaisesRegex(ValueError, "cannot be materialized as training"):
            plan_capture_view(directory, task, mode="head")
        from exp.derive_capture_views import plan_views
        # The public view builder rejects the purpose before any view output.
        evaluation_manifest = directory.parents[1] / "eval_manifest.json"
        evaluation_manifest.write_text(json.dumps({"purpose": "diagnostics"}))
        (directory.parents[1] / "summary.json").write_text("{}")
        (directory.parents[1] / "task_results.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "training capture"):
            plan_views(directory.parents[1])
        renamed = self.root / "accidental_training_copy"
        renamed.mkdir()
        shutil.copyfile(directory / "candidate_000000.pt", renamed / "sample_000000.pt")
        legacy = self.root / "legacy_protocol.json"
        legacy.write_text("{}")
        dataset, _ = dataset_types(legacy)
        with self.assertRaisesRegex(ValueError, "cannot enter QAD training"):
            dataset(renamed)
        with self.assertRaisesRegex(ValueError, "not a captured student rollout"):
            prepare_observation_source(self.teacher, input_dir=renamed)

    def test_each_diagnostic_marker_blocks_even_a_renamed_legacy_qad_sample(self):
        root = self.root / "legacy_samples"
        root.mkdir()
        legacy = self.root / "legacy_protocol.json"
        legacy.write_text("{}")
        dataset, _ = dataset_types(legacy)
        for marker in ({"source_kind": "diagnostic_rollout"}, {"checkpoint_role": "diagnostic_reference"},
                       {"training_eligible": False}):
            torch.save({**marker, "inputs": {}}, root / "sample_000000.pt")
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "cannot enter QAD training"):
                dataset(root)

    def test_candidate_role_or_training_flag_tampering_rejected(self):
        directory, task, _ = self.collect()
        path = directory / "candidate_000000.pt"
        original = path.read_bytes()
        for change in ({"training_eligible": True}, {"checkpoint_role": "teacher"},
                       {"source_kind": "student_rollout"}, {"diagnostic_reference_sha256": "0" * 64}):
            sample = torch.load(io.BytesIO(original), weights_only=True)
            sample.update(change)
            torch.save(sample, path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                _verify_finalized(directory, task)
        path.write_bytes(original)

    def test_direct_install_rejects_missing_reference_before_gr00t_import(self):
        with self.assertRaisesRegex(ValueError, "audited BF16"):
            install_capture(self.root / "invalid", self.teacher,
                capture_metadata={"purpose": "diagnostics", "source_kind": "diagnostic_rollout"})
        self.assertFalse((self.root / "invalid").exists())

    def test_server_reaudits_before_seeding_and_propagates_diagnostic_role(self):
        manifest = self.audited_manifest()
        env = {"GR00T_EVAL_SEED": "10990000", "OPD_CAPTURE_DIR": str(self.root / "out"),
            "FP4VLA_CAPTURE_PURPOSE": "diagnostics", "FP4VLA_CAPTURE_TASK_NAME": evaluation.TASKS[0],
            "FP4VLA_CAPTURE_PROTOCOL_FILE": str(self.protocol),
            "FP4VLA_CAPTURE_PROTOCOL_SHA256": manifest["protocol_sha256"],
            "FP4VLA_CAPTURE_INIT_STATE_INDICES": "29", "FP4VLA_CAPTURE_SAMPLING_MODE": "full_trajectory_candidates",
            "FP4VLA_DIAGNOSTIC_REFERENCE_JSON": json.dumps(manifest["diagnostic_reference"])}
        fake = ModuleType("gr00t.utils.determinism")
        events = []
        fake.seed_everything = Mock(side_effect=lambda seed: events.append("seed"))
        audit = server.validate_diagnostic_capture
        def checked(*args):
            events.append("audit")
            return audit(*args)
        with patch.dict(sys.modules, {fake.__name__: fake}), patch.dict("os.environ", env, clear=True), \
                patch.object(sys, "argv", ["serve_recovery", "--model-path", str(self.teacher)]), \
                patch.object(server, "validate_diagnostic_capture", side_effect=checked), \
                patch("capture_onpolicy.install_capture") as install, patch.object(server.runpy, "run_path"):
            server.main()
            self.assertEqual(events, ["audit", "seed"])
            metadata = install.call_args.kwargs["capture_metadata"]
            self.assertEqual(metadata["source_kind"], "diagnostic_rollout")
            self.assertEqual(metadata["checkpoint_role"], "diagnostic_reference")
            self.assertIs(metadata["training_eligible"], False)
            self.assertEqual(metadata["diagnostic_reference"], manifest["diagnostic_reference"])
            env["FP4VLA_DIAGNOSTIC_REFERENCE_JSON"] = "{}"
            with patch.dict("os.environ", env, clear=True), self.assertRaisesRegex(ValueError, "identity changed"):
                server.main()
            self.assertEqual(fake.seed_everything.call_count, 1)


if __name__ == "__main__":
    unittest.main()
