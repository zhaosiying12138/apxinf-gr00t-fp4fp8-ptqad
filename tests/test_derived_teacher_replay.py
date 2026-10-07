"""Real CPU integration of paired-view provenance with the training audit entry."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from exp.derive_capture_views import audit_training_view, derive_views
from exp.verify_teacher_replay import audit_replay
from rl.derived_replay_guard import audit_report_sha256, prepare_derived_replay
from tests import test_derive_capture_views as fixtures


class DerivedTeacherReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture_count = 0

    def source(self, purpose="teacher_supervision", success=(True, False)):
        return fixtures.DeriveCaptureViewsTests.fixture(
            self, purpose=purpose, formal_teacher=True, success=success)

    def derive(self, source, windows=4):
        output = self.root / (source.name + "_paired")
        derive_views(source, output, windows_per_episode=windows)
        manifest = json.loads((source / "eval_manifest.json").read_text())
        return output, Path(manifest["protocol_file"]), Path(manifest["checkpoint"])

    def rebind_protocol(self, source, change):
        """Freeze a different fixture protocol before deriving any views."""
        path = source / "eval_manifest.json"
        manifest = json.loads(path.read_text())
        protocol = Path(manifest["protocol_file"])
        payload = json.loads(protocol.read_text())
        change(payload)
        fixtures.write_json(protocol, payload)
        manifest["protocol_sha256"] = fixtures.digest(protocol)
        fixtures.write_json(path, manifest)
        for task in fixtures.TASKS:
            path = source / "observations" / task / "capture_manifest.json"
            capture = json.loads(path.read_text())
            capture["protocol_sha256"] = manifest["protocol_sha256"]
            fixtures.write_json(path, capture)

    def test_real_dispatch_reports_are_deterministic_and_source_is_unchanged(self):
        source = self.source()
        paired, protocol, teacher = self.derive(source)
        before = fixtures.tree_hashes(self.root)
        for mode in ("head", "stratified"):
            with self.subTest(mode=mode):
                view = paired / mode
                direct = audit_training_view(view, protocol, teacher, minimum_episodes=1)
                through_entry = audit_replay(view, protocol, teacher, minimum_episodes=1)
                observations = audit_replay(view / "observations", protocol, teacher, minimum_episodes=1)
                self.assertEqual(direct, through_entry)
                self.assertEqual(direct, observations)
                self.assertEqual(json.loads(json.dumps(direct, sort_keys=True)), direct)
                self.assertEqual(direct["status"], "verified")
                self.assertEqual(direct["task_count"], 10)
                self.assertEqual(direct["sample_count"], 40)
                self.assertEqual(direct["selection_mode"], mode)
                self.assertEqual(direct["teacher_weights"]["model.safetensors"]["sha256"],
                                 fixtures.digest(teacher / "model.safetensors"))
        self.assertEqual(before, fixtures.tree_hashes(self.root))

    def test_minimum_successful_episode_gate_is_enforced(self):
        paired, protocol, teacher = self.derive(self.source())
        with self.assertRaisesRegex(ValueError, "Too few successful episodes"):
            audit_replay(paired / "head", protocol, teacher, minimum_episodes=2)

    def test_collection_view_cannot_be_used_as_teacher_qad_supervision(self):
        paired, protocol, teacher = self.derive(self.source(purpose="collection"))
        with self.assertRaisesRegex(ValueError, "successful training rollouts"):
            audit_replay(paired / "head", protocol, teacher, minimum_episodes=1)

    def test_tensor_tampering_fails_even_after_rewriting_the_output_hash(self):
        paired, protocol, teacher = self.derive(self.source())
        task = paired / "head/observations" / fixtures.TASKS[0]
        path = task / "sample_000000.pt"
        sample = torch.load(path, map_location="cpu", weights_only=True)
        sample["inputs"]["state"] = sample["inputs"]["state"] + 1
        torch.save(sample, path)
        manifest_path = task / "capture_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        row = next(row for row in manifest["selected_sources"] if row["output_filename"] == path.name)
        row.update(output_sha256=fixtures.digest(path), output_bytes=path.stat().st_size)
        fixtures.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "tensor.*differ|Tensor.*differ"):
            audit_replay(paired / "head", protocol, teacher, minimum_episodes=1)

    def test_training_guard_reaudits_real_views_after_driver_json_transport(self):
        source = self.source(success=(True, True))
        paired, protocol, teacher = self.derive(source)
        view = paired / "stratified"
        driver_report = audit_replay(view, protocol, teacher)
        # The launcher receives subprocess stdout, so integer dictionary keys
        # would change here and invalidate a naive in-process report signature.
        transferred_report = json.loads(json.dumps(driver_report, sort_keys=True))
        self.assertEqual(driver_report, transferred_report)
        expected_sha = audit_report_sha256(transferred_report)
        for root in (view, view / "observations"):
            with self.subTest(root=str(root)):
                guarded = prepare_derived_replay(root, protocol, {
                    "QAD_CAPTURE_TEACHER": str(teacher),
                    "QAD_CAPTURE_AUDIT_SHA256": expected_sha})
                self.assertEqual(guarded.report, transferred_report)
                self.assertEqual(audit_report_sha256(guarded.report), expected_sha)
                self.assertEqual(len(guarded.paths), 70)
                sample = guarded.load_sample(guarded.paths[0])
                original_path = (source / "observations" / sample["task_name"]
                                 / sample["source_candidate_filename"])
                original = torch.load(original_path, map_location="cpu", weights_only=True)
                for key, tensor in original["inputs"].items():
                    self.assertEqual(sample["inputs"][key].dtype, tensor.dtype)
                    self.assertTrue(torch.equal(sample["inputs"][key], tensor))

    def test_specified_teacher_and_protocol_must_match_original_capture(self):
        paired, protocol, teacher = self.derive(self.source())
        other_teacher = self.root / "other_teacher"
        shutil.copytree(teacher, other_teacher)
        with self.assertRaisesRegex(ValueError, "specified teacher"):
            audit_replay(paired / "head", protocol, other_teacher, minimum_episodes=1)
        other_protocol = self.root / "other_protocol.json"
        changed = json.loads(protocol.read_text())
        changed["unrelated_metadata"] = "different frozen bytes"
        fixtures.write_json(other_protocol, changed)
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            audit_replay(paired / "head", other_protocol, teacher, minimum_episodes=1)

    def test_rewriting_only_paired_protocol_sha_cannot_relabel_original_rollout(self):
        paired, protocol, teacher = self.derive(self.source())
        other_protocol = self.root / "relabelled_protocol.json"
        changed = json.loads(protocol.read_text())
        changed["another_frozen_decision"] = "must not relabel prior capture"
        fixtures.write_json(other_protocol, changed)
        receipt_path = paired / "views_manifest.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["protocol_sha256"] = fixtures.digest(other_protocol)
        fixtures.write_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "Original rollout protocol differs"):
            audit_replay(paired / "head", other_protocol, teacher, minimum_episodes=1)

    def test_paired_root_and_individual_task_are_not_training_dataset_roots(self):
        paired, protocol, teacher = self.derive(self.source())
        with self.assertRaisesRegex(ValueError, "paired view root"):
            audit_replay(paired, protocol, teacher, minimum_episodes=1)
        with self.assertRaisesRegex(ValueError, "Training root must select"):
            audit_training_view(paired / "head/observations" / fixtures.TASKS[0],
                                protocol, teacher, minimum_episodes=1)

    def test_mode_and_window_budget_must_be_frozen_and_match_derivation(self):
        changes = {
            "missing": lambda p: p.pop("capture_views"),
            "mode": lambda p: p["capture_views"].update(modes=["head"]),
            "window_missing": lambda p: p["capture_views"].pop("windows_per_episode"),
            "window_mismatch": lambda p: p["capture_views"].update(windows_per_episode=3),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                source = self.source()
                self.rebind_protocol(source, change)
                paired, protocol, teacher = self.derive(source)
                with self.assertRaisesRegex(ValueError, "freeze capture_views|selection plan"):
                    audit_replay(paired / "stratified", protocol, teacher, minimum_episodes=1)

    def test_non_bf16_execution_or_quantized_teacher_is_rejected(self):
        source = self.source()
        manifest_path = source / "eval_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["environment_summary"]["variables"]["FP4VLA_W4A4"] = "1"
        fixtures.write_json(manifest_path, manifest)
        paired, protocol, teacher = self.derive(source)
        with self.assertRaisesRegex(ValueError, "explicit BF16 execution provenance"):
            audit_replay(paired / "head", protocol, teacher, minimum_episodes=1)
        paired, protocol, teacher = self.derive(self.source())
        fixtures.write_json(teacher / "ptq_recipe.json", {"quantized": True})
        with self.assertRaisesRegex(ValueError, "Capture checkpoint bytes differ"):
            audit_replay(paired / "head", protocol, teacher, minimum_episodes=1)

    def test_hidden_extra_training_sample_is_rejected(self):
        paired, protocol, teacher = self.derive(self.source())
        extra = paired / ".hidden/sample_000000.pt"
        extra.parent.mkdir()
        shutil.copyfile(paired / "head/observations" / fixtures.TASKS[0] / "sample_000000.pt", extra)
        with self.assertRaisesRegex(ValueError, "Unexpected training samples"):
            audit_replay(paired / "head", protocol, teacher, minimum_episodes=1)


if __name__ == "__main__":
    unittest.main()
