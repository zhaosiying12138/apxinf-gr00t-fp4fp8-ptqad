"""CPU tests for derived teacher views at the actual training-data boundary."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

from exp import run_high_fp4_v3 as driver_module
from rl.derived_replay_guard import audit_report_sha256, prepare_derived_replay


ROOT = Path(__file__).resolve().parents[1]


def dataset_types(protocol):
    """Execute the real dataset/collator definitions without importing Trainer."""
    source = ast.parse((ROOT / "rl/lora_qad.py").read_text())
    classes = [node for node in source.body if isinstance(node, ast.ClassDef)
               and node.name in ("CapturedStateDataset", "CapturedStateCollator")]
    namespace = {"torch": torch, "Path": Path, "PROTOCOL_FILE": protocol,
                 "prepare_derived_replay": prepare_derived_replay}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "lora_qad_dataset", "exec"), namespace)
    return namespace["CapturedStateDataset"], namespace["CapturedStateCollator"]


class DerivedReplayTrainingGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps({"capture_views": {
            "modes": ["head", "stratified"], "windows_per_episode": 4}}))
        self.paired = self.root / "paired"
        self.mode = self.paired / "head"
        self.observations = self.mode / "observations"
        self.task = "task_a"
        (self.observations / self.task).mkdir(parents=True)
        self.marker = self.paired / "views_manifest.json"
        self.marker.write_text(json.dumps({"schema": "fp4vla_paired_capture_views_v1"}))
        self.teacher = self.root / "teacher"
        self.teacher.mkdir()
        self.inputs = {"embodiment_id": torch.tensor([2]),
                       "state": torch.ones(1, 1, 132, dtype=torch.bfloat16),
                       "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
                       "attention_mask": torch.ones(1, 3, dtype=torch.int64),
                       "pixel_values": torch.arange(12, dtype=torch.bfloat16).reshape(4, 3),
                       "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
                       "action": torch.ones(1, 40, 132, dtype=torch.float32),
                       "action_mask": torch.zeros(1, 40, 132)}
        self.inputs["action_mask"][:, :16, :7] = 1
        self.paths = []
        sources = []
        for episode in range(2):
            path = self.observations / self.task / f"sample_{episode:06d}.pt"
            torch.save({"inputs": self.inputs, "task_text": "fixture",
                        "episode_index": episode, "source_kind": "teacher_rollout"}, path)
            self.paths.append(path)
            sources.append({"path": str(path.relative_to(self.observations)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.report = {"format": "derived_teacher_replay_audit_v1", "status": "verified",
                       "observations": str(self.observations), "task_count": 1, "sample_count": 2,
                       "tasks": {self.task: {"source_files": sources}}}
        self.audit_module = ModuleType("exp.derive_capture_views")
        self.audit_module.audit_training_view = Mock(side_effect=lambda *a, **k: copy.deepcopy(self.report))
        self.env = {"QAD_CAPTURE_TEACHER": str(self.teacher),
                    "QAD_CAPTURE_AUDIT_SHA256": audit_report_sha256(self.report)}
        self.module_patch = patch.dict(sys.modules, {self.audit_module.__name__: self.audit_module})
        self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop()
        self.tmp.cleanup()

    def guard(self, selected=None, env=None):
        return prepare_derived_replay(selected or self.mode, self.protocol,
                                      self.env if env is None else env)

    def test_both_supported_roots_reaudit_and_use_frozen_inventory(self):
        for selected in (self.mode, self.observations):
            with self.subTest(root=selected):
                self.audit_module.audit_training_view.reset_mock()
                guarded = self.guard(selected)
                self.audit_module.audit_training_view.assert_called_once_with(
                    selected, self.protocol, self.teacher, minimum_episodes=2)
                self.assertEqual(guarded.paths, self.paths)
                self.assertEqual(guarded.report, self.report)

    def test_report_signature_uses_exact_driver_serialization(self):
        report = {"z": {"中文": 2}, "a": 1}
        expected = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        self.assertEqual(audit_report_sha256(report), expected)
        self.assertEqual(audit_report_sha256(dict(reversed(list(report.items())))), expected)

    def test_direct_derived_training_requires_both_environment_bindings(self):
        for key in self.env:
            with self.subTest(key=key):
                env = dict(self.env)
                env.pop(key)
                with self.assertRaisesRegex(ValueError, "requires QAD_CAPTURE_TEACHER"):
                    self.guard(env=env)
        self.audit_module.audit_training_view.assert_not_called()

    def test_different_report_or_invalid_audit_is_rejected(self):
        self.report["source_changed"] = True
        with self.assertRaisesRegex(ValueError, "audit SHA differs"):
            self.guard()
        for key, value in (("format", "successful_teacher_replay_audit_v1"), ("status", "failed")):
            with self.subTest(key=key):
                modified = {**self.report, key: value}
                self.audit_module.audit_training_view.side_effect = lambda *a, **k: modified
                with self.assertRaisesRegex(ValueError, "expected format"):
                    self.guard(env={**self.env, "QAD_CAPTURE_AUDIT_SHA256": audit_report_sha256(modified)})

    def test_auditor_failure_cannot_fall_back_to_legacy(self):
        self.audit_module.audit_training_view.side_effect = ValueError("Collection is not teacher supervision")
        with self.assertRaisesRegex(ValueError, "Collection"):
            self.guard()

    def test_missing_marker_rejected_for_new_protocol(self):
        self.marker.unlink()
        with self.assertRaisesRegex(ValueError, "marker is missing"):
            self.guard()
        self.audit_module.audit_training_view.assert_not_called()

    def test_unknown_schema_and_paired_root_rejected(self):
        with self.assertRaisesRegex(ValueError, "not the paired view root"):
            self.guard(self.paired)
        self.marker.write_text('{"schema":"unknown"}')
        with self.assertRaisesRegex(ValueError, "Unknown paired"):
            self.guard()

    def test_unlisted_sample_or_different_observation_root_rejected(self):
        extra = self.observations / self.task / "sample_999999.pt"
        extra.write_bytes(b"unlisted")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.guard()
        extra.unlink()
        self.report["observations"] = str(self.root / "different")
        self.env["QAD_CAPTURE_AUDIT_SHA256"] = audit_report_sha256(self.report)
        with self.assertRaisesRegex(ValueError, "observations differ"):
            self.guard()

    def test_source_path_escape_or_duplicate_is_rejected(self):
        sources = self.report["tasks"][self.task]["source_files"]
        original = copy.deepcopy(sources)
        for changed in (("path", "../outside/sample_000000.pt"), ("path", str(self.paths[0])),
                        ("sha256", "not-a-sha")):
            with self.subTest(changed=changed):
                sources[:] = copy.deepcopy(original)
                sources[0][changed[0]] = changed[1]
                self.env["QAD_CAPTURE_AUDIT_SHA256"] = audit_report_sha256(self.report)
                with self.assertRaisesRegex(ValueError, "invalid source"):
                    self.guard()
        sources[:] = [original[0], original[0]]
        self.env["QAD_CAPTURE_AUDIT_SHA256"] = audit_report_sha256(self.report)
        with self.assertRaisesRegex(ValueError, "repeats a source"):
            self.guard()

    def test_changed_sample_rejected_before_deserialization(self):
        guarded = self.guard()
        self.paths[0].write_bytes(b"changed after audit")
        with patch.object(torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "changed after derived audit"):
                guarded.load_sample(self.paths[0])
            loader.assert_not_called()

    def test_deserialization_uses_verified_bytes_even_if_file_changes(self):
        guarded = self.guard()
        original_load = torch.load

        def replace_after_read(value, **kwargs):
            self.assertIsInstance(value, io.BytesIO)
            self.paths[0].write_bytes(b"changed after byte verification")
            return original_load(value, **kwargs)

        with patch.object(torch, "load", side_effect=replace_after_read):
            sample = guarded.load_sample(self.paths[0])
        torch.testing.assert_close(sample["inputs"]["state"], self.inputs["state"])

    def test_real_dataset_collator_preserves_every_tensor_dtype_and_shape(self):
        Dataset, Collator = dataset_types(self.protocol)
        with patch.dict(os.environ, self.env, clear=True):
            dataset = Dataset(self.mode)
        self.assertEqual(len(dataset), 2)
        restored = Collator()([dataset[0]])["inputs"]
        self.assertEqual(set(restored), set(self.inputs))
        for key in self.inputs:
            with self.subTest(key=key):
                torch.testing.assert_close(restored[key], self.inputs[key], rtol=0, atol=0)
                self.assertEqual(restored[key].dtype, self.inputs[key].dtype)
                self.assertEqual(restored[key].shape, self.inputs[key].shape)

    def test_legacy_dataset_retains_original_loading_behavior(self):
        self.marker.unlink()
        self.protocol.write_text('{"version":12}')
        Dataset, Collator = dataset_types(self.protocol)
        with patch.dict(os.environ, {}, clear=True):
            dataset = Dataset(self.observations)
        self.assertIsNone(dataset.derived_guard)
        torch.testing.assert_close(Collator()([dataset[0]])["inputs"]["action"], self.inputs["action"])
        self.audit_module.audit_training_view.assert_not_called()

    def test_driver_emits_derived_identity_without_changing_legacy_request(self):
        driver = driver_module.Driver.__new__(driver_module.Driver)
        for key, value in {"base": self.teacher, "dataset": self.root / "demo", "batch": 16,
                           "rank": 32, "alpha": 64, "scope": "all_ordinary_linear", "every": 4,
                           "seed": 42, "protocol_path": self.protocol, "w4a4": True,
                           "capture_dataset": self.mode, "capture_dataset_identity": {"fixture": "data"},
                           "protocol": {"sha256": "protocol"}, "py": Path("python"),
                           "groot": self.root, "teacher_capture_identity": self.report}.items():
            setattr(driver, key, value)
        driver.run_logged = Mock()
        for derived in (True, False):
            with self.subTest(derived=derived):
                driver.teacher_capture_identity = (self.report if derived else {
                    "format": "successful_teacher_replay_audit_v1"})
                work = self.root / ("derived_work" if derived else "legacy_work")
                with patch.object(driver_module, "model_id", return_value={"model": "fixture"}):
                    driver.train(work, "qad", self.root / "ptq", 1e-4, 100)
                env = driver.run_logged.call_args.args[-1]
                if derived:
                    self.assertEqual(env["QAD_CAPTURE_TEACHER"], str(self.teacher))
                    self.assertEqual(env["QAD_CAPTURE_AUDIT_SHA256"], audit_report_sha256(self.report))
                else:
                    self.assertNotIn("QAD_CAPTURE_TEACHER", env)
                    self.assertNotIn("QAD_CAPTURE_AUDIT_SHA256", env)
                request = json.loads((work / "orchestrator_training_request.json").read_text())
                self.assertEqual(request["environment"], env)


if __name__ == "__main__":
    unittest.main()
