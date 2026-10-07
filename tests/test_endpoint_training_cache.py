"""CPU checks at the real matched-cache training boundary."""
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

from rl.endpoint_cache_guard import (prepare_endpoint_training, prepare_endpoint_training_cache,
                                     sha, state_distillation_config)
from rl.probe_distill import ProbeAnchor


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EndpointTrainingCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.teacher, self.base, self.initial, self.data = (self.root / name for name in (
            "teacher", "base", "checkpoint-2000", "stratified"))
        for root in (self.teacher, self.base, self.initial, self.data):
            root.mkdir()
        for name in ("config.json", "statistics.json"):
            write(self.teacher / name, {"name": name})
        (self.teacher / "model.safetensors").write_bytes(b"identity fixture")
        write(self.initial / "recovery_manifest.json", {"base": str(self.base)})
        self.protocol = self.root / "protocol.json"
        self.protocol_data = {"endpoint_distillation": {}, "selection": {
            "qad_learning_rates": [5e-5], "continuation_optimizer_steps": 2000, "opd_every": 1,
            "train_seed": 123, "rank": 32, "alpha": 64, "recovery_scope": "all_ordinary_linear",
            "effective_demo_batch": 16}, "state_distillation": {
            "arms": ["continued_qad", "teacher_state_kd", "student_state_opd"],
            "starting_qad_view": "stratified", "teacher_velocity_weight": 0.25,
            "probe_every": 1, "optimizer_steps": 2000, "learning_rate": 5e-5, "max_grad_norm": 0.25}}
        write(self.protocol, self.protocol_data)
        self.reference = {"schema": "fp4vla_probe_endpoint_bundle_v1", "root": str(self.root / "bundle"),
            "role": "teacher", "source_kind": "teacher_rollout", "pairs_per_arm": 2,
            "protocol_sha256": sha(self.protocol), "endpoint_policy_training_checkpoint": str(self.initial),
            "endpoint_policy_training_identity": {"capture_dataset": str(self.data),
                "capture_dataset_sha256": "data-sha", "capture_audit_sha256": "audit-sha"}}
        self.raw = []
        for index in range(2):
            action = torch.arange(40 * 132, dtype=torch.float32).reshape(1, 40, 132) + index
            mask = torch.zeros_like(action)
            mask[:, :16, :7] = 1
            self.raw.append({"source_kind": "teacher_rollout", "student_checkpoint": str(self.teacher),
                "episode_success": True, "endpoint": {"pair_index": index, "velocity_seed": 900 + index},
                "inputs": {"state": torch.ones(1, 1, 132, dtype=torch.bfloat16),
                           "action": action, "action_mask": mask}})
        weights = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in self.teacher.glob("*.safetensors")}
        self.cache = {"version": 3, "metadata": {"endpoint_bundle": copy.deepcopy(self.reference),
            "teacher": str(self.teacher), "teacher_config_sha256": sha(self.teacher / "config.json"),
            "teacher_statistics_sha256": sha(self.teacher / "statistics.json"), "teacher_weights": weights,
            "model_dtype": "float32", "autocast_dtype": "bfloat16", "eval_mode": True,
            "labeling_implementation_sha256": sha(ROOT / "rl/opd_probe_cache.py"),
            "replay_implementation_sha256": sha(ROOT / "rl/probe_distill.py"),
            "source_kind": "teacher_rollout", "count": 2, "requested_count": 2},
            "samples": [{"inputs": copy.deepcopy(raw["inputs"]), "seed": raw["endpoint"]["velocity_seed"],
                         "pred": torch.zeros_like(raw["inputs"]["action"]),
                         "provenance": {key: value for key, value in raw.items() if key != "inputs"}}
                        for raw in self.raw]}
        self.cache_path = self.root / "teacher_probes.pt"
        self.save()
        self.env = {"QAD_ENDPOINT_ROLE": "teacher", "QAD_CAPTURE_TEACHER": str(self.teacher),
            "QAD_INIT_ADAPTER": str(self.initial), "QAD_CAPTURE_DATASET": str(self.data),
            "QAD_CAPTURE_DATASET_SHA256": "data-sha", "QAD_CAPTURE_AUDIT_SHA256": "audit-sha",
            "GR00T_BASE_CKPT": str(self.base), "QAD_STEPS": "2000", "QAD_LR": "5e-05",
            "QAD_OPD_MSE_W": "0.25", "OPD_EVERY": "1", "QAD_MAX_GRAD_NORM": "0.25",
            "TRAIN_SEED": "123", "QAD_LORA_R": "32", "QAD_LORA_ALPHA": "64",
            "QAD_GLOBAL_BATCH": "16", "QAD_MICRO_BATCH": "1", "QAD_LORA_SCOPE": "all_ordinary_linear",
            "QAD_W4A4": "1", "FP4VLA_QUANT": "0", "FP4VLA_W4A4": "1", "FP4VLA_W4A4_ADAPTER": "1",
            "FP4VLA_SATURATE_F16_ACTIVATIONS": "1", "QAD_ACTIVATION_CHECKPOINTING": "1"}
        self.loader = Mock(side_effect=lambda *a, **k: (copy.deepcopy(self.raw), copy.deepcopy(self.reference)))
        self.bundle_module = ModuleType("exp.probe_endpoint_bundle")
        self.bundle_module.load_endpoint_bundle = self.loader
        patcher = patch.dict(sys.modules, {self.bundle_module.__name__: self.bundle_module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self):
        torch.save(self.cache, self.cache_path)

    def audit(self, env=None):
        return prepare_endpoint_training_cache(self.cache_path, self.protocol, self.env if env is None else env)

    def test_complete_teacher_cache_binds_invocation_and_actual_loaded_bytes(self):
        report = self.audit()
        self.loader.assert_called_once_with(self.reference["root"], "teacher", teacher_checkpoint=self.teacher)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["cache_sha256"], sha(self.cache_path))
        anchor = ProbeAnchor(self.cache_path, expected_sha256=report["cache_sha256"])
        self.assertEqual(anchor.samples[1]["seed"], 901)
        self.assertTrue(torch.equal(anchor.samples[0]["inputs"]["action"], self.raw[0]["inputs"]["action"]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_student_cache_retains_failed_observation_and_its_source_identity(self):
        self.env["QAD_ENDPOINT_ROLE"] = "student"
        self.reference.update(role="student", source_kind="student_rollout")
        self.cache["metadata"].update(endpoint_bundle=copy.deepcopy(self.reference), source_kind="student_rollout")
        for raw, stored in zip(self.raw, self.cache["samples"]):
            raw.update(source_kind="student_rollout", student_checkpoint=str(self.root / "qad_export"), episode_success=False)
            stored["provenance"] = {key: value for key, value in raw.items() if key != "inputs"}
        self.save()
        report = self.audit()
        self.assertEqual(report["source_kind"], "student_rollout")
        self.assertFalse(self.cache["samples"][0]["provenance"]["episode_success"])

    def test_missing_or_wrong_arm_and_unmatched_legacy_cache_are_rejected(self):
        for value in (None, "continued_qad", "student"):
            env = {**self.env, "QAD_ENDPOINT_ROLE": value}
            with self.subTest(role=value), self.assertRaises(ValueError):
                self.audit(env)
        self.cache["metadata"].pop("endpoint_bundle")
        self.save()
        with self.assertRaisesRegex(ValueError, "lacks the intended"):
            self.audit()

    def test_protocol_or_source_bundle_change_is_rejected(self):
        self.reference["source_changed"] = True
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            self.audit()
        self.reference.pop("source_changed")
        self.protocol_data["extra"] = "new protocol identity"
        write(self.protocol, self.protocol_data)
        with self.assertRaisesRegex(ValueError, "protocols differ"):
            self.audit()

    def test_tensor_change_even_in_padding_or_dtype_is_rejected(self):
        original = copy.deepcopy(self.cache)
        for field in ("padding", "state_dtype", "mask"):
            with self.subTest(field=field):
                self.cache = copy.deepcopy(original)
                inputs = self.cache["samples"][0]["inputs"]
                if field == "padding":
                    inputs["action"][0, 39, 131] += 1
                elif field == "state_dtype":
                    inputs["state"] = inputs["state"].float()
                else:
                    inputs["action_mask"][0, 0, 0] = 0
                self.save()
                with self.assertRaisesRegex(ValueError, "inputs differ"):
                    self.audit()

    def test_changed_seed_provenance_missing_budget_or_nonfinite_prediction_rejected(self):
        original = copy.deepcopy(self.cache)
        for field in ("seed", "provenance", "count", "prediction"):
            with self.subTest(field=field):
                self.cache = copy.deepcopy(original)
                if field == "seed":
                    self.cache["samples"][0]["seed"] += 1
                elif field == "provenance":
                    self.cache["samples"][0]["provenance"]["source_kind"] = "student_rollout"
                elif field == "count":
                    self.cache["samples"].pop()
                else:
                    self.cache["samples"][0]["pred"][0, 0, 0] = float("nan")
                self.save()
                with self.assertRaises(ValueError):
                    self.audit()

    def test_wrong_qad_base_data_budget_or_numerics_rejected(self):
        changes = {"QAD_INIT_ADAPTER": str(self.root / "other_checkpoint"),
            "GR00T_BASE_CKPT": str(self.root / "other_base"), "QAD_CAPTURE_DATASET": str(self.root / "head"),
            "QAD_CAPTURE_AUDIT_SHA256": "wrong", "QAD_CAPTURE_DATASET_SHA256": "wrong",
            "QAD_STEPS": "1000", "QAD_LR": "0.0001", "QAD_OPD_MSE_W": "1.0", "OPD_EVERY": "4",
            "QAD_MAX_GRAD_NORM": "1", "QAD_LORA_R": "16", "TRAIN_SEED": "999",
            "QAD_GLOBAL_BATCH": "8", "FP4VLA_SATURATE_F16_ACTIVATIONS": "0", "QAD_W4A4": "0"}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit({**self.env, key: value})

    def test_changed_teacher_weights_or_numerical_labels_rejected(self):
        self.cache["metadata"]["model_dtype"] = "bfloat16"
        self.save()
        with self.assertRaisesRegex(ValueError, "numerical convention"):
            self.audit()
        self.cache["metadata"]["model_dtype"] = "float32"
        self.save()
        (self.teacher / "model.safetensors").write_bytes(b"changed teacher")
        with self.assertRaisesRegex(ValueError, "teacher weights"):
            self.audit()

    def test_legacy_protocol_does_not_read_or_audit_legacy_cache(self):
        write(self.protocol, {"version": 12})
        self.cache_path.unlink()
        self.assertIsNone(prepare_endpoint_training_cache(self.cache_path, self.protocol, {}))
        self.loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "requires a matched-state protocol"):
            self.audit()

    def test_matched_anchor_requires_audit_and_rejects_post_audit_file_change(self):
        with self.assertRaisesRegex(ValueError, "requires its training audit hash"):
            ProbeAnchor(self.cache_path)
        report = self.audit()
        self.cache_path.write_bytes(b"changed after audit")
        with patch.object(torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "changed after training audit"):
                ProbeAnchor(self.cache_path, expected_sha256=report["cache_sha256"])
            loader.assert_not_called()

    def test_matched_anchor_deserializes_only_verified_bytes(self):
        digest = sha(self.cache_path)
        original_load = torch.load
        def change_file_after_read(value, **kwargs):
            self.assertIsInstance(value, io.BytesIO)
            self.cache_path.write_bytes(b"changed file after read")
            return original_load(value, **kwargs)
        with patch.object(torch, "load", side_effect=change_file_after_read):
            anchor = ProbeAnchor(self.cache_path, expected_sha256=digest)
        self.assertEqual(len(anchor.samples), 2)

    def test_protocol_rejects_unmatched_or_invalid_extra_training_budget(self):
        for key, value in (("optimizer_steps", 1000), ("learning_rate", 1e-4),
                           ("teacher_velocity_weight", float("nan")), ("probe_every", True)):
            protocol = copy.deepcopy(self.protocol_data)
            protocol["state_distillation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                state_distillation_config(protocol)

    def test_zero_nan_or_unlabeled_continuation_cannot_bypass_kd_guard(self):
        for role in ("teacher", "student"):
            for weight in (0, float("nan"), float("inf"), -1):
                with self.subTest(role=role, weight=weight), self.assertRaises(ValueError):
                    prepare_endpoint_training(self.protocol, weight, {
                        **self.env, "QAD_ENDPOINT_ROLE": role, "OPD_CACHE_PATH": str(self.cache_path)})
        for env in ({"QAD_INIT_ADAPTER": str(self.initial)}, {"OPD_CACHE_PATH": str(self.cache_path)}):
            with self.assertRaisesRegex(ValueError, "requires an explicit role"):
                prepare_endpoint_training(self.protocol, 0.0, env)
        self.assertIsNone(prepare_endpoint_training(self.protocol, 0.0, {}))

    def test_continued_control_requires_same_qad_data_and_complete_budget(self):
        env = {**self.env, "QAD_ENDPOINT_ROLE": "continued", "QAD_OPD_MSE_W": "0.0",
               "QAD_ENDPOINT_BUNDLE": self.reference["root"]}
        report = prepare_endpoint_training(self.protocol, 0.0, env)
        self.assertEqual(report["role"], "continued")
        self.assertEqual(report["sample_count"], 0)
        for key, value in (("QAD_STEPS", "1000"), ("QAD_INIT_ADAPTER", str(self.root / "different")),
                           ("OPD_CACHE_PATH", str(self.cache_path)), ("QAD_ENDPOINT_BUNDLE", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_endpoint_training(self.protocol, 0.0, {**env, key: value})

    def test_old_labeler_or_replay_source_cache_is_rejected(self):
        for field in ("labeling_implementation_sha256", "replay_implementation_sha256"):
            previous = self.cache["metadata"][field]
            self.cache["metadata"][field] = "0" * 64
            self.save()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "implementation changed"):
                self.audit()
            self.cache["metadata"][field] = previous

    def test_real_hook_passes_audit_hash_to_probe_before_trainer_initialization(self):
        """Execute the actual install function with tiny import stubs, no model."""
        source = ast.parse((ROOT / "rl/lora_qad.py").read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "install_trainer_hooks")
        class Trainer:
            def __init__(self):
                raise AssertionError("No trainer should be initialized by hook installation")
        class Pipeline:
            _create_model = _create_dataset = _create_collator = lambda *args: None
        modules = {}
        for name in ("gr00t.experiment.trainer", "gr00t.model.gr00t_n1d7.setup", "transformers"):
            modules[name] = ModuleType(name)
        modules["gr00t.experiment.trainer"].Gr00tTrainer = Trainer
        modules["gr00t.model.gr00t_n1d7.setup"].Gr00tN1d7Pipeline = Pipeline
        modules["transformers"].TrainerCallback = object
        anchor = Mock()
        anchor.meta = {"source_kind": "teacher_rollout"}
        anchor.samples = [1, 2]
        install = Mock(return_value=anchor)
        namespace = {"os": os, "MSE_W": 0.25, "PROTOCOL_FILE": self.protocol,
            "prepare_endpoint_training": prepare_endpoint_training,
            "_endpoint_cache_audit": [None], "install_sequential_probe": install,
            "install_gradient_audit": Mock()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "real_install_hooks", "exec"), namespace)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {
                **self.env, "OPD_CACHE_PATH": str(self.cache_path)}, clear=True):
            namespace["install_trainer_hooks"]()
        install.assert_called_once_with(Trainer, str(self.cache_path), 0.25, 1, cache_sha256=sha(self.cache_path))
        self.assertEqual(namespace["_endpoint_cache_audit"][0]["status"], "verified")


if __name__ == "__main__":
    unittest.main()
