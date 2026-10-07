"""CPU tests for frozen PTQ reference boundaries; source audit is delegated."""
import copy
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from exp import run_high_fp4_v3 as driver
from exp.ptq_sampling_reference import PTQReferenceError, validate_reference


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PTQSamplingReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_path = self.root / "source_protocol.json"
        self.selection_path = self.root / "selection.json"
        self.target_path = self.root / "target_protocol.json"
        part = lambda seed, indices: {"seed": seed, "init_state_indices": indices,
                                      "episodes_per_task": len(indices)}
        self.source = {"version": 12, "w4a4": True,
            "partitions": {"development": part(940000, [4, 5]),
                "teacher_supervision": part(950000, [20, 21]),
                "collection": part(960000, [20, 21]), "heldout": part(970000, [24, 25])},
            "selection": {"pressure_candidates": ["rtn"],
                "pressure_candidate_checkpoints": {"rtn": "/fixed/ptq"},
                "pressure_rule": {"choose": "highest_fp4", "min_drop_from_bf16": 0.05,
                    "min_absolute_success": 0.3, "selection_metric": "micro_success_rate"},
                "recovery_scope": "all_ordinary_linear", "qad_learning_rates": [5e-5, 1e-4]},
            "quantization_scope": {"weight_format": "NVFP4", "activation_format": "NVFP4"},
            "evaluation_contract": {"n_envs": 1, "n_action_steps": 8, "max_episode_steps": 720}}
        write(self.source_path, self.source)
        self.validated = {"protocol_sha256": sha(self.source_path), "selected_recipe": "rtn",
                          "selected_ptq_checkpoint": "/fixed/ptq",
                          "arms": {"rtn": {"successes": 41, "episodes": 50}}}
        write(self.selection_path, self.validated)
        self.target = copy.deepcopy(self.source)
        self.target["selection"]["qad_learning_rates"] = [1e-4]
        self.target["capture_views"] = {"modes": ["head", "stratified"], "windows_per_episode": 4}
        self.target["ptq_reference"] = {
            "scope": "fixed_base_for_sampling_ablation", "allowed_until": "qad_dev",
            "source_protocol": {"path": str(self.source_path), "sha256": sha(self.source_path)},
            "source_selection": {"path": str(self.selection_path), "sha256": sha(self.selection_path)}}

    def normalized(self):
        write(self.target_path, self.target)
        return driver.load_protocol(self.target_path)

    def validate(self, **kwargs):
        protocol = kwargs.pop("protocol", None) or self.normalized()
        return validate_reference(kwargs.pop("path", self.selection_path), protocol)

    def test_original_selection_is_fully_audited_and_not_relabelled_or_modified(self):
        normalized = self.normalized()
        before = {p: p.read_bytes() for p in (self.source_path, self.selection_path, self.target_path)}
        with patch.object(driver, "validate_ptq", return_value=self.validated) as audit:
            result = self.validate(protocol=normalized)
        source_protocol = driver.load_protocol(self.source_path)
        audit.assert_called_once_with(self.selection_path, source_protocol)
        self.assertEqual({k: result[k] for k in self.validated}, self.validated)
        self.assertNotEqual(result["protocol_sha256"], normalized["sha256"])
        reference = result["reference_provenance"]
        self.assertEqual(reference["source_protocol"]["sha256"], sha(self.source_path))
        self.assertEqual(reference["target_protocol_sha256"], normalized["sha256"])
        self.assertIs(reference["source_score_is_new_evaluation"], False)
        self.assertEqual(reference["allowed_until"], "qad_dev")
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertNotIn("reference_provenance", self.validated)

    def test_relative_reference_paths_use_target_protocol_directory(self):
        self.target["ptq_reference"]["source_protocol"]["path"] = self.source_path.name
        self.target["ptq_reference"]["source_selection"]["path"] = self.selection_path.name
        with patch.object(driver, "validate_ptq", return_value=self.validated):
            self.assertEqual(self.validate()["reference_provenance"]["source_selection"]["path"],
                             str(self.selection_path))

    def test_source_identity_hash_and_requested_path_mismatches_are_rejected(self):
        for source in ("source_protocol", "source_selection"):
            with self.subTest(source=source):
                original = self.target["ptq_reference"][source]["sha256"]
                self.target["ptq_reference"][source]["sha256"] = "0" * 64
                with patch.object(driver, "validate_ptq") as audit:
                    with self.assertRaisesRegex(PTQReferenceError, "SHA differs"):
                        self.validate()
                    audit.assert_not_called()
                self.target["ptq_reference"][source]["sha256"] = original
        other = self.root / "other_selection.json"
        other.write_bytes(self.selection_path.read_bytes())
        with self.assertRaisesRegex(PTQReferenceError, "selection path differs"):
            self.validate(path=other)

    def test_nested_source_reference_is_rejected(self):
        self.source["ptq_reference"] = None
        write(self.source_path, self.source)
        self.target["ptq_reference"]["source_protocol"]["sha256"] = sha(self.source_path)
        with self.assertRaisesRegex(PTQReferenceError, "Nested PTQ references"):
            self.validate()

    def test_all_evaluation_and_pressure_boundaries_are_frozen(self):
        original = copy.deepcopy(self.target)
        changes = {
            "development": lambda p: p["partitions"]["development"].update(seed=940001),
            "development_order": lambda p: p["partitions"]["development"].update(init_state_indices=[5, 4]),
            "w4a4": lambda p: p.update(w4a4=False),
            "quantization_scope": lambda p: p["quantization_scope"].update(activation_format="BF16"),
            "evaluation_contract": lambda p: p["evaluation_contract"].update(n_action_steps=16),
            "pressure_candidates": lambda p: p["selection"].update(pressure_candidates=["another"]),
            "pressure_rule": lambda p: p["selection"]["pressure_rule"].update(min_drop_from_bf16=0.01),
            "pressure_checkpoint": lambda p: p["selection"]["pressure_candidate_checkpoints"].update(rtn="/other/ptq"),
        }
        for name, change in changes.items():
            with self.subTest(boundary=name):
                self.target = copy.deepcopy(original)
                change(self.target)
                with patch.object(driver, "validate_ptq") as audit:
                    with self.assertRaises(PTQReferenceError):
                        self.validate()
                    audit.assert_not_called()

    def test_ablation_scope_stage_views_and_single_learning_rate_are_required(self):
        original = copy.deepcopy(self.target)
        changes = [lambda p: p["ptq_reference"].update(scope="general_reuse"),
                   lambda p: p["ptq_reference"].update(allowed_until="heldout"),
                   lambda p: p.pop("capture_views"),
                   lambda p: p["capture_views"].update(modes=["head"]),
                   lambda p: p["capture_views"].update(windows_per_episode=True)]
        changes.extend(lambda p, lr=lr: p["selection"].update(qad_learning_rates=lr)
                       for lr in ([], [5e-5, 1e-4], [0], [True], ["1e-4"]))
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                self.target = copy.deepcopy(original)
                change(self.target)
                with patch.object(driver, "validate_ptq") as audit:
                    with self.assertRaises(PTQReferenceError):
                        self.validate()
                    audit.assert_not_called()

    def test_stale_normalization_and_source_audit_failures_cannot_be_bypassed(self):
        normalized = self.normalized()
        normalized["sha256"] = "0" * 64
        with self.assertRaisesRegex(PTQReferenceError, "Normalized target protocol differs"):
            self.validate(protocol=normalized)
        with patch.object(driver, "validate_ptq", side_effect=driver.OrchestrationError("changed model bytes")):
            with self.assertRaisesRegex(driver.OrchestrationError, "changed model bytes"):
                self.validate()

    def test_source_bytes_changed_during_full_validation_are_rejected(self):
        def change_source(*args):
            self.selection_path.write_text(self.selection_path.read_text() + "\n")
            return self.validated
        with patch.object(driver, "validate_ptq", side_effect=change_source):
            with self.assertRaisesRegex(PTQReferenceError, "changed during audit"):
                self.validate()


class PTQSamplingDriverTests(unittest.TestCase):
    """Exercise the real Driver entry points without launching subprocesses."""

    normalized = PTQSamplingReferenceTests.normalized

    def setUp(self):
        PTQSamplingReferenceTests.setUp(self)
        self.validated.update(selection_sha256=sha(self.selection_path),
                              selected_ptq_recipe_sha256="r" * 64)
        self.target["selection"].update(
            train_seed=42, qad_optimizer_steps=5, continuation_optimizer_steps=5,
            rank=32, alpha=64, effective_demo_batch=16, opd_every=4,
            opd_weights=[0.25, 1.0])
        self.protocol = self.normalized()

    def args(self, until="qad_dev", validate_only=False):
        return argparse.Namespace(
            protocol_file=str(self.target_path), ptq_selection=str(self.selection_path),
            run_dir=str(self.root / "run"), until=until, validate_only=validate_only,
            gr00t_repo=str(self.root / "groot"), python=sys.executable,
            rollout_python=sys.executable, base=str(self.root / "base"),
            dataset=str(self.root / "dataset"), capture_dataset=str(self.root / "captures"),
            port_base=5790, adopt_complete=False, cleanup_duplicates=False)

    def driver_context(self):
        context = ExitStack()
        context.enter_context(patch.object(driver, "validate_ptq", return_value=self.validated))
        context.enter_context(patch.object(driver, "capture_dataset_id", return_value={"data": "sha"}))
        context.enter_context(patch.object(driver, "validate_teacher_capture", return_value={"audit": "sha"}))
        context.enter_context(patch.object(driver, "model_id", return_value={"weights": "sha"}))
        return context

    def test_constructor_rejects_unbounded_reference_before_any_run_directory_or_source_audit(self):
        for until in ("all", "recovery_dev", "qad_selection", "opd_selection"):
            with self.subTest(until=until), patch.object(driver, "resolve_ptq_selection") as resolve, \
                    patch.object(driver.Driver, "load_state") as load, \
                    patch.object(driver, "validate_teacher_capture") as capture:
                with self.assertRaisesRegex(driver.OrchestrationError, "restricted to --until qad_dev"):
                    driver.Driver(self.args(until))
                resolve.assert_not_called()
                load.assert_not_called()
                capture.assert_not_called()
                self.assertFalse((self.root / "run").exists())

    def test_validate_only_all_preserves_source_protocol_and_records_reference(self):
        before = self.selection_path.read_bytes()
        with self.driver_context():
            instance = driver.Driver(self.args("all", validate_only=True))
            with patch.object(instance, "stage") as stage:
                result = instance.run("all")
                stage.assert_not_called()
        state = json.loads((self.root / "run/run_manifest.json").read_text())
        self.assertEqual(result["status"], "validated_only")
        self.assertEqual(state["protocol_sha256"], self.protocol["sha256"])
        self.assertEqual(state["selection_protocol_sha256"], self.validated["protocol_sha256"])
        self.assertNotEqual(state["selection_protocol_sha256"], state["protocol_sha256"])
        self.assertEqual(state["reference_provenance"], instance.selection["reference_provenance"])
        self.assertEqual(result["reference_provenance"], state["reference_provenance"])
        self.assertEqual(before, self.selection_path.read_bytes())

    def test_run_rejects_changed_boundary_before_train_even_without_constructor(self):
        instance = driver.Driver.__new__(driver.Driver)
        instance.a = argparse.Namespace(validate_only=False)
        instance.protocol = self.protocol
        instance.train = Mock()
        instance.stage = Mock()
        for until in ("all", "recovery_dev", "qad_selection", "opd_selection"):
            with self.subTest(until=until), self.assertRaisesRegex(
                    driver.OrchestrationError, "restricted to --until qad_dev"):
                instance.run(until)
        instance.train.assert_not_called()
        instance.stage.assert_not_called()

    def test_qad_dev_runs_one_frozen_learning_rate_and_stops_before_selection(self):
        with self.driver_context():
            instance = driver.Driver(self.args())
            for name in ("train", "train_verify", "merge", "merge_verify", "evaluate",
                         "eval_verify", "cleanup_dupes", "select", "cache"):
                setattr(instance, name, Mock())

            def stage(name, produce, verify):
                path = self.root / "fake_stages" / name
                produce(path)
                verify(path)
                return path

            instance.stage = Mock(side_effect=stage)
            result = instance.run("qad_dev")
        self.assertIs(result, instance.state)
        self.assertEqual([call.args[0] for call in instance.stage.call_args_list],
                         ["train_qad_lr_0.0001", "merge_qad_lr_0.0001", "dev_qad_lr_0.0001"])
        instance.train.assert_called_once()
        self.assertEqual(instance.train.call_args.args[3], 1e-4)
        instance.evaluate.assert_called_once()
        self.assertEqual(instance.evaluate.call_args.args[3], "development")
        instance.select.assert_not_called()
        instance.cache.assert_not_called()

    def test_freeze_check_reaudits_reference_and_rejects_changed_provenance(self):
        with self.driver_context():
            instance = driver.Driver(self.args())
            with patch.object(driver, "resolve_ptq_selection", wraps=driver.resolve_ptq_selection) as resolve:
                instance.freeze_check()
                resolve.assert_called_once_with(instance.selection_path, instance.protocol)
            changed = copy.deepcopy(instance.selection)
            changed["reference_provenance"]["source_score_is_new_evaluation"] = True
            with patch.object(driver, "resolve_ptq_selection", return_value=changed):
                with self.assertRaisesRegex(driver.OrchestrationError, "reference provenance changed"):
                    instance.freeze_check()

    def test_resume_rejects_reference_or_original_protocol_relabelling(self):
        with self.driver_context():
            instance = driver.Driver(self.args())
            manifest = instance.run_dir / "run_manifest.json"
            original = json.loads(manifest.read_text())
            for field, value in (("reference_provenance", None),
                                 ("selection_protocol_sha256", self.protocol["sha256"])):
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    changed[field] = value
                    write(manifest, changed)
                    with self.assertRaisesRegex(driver.OrchestrationError, "resume identity changed: " + field):
                        instance.load_state()
            write(manifest, original)
            self.assertEqual(instance.load_state(), original)

    def test_normal_protocol_still_uses_original_validator(self):
        source_protocol = driver.load_protocol(self.source_path)
        with patch.object(driver, "validate_ptq", return_value=self.validated) as audit:
            self.assertIs(driver.resolve_ptq_selection(self.selection_path, source_protocol), self.validated)
        audit.assert_called_once_with(self.selection_path, source_protocol)
        driver.check_ptq_reference_stage(source_protocol, "all", False)

    def test_constructor_without_stage_namespace_fields_keeps_normal_protocol_compatibility(self):
        self.target.pop("ptq_reference")
        self.normalized()
        args = self.args()
        del args.until
        del args.validate_only
        with self.driver_context():
            instance = driver.Driver(args)
        self.assertEqual(instance.selection, self.validated)

    def test_reference_without_until_defaults_to_rejected_all(self):
        args = self.args()
        del args.until
        with self.assertRaisesRegex(driver.OrchestrationError, "restricted to --until qad_dev"):
            driver.Driver(args)
        self.assertFalse((self.root / "run").exists())


if __name__ == "__main__":
    unittest.main()
