"""CPU integration of Driver training requests with the existing endpoint guard."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from exp import run_high_fp4_v3 as driver_module
from rl.endpoint_cache_guard import prepare_endpoint_training
from tests import test_endpoint_training_cache as guard_tests
from tests import test_ptq_sampling_reference as reference_tests


class DriverEndpointTrainingTests(unittest.TestCase):
    def setUp(self):
        # Existing guard fixture mocks only the separately tested bundle reader.
        # Cache bytes, role/budget checks and the continuation audit stay real.
        self.fixture = guard_tests.EndpointTrainingCacheTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        for root in (f.base, f.initial):
            for name in ("config.json", "statistics.json", "ptq_recipe.json"):
                driver_module.jwrite(root / name, {"fixture": name})
            (root / "model.safetensors").write_bytes(b"fixture checkpoint weights")
        self.driver = driver_module.Driver.__new__(driver_module.Driver)
        values = {"base": f.teacher, "dataset": f.root / "demo", "batch": 16,
            "rank": 32, "alpha": 64.0, "scope": "all_ordinary_linear", "every": 1,
            "seed": 123, "protocol_path": f.protocol, "w4a4": True,
            "capture_dataset": f.data, "capture_dataset_identity": {"fixture": "capture identity"},
            "protocol": {"data": f.protocol_data, "sha256": driver_module.sha(f.protocol)},
            "py": Path(sys.executable), "groot": f.root,
            "teacher_capture_identity": {"format": "derived_teacher_replay_audit_v1", "fixture": "audit"},
            "selection": {"selected_ptq_checkpoint": str(f.base)}}
        for key, value in values.items():
            setattr(self.driver, key, value)
        identity = f.reference["endpoint_policy_training_identity"]
        identity["capture_dataset_sha256"] = self.canonical(self.driver.capture_dataset_identity)
        identity["capture_audit_sha256"] = self.canonical(self.driver.teacher_capture_identity)
        f.cache["metadata"]["endpoint_bundle"] = copy.deepcopy(f.reference)
        f.save()
        self.driver.run_logged = Mock(side_effect=self.complete_training_fixture)
        self.audit_calls = []
        patcher = patch.object(driver_module.subprocess, "run", side_effect=self.cpu_audit_process)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def canonical(value):
        import hashlib
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def cpu_audit_process(self, command, **kwargs):
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(command[:2], [sys.executable, "-c"])
        self.assertEqual(Path(command[3]), driver_module.ROOT)
        weight, environment = json.loads(command[5]), json.loads(command[6])
        self.audit_calls.append(copy.deepcopy(environment))
        try:
            result = prepare_endpoint_training(command[4], weight, environment)
        except ValueError as exc:
            raise subprocess.CalledProcessError(1, command, stderr=str(exc)) from exc
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(result), stderr="")

    def complete_training_fixture(self, name, command, cwd, environment):
        """Write the trainer receipt contract without loading GR00T or running GPU."""
        root = Path(environment["QAD_OUT"])
        steps = int(environment["QAD_STEPS"])
        checkpoint = root / f"checkpoint-{steps}"
        rec = {"base": str(self.fixture.base), "rank": 32, "alpha": 64.0,
            "scope": self.driver.scope, "train_seed": 123,
            "protocol_sha256": self.driver.protocol["sha256"], "optimizer_resumed": False,
            "probe_weight": float(environment["QAD_OPD_MSE_W"]), "probe_every": 1,
            "micro_batch": 1, "effective_global_batch": 16, "gradient_accumulation_steps": 16,
            "w4a4_enabled": True, "execution_mode": "W4A4 numerical QDQ + BF16 LoRA residual",
            "initial_adapter": environment.get("QAD_INIT_ADAPTER"),
            "capture_dataset": str(self.fixture.data),
            "capture_dataset_sha256": self.canonical(self.driver.capture_dataset_identity),
            "probe_cache": environment.get("OPD_CACHE_PATH"),
            "probe_cache_sha256": driver_module.sha(Path(environment["OPD_CACHE_PATH"])) if environment.get("OPD_CACHE_PATH") else None,
            "environment_summary": driver_module.environment_summary(environment)}
        if rec["initial_adapter"] is not None:
            rec.update(max_grad_norm=0.25, f16_activation_saturation=True)
        for filename, field in (("config.json", "base_config_sha256"),
                                ("statistics.json", "base_statistics_sha256"),
                                ("ptq_recipe.json", "base_recipe_sha256")):
            rec[field] = driver_module.sha(self.fixture.base / filename)
        endpoint = prepare_endpoint_training(self.fixture.protocol, rec["probe_weight"], environment)
        if endpoint is not None:
            rec["endpoint_training_identity"] = endpoint
        driver_module.jwrite(root / "recovery_manifest.json", rec)
        driver_module.jwrite(checkpoint / "recovery_manifest.json", rec)
        (checkpoint / "model.safetensors").write_bytes(b"completed fixture adapter")
        driver_module.jwrite(root / "runtime_metrics.json", {
            "status": "completed", "global_steps": steps, "requested_optimizer_steps": steps})

    def stage(self, role="teacher", *, bundle=True):
        f = self.fixture
        if role == "student":
            f.reference.update(role="student", source_kind="student_rollout")
            f.cache["metadata"].update(endpoint_bundle=copy.deepcopy(f.reference), source_kind="student_rollout")
            for raw, stored in zip(f.raw, f.cache["samples"]):
                raw.update(source_kind="student_rollout", student_checkpoint=str(f.root / "qad"), episode_success=False)
                stored["provenance"] = {key: value for key, value in raw.items() if key != "inputs"}
            f.save()
        path = f.root / ("stage_" + role)
        options = {"initial": f.initial, "weight": 0.0 if role == "continued" else 0.25,
                   "cache": None if role == "continued" else f.cache_path, "endpoint_role": role,
                   "endpoint_bundle": f.reference["root"] if bundle else None}
        result = self.driver.train(path, "train_" + role, f.base, 5e-5, 2000, **options)
        return path, options, result

    def test_teacher_request_and_trainer_and_recomputed_audit_match(self):
        path, options, result = self.stage()
        verified = self.driver.train_verify(path, 2000, lr=5e-5, **options)
        request = driver_module.jread(path / "orchestrator_training_request.json")
        rec = driver_module.jread(path / "recovery_manifest.json")
        self.assertEqual(request["endpoint_training_identity"], rec["endpoint_training_identity"])
        self.assertEqual(result["endpoint_training_identity"], verified["endpoint_training_identity"])
        self.assertEqual(verified["endpoint_training_identity"]["role"], "teacher")
        self.assertEqual(request["environment"]["QAD_ENDPOINT_ROLE"], "teacher")
        self.assertEqual(request["environment"]["QAD_ENDPOINT_BUNDLE"], self.fixture.reference["root"])
        self.assertEqual(len(self.audit_calls), 2)

    def test_student_cache_is_wired_without_relabelling_and_bundle_is_optional_for_cache_role(self):
        path, options, _ = self.stage("student", bundle=False)
        result = self.driver.train_verify(path, 2000, lr=5e-5, **options)
        self.assertEqual(result["endpoint_training_identity"]["source_kind"], "student_rollout")
        self.assertNotIn("QAD_ENDPOINT_BUNDLE", self.driver.run_logged.call_args.args[-1])

    def test_continued_qad_uses_the_common_bundle_without_a_velocity_cache(self):
        path, options, _ = self.stage("continued")
        result = self.driver.train_verify(path, 2000, lr=5e-5, **options)
        report = result["endpoint_training_identity"]
        self.assertEqual((report["role"], report["sample_count"], report["cache_path"]), ("continued", 0, None))
        self.assertNotIn("OPD_CACHE_PATH", self.driver.run_logged.call_args.args[-1])

    def test_wrong_role_budget_or_bundle_rejected_before_training_or_request_write(self):
        f = self.fixture
        cases = ({"endpoint_role": "unknown"}, {"endpoint_role": "student"},
                 {"steps": 1000}, {"endpoint_bundle": f.root / "different_bundle"})
        for index, changed in enumerate(cases):
            path = f.root / f"invalid_{index}"
            opts = {"initial": f.initial, "weight": 0.25, "cache": f.cache_path,
                    "endpoint_role": "teacher", "endpoint_bundle": f.reference["root"]}
            opts.update(changed)
            steps = opts.pop("steps", 2000)
            with self.subTest(changed=changed), self.assertRaises(driver_module.OrchestrationError):
                self.driver.train(path, "invalid", f.base, 5e-5, steps, **opts)
            self.assertFalse(path.exists())
        self.driver.run_logged.assert_not_called()

    def test_verifier_requires_explicit_stage_role_and_original_trainer_identity(self):
        path, options, _ = self.stage()
        with self.assertRaisesRegex(driver_module.OrchestrationError, "explicit stage request"):
            self.driver.train_verify(path, 2000, initial=options["initial"], lr=5e-5,
                                     weight=0.25, cache=options["cache"])
        rec_path = path / "recovery_manifest.json"
        rec = driver_module.jread(rec_path)
        rec["endpoint_training_identity"]["role"] = "student"
        driver_module.jwrite(rec_path, rec)
        with self.assertRaisesRegex(driver_module.OrchestrationError, "training endpoint identity differs"):
            self.driver.train_verify(path, 2000, lr=5e-5, **options)

    def test_legacy_train_adds_no_endpoint_keys_and_skips_the_new_audit(self):
        f = self.fixture
        driver_module.jwrite(f.protocol, {"version": 12})
        self.driver.protocol = {"data": {"version": 12}, "sha256": driver_module.sha(f.protocol)}
        path = f.root / "legacy"
        result = self.driver.train(path, "legacy", f.base, 5e-5, 2000)
        self.driver.train_verify(path, 2000, lr=5e-5)
        request = driver_module.jread(path / "orchestrator_training_request.json")
        self.assertNotIn("endpoint_training_identity", request)
        self.assertNotIn("endpoint_training_identity", result)
        self.assertFalse(any("ENDPOINT" in key for key in request["environment"]))
        self.assertEqual(self.audit_calls, [])

    def test_reference_run_boundary_still_rejects_recovery_before_any_stage(self):
        self.driver.protocol["data"]["ptq_reference"] = {"allowed_until": "qad_dev"}
        self.driver.a = type("Args", (), {"validate_only": False})()
        self.driver.stage = Mock()
        with self.assertRaisesRegex(driver_module.OrchestrationError, "restricted to --until qad_dev"):
            self.driver.run("recovery_dev")
        self.driver.stage.assert_not_called()

    def test_cpu_preflight_environment_is_identical_to_actual_training_request(self):
        f = self.fixture
        path = f.root / "stage_teacher"
        expected = self.driver.training_environment(path, f.base, 5e-5, 2000,
            initial=f.initial, weight=0.25, cache=f.cache_path,
            endpoint_role="teacher", endpoint_bundle=f.reference["root"])
        self.assertFalse(path.exists())
        self.stage()
        request = driver_module.jread(path / "orchestrator_training_request.json")
        self.assertEqual(request["environment"], expected)
        self.assertEqual(self.driver.run_logged.call_args.args[-1], expected)


class ExplicitDriverScopeTests(unittest.TestCase):
    def setUp(self):
        self.reference = reference_tests.PTQSamplingDriverTests()
        self.reference.setUp()
        self.addCleanup(self.reference.doCleanups)
        self.declaration = {"scope": "development_only",
            "arms": ["continued_qad", "teacher_state_kd", "student_state_opd"],
            "source": "frozen_stratified_qad_endpoint_bundle",
            "allowed_stages": ["train", "merge", "development", "paired_summary"]}
        self.reference.target["state_distillation_execution"] = self.declaration
        self.reference.normalized()

    def construct(self):
        return driver_module.Driver(self.reference.args(driver_module.STATE_DISTILLATION_SCOPE),
                                     execution_scope=driver_module.STATE_DISTILLATION_SCOPE)

    def test_explicit_frozen_scope_constructs_without_relaxing_legacy_run(self):
        with self.reference.driver_context():
            instance = self.construct()
        self.assertEqual(instance.state["execution_scope"], driver_module.STATE_DISTILLATION_SCOPE)
        self.assertEqual(instance.state["execution_scope_declaration"], self.declaration)
        self.assertEqual(instance.state["selection_protocol_sha256"], self.reference.validated["protocol_sha256"])
        for until in ("qad_dev", "recovery_dev", "all"):
            with self.subTest(until=until), self.assertRaisesRegex(driver_module.OrchestrationError, "dedicated development"):
                instance.run(until)

    def test_scope_requires_exact_predeclared_bounds_before_creating_outputs(self):
        for key, value in (("scope", "heldout"), ("arms", ["student_state_opd"]),
                           ("allowed_stages", ["train", "merge", "heldout", "paired_summary"])):
            original = copy.deepcopy(self.declaration)
            self.reference.target["state_distillation_execution"] = {**original, key: value}
            self.reference.normalized()
            with self.subTest(key=key), patch.object(driver_module, "resolve_ptq_selection") as resolve:
                with self.assertRaisesRegex(driver_module.OrchestrationError, "does not authorize"):
                    self.construct()
                resolve.assert_not_called()
                self.assertFalse((self.reference.root / "run").exists())

    def test_scope_rejects_extra_stages_wrong_arms_and_non_development_evaluation(self):
        with self.reference.driver_context():
            instance = self.construct()
        with patch.object(instance, "freeze_check") as freeze:
            for name in ("heldout_qad_opd", "collection_qad", "teacher_cache", "select_opd_weight", "dev_qad"):
                with self.subTest(stage=name), self.assertRaisesRegex(driver_module.OrchestrationError, "outside the explicit"):
                    instance.stage(name, Mock(), Mock())
            freeze.assert_not_called()
        for purpose in ("heldout", "collection", "teacher_supervision"):
            with self.subTest(purpose=purpose), self.assertRaisesRegex(driver_module.OrchestrationError, "development evaluation only"):
                instance.evaluate(self.reference.root / "invalid", "bad", self.reference.root, purpose)
        with self.assertRaisesRegex(driver_module.OrchestrationError, "explicit matched arm"):
            instance.train(self.reference.root / "invalid", "train_teacher_state_kd", self.reference.root,
                           5e-5, 2000, endpoint_role="student")

    def test_scope_and_declaration_cannot_be_relabelled_on_resume(self):
        with self.reference.driver_context():
            instance = self.construct()
            with self.assertRaisesRegex(driver_module.OrchestrationError, "resume identity changed: execution_scope"):
                driver_module.Driver(self.reference.args("qad_dev"))
            manifest = instance.run_dir / "run_manifest.json"
            state = driver_module.jread(manifest)
            state["execution_scope_declaration"]["scope"] = "heldout"
            driver_module.jwrite(manifest, state)
            with self.assertRaisesRegex(driver_module.OrchestrationError, "resume identity changed: execution_scope_declaration"):
                self.construct()


if __name__ == "__main__":
    unittest.main()
