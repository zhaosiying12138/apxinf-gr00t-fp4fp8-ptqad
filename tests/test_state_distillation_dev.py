"""CPU orchestration tests: real stage receipts, resume and raw rollout audit.

Physical training/export and the separately tested full endpoint reader are
replaced at their boundaries. No model, simulator or GPU worker is launched.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from exp import run_high_fp4_v3 as driver_module
from exp import run_state_distillation_dev as entry
from tests import test_ptq_sampling_reference as reference_tests


class StateDistillationDevTests(unittest.TestCase):
    def setUp(self):
        self.fixture = reference_tests.PTQSamplingDriverTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.root = f.root
        self.bundle = self.root / "endpoint_bundle"
        self.bundle.mkdir()
        self.teacher = self.root / "teacher"
        self.base = self.root / "ptq"
        self.qad = self.root / "qad"
        self.initial = self.root / "qad_training/checkpoint-5"
        for folder in (self.teacher, self.base, self.qad, self.initial):
            folder.mkdir(parents=True)
            (folder / "model.safetensors").write_bytes(folder.name.encode())
        self.caches = {role: self.root / (role + "_cache.json") for role in ("teacher", "student")}
        for role, path in self.caches.items():
            driver_module.jwrite(path, {"role": role})
        f.source["selection"]["pressure_candidate_checkpoints"]["rtn"] = str(self.base)
        f.target["selection"]["pressure_candidate_checkpoints"]["rtn"] = str(self.base)
        driver_module.jwrite(f.source_path, f.source)
        f.validated.update(selected_ptq_checkpoint=str(self.base), protocol_sha256=entry.sha(f.source_path))
        driver_module.jwrite(f.selection_path, f.validated)
        f.validated["selection_sha256"] = entry.sha(f.selection_path)
        f.target["ptq_reference"]["source_protocol"]["sha256"] = entry.sha(f.source_path)
        f.target["ptq_reference"]["source_selection"]["sha256"] = entry.sha(f.selection_path)
        f.target["state_distillation_execution"] = {
            "scope": "development_only", "arms": list(entry.ARMS),
            "source": "frozen_stratified_qad_endpoint_bundle",
            "allowed_stages": ["train", "merge", "development", "paired_summary"]}
        f.target["state_distillation"] = {
            "arms": list(entry.ARMS), "starting_qad_view": "stratified",
            "teacher_velocity_weight": 0.25, "probe_every": 4,
            "optimizer_steps": 5, "learning_rate": 1e-4, "max_grad_norm": 0.25}
        f.protocol = f.normalized()
        self.plan = {"schema": "fp4vla_probe_endpoint_plan_v1", "status": "frozen_cpu_plan",
            "teacher_checkpoint": str(self.teacher), "protocol_sha256": f.protocol["sha256"],
            "qad": {"checkpoint": str(self.qad), "base": str(self.base),
                "training_checkpoint": str(self.initial),
                "checkpoint_files": {"model.safetensors": entry.identity(self.qad / "model.safetensors")},
                "training_identity": {"capture_dataset": str(self.root / "stratified")}},
            "sources": {role: {"source_directory": str(self.root / (role + "_source")),
                "view": str(self.root / (role + "_views") / "stratified")} for role in self.caches}}
        self.write_plan()
        self.args = entry.parse_args([
            "--run-dir", str(self.root / "run"), "--protocol-file", str(f.target_path),
            "--ptq-selection", str(f.selection_path), "--endpoint-bundle", str(self.bundle),
            "--teacher-cache", str(self.caches["teacher"]), "--student-cache", str(self.caches["student"]),
            "--python", sys.executable, "--rollout-python", sys.executable,
            "--gr00t-repo", str(self.root / "gr00t")])
        self.events = []
        self.audit_override = None
        self.enterContext(f.driver_context())
        for name, callback in (
            ("_endpoint_training_identity", self.audit), ("train", self.train),
            ("train_verify", self.train_verify), ("merge", self.merge),
            ("merge_verify", self.merge_verify), ("evaluate", self.evaluate)):
            self.enterContext(patch.object(driver_module.Driver, name, autospec=True, side_effect=callback))
        self.forbidden = {}
        for name in ("select", "cache", "cache_verify", "cleanup_dupes", "run", "run_logged"):
            self.forbidden[name] = self.enterContext(patch.object(driver_module.Driver, name,
                side_effect=AssertionError("Unexpected legacy or process path: " + name)))

    def write_plan(self):
        entry.jwrite(self.bundle / "plan.json", self.plan)

    def audit(self, driver, weight, environment):
        role = environment["QAD_ENDPOINT_ROLE"]
        self.events.append("audit_" + role)
        self.assertEqual(environment["GR00T_BASE_CKPT"], str(self.base))
        self.assertEqual(environment["QAD_INIT_ADAPTER"], str(self.initial))
        self.assertEqual(environment["QAD_ENDPOINT_BUNDLE"], str(self.bundle))
        cache = environment.get("OPD_CACHE_PATH")
        if role == "continued":
            self.assertIsNone(cache)
            self.assertEqual(weight, 0)
        else:
            if entry.jread(Path(cache))["role"] != role:
                raise entry.OrchestrationError("cache role differs")
            self.assertEqual(weight, 0.25)
        result = {"status": "verified", "role": role, "initial_adapter": str(self.initial),
            "cache_identity": entry.identity(Path(cache)) if cache else None,
            "endpoint_bundle": {"root": str(self.bundle),
                "plan_sha256": entry.sha(self.bundle / "plan.json"),
                "endpoint_policy_checkpoint": str(self.qad),
                "endpoint_policy_files": self.plan["qad"]["checkpoint_files"]}}
        if self.audit_override:
            self.audit_override(result)
        return result

    def train(self, driver, path, name, base, lr, steps, **options):
        self.events.append(name)
        self.assertEqual(self.events[:3], ["audit_continued", "audit_teacher", "audit_student"])
        environment = driver.training_environment(path, base, lr, steps, **options)
        self.assertEqual((base, options["initial"], lr, steps), (self.base, self.initial, 1e-4, 5))
        self.assertEqual(name, "train_" + driver_module.STATE_DISTILLATION_ROLES[options["endpoint_role"]])
        entry.jwrite(path / "fixture_request.json", environment)
        entry.jwrite(path / "runtime_metrics.json", {"status": "completed", "global_steps": steps})
        checkpoint = path / f"checkpoint-{steps}"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(name.encode())
        return {}

    def train_verify(self, driver, path, steps, *, lr, **options):
        environment = driver.training_environment(path, self.base, lr, steps, **options)
        if entry.jread(path / "fixture_request.json") != environment:
            raise entry.OrchestrationError("training request changed")
        metrics = entry.jread(path / "runtime_metrics.json")
        if metrics != {"status": "completed", "global_steps": steps}:
            raise entry.OrchestrationError("incomplete final checkpoint")
        return {"training_identity": entry.identity(path / f"checkpoint-{steps}" / "model.safetensors"),
                "runtime_identity": entry.identity(path / "runtime_metrics.json")}

    def merge(self, driver, path, name, base, checkpoint, steps):
        self.events.append(name)
        entry.jwrite(path / "merge_manifest.json", {"training_checkpoint": str(checkpoint)})
        (path / "model.safetensors").write_bytes((checkpoint / "model.safetensors").read_bytes())
        return {}

    def merge_verify(self, driver, path):
        return {"merge_manifest_sha256": entry.sha(path / "merge_manifest.json"),
                "output_identity": entry.identity(path / "model.safetensors")}

    def evaluate(self, driver, path, name, checkpoint, purpose, offset):
        self.events.append(name)
        self.assertEqual(purpose, "development")
        self.assertEqual(offset, entry.ARMS.index(name.removeprefix("dev_")))
        # Each toy task has a discordant pair in the first comparison.
        outcomes = {entry.ARMS[0]: [True, False], entry.ARMS[1]: [False, True],
                    entry.ARMS[2]: [True, True]}[name.removeprefix("dev_")]
        self.write_evaluation(driver, path, checkpoint, outcomes)
        return {}

    def write_evaluation(self, driver, path, checkpoint, outcomes):
        part = driver.protocol["partitions"]["development"]
        manifest = {"out": str(path), "video_root": str(path / "videos"), "checkpoint": str(checkpoint),
            "protocol_file": str(driver.protocol_path), "protocol_sha256": driver.protocol["sha256"],
            "purpose": "development", "seed": part["seed"], "episodes": 2,
            "init_state_indices": part["init_state_indices"], "tasks": list(entry.TASKS),
            "task_count": 10, "n_envs": 1, "task_seed_stride": 1000, "episode_seed_stride": 1,
            "server_seed_offset": 10000000, "settle_steps": 10, "n_action_steps": 8,
            "max_episode_steps": 720, "initial_state_protocol": "libero10_official_bank_v1",
            "environment_summary": {"variables": {"FP4VLA_QUANT": "0", "FP4VLA_W4A4": "1",
                "FP4VLA_W4A4_ADAPTER": "1", "FP4VLA_SATURATE_F16_ACTIVATIONS": "1"}},
            "recovery_contract": {"max_grad_norm": None, "f16_activation_saturation": False}}
        entry.jwrite(path / "eval_manifest.json", manifest)
        rows = {}
        for ti, task in enumerate(entry.TASKS):
            resets = [{"episode_index": i, "seed": part["seed"] + 1000 * ti + i,
                "init_state_index": part["init_state_indices"][i], "settle_steps": 10,
                "initial_state_sha256": "a" * 64, "restored_state_sha256": "b" * 64,
                "init_state_bank_sha256": "c" * 64} for i in range(2)]
            log = path / (task + ".log")
            import json
            log.write_text("".join("FP4VLA_EPISODE_RESET " + json.dumps(row) + "\n" for row in resets)
                           + f"results: ('{task}', {outcomes})\n")
            rows[task] = {**driver_module.parse_log(log), "returncode": 0, "seed": part["seed"] + 1000 * ti}
        entry.jwrite(path / "task_results.json", rows)
        entry.jwrite(path / "summary.json", {"tasks_complete": 10, "total_successes": sum(outcomes) * 10,
            "total_episodes": 20, "macro_success_rate": sum(outcomes) / 2,
            "micro_success_rate": sum(outcomes) / 2, "purpose": "development", "seed": part["seed"],
            "score_definition": "macro=ten-task mean; micro=total successes/episodes"})

    def test_fixed_serial_arms_and_raw_paired_contrasts(self):
        instance = entry.StateDistillationDev(self.args)
        self.assertEqual(instance.driver.execution_scope, driver_module.STATE_DISTILLATION_SCOPE)
        self.assertEqual(instance.driver.a.until, driver_module.STATE_DISTILLATION_SCOPE)
        self.assertEqual(instance.driver.protocol["data"]["ptq_reference"]["allowed_until"], "qad_dev")
        report = instance.run()
        expected = [prefix + "_" + arm for arm in entry.ARMS for prefix in ("train", "merge", "dev")]
        self.assertEqual([item for item in self.events if not item.startswith("audit_")], expected)
        self.assertEqual(list(instance.driver.state["stages"]), expected + ["paired_summary"])
        self.assertFalse(report["selection_performed"])
        self.assertFalse(report["heldout_executed"])
        self.assertEqual(report["scope"], "development_only")
        contrast = report["contrasts"]["teacher_state_kd_vs_continued_qad"]
        self.assertEqual((contrast["only_treatment_success"], contrast["only_baseline_success"]), (10, 10))
        self.assertEqual(contrast["success_rate_difference_treatment_minus_baseline"], 0)
        self.assertEqual(contrast["per_task"][entry.TASKS[0]]["discordant_count"], 2)
        opd = report["contrasts"]["student_state_opd_vs_teacher_state_kd"]
        self.assertEqual((opd["paired_count"], opd["only_treatment_success"], opd["only_baseline_success"]), (20, 10, 0))
        for function in self.forbidden.values():
            function.assert_not_called()

    def test_validate_only_audits_all_inputs_without_any_stage(self):
        self.args.validate_only = True
        instance = entry.StateDistillationDev(self.args)
        result = instance.run()
        self.assertEqual(result["status"], "validated_only")
        self.assertEqual(self.events, ["audit_continued", "audit_teacher", "audit_student"])
        self.assertEqual(instance.driver.state["stages"], {})
        self.assertEqual(list(instance.driver.art.iterdir()), [])

    def test_resume_reaudits_all_inputs_but_reuses_completed_stages(self):
        original = entry.StateDistillationDev(self.args).run()
        self.events.clear()
        resumed = entry.StateDistillationDev(self.args).run()
        self.assertEqual(resumed, original)
        self.assertTrue(self.events)
        self.assertTrue(all(event.startswith("audit_") for event in self.events))

    def test_wrong_plan_protocol_base_or_audited_initial_is_rejected_before_training(self):
        original = copy.deepcopy(self.plan)
        for field in ("protocol", "base", "initial", "endpoint_files"):
            with self.subTest(field=field):
                self.plan = copy.deepcopy(original)
                self.args.run_dir = str(self.root / ("bad_" + field))
                self.audit_override = None
                if field == "protocol":
                    self.plan["protocol_sha256"] = "0" * 64
                elif field == "base":
                    self.plan["qad"]["base"] = str(self.root / "other_base")
                elif field == "initial":
                    self.audit_override = lambda report: report.update(initial_adapter=str(self.root / "other_initial"))
                else:
                    self.audit_override = lambda report: report["endpoint_bundle"].update(endpoint_policy_files={})
                self.write_plan()
                with self.assertRaises(entry.OrchestrationError):
                    entry.StateDistillationDev(self.args)
        self.assertFalse(any(item.startswith("train_") for item in self.events))

    def test_swapped_or_same_caches_are_rejected_before_first_training(self):
        self.args.teacher_cache, self.args.student_cache = self.args.student_cache, self.args.teacher_cache
        with self.assertRaisesRegex(entry.OrchestrationError, "cache role"):
            entry.StateDistillationDev(self.args)
        self.args.teacher_cache = self.args.student_cache
        with self.assertRaisesRegex(entry.OrchestrationError, "distinct velocity caches"):
            entry.StateDistillationDev(self.args)
        self.assertFalse(any(item.startswith("train_") for item in self.events))

    def test_changed_input_binding_cannot_resume(self):
        entry.StateDistillationDev(self.args)
        entry.jwrite(self.caches["student"], {"role": "student", "changed_bytes": True})
        with self.assertRaisesRegex(entry.OrchestrationError, "inputs changed"):
            entry.StateDistillationDev(self.args)

    def test_incomplete_unmarked_stage_and_wrong_final_step_cannot_resume(self):
        instance = entry.StateDistillationDev(self.args)
        training = instance.driver.art / "train_continued_qad"
        training.mkdir()
        with self.assertRaisesRegex(entry.OrchestrationError, "incomplete stage"):
            instance.run()
        self.assertEqual(instance.driver.state["stages"], {})
        self.args.adopt_complete = True
        self.train(instance.driver, training, "train_continued_qad", self.base, 1e-4, 5, **instance.options("continued"))
        entry.jwrite(training / "runtime_metrics.json", {"status": "completed", "global_steps": 4})
        with self.assertRaisesRegex(entry.OrchestrationError, "incomplete final checkpoint"):
            entry.StateDistillationDev(self.args).run()

    def test_changed_raw_outcome_and_summary_are_detected(self):
        instance = entry.StateDistillationDev(self.args)
        instance.run()
        result_path = instance.driver.art / "dev_teacher_state_kd/task_results.json"
        original = result_path.read_bytes()
        data = entry.jread(result_path)
        data[entry.TASKS[0]]["results"][0] = True
        entry.jwrite(result_path, data)
        with self.assertRaises((entry.OrchestrationError, ValueError)):
            instance.paired_report()
        result_path.write_bytes(original)
        destination = instance.driver.art / "paired_summary"
        summary = entry.jread(destination / "paired_comparison.json")
        summary["selection_performed"] = True
        entry.jwrite(destination / "paired_comparison.json", summary)
        with self.assertRaisesRegex(entry.OrchestrationError, "differs from recomputed"):
            instance.verify_summary(destination)

    def test_different_reset_identity_is_rejected_even_if_each_arm_is_valid(self):
        instance = entry.StateDistillationDev(self.args)
        instance.run()
        folder = instance.driver.art / "dev_student_state_opd"
        task = entry.TASKS[0]
        log = folder / (task + ".log")
        log.write_text(log.read_text().replace("b" * 64, "d" * 64))
        rows = entry.jread(folder / "task_results.json")
        rows[task].update(driver_module.parse_log(log))
        entry.jwrite(folder / "task_results.json", rows)
        # Its own raw outcomes remain valid, but cross-arm pairing must fail.
        instance.driver.eval_verify(folder, "development", instance.driver.art / "merge_student_state_opd")
        with self.assertRaisesRegex(entry.OrchestrationError, "different environment reset"):
            instance.paired_report()

    def test_export_cannot_be_substituted_across_arms(self):
        instance = entry.StateDistillationDev(self.args)
        instance.run()
        path = instance.driver.art / "merge_student_state_opd/merge_manifest.json"
        entry.jwrite(path, {"training_checkpoint": str(instance.driver.art / "train_teacher_state_kd/checkpoint-5")})
        with self.assertRaisesRegex(entry.OrchestrationError, "another continuation arm"):
            instance.paired_report()

    def test_output_cannot_overlap_frozen_evidence(self):
        for target in (self.bundle, self.bundle / "run", self.root, self.initial.parent / "run",
                       self.root / "teacher_views" / "output"):
            with self.subTest(target=target), self.assertRaises(entry.OrchestrationError):
                entry.external_output(target, self.bundle, self.plan, self.caches)

    def test_cli_has_no_heldout_selection_or_budget_override(self):
        mandatory = ["--run-dir", "run", "--protocol-file", "protocol", "--ptq-selection", "selection",
            "--endpoint-bundle", "bundle", "--teacher-cache", "teacher", "--student-cache", "student"]
        import contextlib
        import io
        for option in ("--until", "--initial", "--base", "--weight", "--steps"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.parse_args(mandatory + [option, "heldout"])


if __name__ == "__main__":
    unittest.main()
