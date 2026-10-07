"""Complete CPU fixtures for matched observation/seed plans; no model loads."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from eval.run_recovery_eval import TASKS, capture_sampling_config, finalize_capture_samples, parse_log
from exp.action_chunk_diagnostics import checkpoint_files
from exp.derive_capture_views import audit_training_view, derive_views
from exp.probe_endpoint_plan import endpoint_config, freeze_endpoint_plan
from exp import probe_endpoint_plan as planner
from exp.run_high_fp4_v3 import capture_dataset_id, model_id


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_read(path):
    return json.loads(path.read_text())


class ProbeEndpointPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = self.root / "protocol.json"
        capture = {"mode": "full_trajectory_candidates", "every_server_calls": 4,
                   "safety_candidates_per_episode": 181}
        self.config = {"view_mode": "stratified",
            "pairing_rule": "task_init_state_teacher_success_student_any",
            "identity_order": "task_order_then_init_state_index",
            "window_pairing": "episode_call_rank_prefix_min", "windows_per_identity": 4,
            "minimum_common_identities_per_task": 2, "minimum_pairs_per_task": 2,
            "maximum_pairs_per_task": 16, "endpoint_seed": 800, "velocity_seed": 900,
            "seed_rule": "base_plus_global_pair_index"}
        self.protocol_data = {"capture_views": {"modes": ["head", "stratified"], "windows_per_episode": 4},
            "selection": {"qad_optimizer_steps": 2000, "qad_learning_rates": [5e-5], "rank": 32,
                "alpha": 64, "effective_demo_batch": 16, "train_seed": 20261006,
                "opd_every": 1, "recovery_scope": "all_ordinary_linear"},
            "endpoint_distillation": self.config, "partitions": {
                "teacher_supervision": {"seed": 950000, "episodes_per_task": 3,
                    "init_state_indices": [20, 21, 22], "capture_sampling": capture},
                "collection": {"seed": 960000, "episodes_per_task": 3,
                    "init_state_indices": [20, 21, 22], "capture_sampling": capture},
                "development": {"seed": 910000, "episodes_per_task": 2, "init_state_indices": [4, 5]},
                "heldout": {"seed": 970000, "episodes_per_task": 2, "init_state_indices": [24, 25]}}}
        self.protocol_data.update(copy.deepcopy(getattr(self, "protocol_overrides", {})))
        write(self.protocol, self.protocol_data)
        self.teacher, self.base, self.training, self.qad = (self.root / name for name in (
            "teacher", "base", "training/checkpoint-2000", "qad"))
        for checkpoint in (self.teacher, self.base, self.training, self.qad):
            checkpoint.mkdir(parents=True)
            for name in ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json"):
                write(checkpoint / name, {"fixture": name})
            (checkpoint / "model.safetensors").write_bytes(checkpoint.name.encode())
        write(self.base / "ptq_recipe.json", {"fixture": "frozen W4A4 PTQ"})
        self.teacher_source = self.make_source("teacher_supervision", self.teacher, (17, 9, 5), [True, True, False])
        self.teacher_views, self.student_views = self.root / "teacher_views", self.root / "student_views"
        derive_views(self.teacher_source, self.teacher_views)
        capture = self.teacher_views / "stratified"
        audit = audit_training_view(capture, self.protocol, self.teacher)
        dataset = capture_dataset_id(capture)
        numeric_env = {"QAD_STEPS": "2000", "QAD_LR": "5e-05", "QAD_LORA_R": "32",
            "QAD_LORA_ALPHA": "64.0", "QAD_LORA_SCOPE": "all_ordinary_linear", "TRAIN_SEED": "20261006",
            "OPD_EVERY": "1", "QAD_OPD_MSE_W": "0.0", "FP4VLA_QUANT": "0", "FP4VLA_W4A4": "1",
            "FP4VLA_W4A4_ADAPTER": "0", "FP4VLA_SATURATE_F16_ACTIVATIONS": "0",
            "QAD_W4A4": "1", "QAD_ACTIVATION_CHECKPOINTING": "1"}
        recovery = {"base": str(self.base), "w4a4_enabled": True, "protocol_sha256": sha(self.protocol),
                    "protocol_file": str(self.protocol), "rank": 32, "alpha": 64.0,
                    "scope": "all_ordinary_linear", "train_seed": 20261006, "seed": 20261006,
                    "initial_adapter": None, "optimizer_resumed": False, "probe_weight": 0.0,
                    "probe_every": 1, "probe_cache": None, "probe_cache_sha256": None,
                    "probe_action_mask": None, "objective": "demo flow loss", "micro_batch": 1,
                    "gradient_accumulation_steps": 16, "effective_global_batch": 16,
                    "execution_mode": "W4A4 numerical QDQ + BF16 LoRA residual",
                    "capture_dataset": str(capture), "capture_dataset_samples": audit["sample_count"],
                    "capture_dataset_sha256": planner.canonical_sha(dataset),
                    "environment_summary": {"variables": numeric_env},
                    "base_config_sha256": sha(self.base / "config.json"),
                    "base_statistics_sha256": sha(self.base / "statistics.json"),
                    "base_recipe_sha256": sha(self.base / "ptq_recipe.json")}
        for checkpoint in (self.training, self.training.parent, self.qad):
            write(checkpoint / "recovery_manifest.json", recovery)
        env = {**numeric_env, "QAD_OUT": str(self.training.parent), "QAD_CAPTURE_DATASET": str(capture),
            "QAD_CAPTURE_DATASET_SHA256": planner.canonical_sha(dataset), "QAD_CAPTURE_TEACHER": str(self.teacher),
            "QAD_CAPTURE_AUDIT_SHA256": planner.canonical_sha(audit), "QAD_GLOBAL_BATCH": "16", "QAD_MICRO_BATCH": "1"}
        write(self.training.parent / "orchestrator_training_request.json", {"protocol_sha256": sha(self.protocol),
            "base_identity": model_id(self.base), "initial_adapter_identity": None, "cache_identity": None,
            "capture_dataset_identity": dataset, "teacher_capture_identity": audit, "environment": env})
        write(self.training.parent / "runtime_metrics.json", {
            "status": "completed", "global_steps": 2000, "requested_optimizer_steps": 2000})
        weight = lambda folder: {"model.safetensors": {"bytes": (folder / "model.safetensors").stat().st_size,
                                                       "sha256": sha(folder / "model.safetensors")}}
        write(self.qad / "merge_manifest.json", {"status": "complete", "base": str(self.base),
            "training_checkpoint": str(self.training), "base_weights": weight(self.base),
            "training_weights": weight(self.training), "recovery_manifest": recovery,
            "recovery_manifest_source": str(self.training / "recovery_manifest.json"),
            "recovery_manifest_sha256": sha(self.training / "recovery_manifest.json")})
        self.student_source = self.make_source("collection", self.qad, (9, 17, 5), [False, True, False])
        derive_views(self.student_source, self.student_views)

    def rewrite_recovery(self, recovery):
        for checkpoint in (self.training, self.training.parent, self.qad):
            write(checkpoint / "recovery_manifest.json", recovery)
        merge = json_read(self.qad / "merge_manifest.json")
        merge.update(recovery_manifest=recovery, recovery_manifest_sha256=sha(self.training / "recovery_manifest.json"))
        write(self.qad / "merge_manifest.json", merge)

    def check_qad_identity(self):
        return planner.qad_identity(self.qad, sha(self.protocol), self.protocol_data,
                                    self.teacher_views / "stratified", self.teacher)

    def make_source(self, purpose, checkpoint, lengths, outcomes, *, name=None, bank_salt="", mask_width=7):
        source = self.root / (name or purpose)
        source.mkdir()
        part = self.protocol_data["partitions"][purpose]
        teacher = purpose == "teacher_supervision"
        kind, role = ("teacher_rollout", "teacher") if teacher else ("student_rollout", "student")
        config = capture_sampling_config(part, purpose, 3)
        switches = {"FP4VLA_QUANT": "0", "FP4VLA_W4A4": "0" if teacher else "1",
                    "FP4VLA_W4A4_ADAPTER": "0" if teacher else "1",
                    "FP4VLA_SATURATE_F16_ACTIVATIONS": "0" if teacher else "1"}
        write(source / "eval_manifest.json", {"purpose": purpose, "tasks": TASKS, "n_envs": 1,
            "episodes": 3, "protocol_file": str(self.protocol), "protocol_sha256": sha(self.protocol),
            "init_state_indices": part["init_state_indices"], "seed": part["seed"], "max_episode_steps": 720,
            "checkpoint": str(checkpoint), "capture_sampling": config,
            "checkpoint_files": checkpoint_files(checkpoint), "environment_summary": {"variables": switches}})
        task_results = {}
        for ti, task in enumerate(TASKS):
            seed = part["seed"] + 1000 * ti
            folder = source / "observations" / task
            folder.mkdir(parents=True)
            capture = {"task_name": task, "purpose": purpose, "source_kind": kind, "checkpoint_role": role,
                "student_checkpoint": str(checkpoint), "student_config_sha256": sha(checkpoint / "config.json"),
                "student_statistics_sha256": sha(checkpoint / "statistics.json"),
                "protocol_sha256": sha(self.protocol), "seed": seed, "init_state_indices": "20,21,22",
                "sampling_mode": "full_trajectory_candidates", "n_envs": 1, "every_server_calls": 4,
                "per_episode_limit": 181, "per_task_limit": 543, "total_limit": 543,
                "sampling_interval_basis": "episode_call", "candidate_prefix": "candidate_"}
            write(folder / "capture_manifest.json", capture)
            resets = [{"task_name": task, "episode_index": ep, "seed": seed + ep,
                "init_state_index": 20 + ep, "settle_steps": 10,
                "initial_state_sha256": hashlib.sha256(f"initial {ti} {ep}".encode()).hexdigest(),
                "restored_state_sha256": hashlib.sha256(f"restored {seed} {ep}".encode()).hexdigest(),
                "init_state_bank_sha256": hashlib.sha256(f"bank {ti}{bank_salt}".encode()).hexdigest()} for ep in range(3)]
            log = source / f"{task}.log"
            log.write_text("".join("FP4VLA_EPISODE_RESET " + json.dumps(reset) + "\n" for reset in resets)
                           + f"results: ('libero_sim/{task}', {outcomes})\n")
            result = {**parse_log(log), "seed": seed, "returncode": 0}
            count = prefix = 0
            for ep, queries in enumerate(lengths):
                for call in range(1, queries + 1, 4):
                    action = torch.zeros(1, 16, 8)
                    mask = torch.ones_like(action)
                    mask[:, :, mask_width:] = 0
                    inputs = {"embodiment_id": torch.zeros(1, dtype=torch.int64),
                        "state": torch.full((1, 1, 8), ep, dtype=torch.bfloat16),
                        "input_ids": torch.tensor([[ti, count]], dtype=torch.int64),
                        "attention_mask": torch.ones(1, 2, dtype=torch.int64),
                        "pixel_values": torch.zeros(4, 3, dtype=torch.bfloat16),
                        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
                        "action": action, "action_mask": mask}
                    sample = {"source_kind": kind, "checkpoint_role": role,
                        "sampling_mode": "full_trajectory_candidates", "task_name": task,
                        "student_checkpoint": str(checkpoint), "student_statistics_sha256": capture["student_statistics_sha256"],
                        "episode_index": ep, "episode_seed": seed + ep, "init_state_index": 20 + ep,
                        "reset_identity": resets[ep], "episode_success": None, "capture_index": count,
                        "episode_call": call, "server_call": prefix + call, "inputs": inputs}
                    torch.save(sample, folder / f"candidate_{count:06d}.pt")
                    count += 1
                prefix += queries
            write(folder / "capture_counts.json", {"calls": sum(lengths), "saved": count,
                "candidates_saved": count, "episode_query_counts": {str(ep): length for ep, length in enumerate(lengths)},
                "safety_limit_hit": False})
            result["capture"] = finalize_capture_samples(folder, task, result, purpose, "full_trajectory_candidates")
            task_results[task] = result
        write(source / "task_results.json", task_results)
        write(source / "summary.json", {"tasks_complete": 10, "total_episodes": 30,
            "total_successes": 10 * sum(outcomes), "purpose": purpose, "checkpoint_files_verified_unchanged": True})
        return source

    def freeze(self, **changes):
        values = {"teacher_view": self.teacher_views / "stratified", "student_view": self.student_views / "stratified",
                  "protocol_file": self.protocol, "qad_checkpoint": self.qad, "teacher_checkpoint": self.teacher}
        values.update(changes)
        return freeze_endpoint_plan(**values)

    def test_complete_real_double_chain_matches_rank_budget_and_retains_failed_states(self):
        before = {str(p): sha(p) for root in (self.teacher_source, self.student_source, self.teacher_views, self.student_views)
                  for p in root.rglob("*") if p.is_file()}
        plan = self.freeze()
        self.assertEqual(plan["pairs_per_arm"], 60)
        self.assertTrue(all(row["pairs"] == 6 for row in plan["task_budgets"].values()))
        self.assertEqual(sum(not pair["student"]["provenance"]["episode_success"] for pair in plan["pairs"]), 30)
        self.assertEqual(plan["qad"]["checkpoint_files"], checkpoint_files(self.qad))
        for index, pair in enumerate(plan["pairs"]):
            self.assertEqual(pair["endpoint_seed"], 800 + index)
            self.assertEqual(pair["velocity_seed"], 900 + index)
            self.assertEqual(pair["teacher"]["source_kind"], "teacher_rollout")
            self.assertEqual(pair["student"]["source_kind"], "student_rollout")
            self.assertNotEqual(pair["teacher"]["provenance"]["episode_seed"], pair["student"]["provenance"]["episode_seed"])
            self.assertEqual(pair["teacher"]["source_checkpoint"], str(self.teacher))
            self.assertEqual(pair["student"]["source_checkpoint"], str(self.qad))
        self.assertEqual(before, {path: sha(Path(path)) for path in before})
        self.assertEqual(plan, self.freeze(teacher_view=self.teacher_views / "stratified/observations",
                                           student_view=self.student_views / "stratified/observations"))
        self.assertFalse(torch.cuda.is_initialized())

    def test_output_is_new_external_json_and_default_writes_nothing(self):
        output = self.root / "plans/frozen.json"
        plan = self.freeze(out=output)
        self.assertEqual(json_read(output), plan)
        with self.assertRaises(FileExistsError):
            self.freeze(out=output)
        with self.assertRaisesRegex(ValueError, "outside source evidence"):
            self.freeze(out=self.teacher_source / "endpoint.json")

    def test_wrong_mode_or_source_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "single head/stratified"):
            self.freeze(student_view=self.student_views / "head")
        with self.assertRaises(ValueError):
            self.freeze(teacher_view=self.student_views / "stratified", student_view=self.teacher_views / "stratified")

    def test_changed_raw_log_or_derived_bytes_is_rejected(self):
        log = self.student_source / f"{TASKS[0]}.log"
        original = log.read_bytes()
        log.write_bytes(original + b"changed\n")
        with self.assertRaises(ValueError):
            self.freeze()
        log.write_bytes(original)
        sample = next((self.student_views / "stratified/observations").rglob("sample_*.pt"))
        payload = torch.load(sample, map_location="cpu", weights_only=True)
        payload["inputs"]["state"].add_(1)
        torch.save(payload, sample)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_unselected_candidate_change_is_rejected(self):
        sample = self.student_source / "observations" / TASKS[0] / "candidate_000007.pt"
        payload = torch.load(sample, map_location="cpu", weights_only=True)
        payload["inputs"]["state"].add_(1)
        torch.save(payload, sample)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_shared_indices_from_different_initial_state_banks_are_rejected(self):
        source = self.make_source("collection", self.qad, (9, 17, 5), [False, True, False],
                                  name="different_bank", bank_salt="changed")
        views = self.root / "different_bank_views"
        derive_views(source, views)
        with self.assertRaisesRegex(ValueError, "bank identity differs"):
            self.freeze(student_view=views / "stratified")

    def test_different_valid_action_masks_are_rejected(self):
        source = self.make_source("collection", self.qad, (9, 17, 5), [False, True, False],
                                  name="different_mask", mask_width=6)
        views = self.root / "different_mask_views"
        derive_views(source, views)
        with self.assertRaisesRegex(ValueError, "valid-action mask differs"):
            self.freeze(student_view=views / "stratified")

    def test_pair_budget_bounds_are_enforced_without_dropping_windows(self):
        for field, value in (("minimum_pairs_per_task", 7), ("maximum_pairs_per_task", 5),
                             ("minimum_common_identities_per_task", 3)):
            with self.subTest(field=field):
                # Keep the full provenance fixture frozen while independently
                # exercising the post-audit accounting gates with strict bounds.
                changed = {**self.config, field: value}
                with patch.object(planner, "endpoint_config", return_value=changed):
                    with self.assertRaises(ValueError):
                        self.freeze()

    def test_capture_time_weight_identity_is_required_and_cannot_be_replaced(self):
        manifest_path = self.student_source / "eval_manifest.json"
        original = json_read(manifest_path)
        missing = dict(original)
        missing.pop("checkpoint_files")
        write(manifest_path, missing)
        with self.assertRaises(ValueError):
            self.freeze()
        write(manifest_path, original)
        (self.training / "model.safetensors").write_bytes(b"changed real loaded adapter")
        with self.assertRaises(ValueError):
            self.freeze()

    def test_protocol_cannot_be_relabeled_after_source_collection(self):
        changed = copy.deepcopy(self.protocol_data)
        changed["endpoint_distillation"]["endpoint_seed"] += 1
        write(self.protocol, changed)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_config_requires_explicit_budgets_rules_and_integer_seeds(self):
        for field, value in (("endpoint_seed", True), ("velocity_seed", -1),
                             ("minimum_pairs_per_task", 0), ("maximum_pairs_per_task", 1),
                             ("windows_per_identity", 5), ("window_pairing", "choose_best")):
            with self.subTest(field=field):
                protocol = copy.deepcopy(self.protocol_data)
                protocol["endpoint_distillation"][field] = value
                with self.assertRaises(ValueError):
                    endpoint_config(protocol)
        for remove in (False, True):
            protocol = copy.deepcopy(self.protocol_data)
            if remove:
                protocol["endpoint_distillation"].pop("maximum_pairs_per_task")
            else:
                protocol["endpoint_distillation"]["unknown"] = 1
            with self.assertRaises(ValueError):
                endpoint_config(protocol)

    def test_opd_and_continued_checkpoints_cannot_be_the_pure_qad_start(self):
        original = json_read(self.training / "recovery_manifest.json")
        for change in ({"probe_weight": 0.25}, {"initial_adapter": str(self.root / "prior_qad")},
                       {"optimizer_resumed": True}):
            with self.subTest(change=change):
                self.rewrite_recovery({**original, **change})
                with self.assertRaisesRegex(ValueError, "must be pure QAD"):
                    self.check_qad_identity()

    def test_head_trained_qad_cannot_be_the_stratified_start(self):
        recovery = json_read(self.training / "recovery_manifest.json")
        recovery["capture_dataset"] = str(self.teacher_views / "head")
        self.rewrite_recovery(recovery)
        with self.assertRaisesRegex(ValueError, "stratified view"):
            self.check_qad_identity()

    def test_training_budget_rank_scope_and_actual_learning_rate_match_protocol(self):
        original = json_read(self.training / "recovery_manifest.json")
        for field, value in (("rank", 16), ("scope", "head"), ("train_seed", 42), ("alpha", 32),
                             ("effective_global_batch", 8), ("probe_every", 4)):
            with self.subTest(field=field):
                self.rewrite_recovery({**original, field: value})
                with self.assertRaisesRegex(ValueError, "frozen protocol training settings"):
                    self.check_qad_identity()
        self.rewrite_recovery(original)
        runtime = self.training.parent / "runtime_metrics.json"
        write(runtime, {"status": "completed", "global_steps": 1900, "requested_optimizer_steps": 2000})
        with self.assertRaisesRegex(ValueError, "update budget"):
            self.check_qad_identity()
        write(runtime, {"status": "completed", "global_steps": 2000, "requested_optimizer_steps": 2000})
        request_path = self.training.parent / "orchestrator_training_request.json"
        request = json_read(request_path)
        request["environment"]["QAD_LR"] = "0.0001"
        write(request_path, request)
        changed = copy.deepcopy(original)
        changed["environment_summary"]["variables"]["QAD_LR"] = "0.0001"
        self.rewrite_recovery(changed)
        with self.assertRaisesRegex(ValueError, "setting differs from protocol: QAD_LR"):
            self.check_qad_identity()

    def test_training_capture_identity_and_audit_must_match_real_stratified_view(self):
        original = json_read(self.training / "recovery_manifest.json")
        self.rewrite_recovery({**original, "capture_dataset_sha256": "a" * 64})
        with self.assertRaisesRegex(ValueError, "dataset identity differs"):
            self.check_qad_identity()
        self.rewrite_recovery(original)
        request_path = self.training.parent / "orchestrator_training_request.json"
        request = json_read(request_path)
        request["environment"]["QAD_CAPTURE_AUDIT_SHA256"] = "b" * 64
        write(request_path, request)
        with self.assertRaisesRegex(ValueError, "pure stratified starting point"):
            self.check_qad_identity()


if __name__ == "__main__":
    unittest.main()
