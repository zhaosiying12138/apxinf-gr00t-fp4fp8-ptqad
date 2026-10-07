"""CPU teacher-cache checks for explicit matched endpoint bundles."""
import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
import opd_probe_cache as cache_builder
from gr00t_runtime import action_mask
from probe_distill import CACHE_VERSION, ProbeAnchor, flow_signature, prediction, replay_context, require_full_model


class ToyFullTeacher(nn.Module):
    """Small stochastic flow with the complete architecture contract exposed."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.75))
        self.dropout = nn.Dropout(0.5)
        self.backbone = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(layers=range(16))))
        self.action_head = SimpleNamespace(model=SimpleNamespace(transformer_blocks=range(32)),
            vl_self_attention=SimpleNamespace(transformer_blocks=range(4)))
        self.config = SimpleNamespace(noise_beta_alpha=1.5, noise_beta_beta=1.0, noise_s=0.999,
            num_timestep_buckets=1000, action_horizon=3, max_action_dim=4, state_history_length=1)
        self.seen = []

    def forward(self, inputs):
        action = inputs["action"]
        self.seen.append({key: value.clone() for key, value in inputs.items()})
        noise = torch.randn_like(action)
        t = torch.distributions.Beta(1.5, 1.0).sample((1,)).reshape(1, 1, 1)
        # Padding affects the full forward, even though it is excluded from MSE.
        value = (1 - t) * noise + t * action + action.mean() + inputs["state"].float().mean()
        return {"pred_actions": self.dropout(value) * self.weight}


class EndpointCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec = {"horizon": 2, "dimensions": 2}
        self.model = ToyFullTeacher()

    def source(self, role="student"):
        source = "student_rollout" if role == "student" else "teacher_rollout"
        samples = []
        for index in range(4):
            action = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + index
            samples.append({"source_kind": source, "checkpoint_role": role,
                "student_checkpoint": "/source/qad" if role == "student" else "/source/bf16",
                "task_name": "task_a" if index < 2 else "task_b", "episode_index": index // 2,
                "endpoint": {"pair_index": index, "endpoint_seed": 100 + index,
                    "velocity_seed": 713 + index * 7, "policy": "/endpoint/common_qad"},
                "inputs": {"action": action, "action_mask": action_mask(action, self.spec),
                    "state": torch.full((1, 1, 3), index, dtype=torch.bfloat16),
                    "input_ids": torch.tensor([[index, 3]], dtype=torch.int64)}})
        provenance = {"schema": "fp4vla_probe_endpoint_bundle_v1", "root": "/frozen/bundle",
            "role": role, "source_kind": source, "plan_sha256": "a" * 64,
            "protocol_sha256": "b" * 64, "pairs_per_arm": 4,
            "task_budgets": {"task_a": {"pairs": 2}, "task_b": {"pairs": 2}},
            "endpoint_policy_checkpoint": "/endpoint/common_qad", "endpoint_policy_files": {"weight": "sha"},
            "source_observation_files": [{"path": f"/source/{role}/{i}.pt", "sha256": "c" * 64} for i in range(4)],
            "source_endpoint_files": [{"path": f"/endpoint/{role}/{i}.pt", "sha256": "d" * 64} for i in range(4)]}
        return samples, provenance

    def label(self, raw=None, provenance=None, **kwargs):
        if raw is None:
            raw, provenance = self.source()
        return cache_builder.label_observations(self.model, raw, action_spec=self.spec,
            autocast_dtype="none", endpoint_bundle=provenance, **kwargs)

    def test_cli_forbids_implicit_pair_budget_or_seed_override_and_source_mix(self):
        args = cache_builder.parse_args(["--endpoint-bundle", "/bundle", "--endpoint-role", "teacher"])
        self.assertIsNone(args.count)
        self.assertIsNone(args.seed)
        bad = (["--endpoint-bundle", "/bundle"], ["--endpoint-role", "student"],
               ["--input-dir", "/input", "--endpoint-bundle", "/bundle", "--endpoint-role", "student"],
               ["--endpoint-bundle", "/bundle", "--endpoint-role", "student", "--count", "8"],
               ["--endpoint-bundle", "/bundle", "--endpoint-role", "student", "--seed", "20260929"])
        for argv in bad:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cache_builder.parse_args(argv)
        legacy = cache_builder.parse_args([])
        self.assertEqual((legacy.count, legacy.seed), (8, 20260929))
        explicit = cache_builder.parse_args(["--input-dir", "/input", "--count", "11", "--seed", "17"])
        self.assertEqual((explicit.count, explicit.seed), (11, 17))

    def test_endpoint_source_loader_retains_teacher_role_and_separate_endpoint_identity(self):
        raw, provenance = self.source("teacher")
        loader = Mock(return_value=(raw, provenance))
        with patch.dict(sys.modules, {"exp.probe_endpoint_bundle": SimpleNamespace(load_endpoint_bundle=loader)}):
            result, metadata = cache_builder.prepare_observation_source(
                self.root, endpoint_bundle="/bundle", endpoint_role="teacher")
        loader.assert_called_once_with("/bundle", "teacher", teacher_checkpoint=self.root)
        self.assertIs(result, raw)
        self.assertEqual(metadata["source_kind"], "teacher_rollout")
        self.assertEqual(metadata["endpoint_bundle"], provenance)
        self.assertNotIn("student_checkpoint", metadata)
        self.assertEqual(raw[0]["student_checkpoint"], "/source/bf16")
        self.assertEqual(metadata["endpoint_bundle"]["endpoint_policy_checkpoint"], "/endpoint/common_qad")
        self.assertEqual(metadata["requested_count"], 4)
        with self.assertRaisesRegex(ValueError, "forbid count/seed"):
            cache_builder.prepare_observation_source(self.root, endpoint_bundle="/bundle", endpoint_role="student", count=4)

    def test_full_inputs_and_velocity_seeds_are_exact_rng_and_mode_are_preserved(self):
        raw, provenance = self.source()
        original = copy.deepcopy(raw)
        self.model.train()
        self.model.dropout.eval()
        rng = torch.random.get_rng_state().clone()
        result = self.label(raw, provenance)
        self.assertEqual(len(result), 4)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.dropout.training)
        for index, record in enumerate(result):
            self.assertEqual(record["seed"], raw[index]["endpoint"]["velocity_seed"])
            self.assertEqual(record["provenance"], {k: v for k, v in raw[index].items() if k != "inputs"})
            for key, tensor in original[index]["inputs"].items():
                self.assertEqual(record["inputs"][key].dtype, tensor.dtype)
                self.assertTrue(torch.equal(record["inputs"][key], tensor))
                self.assertTrue(torch.equal(raw[index]["inputs"][key], tensor))
                self.assertTrue(torch.equal(self.model.seen[index][key], tensor))
            with replay_context(self.model, record["seed"], "none"), torch.no_grad():
                expected = prediction(self.model(original[index]["inputs"]))
            self.assertTrue(torch.equal(record["pred"], expected))
            self.assertFalse(record["pred"].requires_grad)

    def test_equal_pair_budget_and_velocity_seeds_for_both_observation_roles(self):
        teacher, teacher_provenance = self.source("teacher")
        student, student_provenance = self.source("student")
        left = self.label(teacher, teacher_provenance)
        right = self.label(student, student_provenance)
        self.assertEqual([s["seed"] for s in left], [s["seed"] for s in right])
        self.assertEqual(len(left), len(right))
        self.assertTrue(all(s["provenance"]["source_kind"] == "teacher_rollout" for s in left))
        self.assertTrue(all(s["provenance"]["source_kind"] == "student_rollout" for s in right))

    def test_clipped_reordered_or_malformed_budget_fails_before_teacher_forward(self):
        changes = [lambda raw, p: raw.pop(), lambda raw, p: raw.reverse(),
            lambda raw, p: p["task_budgets"]["task_a"].update(pairs=1),
            lambda raw, p: p["source_endpoint_files"].pop(),
            lambda raw, p: raw[0]["endpoint"].update(velocity_seed=None),
            lambda raw, p: raw[0].update(source_kind="teacher_rollout")]
        for change in changes:
            raw, provenance = self.source()
            change(raw, provenance)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.label(raw, provenance)
        self.assertEqual(self.model.seen, [])
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            self.label(seed=713)

    def test_mask_padding_architecture_and_nonfinite_output_remain_guarded(self):
        raw, provenance = self.source()
        raw[0]["inputs"]["action_mask"][0, 2, 3] = 1
        with self.assertRaisesRegex(ValueError, "frozen LIBERO"):
            self.label(raw, provenance)
        self.assertEqual(self.model.seen, [])
        raw, provenance = self.source()
        raw[0]["inputs"]["action_mask"] = raw[0]["inputs"]["action_mask"].to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "frozen LIBERO"):
            self.label(raw, provenance)
        self.model.backbone.model.language_model.layers = range(15)
        with self.assertRaisesRegex(ValueError, "complete GR00T"):
            self.label()
        self.model.backbone.model.language_model.layers = range(16)
        with torch.no_grad():
            self.model.weight.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "Invalid teacher velocity"):
            self.label()

    def test_version_three_cache_replays_in_probe_anchor_with_flow_checks(self):
        raw, provenance = self.source("teacher")
        samples = self.label(raw, provenance)
        path = self.root / "cache.pt"
        torch.save({"version": CACHE_VERSION, "samples": samples, "metadata": {
            "source_kind": "teacher_rollout", "endpoint_bundle": provenance,
            "architecture": require_full_model(self.model), "flow_config": flow_signature(self.model),
            "model_dtype": "float32", "autocast_dtype": "none"}}, path)
        anchor = ProbeAnchor(path, expected_sha256=cache_builder.file_sha256(path))
        anchor.validate_model(self.model)
        for index, sample in enumerate(samples):
            with replay_context(self.model, sample["seed"], "none"):
                self.assertEqual(anchor.loss(self.model, index).item(), 0)
        wrong = ToyFullTeacher()
        wrong.config.noise_s = 0.5
        with self.assertRaisesRegex(ValueError, "flow/noise"):
            ProbeAnchor(path, expected_sha256=cache_builder.file_sha256(path)).validate_model(wrong)

    def test_legacy_input_round_robin_count_and_sequential_seed_are_unchanged(self):
        teacher, student, captures = (self.root / name for name in ("teacher", "student", "captures"))
        for checkpoint in (teacher, student):
            checkpoint.mkdir()
            (checkpoint / "statistics.json").write_text("{}")
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model.safetensors").write_bytes(b"fixture")
        stats = cache_builder.file_sha256(teacher / "statistics.json")
        for task in ("a", "b"):
            folder = captures / task
            folder.mkdir(parents=True)
            for index in range(2):
                raw = self.source()[0][index]
                raw.pop("endpoint")
                raw.update(student_checkpoint=str(student), student_statistics_sha256=stats,
                           task_name=task, sample_index=index)
                torch.save(raw, folder / f"sample_{index:06d}.pt")
        raw, metadata = cache_builder.prepare_observation_source(teacher, input_dir=captures, count=3, seed=41)
        self.assertEqual([(s["task_name"], s["sample_index"]) for s in raw], [("a", 0), ("b", 0), ("a", 1)])
        self.assertEqual(metadata["student_checkpoint"], str(student))
        self.assertEqual(metadata["requested_count"], 3)
        samples = cache_builder.label_observations(self.model, raw, action_spec=self.spec, seed=41, autocast_dtype="none")
        self.assertEqual([s["seed"] for s in samples], [41, 42, 43])

    def test_real_bundle_to_velocity_cache_to_training_guard_and_probe_anchor(self):
        from exp import probe_endpoint_bundle as bundle
        from rl.endpoint_cache_guard import prepare_endpoint_training
        from tests.test_probe_endpoint_bundle import SPEC, fixture, toy_loader

        selection = {"qad_optimizer_steps": 2000, "continuation_optimizer_steps": 2000,
            "qad_learning_rates": [5e-5], "rank": 32, "alpha": 64,
            "effective_demo_batch": 16, "train_seed": 20261006,
            "opd_every": 1, "recovery_scope": "all_ordinary_linear"}
        state_distillation = {"arms": ["continued_qad", "teacher_state_kd", "student_state_opd"],
            "starting_qad_view": "stratified", "teacher_velocity_weight": 0.25,
            "probe_every": 1, "optimizer_steps": 2000, "learning_rate": 5e-5, "max_grad_norm": 0.25}
        source, plan_file, plan = fixture(self, protocol_overrides={
            "selection": selection, "state_distillation": state_distillation})
        bundle_path = source.root / "endpoint_bundle"
        environment = {"QAD_CAPTURE_TEACHER": str(source.teacher),
            "QAD_INIT_ADAPTER": str(source.training), "GR00T_BASE_CKPT": str(source.base),
            "QAD_STEPS": "2000", "QAD_LR": "5e-05", "QAD_OPD_MSE_W": "0.25",
            "OPD_EVERY": "1", "QAD_MAX_GRAD_NORM": "0.25", "TRAIN_SEED": "20261006",
            "QAD_LORA_R": "32", "QAD_LORA_ALPHA": "64", "QAD_GLOBAL_BATCH": "16",
            "QAD_MICRO_BATCH": "1", "QAD_LORA_SCOPE": "all_ordinary_linear", "QAD_W4A4": "1",
            "FP4VLA_QUANT": "0", "FP4VLA_W4A4": "1", "FP4VLA_W4A4_ADAPTER": "1",
            "FP4VLA_SATURATE_F16_ACTIVATIONS": "1", "QAD_ACTIVATION_CHECKPOINTING": "1"}
        self.model.config.action_horizon = 16
        self.model.config.max_action_dim = 8
        caches, receipts = {}, {}
        # Only GR00T loading/spec parsing use a toy; capture, plan, bundle,
        # cache input auditing, stochastic velocity replay and guard are real.
        with toy_loader():
            bundle.generate_endpoint_bundle(plan_file, bundle_path, gr00t=source.root)
            for role in ("teacher", "student"):
                raw, metadata = cache_builder.prepare_observation_source(
                    source.teacher, endpoint_bundle=bundle_path, endpoint_role=role)
                samples = cache_builder.label_observations(self.model, raw,
                    action_spec=SPEC, endpoint_bundle=metadata["endpoint_bundle"], autocast_dtype="bfloat16")
                weights = {p.name: {"bytes": p.stat().st_size, "sha256": cache_builder.file_sha256(p)}
                           for p in source.teacher.glob("*.safetensors")}
                metadata.update(teacher=str(source.teacher), teacher_weights=weights,
                    teacher_config_sha256=cache_builder.file_sha256(source.teacher / "config.json"),
                    teacher_statistics_sha256=cache_builder.file_sha256(source.teacher / "statistics.json"),
                    labeling_implementation_sha256=cache_builder.file_sha256(cache_builder.__file__),
                    replay_implementation_sha256=cache_builder.file_sha256(Path(cache_builder.__file__).with_name("probe_distill.py")),
                    architecture=require_full_model(self.model), flow_config=flow_signature(self.model),
                    model_dtype="float32", autocast_dtype="bfloat16", eval_mode=True, action_mask=SPEC)
                cache_path = source.root / f"{role}_velocity.pt"
                torch.save({"version": CACHE_VERSION, "metadata": metadata, "samples": samples}, cache_path)
                frozen = metadata["endpoint_bundle"]["endpoint_policy_training_identity"]
                environment.update(QAD_CAPTURE_DATASET=frozen["capture_dataset"],
                    QAD_CAPTURE_DATASET_SHA256=frozen["capture_dataset_sha256"],
                    QAD_CAPTURE_AUDIT_SHA256=frozen["capture_audit_sha256"])
                env = {**environment, "QAD_ENDPOINT_ROLE": role, "OPD_CACHE_PATH": str(cache_path)}
                receipts[role] = prepare_endpoint_training(source.protocol, 0.25, env)
                anchor = ProbeAnchor(cache_path, expected_sha256=receipts[role]["cache_sha256"])
                anchor.validate_model(self.model)
                for index in (0, len(samples) - 1):
                    with replay_context(self.model, samples[index]["seed"], "bfloat16"):
                        self.assertEqual(anchor.loss(self.model, index).item(), 0)
                self.assertEqual(receipts[role]["sample_count"], plan["pairs_per_arm"])
                self.assertEqual(metadata["source_kind"], role + "_rollout")
                self.assertEqual(len(samples), 60)
                self.assertEqual(samples[0]["provenance"]["student_checkpoint"],
                                 str(source.teacher if role == "teacher" else source.qad))
                self.assertTrue(samples[0]["inputs"]["action"][:, :, 7:].abs().sum().item() > 0)
                caches[role] = (cache_path, samples, env)
            continued = prepare_endpoint_training(source.protocol, 0.0, {
                **environment, "QAD_ENDPOINT_ROLE": "continued", "QAD_OPD_MSE_W": "0.0",
                "QAD_ENDPOINT_BUNDLE": str(bundle_path)})
            self.assertEqual(continued["initial_adapter"], receipts["teacher"]["initial_adapter"])
            self.assertEqual(continued["initial_adapter"], receipts["student"]["initial_adapter"])
            self.assertEqual([s["seed"] for s in caches["teacher"][1]], [s["seed"] for s in caches["student"][1]])
            self.assertEqual(sum(not s["provenance"]["episode_success"] for s in caches["student"][1]), 30)
            wrong = {**caches["teacher"][2], "QAD_ENDPOINT_ROLE": "student"}
            with self.assertRaises(ValueError):
                prepare_endpoint_training(source.protocol, 0.25, wrong)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
