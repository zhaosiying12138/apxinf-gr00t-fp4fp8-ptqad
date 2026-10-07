"""CPU integration of real capture audits with complete toy action sampling."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from exp import probe_endpoint_bundle as bundle
from rl.capture_sampling import _require_same_sample
from rl.gr00t_runtime import action_mask
from rl.probe_distill import file_sha256
from tests import test_probe_endpoint_plan as plan_tests


SPEC = {"embodiment": "libero_sim", "horizon": 16, "dimensions": 7,
        "keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "key_dimensions": [1] * 7, "delta_indices": list(range(16)),
        "policy": "prefix horizon and concatenated processor action keys; exclude padding"}


class ToyEndpointFlow(nn.Module):
    """Full sampler whose observation-dependent output includes nonzero padding."""
    def __init__(self, shape=(1, 16, 8)):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.action_head = nn.Module()
        self.action_head.action_encoder = nn.Identity()
        self.action_head.num_inference_timesteps = 4
        self.shape = shape
        self.calls = []

    def forward(self, inputs):
        raise AssertionError("Endpoint generation must not call training velocity forward")

    def get_action(self, inputs):
        if "action" in inputs or "action_mask" in inputs:
            raise AssertionError("Old actions would accidentally enable RTC")
        if torch.is_autocast_enabled("cpu"):
            raise AssertionError("Formal inference has no outer autocast")
        self.calls.append({key: value.clone() for key, value in inputs.items()})
        action = torch.randn(self.shape)
        increment = 0.25 * self.weight + inputs["input_ids"].float().mean() / 10
        for _ in range(self.action_head.num_inference_timesteps):
            action = self.action_head.action_encoder(action) + increment
        return {"action_pred": action}


def fixture(testcase, protocol_overrides=None):
    """Reusable real endpoint source fixture; no source audits are mocked."""
    source = plan_tests.ProbeEndpointPlanTests()
    source.protocol_overrides = protocol_overrides or {}
    source.setUp()
    testcase.addCleanup(source.doCleanups)
    plan_file = source.root / "frozen_endpoint_plan.json"
    plan = source.freeze(out=plan_file)
    return source, plan_file, plan


def toy_runtime():
    return {"loader": "serve_recovery_local_callback", "rpc_opened": False,
            "source_files": {str(Path(__file__).resolve()): file_sha256(__file__)},
            "architecture": {"fixture": "CPU complete sampler"},
            "w4a4_activation_count": 1, "model_parameter_dtype": "torch.float32", "device": "cpu"}


@contextlib.contextmanager
def toy_loader(model=None):
    model = model or ToyEndpointFlow()

    def run(plan, gr00t, callback):
        callback(model)
        return toy_runtime()

    with patch.object(bundle, "libero_action_spec", return_value=copy.deepcopy(SPEC)), \
            patch.object(bundle, "_run_qad_policy", side_effect=run), \
            contextlib.redirect_stdout(io.StringIO()):
        yield model


class EndpointBundleTests(unittest.TestCase):
    def setUp(self):
        self.source, self.plan_file, self.plan = fixture(self)
        self.output = self.source.root / "bundle"
        self.model = ToyEndpointFlow().train()
        self.model.action_head.action_encoder.eval()
        self.context = toy_loader(self.model)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def generate(self):
        return bundle.generate_endpoint_bundle(self.plan_file, self.output, gr00t=self.source.root)

    def load(self, role="teacher"):
        return bundle.load_endpoint_bundle(self.output, role, teacher_checkpoint=self.source.teacher)

    def reseal_sample(self, role, mutate):
        path = self.output / role / "sample_000000.pt"
        raw = torch.load(path, weights_only=True)
        mutate(raw)
        torch.save(raw, path)
        manifest = bundle.read_json(self.output / "manifest.json")
        manifest["records"][0][role].update({key: value for key, value in bundle.identity(path).items() if key != "path"})
        plan_tests.write(self.output / "manifest.json", manifest)

    def test_real_audits_and_complete_sampling_preserve_origin_full_actions_and_rng(self):
        before = torch.random.get_rng_state().clone()
        report = self.generate()
        self.assertEqual(report["pairs_per_arm"], 60)
        self.assertEqual(len(self.model.calls), 120)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.action_head.action_encoder.training)
        self.assertEqual(len(self.model.action_head.action_encoder._forward_pre_hooks), 0)
        for role, checkpoint in (("teacher", self.source.teacher), ("student", self.source.qad)):
            raws, provenance = self.load(role)
            self.assertEqual(len(raws), 60)
            self.assertEqual(provenance["source_kind"], role + "_rollout")
            self.assertEqual(provenance["endpoint_policy_checkpoint"], str(self.source.qad))
            self.assertEqual(provenance["endpoint_policy_training_checkpoint"], str(self.source.training))
            self.assertEqual(provenance["pairs_per_arm"], 60)
            self.assertTrue(all(row["pairs"] == 6 for row in provenance["task_budgets"].values()))
            self.assertEqual(len(provenance["source_observation_files"]), 60)
            self.assertEqual(len(provenance["source_endpoint_files"]), 60)
            json.dumps(provenance, allow_nan=False)
            for index, (raw, pair) in enumerate(zip(raws, self.plan["pairs"])):
                self.assertEqual(raw["source_kind"], role + "_rollout")
                self.assertEqual(raw["student_checkpoint"], str(checkpoint))
                self.assertEqual(raw["endpoint"]["pair_index"], index)
                self.assertEqual(raw["endpoint"]["endpoint_seed"], 800 + index)
                self.assertEqual(raw["endpoint"]["velocity_seed"], 900 + index)
                json.dumps({key: value for key, value in raw.items() if key != "inputs"}, allow_nan=False)
                inputs = raw["inputs"]
                self.assertEqual(inputs["action"].dtype, torch.float32)
                self.assertEqual(inputs["action_mask"].dtype, torch.float32)
                self.assertEqual(inputs["action_mask"].sum().item(), 16 * 7)
                self.assertTrue(inputs["action"][:, :, 7:].abs().sum().item() > 0)
                original = torch.load(pair[role]["path"], weights_only=True)
                for key in inputs.keys() - {"action", "action_mask"}:
                    _require_same_sample(inputs[key], original["inputs"][key], key)
                trace = torch.load(self.output / raw["endpoint"]["trace"]["path"], weights_only=True)
                self.assertTrue(torch.equal(trace["teacher"]["initial_noise"], trace["student"]["initial_noise"]))
                self.assertTrue(torch.equal(inputs["action"], trace[role]["action_pred"]))
            if role == "student":
                self.assertEqual(sum(not raw["episode_success"] for raw in raws), 30)
        self.assertFalse(torch.cuda.is_initialized())

    def test_temporal_padding_is_not_zeroed_or_sent_back_as_rtc(self):
        model = ToyEndpointFlow((1, 40, 132))
        original = torch.full((1, 40, 132), 17.0)
        inputs = {"action": original, "action_mask": action_mask(original, SPEC),
                  "input_ids": torch.tensor([[1, 2]]), "state": torch.zeros(1, 1, 132)}
        result = bundle.infer_chunk(model, inputs, 15, SPEC)
        self.assertTrue(result["action_pred"][:, 16:, :].abs().sum().item() > 0)
        self.assertTrue(result["action_pred"][:, :16, 7:].abs().sum().item() > 0)
        self.assertTrue(torch.equal(inputs["action"], original))
        self.assertEqual(result["integration_steps"], 4)

    def test_wrong_role_teacher_and_existing_or_nested_outputs_fail(self):
        self.generate()
        with self.assertRaisesRegex(ValueError, "role"):
            bundle.load_endpoint_bundle(self.output, "other", teacher_checkpoint=self.source.teacher)
        with self.assertRaisesRegex(ValueError, "teacher differs"):
            bundle.load_endpoint_bundle(self.output, "teacher", teacher_checkpoint=self.source.qad)
        with self.assertRaises(FileExistsError):
            self.generate()
        for root in (self.source.teacher_source, self.source.qad, self.source.training,
                     self.source.student_views):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "outside source"):
                bundle.generate_endpoint_bundle(self.plan_file, root / "new_bundle", gr00t=self.source.root)

    def test_incomplete_missing_extra_and_symlink_artifacts_fail(self):
        self.generate()
        unexpected = self.output / "incomplete.json"
        unexpected.write_text("{}")
        with self.assertRaisesRegex(ValueError, "extra files"):
            self.load()
        unexpected.unlink()
        extra_directory = self.output / "unused"
        extra_directory.mkdir()
        with self.assertRaisesRegex(ValueError, "extra files"):
            self.load()
        extra_directory.rmdir()
        link = self.output / "alias.pt"
        link.symlink_to(self.output / "teacher/sample_000000.pt")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.load()
        link.unlink()
        (self.output / "student/sample_000000.pt").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.load()

    def test_bytes_tamper_and_resealed_action_tamper_fail_even_in_other_role(self):
        self.generate()
        path = self.output / "student/sample_000000.pt"
        original = path.read_bytes()
        path.write_bytes(original + b"extra")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.load("teacher")
        path.write_bytes(original)
        self.reseal_sample("student", lambda raw: raw["inputs"]["action"].add_(1))
        with self.assertRaises(ValueError):
            self.load("teacher")

    def test_resealed_noise_mismatch_rejected_even_when_seeds_match(self):
        self.generate()
        path = self.output / "traces/pair_000000.pt"
        trace = torch.load(path, weights_only=True)
        trace["student"]["initial_noise"].add_(1)
        torch.save(trace, path)
        manifest = bundle.read_json(self.output / "manifest.json")
        manifest["records"][0]["trace"].update({key: value for key, value in bundle.identity(path).items() if key != "path"})
        plan_tests.write(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "identical actual initial noise"):
            self.load()

    def test_resealed_origin_provenance_and_non_action_inputs_cannot_change(self):
        self.generate()
        original_manifest = (self.output / "manifest.json").read_bytes()
        path = self.output / "teacher/sample_000000.pt"
        original = path.read_bytes()
        for mutation in (lambda raw: raw.update(source_kind="student_rollout"),
                         lambda raw: raw.update(student_checkpoint=str(self.source.qad)),
                         lambda raw: raw["inputs"]["state"].add_(1),
                         lambda raw: raw["inputs"].update(pixel_values=raw["inputs"]["pixel_values"].float()),
                         lambda raw: raw["inputs"]["action_mask"].fill_(1)):
            with self.subTest(mutation=mutation):
                self.reseal_sample("teacher", mutation)
                with self.assertRaises(ValueError):
                    self.load()
                path.write_bytes(original)
                (self.output / "manifest.json").write_bytes(original_manifest)

    def test_original_plan_source_and_actual_adapter_are_reaudited(self):
        self.generate()
        for path in (self.plan_file, Path(self.plan["pairs"][0]["student"]["path"]),
                     self.source.training / "model.safetensors"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    self.load()
                path.write_bytes(original)

    def test_generation_rejects_noise_mismatch_and_leaves_incomplete_receipt(self):
        calls = []
        real = bundle.infer_chunk

        def inconsistent(*args):
            result = real(*args)
            if calls:
                result["initial_noise"] = result["initial_noise"] + 1
            calls.append(True)
            return result

        with patch.object(bundle, "infer_chunk", side_effect=inconsistent):
            with self.assertRaisesRegex(ValueError, "identical actual initial noise"):
                self.generate()
        self.assertTrue((self.output / "incomplete.json").is_file())
        self.assertFalse((self.output / "manifest.json").exists())

    def test_missing_sampler_callback_and_changed_plan_cannot_complete(self):
        with patch.object(bundle, "_run_qad_policy", return_value=toy_runtime()):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.generate()
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertTrue((self.output / "incomplete.json").exists())
        self.output = self.source.root / "mutated_plan_bundle"
        real = bundle.infer_chunk

        def mutate(*args):
            result = real(*args)
            if self.plan_file.read_bytes().endswith(b"\n"):
                self.plan_file.write_bytes(self.plan_file.read_bytes() + b" ")
            return result

        with patch.object(bundle, "infer_chunk", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                self.generate()
        self.assertFalse((self.output / "manifest.json").exists())


class FormalEndpointLoaderTests(unittest.TestCase):
    """Exercise loader wiring and cleanup while GR00T construction is stubbed."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.qad, self.base = self.root / "qad", self.root / "base"
        self.qad.mkdir()
        self.base.mkdir()
        self.modalities = {"libero_sim": {"fixture": "saved modalities"}}
        self.statistics = {"action": {"x": {"mean": [0], "std": [1]}}}
        self.mapping = {"libero_sim": 1}
        plan_tests.write(self.qad / "processor_config.json", {
            "processor_kwargs": {"modality_configs": self.modalities}})
        plan_tests.write(self.base / "statistics.json", {"libero_sim": self.statistics})
        plan_tests.write(self.base / "embodiment_id.json", self.mapping)
        self.protocol = self.root / "protocol.json"
        plan_tests.write(self.protocol, {"quantization_scope": {"activation_linear_count": 1}})
        self.plan = {"qad": {"checkpoint": str(self.qad), "base": str(self.base)},
            "protocol_file": str(self.protocol), "endpoint_distillation": {"endpoint_seed": 123}}
        self.model = ToyEndpointFlow()
        self.model.action_head.action_encoder._fp4vla_w4a4 = True

        class StateActionProcessor:
            pass

        class Processor:
            pass

        class Policy:
            pass

        self.policy = Policy()
        self.policy.model = self.model
        self.policy.processor = Processor()
        self.policy.processor.modality_configs = self.modalities
        self.policy.processor.state_action_processor = StateActionProcessor()
        self.policy.processor.state_action_processor.statistics = {"libero_sim": self.statistics}
        self.policy.processor.embodiment_id_mapping = self.mapping
        self.modules = {name: types.ModuleType(name) for name in (
            "gr00t", "gr00t.policy", "gr00t.policy.server_client", "gr00t.data", "gr00t.data.utils")}
        self.modules["gr00t.policy"].server_client = self.modules["gr00t.policy.server_client"]
        self.original_server = object()
        self.modules["gr00t.policy.server_client"].PolicyServer = self.original_server
        self.modules["gr00t.data.utils"].parse_modality_configs = lambda value: value

    def test_formal_loader_uses_callback_and_restores_process_context(self):
        seen = []
        original_path, original_argv, original_cwd = list(sys.path), sys.argv, Path.cwd()
        environment = {"OPD_CAPTURE_DIR": "must_not_capture", "FP4VLA_CAPTURE_TASK_NAME": "unused",
                       "FP4VLA_LOG_DIR": "must_not_log", "FP4VLA_W4A4": "old", "GR00T_EVAL_SEED": "old"}

        def launch(path, run_name):
            self.assertEqual(path, str(bundle.ROOT / "eval/serve_recovery.py"))
            self.assertEqual(run_name, "__main__")
            self.assertEqual(Path.cwd(), self.root)
            self.assertEqual(sys.argv[sys.argv.index("--model-path") + 1], str(self.qad))
            self.assertEqual(os.environ["FP4VLA_W4A4_ADAPTER"], "1")
            self.assertEqual(os.environ["FP4VLA_SATURATE_F16_ACTIVATIONS"], "1")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["GR00T_EVAL_SEED"], "123")
            self.assertFalse(any(key in os.environ for key in environment if "CAPTURE" in key or "LOG" in key))
            with self.modules["gr00t.policy.server_client"].PolicyServer(policy=self.policy) as server:
                server.run()

        with patch.dict(sys.modules, self.modules), patch.dict(os.environ, environment), \
                patch.object(bundle, "require_full_model", return_value={"fixture": "complete"}), \
                patch.object(bundle.runpy, "run_path", side_effect=launch):
            previous_environment = dict(os.environ)
            runtime = bundle._run_qad_policy(self.plan, self.root, seen.append)
            self.assertEqual(dict(os.environ), previous_environment)
        self.assertEqual(seen, [self.model])
        self.assertIs(self.modules["gr00t.policy.server_client"].PolicyServer, self.original_server)
        self.assertEqual(sys.path, original_path)
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual(runtime["loader"], "serve_recovery_local_callback")
        self.assertEqual(runtime["w4a4_activation_count"], 1)
        self.assertFalse(runtime["rpc_opened"])
        self.assertTrue(runtime["source_files"])

    def test_missing_callback_or_wrong_loaded_activation_scope_fails_and_restores(self):
        for execute, count in ((False, 1), (True, 0)):
            with self.subTest(execute=execute):
                self.model.action_head.action_encoder._fp4vla_w4a4 = bool(count)

                def launch(*args, **kwargs):
                    if execute:
                        with self.modules["gr00t.policy.server_client"].PolicyServer(policy=self.policy) as server:
                            server.run()

                before_env, before_cwd, before_path = dict(os.environ), Path.cwd(), list(sys.path)
                with patch.dict(sys.modules, self.modules), \
                        patch.object(bundle, "require_full_model", return_value={"fixture": "complete"}), \
                        patch.object(bundle.runpy, "run_path", side_effect=launch):
                    with self.assertRaises(ValueError):
                        bundle._run_qad_policy(self.plan, self.root, lambda _: None)
                self.assertEqual(dict(os.environ), before_env)
                self.assertEqual(Path.cwd(), before_cwd)
                self.assertEqual(sys.path, before_path)
                self.assertIs(self.modules["gr00t.policy.server_client"].PolicyServer, self.original_server)


if __name__ == "__main__":
    unittest.main()
