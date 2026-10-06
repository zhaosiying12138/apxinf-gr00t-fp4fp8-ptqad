"""CPU tests for action fidelity metrics; no model weights or CUDA are used."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from exp.action_chunk_diagnostics import bind_final_checkpoint, compare, decode_chunk, freeze_inputs, infer_chunk
from rl.gr00t_runtime import action_mask
from rl.probe_distill import file_sha256


SPEC = {"horizon": 16, "dimensions": 7,
        "keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "key_dimensions": [1] * 7}


class ToyFlow(nn.Module):
    """Separates the training velocity path from a four-step action sampler."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.action_head = nn.Module()
        self.action_head.action_encoder = nn.Identity()
        self.action_head.num_inference_timesteps = 4

    def forward(self, inputs):
        raise AssertionError("Training velocity forward must not be called")

    def get_action(self, inputs):
        if "action" in inputs or "action_mask" in inputs:
            raise AssertionError("Training endpoint would accidentally enable RTC")
        self.observed_autocast = torch.is_autocast_enabled("cpu")
        action = torch.randn(1, 40, 132)
        for _ in range(4):
            action = self.action_head.action_encoder(action) + 0.25 * self.weight
        return {"action_pred": action}


def payload(arm="bf16"):
    sample = {"task": "task_a", "sha256": "sample_hash", "seed": 20}
    return {"format": "gr00t_action_outputs_v1", "arm": arm, "inputs_sha256": "inputs_hash",
            "final_manifest_sha256": "final_hash", "source_files": {"model.py": "model_hash"},
            "input_manifest": {"interpretation": "training_distribution_fit_diagnostic", "samples": [sample]},
            "action_spec": copy.deepcopy(SPEC), "records": [{"sample": sample,
            "initial_noise": torch.zeros(1, 40, 132, dtype=torch.bfloat16),
            "action_pred": torch.zeros(1, 40, 132), "integration_steps": 4,
            "decoded": {k: torch.zeros(1, 16, 1) for k in SPEC["keys"]}}]}


class ActionDiagnosticTests(unittest.TestCase):
    def test_complete_action_sampling_preserves_rng_and_module_modes(self):
        model = ToyFlow().train()
        model.action_head.action_encoder.eval()
        endpoint = torch.full((1, 40, 132), 123.0)
        inputs = {"action": endpoint, "action_mask": action_mask(endpoint, SPEC), "state": torch.zeros(1, 1, 132)}
        before = torch.random.get_rng_state().clone()
        first = infer_chunk(model, inputs, 52, SPEC)
        second = infer_chunk(model, inputs, 52, SPEC)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(model.training)
        self.assertFalse(model.action_head.action_encoder.training)
        self.assertFalse(model.observed_autocast)
        self.assertTrue(torch.equal(first["initial_noise"], second["initial_noise"]))
        torch.testing.assert_close(first["action_pred"], first["initial_noise"] + 1)
        self.assertTrue(torch.equal(inputs["action"], endpoint))
        self.assertEqual(len(model.action_head.action_encoder._forward_pre_hooks), 0)

    def test_padding_excluded_but_later_unexecuted_chunk_steps_count(self):
        ref, cur = payload(), payload("qad")
        cur["records"][0]["action_pred"][:, 16:, :] = 1000
        cur["records"][0]["action_pred"][:, :16, 7:] = 1000
        self.assertEqual(compare(ref, cur)["task_macro"]["normalized_mse"], 0)
        cur["records"][0]["action_pred"][:, 15, 0] = 2
        report = compare(ref, cur)
        self.assertAlmostEqual(report["task_macro"]["normalized_mse"], 4 / 112)
        self.assertAlmostEqual(report["task_macro"]["normalized_x_mse"], 4 / 16)

    def test_actual_noise_mismatch_rejected_even_with_equal_seed(self):
        ref, cur = payload(), payload("ptq")
        cur["records"][0]["initial_noise"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Actual initial noise"):
            compare(ref, cur)

    def test_gripper_neutral_and_open_close_are_three_distinct_commands(self):
        ref, cur = payload(), payload("qad_opd")
        ref["records"][0]["decoded"]["gripper"][:] = 0.5
        cur["records"][0]["decoded"]["gripper"][:] = 0.6
        self.assertEqual(compare(ref, cur)["task_macro"]["libero_gripper_command_disagreement"], 1)
        ref["records"][0]["decoded"]["gripper"][:] = 0.9
        self.assertEqual(compare(ref, cur)["task_macro"]["libero_gripper_command_disagreement"], 0)

    def test_incomplete_run_and_wrong_sample_order_rejected(self):
        for mutation in (lambda p: p["records"].clear(),
                         lambda p: p["records"][0].update(sample={"task": "wrong"}),
                         lambda p: p["records"][0].update(integration_steps=8)):
            ref, cur = payload(), payload("ptq")
            mutation(cur)
            with self.assertRaises(ValueError):
                compare(ref, cur)

    def test_nonfinite_actions_fail_even_when_in_padding(self):
        ref, cur = payload(), payload("ptq")
        cur["records"][0]["action_pred"][:, -1, -1] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            compare(ref, cur)

    def test_different_selected_runs_or_source_versions_rejected(self):
        for field in ("final_manifest_sha256", "source_files"):
            ref, cur = payload(), payload("qad_opd")
            cur[field] = "different"
            with self.assertRaises(ValueError):
                compare(ref, cur)

    def test_final_protocol_mismatch_rejected_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_path = Path(tmp) / "final_manifest.json"
            final_path.write_text(json.dumps({"format": "w4a4_recovery_v12_final_manifest", "protocol_sha256": "other"}))
            with self.assertRaisesRegex(ValueError, "protocols differ"):
                bind_final_checkpoint(final_path, "qad_opd", {"protocol_sha256": "expected"})

    def test_final_selects_opd_and_detects_changed_adapter_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, train, export = (root / name for name in ("ptq", "trained_opd", "deployed_opd"))
            for folder in (base, train, export):
                folder.mkdir()
                (folder / "model.safetensors").write_bytes(folder.name.encode())
            def weight(folder):
                path = folder / "model.safetensors"
                return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            merge = export / "merge_manifest.json"
            merge.write_text(json.dumps({"status": "complete", "base": str(base),
                "training_checkpoint": str(train), "base_weights": {"model.safetensors": weight(base)},
                "training_weights": {"model.safetensors": weight(train)}}))
            def identity(folder):
                return {"path": str(folder), "metadata": ({"merge_manifest.json": file_sha256(merge)} if folder == export else {}),
                        "shards": [{"name": "model.safetensors", **weight(folder)}]}
            selection = root / "selection.json"
            selection.write_text(json.dumps({"protocol_sha256": "protocol_hash",
                "arms": {"rtn": {"model_identity": identity(base)}}}))
            final = root / "final_manifest.json"
            final.write_text(json.dumps({"format": "w4a4_recovery_v12_final_manifest",
                "protocol_sha256": "protocol_hash", "selection_file": str(selection),
                "selection_sha256": file_sha256(selection), "selected_pressure_recipe": "rtn",
                "selected_opd_model_identity": identity(export)}))
            actual, _ = bind_final_checkpoint(final, "qad_opd", {"protocol_sha256": "protocol_hash"})
            self.assertEqual(actual, export)
            (train / "model.safetensors").write_bytes(b"different trained adapter")
            with self.assertRaisesRegex(ValueError, "Deployment weight identity"):
                bind_final_checkpoint(final, "qad_opd", {"protocol_sha256": "protocol_hash"})

    def test_relative_actions_are_not_decoded_without_raw_state(self):
        config = {"processor_kwargs": {"modality_configs": {"libero_sim": {"action": {
            "action_configs": [{"rep": "RELATIVE"}]}}}}}
        self.assertIsNone(decode_chunk(object(), torch.zeros(1, 40, 132), config))

    def test_heldout_capture_rejected_before_loading_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "observations").mkdir()
            (root / "protocol.json").write_text("{}")
            (root / "eval_manifest.json").write_text(json.dumps({"purpose": "heldout"}))
            with self.assertRaisesRegex(ValueError, "held-out"):
                freeze_inputs(root / "observations", root / "protocol.json", 8, 1)

    def test_freeze_binds_real_sample_bytes_and_checks_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "observations/task_a"
            folder.mkdir(parents=True)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"partitions": {
                "teacher_supervision": {"init_state_indices": [20]},
                "heldout": {"init_state_indices": [9]}}}))
            sha = file_sha256(protocol)
            (root / "eval_manifest.json").write_text(json.dumps({"purpose": "teacher_supervision",
                "init_state_indices": [20], "protocol_sha256": sha, "tasks": ["task_a"]}))
            (folder / "capture_manifest.json").write_text(json.dumps({"finalized": True,
                "protocol_sha256": sha, "task_name": "task_a"}))
            sample = {"task_name": "task_a", "init_state_index": 20,
                      "reset_identity": {"init_state_index": 20}, "episode_index": 0,
                      "server_call": 1, "student_statistics_sha256": "statistics_hash"}
            sample_path = folder / "sample_000000.pt"
            torch.save(sample, sample_path)
            manifest = freeze_inputs(root / "observations", protocol, 1, 42)
            self.assertEqual(manifest["samples"][0]["sha256"], file_sha256(sample_path))
            self.assertEqual(manifest["interpretation"], "training_distribution_fit_diagnostic")
            sample["init_state_index"] = 9
            torch.save(sample, sample_path)
            with self.assertRaisesRegex(ValueError, "outside partition"):
                freeze_inputs(root / "observations", protocol, 1, 42)


if __name__ == "__main__":
    unittest.main()
