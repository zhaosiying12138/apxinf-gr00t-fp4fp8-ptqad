"""CPU checks for the auxiliary objective and Trainer integration contract."""
import copy
from pathlib import Path
import sys
import tempfile
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from probe_distill import (CACHE_VERSION, ProbeAnchor, flow_signature, install_sequential_probe,
                           prediction, replay_context, require_full_model, masked_velocity_mse)
from recovery_batch import resolve_batch


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(2, 4) / 10)
        self.lora_B = nn.Parameter(torch.randn(4, 2) / 10)
        self.dropout = nn.Dropout(0.5)
        self.backbone = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=range(16))))
        self.action_head = SimpleNamespace(model=SimpleNamespace(transformer_blocks=range(32)),
                                           vl_self_attention=SimpleNamespace(transformer_blocks=range(4)))
        self.events = []

    def forward(self, inputs):
        self.events.append("forward_train" if self.training else "forward_probe")
        action = inputs["action"]
        noise = torch.randn_like(action)
        t = torch.distributions.Beta(1.5, 1.0).sample((1,)).reshape(1, 1, 1)
        x = self.dropout((1 - t) * noise + t * action)
        pred = x @ self.weight.T + (x @ self.lora_A.T) @ self.lora_B.T
        return {"loss": pred.square().mean(), "pred_actions": pred}


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = Toy()
        self.inputs = {"action": torch.ones(1, 3, 4), "action_mask": torch.ones(1, 3, 4)}
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "probes.pt"
        with replay_context(self.model, 17, "none"), torch.no_grad():
            teacher = prediction(self.model(self.inputs)) + 0.2
        torch.save({"version": CACHE_VERSION, "metadata": {
            "architecture": require_full_model(self.model), "model_dtype": "float32",
            "flow_config": flow_signature(self.model),
            "autocast_dtype": "none", "source_kind": "student_rollout"},
            "samples": [{"inputs": self.inputs, "pred": teacher, "seed": 17}]}, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_gradient_and_rng_and_mode_restoration(self):
        self.model.train()
        self.model.dropout.eval()  # mixed state must also be restored
        before = torch.random.get_rng_state().clone()
        anchor = ProbeAnchor(self.path)
        anchor.validate_model(self.model)
        with replay_context(self.model, 17, "none"):
            loss = anchor.loss(self.model, 0)
            loss.backward()
        self.assertGreater(self.model.lora_B.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.lora_A.grad.abs().sum().item(), 0)
        self.assertIsNone(self.model.weight.grad)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.dropout.training)

    def test_identical_seed_matches_noise_time_without_global_rng_effect(self):
        with replay_context(self.model, 73, "none"):
            a = prediction(self.model(self.inputs)).detach()
        torch.rand(13)
        with replay_context(self.model, 73, "none"):
            b = prediction(self.model(self.inputs)).detach()
        self.assertTrue(torch.equal(a, b))
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with replay_context(self.model, 73, "none"):
                raise RuntimeError("test failure")
        self.assertTrue(self.model.training)

    def test_prediction_contracts(self):
        out = {"pred_actions": torch.zeros(1, 3, 4)}
        self.assertIs(prediction(out), out["pred_actions"])
        self.assertIs(prediction((torch.tensor(0.), out)), out["pred_actions"])
        self.assertIs(prediction((torch.tensor(0.), out["pred_actions"])), out["pred_actions"])
        with self.assertRaises(TypeError):
            prediction({"loss": torch.tensor(0.)})

    def test_sequential_backward_accumulation_and_return_outputs(self):
        class Trainer:
            def __init__(self):
                self.args = SimpleNamespace(gradient_accumulation_steps=2)
                self.current_gradient_accumulation_steps = 2
                self.state = SimpleNamespace(global_step=1)
                self.accelerator = SimpleNamespace(gradient_accumulation_steps=1, num_processes=1,
                    backward=lambda loss: loss.backward())

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                out = model(inputs)
                return (out["loss"], out) if return_outputs else out["loss"]

            def training_step(self, model, inputs, **kwargs):
                model.train()
                loss = self.compute_loss(model, inputs) / 2
                loss.backward()
                model.events.append("main_backward_complete")
                return loss.detach()

        original_compute = Trainer.compute_loss
        trainer = Trainer()
        reference = copy.deepcopy(self.model)
        anchor = ProbeAnchor(self.path)
        initial_rng = torch.random.get_rng_state()
        for _ in range(2):
            loss = trainer.compute_loss(reference, self.inputs) / 2
            loss.backward()
            with replay_context(reference, 17, "none"):
                (0.7 * anchor.loss(reference, 0) / 2).backward()
        expected = reference.lora_B.grad.clone()
        torch.random.set_rng_state(initial_rng)
        install_sequential_probe(Trainer, self.path, weight=0.7, every=2)
        self.assertIs(Trainer.compute_loss, original_compute)
        self.model.events.clear()
        for _ in range(2):
            result = trainer.training_step(self.model, self.inputs)
            self.assertFalse(result.requires_grad)
        self.assertTrue(torch.allclose(expected, self.model.lora_B.grad))
        self.assertEqual(self.model.events, ["forward_train", "main_backward_complete", "forward_probe"] * 2)
        loss, out = trainer.compute_loss(self.model, self.inputs, return_outputs=True)
        self.assertIs(loss, out["loss"])

    def test_reject_legacy_and_mismatched_dtype(self):
        anchor = ProbeAnchor(self.path)
        anchor.meta["model_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "noise dtype"):
            anchor.validate_model(self.model)
        torch.save({"inputs": self.inputs, "pred": torch.zeros(1)}, self.path)
        with self.assertRaisesRegex(ValueError, "Legacy"):
            ProbeAnchor(self.path)

    def test_explicit_batch_contract(self):
        self.assertEqual(resolve_batch({"QAD_MICRO_BATCH": "2", "QAD_GLOBAL_BATCH": "16"}), (2, 8, 16))
        self.assertEqual(resolve_batch({"QAD_MICRO_BATCH": "1", "QAD_GLOBAL_BATCH": "16"}), (1, 16, 16))
        # Legacy names retain the ACTUAL prior upstream meaning (micro x acc).
        self.assertEqual(resolve_batch({"QAD_BSZ": "2", "QAD_ACC": "2"}), (2, 2, 4))
        with self.assertRaises(ValueError):
            resolve_batch({"QAD_MICRO_BATCH": "3", "QAD_GLOBAL_BATCH": "16"})
        with self.assertRaises(ValueError):
            resolve_batch({"QAD_MICRO_BATCH": "2", "QAD_GLOBAL_BATCH": "16", "QAD_ACCUM_STEPS": "2"})

    def test_padding_differences_have_no_loss_or_gradient(self):
        student = torch.zeros(1, 3, 4, requires_grad=True)
        teacher = torch.full_like(student, 100.)
        teacher[:, :2, :2] = 1.
        mask = torch.zeros_like(student)
        mask[:, :2, :2] = 1.
        loss = masked_velocity_mse(student, teacher, mask)
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertEqual(student.grad[mask == 0].abs().sum().item(), 0.)
        self.assertGreater(student.grad[mask == 1].abs().sum().item(), 0.)
        with self.assertRaisesRegex(ValueError, "binary"):
            masked_velocity_mse(student, teacher, torch.zeros_like(mask))

    def test_real_lora_injection_keeps_language_eval_differentiable(self):
        import lora_qad
        model = nn.Module()
        model.action_head = nn.Sequential(nn.Linear(16, 16))
        model.backbone = nn.Module()
        model.backbone.model = nn.Module()
        language = nn.Module()
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.q_proj = nn.Linear(16, 16)
        block.mlp = nn.Module()
        block.mlp.gate_proj = nn.Linear(16, 16)
        language.layers = nn.ModuleList([block])
        model.backbone.model.language_model = language
        model.backbone.model.visual = nn.Linear(16, 16)
        model.requires_grad_(False)  # initial tune_llm=False behavior
        model.eval()
        base = Path(self.tmp.name) / "base"
        base.mkdir()
        (base / "ptq_recipe.json").write_text("{}")
        with patch.dict(os.environ, {"GR00T_BASE_CKPT": str(base)}), patch.object(lora_qad, "SCOPE", "head+lang_all"):
            lora_qad.install_lora(model)
        x = torch.ones(1, 16)
        value = block.mlp.gate_proj(block.self_attn.q_proj(x))
        model.action_head(value).square().mean().backward()
        for module in (block.self_attn.q_proj, block.mlp.gate_proj, model.action_head[0]):
            self.assertGreater(module.lora_B.grad.abs().sum().item(), 0)
            self.assertEqual(module.lora_A.grad.abs().sum().item(), 0)  # zero-B initialization
            self.assertIsNone(module.weight.grad)
        self.assertFalse(language.training)
        self.assertFalse(hasattr(model.backbone.model.visual, "lora_A"))


if __name__ == "__main__":
    unittest.main()
