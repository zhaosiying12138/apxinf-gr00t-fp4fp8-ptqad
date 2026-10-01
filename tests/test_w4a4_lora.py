"""CPU contract tests for the explicit W4A4 + raw-input LoRA seam."""
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from rl.w4a4_lora import install_w4a4_lora


class W4A4LoRATests(unittest.TestCase):
    def _module(self):
        torch.manual_seed(112)
        module = nn.Linear(4, 3, bias=True)
        module.lora_A = nn.Parameter(torch.randn(2, 4) * .2)
        module.lora_B = nn.Parameter(torch.randn(3, 2) * .2)
        return module

    def test_base_gets_qdq_but_residual_gets_original_input(self):
        module = self._module()
        model = nn.Sequential(module)
        x = torch.tensor([[1., .25, -2., .5]])

        def qdq(value):
            result = value.clone()
            result[:, 1] = 0
            return result

        expected_base = F.linear(qdq(x), module.weight, module.bias)
        expected_residual = F.linear(F.linear(x, module.lora_A), module.lora_B) * 2
        wrong_residual = F.linear(F.linear(qdq(x), module.lora_A), module.lora_B) * 2
        patch = install_w4a4_lora(model, qdq, rank=2, alpha=4)
        self.assertEqual(patch.count, 1)
        actual = model(x)
        torch.testing.assert_close(actual, expected_base + expected_residual, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, expected_base + wrong_residual))
        patch.restore()
        torch.testing.assert_close(model(x), F.linear(x, module.weight, module.bias), rtol=0, atol=0)

    def test_native_base_operator_receives_qdq_and_scope_is_explicit(self):
        model = nn.Sequential(self._module(), nn.Linear(4, 3))
        seen = []

        def qdq(value):
            return value + 1

        def native_base(module, value):
            seen.append(value.detach().clone())
            return F.linear(value, module.weight, module.bias)

        patch = install_w4a4_lora(
            model, qdq, rank=2, alpha=4,
            scope=lambda name, _: name == "0",
            base_operator=native_base,
        )
        self.assertEqual(patch.count, 1)
        x = torch.zeros(2, 4)
        model[0](x)
        torch.testing.assert_close(seen[0], torch.ones_like(x), rtol=0, atol=0)
        self.assertFalse(hasattr(model[1], "_w4a4_original_forward"))
        patch.restore()

    def test_rejects_bad_contract_without_touching_modules(self):
        model = nn.Sequential(self._module())
        baseline = model(torch.ones(1, 4)).detach()
        with self.assertRaisesRegex(ValueError, "changed shape"):
            patch = install_w4a4_lora(model, lambda x: x[..., :2], rank=2, alpha=4)
            try:
                model(torch.ones(1, 4))
            finally:
                patch.restore()
        torch.testing.assert_close(model(torch.ones(1, 4)), baseline, rtol=0, atol=0)
        with self.assertRaisesRegex(TypeError, "activation_qdq"):
            install_w4a4_lora(model, None, rank=2, alpha=4)
        with self.assertRaisesRegex(ValueError, "positive"):
            install_w4a4_lora(model, lambda x: x, rank=0, alpha=4)


if __name__ == "__main__":
    unittest.main()
