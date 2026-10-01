"""CPU tests for the audited activation-only W4A4 installer."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from rl.scoped_quant import (
    configure_quant_recipe,
    install_activation_only,
)


class CategorySpecificLinear(nn.Module):
    """Small test double matching GR00T's category module contract."""

    def __init__(self, categories=1, input_dim=18, output_dim=3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(categories, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(categories, output_dim))

    def forward(self, x, cat_ids):
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class ScopedActivationTests(unittest.TestCase):
    def setUp(self):
        configure_quant_recipe(tempfile.mkdtemp())

    def test_pad_qdq_slice_supports_non_multiple_k_and_restores(self):
        model = nn.Sequential(nn.Linear(18, 3, bias=False))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ptq_recipe.json").write_text(json.dumps({
                "layers": {"0": {"actual_method": "nvfp4_rtn"}},
            }))
            configure_quant_recipe(root)
            seen = []

            def qdq(value):
                seen.append(tuple(value.shape))
                return value + 1

            x = torch.zeros(2, 18)
            baseline = model(x).detach()
            report = install_activation_only(model, activation_qdq=qdq)
            actual = model(x)
            self.assertEqual(seen, [(2, 32)])
            self.assertEqual(report.ordinary_w4a4, 1)
            self.assertEqual(report.padded_modules, 1)
            self.assertEqual(report.records[0]["padded_k"], True)
            self.assertTrue(getattr(model[0], "_fp4vla_w4a4"))
            self.assertFalse(torch.equal(actual, baseline))
            report.restore()
            torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)

    def test_category_requires_explicit_active_bank_recipe(self):
        model = CategorySpecificLinear(input_dim=18)
        x = torch.zeros(1, 2, 18)
        cat_ids = torch.zeros(1, dtype=torch.long)
        baseline = model(x, cat_ids).detach()
        # No category_ptq_recipe is loaded: the category remains BF16.
        report = install_activation_only(model, activation_qdq=lambda value: value + 1)
        torch.testing.assert_close(model(x, cat_ids), baseline, rtol=0, atol=0)
        self.assertEqual(report.category_bf16, 1)
        self.assertEqual(report.category_w4a4, 0)
        self.assertEqual(getattr(model, "_fp4vla_activation_format"), "BF16")
        self.assertFalse(getattr(model, "_fp4vla_w4a4"))

    def test_category_active_nvfp4_recipe_gets_padded_qdq(self):
        model = nn.Sequential(CategorySpecificLinear(input_dim=18))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "category_ptq_recipe.json").write_text(json.dumps({
                "active_libero_bank": 0,
                "categories": {
                    "0.W": {"banks": [{"method": "nvfp4_rtn"}]},
                },
            }))
            configure_quant_recipe(root)
            seen = []

            def qdq(value):
                seen.append(tuple(value.shape))
                return value + 1

            report = install_activation_only(model, activation_qdq=qdq)
            actual = model[0](torch.zeros(1, 2, 18), torch.zeros(1, dtype=torch.long))
            self.assertEqual(tuple(actual.shape), (1, 2, 3))
            self.assertEqual(seen, [(1, 2, 32)])
            self.assertEqual(report.category_w4a4, 1)
            self.assertEqual(report.category_bf16, 0)
            self.assertTrue(getattr(model[0], "_fp4vla_w4a4"))
            self.assertEqual(report.records[0]["padded_k"], True)

    def test_ste_forwards_gradient_through_default_native_qdq(self):
        model = nn.Linear(18, 3, bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ptq_recipe.json").write_text(json.dumps({
                "layers": {"": {"actual_method": "nvfp4_rtn"}},
            }))
            # Sequential names are normally used in recipes; install without a
            # gate here to exercise the model-wide default and explicit STE.
            configure_quant_recipe(tempfile.mkdtemp())
            report = install_activation_only(model, ste=True)
            x = torch.randn(2, 18, requires_grad=True)
            loss = model(x).square().sum()
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(report.ste)


if __name__ == "__main__":
    unittest.main()
