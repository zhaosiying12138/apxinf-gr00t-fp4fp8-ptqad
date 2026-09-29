"""CPU-only contracts for the independent CategorySpecificLinear PTQ path."""
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quant" / "ptq"))
from category_fp4 import (ACTIVE_BANK, ARCHITECTURE, EXPECTED_SHAPES, decode_exact,
                          memory_budget, pack_exact, quantize_bank, validate_hessian)


class CategoryFp4Test(unittest.TestCase):
    def test_allowlist_and_architecture_are_frozen(self):
        self.assertEqual(len(EXPECTED_SHAPES), 7)
        self.assertEqual(ACTIVE_BANK, 2)
        self.assertEqual(ARCHITECTURE, {"language_layers": 16, "dit_layers": 32, "vl_layers": 4})
        self.assertEqual(sum(torch.tensor(shape).prod().item() for shape in EXPECTED_SHAPES.values()),
                         325_517_312)

    def test_k_padding_is_internal_and_roundtrips(self):
        # K=132 is the real LIBERO state/action width; encoding uses K=144,
        # while the exported dequantized matrix returns to K=132.
        weight = torch.linspace(-1, 1, 132 * 16, dtype=torch.bfloat16).reshape(132, 16)
        entry = {"H": torch.eye(132), "abs": torch.ones(132), "n": 8, "calls": 1}
        validate_hessian(entry, 132, expected_calls=1)
        output, record, encoding = quantize_bank(weight, entry)
        self.assertEqual(tuple(output.shape), (132, 16))
        self.assertEqual(output.dtype, weight.dtype)
        self.assertEqual(record["padding_k"], 12)
        self.assertEqual(tuple(encoding["packed"].shape), (16, 72))
        self.assertTrue(torch.equal(decode_exact(encoding)[:, :132].T.to(weight.dtype), output))

    def test_unobserved_bank_uses_rtn_and_active_bank_can_compare_gptq(self):
        weight = torch.randn(128, 32, dtype=torch.bfloat16) * 0.05
        output, record, _ = quantize_bank(weight, None)
        self.assertEqual(record["method"], "nvfp4_rtn")
        self.assertFalse(record["calibrated"])
        self.assertEqual(tuple(output.shape), tuple(weight.shape))

    def test_budget_reports_category_padding_and_ties(self):
        entries = {}
        for key, shape in EXPECTED_SHAPES.items():
            entries[key] = {"shape": shape, "dtype": "BF16", "params": int(torch.tensor(shape).prod()),
                            "bytes": int(torch.tensor(shape).prod()) * 2, "eligible": False,
                            "shard": "model.safetensors"}
        # Add one ordinary parent NVFP4 tensor and an explicit tied alias.
        entries["ordinary.weight"] = {"shape": [32, 16], "dtype": "BF16", "params": 512,
                                       "bytes": 1024, "eligible": True, "shard": "model.safetensors"}
        entries["backbone.model.model.language_model.embed_tokens.weight"] = {
            "shape": [16, 16], "dtype": "BF16", "params": 256, "bytes": 512, "eligible": True,
            "shard": "model.safetensors"}
        entries["backbone.model.lm_head.weight"] = {
            "shape": [16, 16], "dtype": "BF16", "params": 256, "bytes": 512, "eligible": True,
            "shard": "model.safetensors"}
        recipe = {"layers": {"ordinary": {"requested_method": "nvfp4_rtn", "actual_method": "nvfp4_rtn"},
                               "backbone.model.model.language_model.embed_tokens":
                               {"requested_method": "nvfp4_rtn", "actual_method": "nvfp4_rtn"}},
                  "tied_weight_aliases": {"backbone.model.lm_head.weight":
                                           "backbone.model.model.language_model.embed_tokens.weight"}}
        budget = memory_budget(entries, recipe)
        self.assertEqual(budget["category"]["padding_elements"], 983_040)
        self.assertEqual(budget["known_tied_alias_deduplicated"]["omitted_physical_alias_elements"], 256)
        self.assertGreater(budget["known_tied_alias_deduplicated"]["fraction_of_all"]["nvfp4"], 0.99)


if __name__ == "__main__":
    unittest.main()
