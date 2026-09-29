"""CPU checks for physical tied aliases and fail-closed Hessian coverage."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
EMBED = "backbone.model.model.language_model.embed_tokens.weight"
HEAD = "backbone.model.lm_head.weight"


class TiedWeights(unittest.TestCase):
    def run_bake(self, base, output, cache):
        return subprocess.run([sys.executable, str(ROOT / "quant/ptq/bake.py"),
            "--base", str(base), "--out", str(output), "--recipe", "calib",
            "--calibration-mode", "required", "--calib", str(cache), "--device", "cpu"],
            env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)

    def fixture(self, root, extra=False):
        base, cache = root / "base", root / "calibration"
        base.mkdir(); cache.mkdir()
        w = torch.linspace(-2, 2, 256).reshape(16, 16).bfloat16()
        weights = {EMBED: w.clone(), HEAD: w.clone()}
        if extra:
            weights["action_head.linear.weight"] = w.clone()
        save_file(weights, str(base / "model.safetensors"))
        (base / "config.json").write_text('{}')
        torch.save({HEAD.removesuffix('.weight'): {"H": torch.eye(16), "n": 1}}, cache / "calib.pt")
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        meta = {"architecture": {"language_layers": 16, "dit_layers": 32, "vl_layers": 4},
                "base_config_sha256": digest(base / "config.json"), "base_statistics_sha256": None,
                "cache_sha256": digest(cache / "calib.pt"),
                "base_weight_files": {"model.safetensors": {"sha256": digest(base / "model.safetensors")}},
                "nonlinear_targets": {EMBED.removesuffix('.weight'): "Embedding"}}
        (cache / "calib_meta.json").write_text(json.dumps(meta))
        return base, cache / "calib.pt"

    def test_tied_output_uses_embedding_even_if_head_has_hessian(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); base, cache = self.fixture(root)
            run = self.run_bake(base, root / "out", cache)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            weights = load_file(str(root / "out/model.safetensors"))
            torch.testing.assert_close(weights[EMBED], weights[HEAD], rtol=0, atol=0)
            plan = json.loads((root / "out/ptq_recipe.json").read_text())
            head = plan["layers"][HEAD.removesuffix('.weight')]
            self.assertEqual(head["alias_of"], EMBED)
            self.assertEqual(head["actual_method"], "nvfp4_rtn")

    def test_missing_real_linear_hessian_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); base, cache = self.fixture(root, extra=True)
            run = self.run_bake(base, root / "out", cache)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("required GPTQ layer missing", run.stderr)

    def test_inconsistent_tied_source_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); base, cache = self.fixture(root)
            weights = load_file(str(base / "model.safetensors"))
            weights[HEAD] += 1
            save_file(weights, str(base / "model.safetensors"))
            metadata_path = cache.parent / "calib_meta.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["base_weight_files"]["model.safetensors"]["sha256"] = hashlib.sha256(
                (base / "model.safetensors").read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            run = self.run_bake(base, root / "out", cache)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("tied source weights differ", run.stderr)


if __name__ == "__main__":
    unittest.main()
