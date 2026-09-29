"""CPU checks for bounded-memory native packing and fail-closed output handling."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quant"))
from nvfp4_convert_packed import ACT, LANG, VISION


class NativePackingTest(unittest.TestCase):
    def test_chunking_preserves_bytes_and_existing_output_is_rejected(self):
        rng = np.random.default_rng(7)
        tensors = {}
        for branch in (LANG, ACT, VISION):
            p = f"{branch}.0"
            for projection in ("q", "k", "v"):
                tensors[f"{p}.self_attn.{projection}_proj.weight"] = rng.normal(0, .1, (8, 32)).astype("float32")
            if branch != VISION:
                for projection in ("gate", "up"):
                    tensors[f"{p}.mlp.{projection}_proj.weight"] = rng.normal(0, .1, (16, 32)).astype("float32")
            if branch == LANG:
                for norm in ("input_layernorm", "post_attention_layernorm"):
                    tensors[f"{p}.{norm}.weight"] = rng.normal(0, .1, 32).astype("float32")
        with tempfile.TemporaryDirectory() as temporary:
            d = Path(temporary)
            save_file(tensors, str(d / "input.safetensors"))
            command = [sys.executable, str(ROOT / "quant/nvfp4_convert_packed.py"),
                       "--ckpt", str(d / "input.safetensors"), "--lang-depth", "1",
                       "--act-depth", "1", "--vision-depth", "1"]
            for size in (1, 512):
                result = subprocess.run(command + ["--out", str(d / f"out{size}"),
                                        "--chunk-rows", str(size)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            a, b = d / "out1", d / "out512"
            self.assertEqual({p.name for p in a.iterdir()}, {p.name for p in b.iterdir()})
            for path in a.iterdir():
                self.assertEqual(path.read_bytes(), (b / path.name).read_bytes(), path.name)
            self.assertEqual(json.loads((a / "manifest.json").read_text())["summary"]["tensors"], 5)
            result = subprocess.run(command + ["--out", str(a)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing existing output", result.stderr)
            result = subprocess.run(command + ["--out", str(d / "missing"), "--lang-depth", "2"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required language layer 1", result.stderr)
            self.assertFalse((d / "missing/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
