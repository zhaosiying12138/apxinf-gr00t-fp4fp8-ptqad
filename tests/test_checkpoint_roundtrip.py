"""CPU integration check across PTQ serialization and additive LoRA export."""
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


class CheckpointRoundtrip(unittest.TestCase):
    def test_bake_and_cross_shard_adapter_export_preserve_base_dtype_and_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, baked, trained, exported = [root / name for name in ("base", "baked", "trained", "exported")]
            base.mkdir()
            trained.mkdir()
            weights = {"action_head.linear.weight": torch.linspace(-.5, .5, 256).reshape(16, 16).bfloat16(),
                       "action_head.linear.bias": torch.zeros(16, dtype=torch.bfloat16),
                       "backbone.other.weight": torch.eye(16, dtype=torch.bfloat16)}
            save_file(weights, str(base / "model.safetensors"))
            for name, value in (("config.json", {"model_type": "test"}),
                                ("statistics.json", {"libero_sim": {"test": [1, 2]}}),
                                ("embodiment_id.json", {"libero_sim": 0})):
                (base / name).write_text(json.dumps(value))
            env = os.environ.copy() | {"CUDA_VISIBLE_DEVICES": ""}

            def run(script, *args):
                result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            run("quant/ptq/bake.py", "--base", base, "--out", baked,
                "--recipe", "fp8", "--calibration-mode", "none", "--device", "cpu")
            baked_weights = load_file(str(baked / "model.safetensors"))
            self.assertEqual(set(weights), set(baked_weights))
            self.assertTrue(all(t.dtype == torch.bfloat16 for t in baked_weights.values()))
            self.assertFalse(torch.equal(weights["action_head.linear.weight"], baked_weights["action_head.linear.weight"]))
            a = torch.linspace(-.1, .1, 32).reshape(2, 16)
            b = torch.linspace(.2, -.2, 32).reshape(16, 2)
            first = {name: value.float() for name, value in baked_weights.items()}
            first["action_head.linear.lora_A"] = a
            second = {"action_head.linear.lora_B": b}
            save_file(first, str(trained / "model-1.safetensors"))
            save_file(second, str(trained / "model-2.safetensors"))
            index = {"weight_map": {**{k: "model-1.safetensors" for k in first},
                                    **{k: "model-2.safetensors" for k in second}}}
            (trained / "model.safetensors.index.json").write_text(json.dumps(index))
            manifest = {"base": str(baked), "rank": 2, "alpha": 4.0, "scope": "head"}
            for key, filename in (("base_config_sha256", "config.json"),
                                  ("base_statistics_sha256", "statistics.json"),
                                  ("base_recipe_sha256", "ptq_recipe.json")):
                manifest[key] = hashlib.sha256((baked / filename).read_bytes()).hexdigest()
            (trained / "recovery_manifest.json").write_text(json.dumps(manifest))
            run("rl/lora_merge_bake.py", "--base", baked, "--ckpt", trained,
                "--out", exported, "--rank", 2, "--alpha", 4)
            actual = load_file(str(exported / "model.safetensors"))
            self.assertEqual(set(actual), set(baked_weights))
            self.assertTrue(all(t.dtype == torch.bfloat16 for t in actual.values()))
            expected = (baked_weights["action_head.linear.weight"].float() + 2 * (b @ a)).bfloat16()
            torch.testing.assert_close(actual["action_head.linear.weight"], expected, rtol=0, atol=0)
            torch.testing.assert_close(actual["backbone.other.weight"], baked_weights["backbone.other.weight"], rtol=0, atol=0)
            self.assertEqual((exported / "statistics.json").read_bytes(), (base / "statistics.json").read_bytes())
            exported_index = json.loads((exported / "model.safetensors.index.json").read_text())
            self.assertEqual(set(exported_index["weight_map"]), set(actual))
            self.assertEqual(exported_index["metadata"]["total_size"], sum(t.numel() * t.element_size() for t in actual.values()))
            wrong_alpha = subprocess.run([sys.executable, str(ROOT / "rl/lora_merge_bake.py"),
                "--base", str(baked), "--ckpt", str(trained), "--out", str(root / "wrong_alpha"),
                "--rank", "2", "--alpha", "5"], env=env, capture_output=True, text=True)
            self.assertNotEqual(wrong_alpha.returncode, 0)
            self.assertIn("rank/alpha differs", wrong_alpha.stderr)


if __name__ == "__main__":
    unittest.main()
