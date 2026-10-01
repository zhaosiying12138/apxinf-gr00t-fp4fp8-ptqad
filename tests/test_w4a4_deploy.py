import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from rl.w4a4_deploy import install_saved_w4a4_adapter


class W4A4DeployTests(unittest.TestCase):
    def test_loads_a_b_and_keeps_raw_residual(self):
        model = nn.Sequential(nn.Linear(16, 3, bias=False))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file({
                "0.lora_A": torch.ones(2, 16),
                "0.lora_B": torch.ones(3, 2),
            }, str(root / "model.safetensors"))
            (root / "recovery_manifest.json").write_text(json.dumps({"rank": 2, "alpha": 4}))
            def qdq(x):
                y = x.clone(); y[..., 0] = 0; return y
            patch = install_saved_w4a4_adapter(
                model, root, activation_qdq=qdq, rank=2, alpha=4)
            x = torch.zeros(1, 16); x[0, 0] = 1
            # Base sees qdq(x); residual sees the original x and therefore is
            # nonzero even though the first activation was quantized away.
            self.assertGreater(float(model(x).detach().abs().sum()), 0)
            self.assertEqual(patch.count, 1)
            patch.restore()


if __name__ == "__main__":
    unittest.main()
