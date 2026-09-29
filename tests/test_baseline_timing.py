"""CPU-only checks for timing-script validation and the strict pi0.5 loader."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import bench_gr00t_pt as groot
import bench_pi05_lerobot as pi05


class BaselineTimingTests(unittest.TestCase):
    def test_gr00t_tuple_and_finite_checks(self):
        self.assertEqual(groot.check_action(({"x": np.ones((1, 16, 1))}, {})), {"x": [1, 16, 1]})
        with self.assertRaises(TypeError):
            groot.check_action({"x": np.ones(1)})
        with self.assertRaises(ValueError):
            groot.check_action(({"x": np.array([np.nan])}, {}))

    def test_dtype_hooks_are_untimed_and_removed(self):
        model = torch.nn.Linear(4, 2).to(dtype=torch.bfloat16)
        _, observed = groot.observe_linear_dtypes(model, lambda: model(torch.ones(1, 4, dtype=torch.bfloat16)))
        self.assertEqual(observed, {"input=bfloat16;weight=bfloat16;output=bfloat16": 1})
        self.assertFalse(model._forward_hooks)
        with self.assertRaises(RuntimeError):
            groot.observe_linear_dtypes(model, lambda: (_ for _ in ()).throw(RuntimeError("stop")))
        self.assertFalse(model._forward_hooks)

    def test_pi05_finite_check(self):
        self.assertEqual(pi05.check_action(torch.ones(1, 50, 7)), [1, 50, 7])
        with self.assertRaises(ValueError):
            pi05.check_action(torch.full((1, 50, 7), float("inf")))

    def test_pi05_loader_overrides_config_and_propagates_loading_error(self):
        calls = []
        fake_config = types.SimpleNamespace(device="mps", dtype="float32", compile_model=True)
        fail = [False]
        class Policy:
            def __init__(self, config):
                self.config = config
                calls.append((config.device, config.dtype, config.compile_model))
            def _fix_pytorch_state_dict_keys(self, state, config):
                return state
            def load_state_dict(self, state, strict):
                calls.append((tuple(state), strict))
                if fail[0]:
                    raise RuntimeError("missing pretrained weights")
            def eval(self): return self
            def to(self, device): calls.append(device); return self
            def requires_grad_(self, value): return self
        config_class = types.SimpleNamespace(from_pretrained=lambda *a, **k: fake_config)
        modules = {"lerobot": types.ModuleType("lerobot"), "lerobot.policies": types.ModuleType("lerobot.policies"),
                   "lerobot.policies.pi05": types.ModuleType("lerobot.policies.pi05"),
                   "lerobot.configs": types.ModuleType("lerobot.configs")}
        modules["lerobot.policies.pi05"].PI05Policy = Policy
        modules["lerobot.configs"].PreTrainedConfig = config_class
        args = types.SimpleNamespace(checkpoint=Path("/cpu-fixture"), device="cuda:0")
        with patch.dict(sys.modules, modules), patch.object(pi05.importlib.metadata, "version", return_value="0.6.1"), \
             patch("safetensors.torch.load_file", return_value={"test": torch.ones(1)}):
            policy, count = pi05.load_policy(args)
            self.assertEqual(count, 1)
            self.assertEqual(calls[0], ("cpu", "bfloat16", False))
            self.assertEqual(calls[1], (("model.test",), True))
            self.assertEqual(policy.config.device, "cuda:0")
            fail[0] = True
            with self.assertRaisesRegex(RuntimeError, "missing pretrained weights"):
                pi05.load_policy(args)


if __name__ == "__main__":
    unittest.main()
