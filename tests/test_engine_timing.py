"""CPU-only evidence-contract tests; never import the native engine."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
import bench_engine as bench


class EngineTimingTests(unittest.TestCase):
    def test_rejects_invalid_actions_or_timing(self):
        valid = {"actions": np.ones((2, 7)), "timing": {"model_ms": 2., "total_ms": 3.}}
        self.assertEqual(bench.check_inference(valid), [2, 7])
        for broken in ({**valid, "actions": np.full((2, 7), np.nan)},
                       {**valid, "timing": {"model_ms": 4., "total_ms": 3.}},
                       {**valid, "timing": {"model_ms": 2., "total_ms": float("inf")}}):
            with self.assertRaises(ValueError):
                bench.check_inference(broken)

    def test_percentiles_preserve_sample_order_and_missing_power(self):
        class Sampler:
            samples, errors = [], ["unavailable"]
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        class Policy:
            metadata = {"image_keys": ["image"], "prompt_key": "prompt"}
            values = iter([1., 30., 10.])
            actions = np.ones((2, 7))
            def infer(self, observation):
                value = next(self.values)
                self.actions.fill(value)  # A native wrapper may reuse a host buffer.
                return {"actions": self.actions, "timing": {"model_ms": value, "total_ms": value + 1}}
        args = types.SimpleNamespace(seed=7, warmup=0, samples=2, telemetry_device="0")
        result = bench.benchmark(Policy(), args, Sampler)
        self.assertEqual(result["lat_model_ms"], [30., 10.])
        self.assertEqual(result["model_ms_p50"], 20.)
        self.assertIsNone(result["power_w_mean"])
        self.assertIsNone(result["vram_mb_peak"])
        self.assertEqual(result["first_actions"], np.ones((2, 7)).tolist())

    def test_output_is_atomic_and_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "result.json"
            with self.assertRaises(ValueError):
                bench.publish_new(out, {"bad": float("nan")})
            self.assertFalse(out.exists())
            bench.publish_new(out, {"accepted": 1})
            with self.assertRaises(FileExistsError):
                bench.publish_new(out, {"accepted": 2})
            self.assertEqual(json.loads(out.read_text()), {"accepted": 1})
            self.assertEqual(list(Path(temporary).glob(".bench-*")), [])

    def test_variant_truth_and_family_specific_dispatch(self):
        policy = types.SimpleNamespace(model_runner=types.SimpleNamespace(model_variant="bf16"),
                                       metadata={"model_variant": "bf16"})
        with self.assertRaises(ValueError):
            bench.verify_variant(policy, "pi05", "nvfp4_static")
        policy = types.SimpleNamespace(model_runner=types.SimpleNamespace(model_variant=None),
                                       metadata={"precision": "bf16"})
        self.assertEqual(bench.verify_variant(policy, "gr00t", "bf16")["status"], "explicit_precision_metadata_only")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text('{"model_type":"Gr00tN1d7"}')
            args = types.SimpleNamespace(model_dir=folder, variant="bf16", extra_kwarg=[f"backbone={folder}"],
                                         device="cuda:0", model_seed=0)
            family, kwargs = bench.prepare_config(args)
            self.assertEqual(family, "gr00t")
            self.assertEqual(kwargs["precision"], "bf16")
            self.assertNotIn("model_variant", kwargs)
            args.extra_kwarg.append("precision=fp8")
            with self.assertRaises(ValueError): bench.prepare_config(args)

    def test_failed_run_closes_policy_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            closed = []
            policy = types.SimpleNamespace(metadata={"model_type": "pi05"}, close=lambda: closed.append(True))
            apxinf = types.ModuleType("apxinf")
            apxinf.AutoPolicy = types.SimpleNamespace(from_pretrained=lambda *a, **k: policy)
            args = types.SimpleNamespace(model_dir=Path(temporary), variant="bf16", out=Path(temporary) / "result.json",
                                         telemetry_device="0")
            with patch.dict(sys.modules, {"apxinf": apxinf}), \
                 patch.object(bench, "parse_args", return_value=args), \
                 patch.object(bench, "prepare_config", return_value=("pi05", {})), \
                 patch.object(bench, "directory_identity", return_value={"files": {"model.safetensors": {}}}), \
                 patch.object(bench, "gpu_identity", return_value={}), \
                 patch.object(bench, "verify_variant", return_value={}), \
                 patch.object(bench, "runtime_identity", return_value={}), \
                 patch.object(bench, "benchmark", side_effect=ValueError("nonfinite action")):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    bench.main()
            self.assertEqual(closed, [True])
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
