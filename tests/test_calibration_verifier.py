"""Small CPU fixtures for read-only calibration acceptance, never real caches."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_calibration", ROOT / "quant/ptq/verify_calibration.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class CalibrationVerifierTests(unittest.TestCase):
    def test_accepts_full_fixture_and_rejects_corruption_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, output = root / "base", root / "calibration"
            base.mkdir(); output.mkdir()
            config = {"select_layer": 16, "diffusion_model_cfg": {"num_layers": 32},
                      "vl_self_attention_cfg": {"num_layers": 4}}
            (base / "config.json").write_text(json.dumps(config))
            prefixes = (("backbone.model.model.language_model.layers", 16),
                        ("action_head.model.transformer_blocks", 32),
                        ("action_head.vl_self_attention.transformer_blocks", 4))
            weights = {f"{prefix}.{i}.projection.weight": torch.zeros(2, 16)
                       for prefix, count in prefixes for i in range(count)}
            shard = base / "model.safetensors"
            save_file(weights, shard)
            entries = {name.removesuffix(".weight"): {"H": torch.eye(16), "abs": torch.ones(16),
                       "n": 128, "calls": 128} for name in weights}
            cache = output / "calib.pt"
            torch.save(entries, cache)
            metadata = {"status": "complete", "version": "full-model-cpu-hessian-v2",
                        "architecture": VERIFY.EXPECTED_ARCHITECTURE, "base": str(base),
                        "base_config_sha256": VERIFY.file_sha256(base / "config.json"),
                        "base_statistics_sha256": None,
                        "base_weight_files": {shard.name: {"bytes": shard.stat().st_size,
                                                            "sha256": VERIFY.file_sha256(shard)}},
                        "windows_requested": 128, "windows_consumed": 128,
                        "batch": 1, "forwards": 128, "model_dtype": "bfloat16",
                        "accumulation_dtype": "float32", "accumulation_device": "cpu",
                        "tf32_matmul": False, "n_layers": len(entries), "recipe_targets": "calib",
                        "rows_per_layer": {k: 128 for k in entries},
                        "calls_per_layer": {k: 128 for k in entries},
                        "H_GB_f32": len(entries) * 16 * 16 * 4 / 1e9,
                        "cache_sha256": VERIFY.file_sha256(cache)}
            meta = output / "calib_meta.json"
            meta.write_text(json.dumps(metadata))
            args = argparse.Namespace(calib=str(cache), meta=None, expected_windows=128,
                                      chunk_rows=3, cpu_threads=1, atol=1e-5, rtol=1e-5, report=None)
            before = {p: VERIFY.file_sha256(p) for p in (cache, meta, shard, base / "config.json")}
            with redirect_stdout(io.StringIO()):
                report = VERIFY.verify(args)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["layers"], 52)
            self.assertEqual(report["max_abs_asymmetry"], 0)
            self.assertFalse(report["cuda_initialized"])
            self.assertEqual(before, {p: VERIFY.file_sha256(p) for p in before})

            first = next(iter(entries))
            for name in ("shape", "nonfinite", "asymmetry", "missing_layer", "wrong_count", "wrong_sha"):
                with self.subTest(corruption=name):
                    bad = {k: {**v, "H": v["H"].clone()} for k, v in entries.items()}
                    if name == "shape":
                        bad[first]["H"] = torch.eye(15)
                    elif name == "nonfinite":
                        bad[first]["H"][0, 0] = float("nan")
                    elif name == "asymmetry":
                        bad[first]["H"][0, 1] = 1
                    elif name == "missing_layer":
                        bad.pop(first)
                    elif name == "wrong_count":
                        bad[first]["n"] = 127
                    torch.save(bad, cache)
                    metadata["cache_sha256"] = "0" * 64 if name == "wrong_sha" else VERIFY.file_sha256(cache)
                    meta.write_text(json.dumps(metadata))
                    with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                        VERIFY.verify(args)


if __name__ == "__main__":
    unittest.main()
