"""Tiny CPU fixtures for portable inventory identity and coverage guards."""
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recipe_inventory", ROOT / "exp/recipe_inventory.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    base = root / "original/base"
    base.mkdir(parents=True)
    shapes = {
        "action_head.proj.weight": [16, 16],
        "backbone.model.model.language_model.layers.0.self_attn.q_proj.weight": [16, 32],
        "backbone.model.model.visual.blocks.0.attn.proj.weight": [8, 16],
        "action_head.position_embedding.weight": [4, 16],
    }
    offset, header = 0, {}
    for key, shape in shapes.items():
        end = offset + shape[0] * shape[1] * 2
        header[key] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header).encode()
    (base / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    write(base / "model.safetensors.index.json", {"weight_map": {key: "model.safetensors" for key in shapes}})
    write(base / "config.json", {"fixture": "model"})
    write(base / "statistics.json", {"fixture": "stats"})
    linear = [key.removesuffix(".weight") for key in shapes if "position_embedding" not in key]
    meta = {
        "base": str(base), "status": "complete", "windows_consumed": 128, "windows_requested": 128,
        "recipe_targets": "calib", "architecture": {"language_layers": 16, "dit_layers": 32, "vl_layers": 4},
        "base_config_sha256": budget.sha(base / "config.json"),
        "base_statistics_sha256": budget.sha(base / "statistics.json"),
        "base_weight_files": {"model.safetensors": {"bytes": (base / "model.safetensors").stat().st_size,
                                                    "sha256": budget.sha(base / "model.safetensors")}},
        "tied_weight_aliases": {}, "nonlinear_targets": {"action_head.position_embedding": "Embedding"},
        "rows_per_layer": {name: 128 for name in linear}, "calls_per_layer": {name: 128 for name in linear},
        "n_layers": len(linear), "forwards": 128,
    }
    path = root / "calib_meta.json"
    write(path, meta)
    return base, path


class RecipeInventoryTest(unittest.TestCase):
    def test_portable_source_path_and_real_linear_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, meta = fixture(root)
            moved = root / "different_clone/weights/base"
            shutil.copytree(base, moved)
            with patch("torch.load", side_effect=AssertionError("no tensor loading")), \
                 patch("torch.cuda._lazy_init", side_effect=AssertionError("no CUDA")):
                result = budget.generate(moved, meta, root / "dev/recipe_inventory.json")
            self.assertEqual(result["base"], str(moved))
            self.assertEqual(result["calibration_provenance"]["recorded_base_path"], str(base))
            self.assertEqual(result["recovery_residual"]["linear_modules"], 2)
            self.assertEqual(result["recovery_residual"]["tensor_elements"], 32 * (16+16+16+32))
            self.assertEqual(result["recovery_residual"]["target_bytes"], 2 * 32 * (16+16+16+32))
            self.assertEqual(result["recipes"]["fp8"]["known_tied_alias_deduplicated"]["source_tensor_bytes"],
                             result["recipes"]["fp8"]["source_tensor_bytes"])

    def test_missing_linear_coverage_refuses_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, path = fixture(root)
            meta = budget.load(path)
            del meta["rows_per_layer"]["action_head.proj"]
            write(path, meta)
            with self.assertRaisesRegex(ValueError, "coverage is missing"):
                budget.generate(base, path, root / "out.json")
            self.assertFalse((root / "out.json").exists())

    def test_smoke_calibration_refuses_formal_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, path = fixture(root)
            meta = budget.load(path)
            meta["windows_consumed"] = 2
            write(path, meta)
            with self.assertRaisesRegex(ValueError, "smoke coverage"):
                budget.generate(base, path, root / "out.json")

    def test_same_length_tampered_weight_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, path = fixture(root)
            weight = base / "model.safetensors"
            content = weight.read_bytes()
            weight.write_bytes(content[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "source weight identity"):
                budget.generate(base, path, root / "out.json")

    def test_refuses_overwrite_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "keep.json"
            out.write_bytes(b"preserve")
            with patch.object(budget, "build", side_effect=AssertionError("should not read")), \
                 self.assertRaises(FileExistsError):
                budget.generate("missing", "missing", out)
            self.assertEqual(out.read_bytes(), b"preserve")

    def test_inconsistent_call_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, path = fixture(root)
            meta = budget.load(path)
            meta["calls_per_layer"]["action_head.proj"] = 127
            write(path, meta)
            with self.assertRaisesRegex(ValueError, "call coverage"):
                budget.generate(base, path, root / "out.json")


if __name__ == "__main__":
    unittest.main()
