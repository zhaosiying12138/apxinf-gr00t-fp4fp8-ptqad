import json
import tempfile
import unittest
from pathlib import Path

from paper import materialize_final_evidence as materialize


class MaterializeFinalEvidenceTests(unittest.TestCase):
    def test_mixed_protocol_requires_nonzero_fp4_and_fp8(self):
        protocol = {"quantization_scope": {"recipe": "mixed"}}
        memory = {"linear_params": 100, "nvfp4_params": 75, "fp8_params": 25,
                  "bf16_params": 0,
                  "fraction_of_eligible_params": {"nvfp4": .75, "fp8": .25, "bf16": 0}}
        self.assertEqual(materialize.category_memory_kind(memory, protocol), "mixed_nvfp4_fp8")
        memory.update(nvfp4_params=100, fp8_params=0,
                      fraction_of_eligible_params={"nvfp4": 1., "fp8": 0., "bf16": 0.})
        with self.assertRaisesRegex(ValueError, "both NVFP4 and FP8"):
            materialize.category_memory_kind(memory, protocol)
        self.assertEqual(materialize.category_memory_kind(memory, {}), "nvfp4_only")

    def test_rejects_inconsistent_category_accounting(self):
        memory = {"linear_params": 100, "nvfp4_params": 75, "fp8_params": 25,
                  "bf16_params": 0,
                  "fraction_of_eligible_params": {"nvfp4": .8, "fp8": .2, "bf16": 0}}
        with self.assertRaisesRegex(ValueError, "fraction disagrees"):
            materialize.category_memory_kind(memory, {})
        memory["linear_params"] = 101
        with self.assertRaisesRegex(ValueError, "do not sum"):
            materialize.category_memory_kind(memory, {})

    def test_requires_final_manifest_name_and_complete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "selection.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "final_manifest.json"):
                materialize.materialize(root / "selection.json", root / "out", training_evidence=root)

    def test_rejects_missing_final_manifest_before_touching_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            with self.assertRaises(FileNotFoundError):
                materialize.materialize(root / "final_manifest.json", out, training_evidence=root)
            self.assertFalse(out.exists())

    def test_selected_recipe_requires_complete_category_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "category_ptq_recipe.json").write_text(json.dumps({"parent": str(root / "missing")}))
            with self.assertRaisesRegex(ValueError, "category PTQ checkpoint has incomplete"):
                materialize._selected_recipe_sources({"selected_ptq_checkpoint": str(checkpoint)})


if __name__ == "__main__":
    unittest.main()
