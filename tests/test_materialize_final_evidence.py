import json
import tempfile
import unittest
from pathlib import Path

from paper import materialize_final_evidence as materialize


class MaterializeFinalEvidenceTests(unittest.TestCase):
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
