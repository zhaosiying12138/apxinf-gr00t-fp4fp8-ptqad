import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper import materialize_final_evidence as materialize


class MaterializeFinalEvidenceTests(unittest.TestCase):
    def test_materializes_complete_supplement_set_and_rejects_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / 'stage'
            stage.mkdir()
            supplement = root / 'supplement'
            for name in materialize.SUPPLEMENT_DIRECTORIES:
                path = supplement / name / 'manifest.json'
                path.parent.mkdir(parents=True)
                path.write_text(name + '\n')
            (supplement / 'frontier_comparison.json').write_text('{}\n')
            mapping = []
            self.assertEqual(
                set(materialize._materialize_supplements(stage, supplement, mapping)),
                set((*materialize.SUPPLEMENT_DIRECTORIES, *materialize.SUPPLEMENT_FILES)))
            self.assertEqual(len(mapping), 5)
            self.assertEqual((stage / 'evidence/runtime/manifest.json').read_text(), 'runtime\n')
            (supplement / 'gptq_reference/manifest.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing gptq_reference'):
                materialize._materialize_supplements(root / 'other-stage', supplement, [])

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

    def test_pairing_mapping_is_unique_and_names_original_round_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            memory = {"linear_params": 100, "nvfp4_params": 100, "fp8_params": 0,
                      "bf16_params": 0,
                      "fraction_of_eligible_params": {"nvfp4": 1., "fp8": 0., "bf16": 0.}}
            (checkpoint / "category_ptq_recipe.json").write_text(json.dumps({
                "parent": str(root / "removed-parent"), "recipe": "all_nvfp4_gptq_category",
                "memory": memory}))
            for name in ("category_bake_manifest.json", "ptq_recipe.json", "bake_manifest.json"):
                (checkpoint / name).write_text("{}")
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            inventory = root / "recipe_inventory.json"
            inventory.write_text("{}")
            paper = root / "paper"
            paper.mkdir()
            (paper / "analysis_plan_w4a4.json").write_text("{}")
            round_dir = root / "heldout_round"
            for name in materialize.collect_pairing_evidence.NAMES:
                source = round_dir / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(json.dumps({"fixture": name}))
            final_path = root / "final_manifest.json"
            final_path.write_text(json.dumps({
                "protocol_file": str(protocol), "selected_ptq_checkpoint": str(checkpoint),
                "heldout_round": str(round_dir)}))

            def copy_pairing(origin, output, protocol_file):
                self.assertEqual(origin, round_dir)
                self.assertEqual(protocol_file, str(protocol))
                for name in materialize.collect_pairing_evidence.NAMES:
                    target = output / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(origin / name, target)

            final_results = {"source": {}, "selected_recipe": "all_nvfp4_gptq_category",
                             "public_arms": {name: {} for name in ("bf16", "ptq", "qad", "qad_opd")}}
            out = root / "out"
            # The fixture isolates manifest assembly from scientific validation,
            # which has separate collector/extractor tests. All copies are real.
            with mock.patch.object(materialize, "PAPER", paper), \
                 mock.patch.object(materialize.extract_final_evidence, "extract", return_value=final_results), \
                 mock.patch.object(materialize.collect_pairing_evidence, "collect", side_effect=copy_pairing), \
                 mock.patch.object(materialize, "_materialize_heldout_raw_logs", return_value={}), \
                 mock.patch.object(materialize, "_materialize_training", return_value={}):
                materialize.materialize(final_path, out, inventory, orchestrator_run=root)

            rows = json.loads((out / "evidence_manifest.json").read_text())["files"]
            paths = [row["published_path"] for row in rows]
            self.assertEqual(len(paths), len(set(paths)))
            pairing = [row for row in rows if row["role"] == "verified_heldout_pairing_copy"]
            self.assertEqual(len(pairing), 16)
            self.assertEqual({row["published_path"] for row in pairing},
                             {"evidence/" + name for name in materialize.collect_pairing_evidence.NAMES})
            for row in pairing:
                name = Path(row["published_path"]).relative_to("evidence")
                self.assertEqual(row["source"], materialize.identity(round_dir / name))
                self.assertEqual((out / row["published_path"]).read_bytes(), (round_dir / name).read_bytes())
            self.assertEqual(sum(row["role"] == "frozen_recipe_inventory" for row in rows), 1)
            self.assertEqual(sum(row["role"].startswith("selected_") for row in rows), 4)


if __name__ == "__main__":
    unittest.main()
