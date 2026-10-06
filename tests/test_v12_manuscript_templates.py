"""CPU prose checks; synthetic contrasts stay in memory and never become evidence."""
from pathlib import Path
import json
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
import update_v12_release as publication


def contrast(difference, interval, p):
    return {"difference_pp": difference, "pointwise_ci_pp": interval,
            "exact_mcnemar_two_sided_p": p, "holm_adjusted_p": p}


def render_fixture():
    # Deliberately lacks complete/status/protocol fields: cannot pass release gates.
    row = {"successes": 8, "episodes": 16, "success_rate": .5,
           "per_task": {"synthetic_task": {"successes": 8, "episodes": 16}}}
    data = {"public_arms": {a: row for a in publication.ARMS if a != "continued_qad"},
            "control": {"continued_qad": row}, "selected_recipe": "rtn_w4a4_category",
            "uncertainty": {"contrasts": {key: contrast(-2, [-5, 1], .7)
                                            for key, _ in publication.CONTRASTS}}}
    inventory = {"recipe": "rtn_all", "schema_version": "w4a4-selected-all-nvfp4-category-v12",
                 "recovery_residual": {"target_bytes": 50},
                 "recipes": {"rtn_all": {"known_tied_alias_deduplicated": {
                     "source_tensor_bytes": 1000, "target_full_bytes": 250}}}}
    return data, inventory


class ManuscriptTemplates(unittest.TestCase):
    def test_uncertain_estimates_never_claim_improvement_or_equivalence(self):
        for item in (contrast(2, [-1, 5], .2), contrast(-2, [-5, 1], .2),
                     contrast(0, [0, 0], 1), contrast(2, [.1, 5], .08)):
            text = publication.contrast_assessment(item)
            self.assertIn("尚不能确认", text)
            self.assertNotIn("支持本设置下的成功率提高", text)
            self.assertNotIn("支持本设置下的成功率下降", text)

    def test_directional_evidence_is_symmetric(self):
        self.assertIn("成功率下降", publication.contrast_assessment(contrast(-5, [-9, -1], .01)))
        self.assertIn("成功率提高", publication.contrast_assessment(contrast(5, [1, 9], .01)))

    def test_templates_preserve_polished_structure_method_and_figures(self):
        for name in publication.TEMPLATE_NAMES:
            template = (publication.TEMPLATES / name).read_text()
            source = (ROOT / "paper/sections" / name).read_text()
            self.assertEqual(re.findall(r"^#{1,2} .+$", template, re.M),
                             re.findall(r"^#{1,2} .+$", source, re.M))
            self.assertEqual(re.findall(r"\{\{fig:[^}]+\}\}", template),
                             re.findall(r"\{\{fig:[^}]+\}\}", source))
            self.assertNotRegex(template, r"xxx|审阅说明|尚未全部完成|正式评测尚未完成")
        self.assertEqual((publication.TEMPLATES / "03-方法.md").read_bytes(),
                         (ROOT / "paper/sections/03-方法.md").read_bytes())

    def test_render_retains_full_tables_budget_and_negative_contrasts(self):
        data, inventory = render_fixture()
        with mock.patch.object(Path, "write_text", side_effect=AssertionError("Pure render must not write")):
            documents = publication.render_main(data, "ACTION DIAGNOSTIC", "GPTQ REFERENCE", inventory)
        experiment = documents["04-实验.md"]
        for _, label in publication.CONTRASTS:
            self.assertIn(label, experiment)
        for label in publication.LABELS.values():
            self.assertIn(label, experiment)
        self.assertIn("synthetic_task", experiment)
        self.assertIn("3.333×", experiment)  # Includes the residual, not 1000/250.
        self.assertIn("-2.00", documents["05-讨论与结论.md"])
        self.assertIn("尚不能确认", documents["05-讨论与结论.md"])
        self.assertIn("OPD 完整追加阶段相对 QAD", documents["05-讨论与结论.md"])
        self.assertIn("OPD 相对同演示预算续训", documents["05-讨论与结论.md"])
        self.assertLess(experiment.index("ACTION DIAGNOSTIC"), experiment.index("GPTQ REFERENCE"))
        for text in documents.values():
            self.assertNotRegex(text, r"@@|xxx|审阅说明|尚未全部完成")

    def test_stale_inventory_cannot_fill_new_results(self):
        data, inventory = render_fixture()
        inventory["recipe"] = "old_recipe"
        with self.assertRaisesRegex(ValueError, "selected final recipe"):
            publication.render_main(data, "diagnostic", "gptq", inventory)

    def test_inventory_names_accept_actual_builder_schema(self):
        from build_recipe_inventory_v11 import build
        data, _ = render_fixture()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            category = {"source_base": {"path": str(root)}, "memory": {
                "eligible_tensor_count": 479, "linear_params": 1000,
                "nvfp4_params": 1000, "fp8_params": 0, "bf16_params": 0}}
            recovery = {"scope": "all_ordinary_linear", "rank": 32, "alpha": 64,
                        "lora_linear_modules": 468, "trainable_parameters": 10}
            records = {"category_ptq_recipe.json": category, "category_bake_manifest.json": {},
                       "ptq_recipe.json": {"recipe": "rtn"}, "bake_manifest.json": {},
                       "config.json": {}, "model.safetensors.index.json": {},
                       "recovery_manifest.json": recovery}
            for name, value in records.items():
                (root / name).write_text(json.dumps(value))
            inventory = build(root, root / "recovery_manifest.json", root / "inventory.json", "rtn_all")
            self.assertEqual(inventory["recipe"], "rtn_all")
            publication.validate_inventory_name(data, inventory)
            inventory["schema_version"] = "v11-selected-all-nvfp4-category"
            with self.assertRaisesRegex(ValueError, "selected final recipe"):
                publication.validate_inventory_name(data, inventory)

    def test_main_verifies_both_archives_before_any_write(self):
        import action_diagnostics_publication as action
        import gptq_reference_publication as gptq
        for failing_archive in ("action", "gptq"):
            with (mock.patch.object(publication, "load", return_value={}),
                  mock.patch.object(publication, "validate_v12_results") as validate,
                  mock.patch.object(action, "load_verified", side_effect=(
                      ValueError("archive rejected") if failing_archive == "action" else None)),
                  mock.patch.object(action, "render", return_value="action"),
                  mock.patch.object(gptq, "load_verified", side_effect=ValueError("archive rejected")),
                  mock.patch.object(Path, "write_text", side_effect=AssertionError("Premature write"))):
                with self.assertRaisesRegex(ValueError, "archive rejected"):
                    publication.main()
                validate.assert_called_once_with({}, root=ROOT)


if __name__ == "__main__":
    unittest.main()
