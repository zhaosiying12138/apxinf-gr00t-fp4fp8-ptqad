"""Explicit synthetic CPU fixtures; never write publication evidence or figures."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


FIGURES = module("frontier_figures", "paper/make_figs.py")
BUILDER = module("final_frontier_builder", "paper/build_final_frontier.py")
RECIPE = "all_nvfp4_gptq_category"
GATES = ("environment_pairing_verified", "protocol_consistency_verified", "source_accounting_verified")


class FrontierFigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="explicit-synthetic-v11-fixture-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evidence = self.root / "paper/evidence"
        (self.evidence / "selected_recipe").mkdir(parents=True)
        # Budget metadata is real; success counts below are deliberately
        # fictional and exist only inside the explicitly named temporary root.
        for name in ("recipe_inventory.json", "selected_recipe/category_memory.json",
                     "selected_recipe/category_ptq_recipe.json", "selected_recipe/category_bake_manifest.json"):
            shutil.copyfile(ROOT / "paper/evidence" / name, self.evidence / name)
        for target in (FIGURES, BUILDER):
            context = patch.object(target, "ROOT", self.root)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(FIGURES, "_DATA_INPUTS", {})
        context.start()
        self.addCleanup(context.stop)
        self.pair = {"fixture_only": True, **dict.fromkeys(GATES, True), "arms": {}}
        # Negative QAD and OPD effects must remain valid chart inputs.
        for name, wins in zip(BUILDER.ARMS, (14, 12, 11, 13, 9)):
            tasks = {f"fixture_task_{i}": {"episodes": 16, "successes": wins,
                                          "success_rate": wins / 16} for i in range(10)}
            self.pair["arms"][name] = {"count": 160, "successes": wins * 10,
                                        "macro_success_rate": wins / 16, "per_task": tasks}
        for name in ("qad", "continued_qad"):
            difference = (self.pair["arms"]["qad_opd"]["successes"] -
                          self.pair["arms"][name]["successes"]) / 160
            self.pair["opd_vs_" + name] = {"paired_count": 160,
                "success_rate_difference_opd_minus_baseline": difference}
        self.pair_path = self.evidence / "paired_comparison.json"
        self.write(self.pair_path, self.pair)
        exposed = {name: {"episodes": row["count"], "successes": row["successes"],
                          "macro_success_rate": row["macro_success_rate"], "per_task": row["per_task"]}
                   for name, row in self.pair["arms"].items()}
        final = {"fixture_only": True, "format": "publication_final_results_v1", "status": "complete",
                 "selected_recipe": RECIPE,
                 "source": {"heldout_comparison": BUILDER.identity(self.pair_path)},
                 "public_arms": {name: row for name, row in exposed.items() if name != "continued_qad"},
                 "control": {"continued_qad": exposed["continued_qad"]}}
        final_path = self.root / "explicit_fixture_final_results.json"
        self.write(final_path, final)
        self.frontier_path = self.evidence / "frontier_comparison.json"
        self.frontier = BUILDER.build(final_path, self.pair_path,
                                      self.evidence / "recipe_inventory.json", self.frontier_path)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def parse_frontier(self, report):
        FIGURES._DATA_INPUTS.clear()
        self.write(self.frontier_path, report)
        return FIGURES.frontier_rows()

    def parse_pair(self, report):
        FIGURES._DATA_INPUTS.clear()
        self.write(self.pair_path, report)
        return FIGURES.paired_rows()

    def test_current_five_arm_builder_output_and_negative_recovery_are_accepted(self):
        data, points = FIGURES.frontier_rows()
        self.assertEqual([p["name"] for p in points], list(BUILDER.ARMS))
        self.assertEqual(points[0]["recipe"], "bf16")
        self.assertLess(points[2]["successes"], points[1]["successes"])
        self.assertLess(points[4]["successes"], points[2]["successes"])
        _, rows = FIGURES.paired_rows()
        self.assertEqual(len(rows), 5)
        self.assertLess(self.pair["opd_vs_continued_qad"]["success_rate_difference_opd_minus_baseline"], 0)
        # Each recovery budget adds the current BF16 residual exactly once.
        for scope in points[1]["encoding_budget"]:
            base = points[1]["encoding_budget"][scope]["total_bytes"]
            for point in points[2:]:
                cost = point["encoding_budget"][scope]
                self.assertEqual(cost["total_bytes"], base + 145997824)

    def test_plot_uses_success_percentage_instead_of_160_episode_success_count(self):
        # Exercise plotting with only an in-memory SVG; never save a fixture
        # chart to either the temporary root or the real publication folder.
        with patch.object(FIGURES, "frontier_label_layout", wraps=FIGURES.frontier_label_layout) as layout, \
                patch.object(FIGURES.SVG, "save") as save:
            FIGURES.ptq_frontier()
        markers, bounds = layout.call_args.args
        _, top, _, bottom = bounds
        self.assertTrue(all(top <= marker["y"] <= bottom for marker in markers))
        bf16 = next(marker for marker in markers if marker["role"] == "bf16")
        self.assertAlmostEqual(bf16["y"], bottom - (bottom - top) * 140 / 160)
        save.assert_called_once_with("ptq_frontier")

    def test_legacy_reference_or_selected_recipe_is_rejected(self):
        for key, value in (("selected_recipe", "head_lang"),
                           ("reference_order", ["fp8", "head_ffn"]),
                           ("references", {"head_lang_vision": {}})):
            with self.subTest(key=key):
                report = copy.deepcopy(self.frontier)
                report[key] = value
                with self.assertRaises(ValueError):
                    self.parse_frontier(report)
        report = copy.deepcopy(self.frontier)
        report.pop("selected_recipe")
        with self.assertRaises(ValueError):
            self.parse_frontier(report)

    def test_both_readers_require_all_integrity_gates(self):
        for gate in GATES:
            with self.subTest(gate=gate):
                report = copy.deepcopy(self.frontier)
                report[gate] = False
                with self.assertRaises(ValueError):
                    self.parse_frontier(report)
                pair = copy.deepcopy(self.pair)
                pair.pop(gate)
                with self.assertRaises(ValueError):
                    self.parse_pair(pair)
                self.write(self.pair_path, self.pair)

    def test_legacy_100_episode_pair_and_frontier_are_rejected(self):
        pair = copy.deepcopy(self.pair)
        for row in pair["arms"].values():
            row.update(count=100, successes=50, macro_success_rate=0.5)
            for task in row["per_task"].values():
                task.update(episodes=10, successes=5, success_rate=0.5)
        with self.assertRaises(ValueError):
            self.parse_pair(pair)
        self.write(self.pair_path, self.pair)
        report = copy.deepcopy(self.frontier)
        report["points"][0].update(count=100, successes=50, macro_success_rate=0.5)
        with self.assertRaises(ValueError):
            self.parse_frontier(report)

    def test_frontier_requires_the_exact_paired_file_and_budget_files(self):
        for key in self.frontier["source"]:
            if key == "final_results":
                continue
            with self.subTest(source=key):
                report = copy.deepcopy(self.frontier)
                report["source"][key]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "source identity"):
                    self.parse_frontier(report)

    def test_arithmetic_consistent_but_different_outcomes_are_rejected(self):
        report = copy.deepcopy(self.frontier)
        row = report["points"][-1]
        row.update(successes=160, macro_success_rate=1.0)
        for task in row["per_task"].values():
            task.update(successes=16, success_rate=1.0)
        with self.assertRaisesRegex(ValueError, "paired heldout"):
            self.parse_frontier(report)

    def test_consistently_changed_adapter_bytes_are_rejected(self):
        report = copy.deepcopy(self.frontier)
        for point in report["points"][2:]:
            for cost in point["encoding_budget"].values():
                cost["residual_bytes"] += 2
                cost["total_bytes"] += 2
                cost["compression_x"] = cost["source_bytes"] / cost["total_bytes"]
        with self.assertRaisesRegex(ValueError, "selected v11 encoding budget"):
            self.parse_frontier(report)

    def test_all_budget_artifacts_cannot_consistently_switch_back_to_fp8(self):
        inventory_path = self.evidence / "recipe_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        memory = inventory["recipes"][RECIPE]
        total = memory["linear_params"]
        memory["nvfp4_params"] -= 16
        memory["fp8_params"] = 16
        memory["fraction_of_eligible_params"].update(nvfp4=(total - 16) / total, fp8=16 / total)
        self.write(inventory_path, inventory)
        recipe_path = self.evidence / "selected_recipe/category_ptq_recipe.json"
        recipe = json.loads(recipe_path.read_text())
        recipe["memory"] = memory
        self.write(recipe_path, recipe)
        memory_path = self.evidence / "selected_recipe/category_memory.json"
        summary = json.loads(memory_path.read_text())
        summary.update(memory=memory, source_recipe_sha256=BUILDER.digest(recipe_path))
        self.write(memory_path, summary)
        bake_path = self.evidence / "selected_recipe/category_bake_manifest.json"
        bake = json.loads(bake_path.read_text())
        bake["memory"] = memory
        self.write(bake_path, bake)
        with self.assertRaisesRegex(ValueError, "100% NVFP4"):
            FIGURES.budget_rows()
        with self.assertRaisesRegex(ValueError, "full NVFP4"):
            BUILDER.load_inventory(inventory_path)

    def test_missing_arm_or_bf16_recipe_mislabel_is_rejected(self):
        report = copy.deepcopy(self.frontier)
        report["points"].pop(3)
        with self.assertRaises(ValueError):
            self.parse_frontier(report)
        report = copy.deepcopy(self.frontier)
        report["points"][0]["recipe"] = RECIPE
        with self.assertRaisesRegex(ValueError, "recipe/role"):
            self.parse_frontier(report)


if __name__ == "__main__":
    unittest.main()
