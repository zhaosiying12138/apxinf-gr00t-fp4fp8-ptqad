"""Check evidence-to-chart identities without writing synthetic result figures."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("frontier_figures", Path(__file__).resolve().parents[1] / "paper/make_figs.py")
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


def fixture(references, selected):
    points = []
    for name in ["bf16", *references, "ptq", "qad", "continued_qad", "qad_opd"]:
        recovery = name in ("qad", "continued_qad", "qad_opd")
        base = 1000 if name == "bf16" else 600 if name in references else 400
        residual = 20 if recovery else 0
        cost = dict(source_bytes=1000, base_bytes=base, residual_bytes=residual,
                    total_bytes=base + residual, compression_x=1000 / (base + residual))
        points.append(dict(name=name, count=100, successes=80, macro_success_rate=.8,
                           role="recovery" if recovery else "bf16" if name == "bf16" else "ptq",
                           per_task={str(i): dict(episodes=10, successes=8, success_rate=.8) for i in range(10)},
                           encoding_budget={scope: copy.deepcopy(cost) for scope in ("physical", "known_alias_deduplicated")}))
    return dict(version=1, status="complete", environment_pairing_verified=True,
                references={name: {} for name in references}, reference_order=references,
                selected_recipe=selected, points=points)


class FrontierFigureTests(unittest.TestCase):
    def test_v3_uses_selected_ptq_budget_with_one_reference(self):
        report = fixture(["head_lang_vision"], "calib")
        with patch.object(FIGURES, "read", return_value=report):
            _, points = FIGURES.frontier_rows()
            self.assertEqual([p["name"] for p in points],
                             ["bf16", "head_lang_vision", "ptq", "qad", "continued_qad", "qad_opd"])
        altered = copy.deepcopy(report)
        # Consistent arithmetic is insufficient: the recovered base must be
        # the selected PTQ point, not the adjacent higher-precision reference.
        cost = altered["points"][3]["encoding_budget"]["physical"]
        cost.update(base_bytes=600, total_bytes=620, compression_x=1000 / 620)
        with patch.object(FIGURES, "read", return_value=altered), self.assertRaises(ValueError):
            FIGURES.frontier_rows()

    def test_missing_frozen_reference_is_rejected(self):
        report = fixture(["head_lang_vision"], "calib")
        report["points"].pop(1)
        with patch.object(FIGURES, "read", return_value=report), self.assertRaises(ValueError):
            FIGURES.frontier_rows()

    def test_legacy_seven_arm_report_remains_readable(self):
        report = fixture(["fp8", "head_ffn"], "head_lang")
        report.pop("reference_order")
        report.pop("selected_recipe")
        with patch.object(FIGURES, "read", return_value=report):
            self.assertEqual(len(FIGURES.frontier_rows()[1]), 7)


if __name__ == "__main__":
    unittest.main()
