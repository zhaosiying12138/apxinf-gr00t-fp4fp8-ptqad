"""Publication guards use synthetic counts, never experimental evidence."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
from v12_publication_contract import PROTOCOL_NAME, PROTOCOL_SHA256, validate_v12_results


def fixture():
    row = {"successes": 80, "episodes": 160, "success_rate": .5, "macro_success_rate": .5,
           "per_task": {f"synthetic_{i}": {"successes": 8, "episodes": 16, "success_rate": .5} for i in range(10)}}
    return {"format": "publication_final_results_v1", "status": "complete",
            "source": {"protocol": {"path": PROTOCOL_NAME, "sha256": PROTOCOL_SHA256}},
            "selected_recipe": "rtn_w4a4_category", "selection": {"selection_uses_heldout": False},
            "public_arms": {arm: copy.deepcopy(row) for arm in ("bf16", "ptq", "qad", "qad_opd")},
            "control": {"continued_qad": copy.deepcopy(row)}}


class PublicationGuardTests(unittest.TestCase):
    def test_accepts_complete_v12_counts(self):
        validate_v12_results(fixture(), root=ROOT)

    def test_rejects_old_or_incomplete_protocol(self):
        for mutation in (lambda d: d.update(status="running"),
                         lambda d: d["source"]["protocol"].update(sha256="0" * 64),
                         lambda d: d["selection"].update(selection_uses_heldout=True),
                         lambda d: d["control"].clear()):
            d = fixture(); mutation(d)
            with self.assertRaises(ValueError): validate_v12_results(d, root=ROOT)

    def test_rejects_missing_task_or_mismatched_rate(self):
        for mutation in (lambda r: r.update(episodes=150),
                         lambda r: r.update(success_rate=.99),
                         lambda r: r["per_task"].pop("synthetic_0"),
                         lambda r: r["per_task"]["synthetic_0"].update(successes=17)):
            d = fixture(); mutation(d["public_arms"]["qad"])
            with self.assertRaises(ValueError): validate_v12_results(d, root=ROOT)

    def test_method_preserves_latex_commands(self):
        value = (ROOT / "paper/templates/v12/03-方法.md").read_text()
        self.assertIn(r"\mathrm{E4M3}", value)
        self.assertIn(r"\frac{\alpha}{r}", value)
        self.assertNotIn("\t", value)
        self.assertNotIn("\x0c", value)
        self.assertNotIn(r"\\mathrm", value)


if __name__ == "__main__": unittest.main()
