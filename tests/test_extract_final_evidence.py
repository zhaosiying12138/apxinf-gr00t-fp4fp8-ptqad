import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper.extract_final_evidence import extract


TASKS = [f"task_{i}" for i in range(10)]


def _arm(successes):
    outcomes = [i < successes for i in range(160)]
    per_task = {}
    episodes = []
    for task_index, task in enumerate(TASKS):
        task_outcomes = outcomes[task_index * 16:(task_index + 1) * 16]
        k = sum(task_outcomes)
        per_task[task] = {"successes": k, "episodes": 16, "success_rate": k / 16}
        episodes.extend({"task": task, "episode_index": i, "success": value}
                        for i, value in enumerate(task_outcomes))
    return {"successes": successes, "count": 160,
            "macro_success_rate": sum(v["success_rate"] for v in per_task.values()) / 10,
            "per_task": per_task, "episodes": episodes}


class FinalEvidenceAdapterTests(unittest.TestCase):
    def _fixture(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        protocol = root / "recovery_protocol_v11_w4a4_category.json"
        protocol.write_text("{\"version\":11,\"w4a4\":true}\n", encoding="utf-8")
        round_dir = root / "artifacts" / "heldout_round"
        round_dir.mkdir(parents=True)
        comparison = {
            "environment_pairing_verified": True,
            "protocol_consistency_verified": True,
            "source_accounting_verified": True,
            "arms": {name: _arm(score) for name, score in
                     (("bf16", 95), ("ptq", 72), ("qad", 89),
                      ("continued_qad", 86), ("qad_opd", 92))},
        }
        comparison_path = round_dir / "paired_comparison.json"
        comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
        final = {
            "format": "w4a4_recovery_v11_final_manifest",
            "protocol_file": str(protocol),
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "selection_uses_heldout": False,
            "selected_pressure_recipe": "calib_category",
            "selected_ptq_checkpoint": str(root / "selected_checkpoint"),
            "heldout_round": str(round_dir),
            "heldout_comparison": {"path": str(comparison_path), "bytes": comparison_path.stat().st_size,
                                   "sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest()},
            "required_arms": ["bf16", "ptq", "qad", "continued_qad", "qad_opd"],
            "selected_qad_learning_rate": 0.0001,
            "selected_opd_weight": 0.25,
        }
        (root / "final_manifest.json").write_text(json.dumps(final), encoding="utf-8")
        return tmp, root

    def test_extracts_only_heldout_public_arms(self):
        tmp, root = self._fixture()
        self.addCleanup(tmp.cleanup)
        result = extract(root)
        self.assertEqual(set(result["public_arms"]), {"bf16", "ptq", "qad", "qad_opd"})
        self.assertEqual(result["control"]["continued_qad"]["successes"], 86)
        self.assertAlmostEqual(result["deltas"]["qad_minus_ptq_pp"], 17 / 160 * 100)
        self.assertAlmostEqual(result["deltas"]["qad_opd_minus_qad_pp"], 3 / 160 * 100)
        self.assertEqual(len(result["uncertainty"]["contrasts"]), 4)
        self.assertNotIn("dev_qad", json.dumps(result))

    def test_rejects_heldout_dependent_selection(self):
        tmp, root = self._fixture()
        self.addCleanup(tmp.cleanup)
        path = root / "final_manifest.json"
        data = json.loads(path.read_text())
        data["selection_uses_heldout"] = True
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "independent"):
            extract(root)

    def test_rejects_incomplete_comparison(self):
        tmp, root = self._fixture()
        self.addCleanup(tmp.cleanup)
        comparison = root / "artifacts" / "heldout_round" / "paired_comparison.json"
        data = json.loads(comparison.read_text())
        data["arms"]["qad_opd"]["count"] = 99
        comparison.write_text(json.dumps(data))
        final = json.loads((root / "final_manifest.json").read_text())
        final["heldout_comparison"]["bytes"] = comparison.stat().st_size
        final["heldout_comparison"]["sha256"] = hashlib.sha256(comparison.read_bytes()).hexdigest()
        (root / "final_manifest.json").write_text(json.dumps(final))
        with self.assertRaisesRegex(ValueError, "complete 160-episode"):
            extract(root)

    def test_rejects_episode_task_totals_that_disagree_with_declared_totals(self):
        tmp, root = self._fixture()
        self.addCleanup(tmp.cleanup)
        comparison = root / "artifacts/heldout_round/paired_comparison.json"
        data = json.loads(comparison.read_text())
        for arm in data["arms"].values():
            for i, row in enumerate(arm["episodes"]):
                row.update(task=TASKS[i // 32], episode_index=i % 32)
        comparison.write_text(json.dumps(data))
        final_path = root / "final_manifest.json"
        final = json.loads(final_path.read_text())
        final["heldout_comparison"].update(bytes=comparison.stat().st_size,
                                         sha256=hashlib.sha256(comparison.read_bytes()).hexdigest())
        final_path.write_text(json.dumps(final))
        with self.assertRaisesRegex(ValueError, "sixteen distinct"):
            extract(root)

    def test_does_not_attach_current_analysis_plan_to_historical_experiment(self):
        tmp, root = self._fixture()
        self.addCleanup(tmp.cleanup)
        path = root / "final_manifest.json"
        final = json.loads(path.read_text())
        final["format"] = "high_fp4_v7_final_manifest"
        path.write_text(json.dumps(final))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            extract(root)


if __name__ == "__main__":
    unittest.main()
