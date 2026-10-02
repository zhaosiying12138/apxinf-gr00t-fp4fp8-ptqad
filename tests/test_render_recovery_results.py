"""Synthetic rendering fixtures stay in temporary storage, never paper evidence."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper import render_recovery_results as module
from eval.run_recovery_eval import TASKS

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def complete_v11_fixture(test):
    """Real comparison/extractor interfaces over explicitly synthetic files."""
    tmp = tempfile.TemporaryDirectory(prefix="explicit-synthetic-v11-render-")
    test.addCleanup(tmp.cleanup)
    run = Path(tmp.name)
    protocol = run / "recovery_protocol_v11_w4a4_category.json"
    protocol.write_bytes((ROOT / "exp/recovery_protocol_v11_w4a4_category.json").read_bytes())
    spec = json.loads(protocol.read_text())
    part = spec["partitions"]["heldout"]
    indices, seed = part["init_state_indices"], part["seed"]
    round_dir = run / "artifacts/heldout_round"
    for arm, score in zip(module.LABELS, (80, 90, 95, 100, 96)):
        folder = round_dir / ("heldout_" + arm)
        manifest = {"fixture_only": True, "purpose": "heldout", "seed": seed,
                    "episodes": 16, "tasks": TASKS, "task_count": 10, "n_envs": 1,
                    "task_seed_stride": 1000, "episode_seed_stride": 1,
                    "server_seed_offset": 10000000, "n_action_steps": 8,
                    "max_episode_steps": 720, "settle_steps": 10,
                    "initial_state_protocol": "libero10_official_bank_v1",
                    "init_state_indices": indices, "protocol_file": str(protocol),
                    "protocol_sha256": module.file_identity(protocol)["sha256"]}
        tasks = {}
        for ti, task in enumerate(TASKS):
            outcomes = [ti * 16 + i < score for i in range(16)]
            tasks[task] = {"returncode": 0, "results": outcomes, "episodes": 16,
                           "successes": sum(outcomes), "success_rate": sum(outcomes) / 16,
                           "seed": seed + ti * 1000,
                           "resets": [{"episode_index": i, "seed": seed + ti * 1000 + i,
                                       "init_state_index": bank, "settle_steps": 10,
                                       "initial_state_sha256": f"{ti * 100 + i:064x}",
                                       "restored_state_sha256": f"{ti * 100 + i + 3000:064x}",
                                       "init_state_bank_sha256": f"{ti + 7000:064x}"}
                                      for i, bank in enumerate(indices)]}
        write_json(folder / "eval_manifest.json", manifest)
        write_json(folder / "task_results.json", tasks)
        write_json(folder / "summary.json", {"tasks_complete": 10, "total_successes": score,
                   "total_episodes": 160, "macro_success_rate": score / 160,
                   "purpose": "heldout", "seed": seed})
    pair_path = round_dir / "paired_comparison.json"
    comparison = module.collect_pairing_evidence.compare_round(round_dir)
    write_json(pair_path, comparison)
    write_json(run / "final_manifest.json", {
        "fixture_only": True, "format": "w4a4_recovery_v11_final_manifest",
        "protocol_file": str(protocol), "protocol_sha256": module.file_identity(protocol)["sha256"],
        "selection_uses_heldout": False, "selected_pressure_recipe": "all_nvfp4_gptq_category",
        "selected_ptq_checkpoint": str(run / "synthetic_checkpoint_identity"),
        "heldout_round": str(round_dir), "heldout_comparison": module.file_identity(pair_path),
        "required_arms": list(module.LABELS), "selected_qad_learning_rate": 5e-5,
        "selected_opd_weight": 0.25})
    final_path = run / "explicit_synthetic_final_results.json"
    module.extract_final_evidence.extract(run, final_path)
    return run, final_path


def fixture():
    arms = {}
    for name, score in zip(module.LABELS, (80, 90, 95, 100, 96)):
        per_task = {f"task_{i}": {"episodes": 16, "successes": min(16, max(0, score - i*16))}
                    for i in range(10)}
        arms[name] = {"episodes": 160, "successes": score, "success_rate": score/160,
                      "per_task": per_task}
    contrasts = {}
    for key, baseline, treatment in (("ptq_vs_bf16", "bf16", "ptq"),
                                     ("qad_vs_ptq", "ptq", "qad"),
                                     ("opd_vs_qad", "qad", "qad_opd"),
                                     ("opd_vs_continued_qad", "continued_qad", "qad_opd")):
        delta = 100*(arms[treatment]["success_rate"] - arms[baseline]["success_rate"])
        contrasts[key] = {"difference_pp": delta, "pointwise_ci_pp": [-3., 8.],
                          "discordant_episodes": 10, "exact_mcnemar_two_sided_p": .5,
                          "holm_adjusted_p": 1., "zero_discordance_warning": False}
    return {"public_arms": {k: v for k, v in arms.items() if k != "continued_qad"},
            "control": {"continued_qad": arms["continued_qad"]}, "uncertainty": {"contrasts": contrasts}}


class RenderResultsTests(unittest.TestCase):
    def test_complete_v11_extract_pairing_and_render_without_mock_validators(self):
        run, final_path = complete_v11_fixture(self)
        output = run / "explicit_synthetic_inserts"
        report = module.build(final_path, output)
        self.assertEqual(set(report["files"]), {"main_table.md", "paired_effects.md", "per_task.md", "summary.md"})
        self.assertIn("100/160", (output / "main_table.md").read_text())
        self.assertIn("96/160", (output / "main_table.md").read_text())
        self.assertIn("-2.50", (output / "paired_effects.md").read_text())
        manifest = json.loads((output / "render_manifest.json").read_text())
        sources = {Path(row["path"]).name for row in manifest["sources"]}
        self.assertTrue({"extract_final_evidence.py", "collect_pairing_evidence.py"} <= sources)
        self.assertTrue(all(module.file_identity(row["path"]) == row for row in manifest["sources"]))

    def test_source_change_during_real_pairing_verification_creates_no_output(self):
        run, final_path = complete_v11_fixture(self)
        output = run / "rejected_inserts"
        source = run / "artifacts/heldout_round/heldout_qad/task_results.json"
        original_audit = module.collect_pairing_evidence.audit

        def audit_then_change(*args, **kwargs):
            result = original_audit(*args, **kwargs)
            source.write_bytes(source.read_bytes() + b"\n")
            return result

        with patch.object(module.collect_pairing_evidence, "audit", side_effect=audit_then_change):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                module.build(final_path, output)
        self.assertFalse(output.exists())

    def test_renderer_rejects_nonselected_recipe_before_reading_run(self):
        with tempfile.TemporaryDirectory(prefix="explicit-synthetic-render-recipe-") as temp:
            path = Path(temp) / "final.json"
            write_json(path, {"format": "publication_final_results_v1", "status": "complete",
                              "selected_recipe": "mixed"})
            with self.assertRaisesRegex(ValueError, "unique all_nvfp4_gptq_category"):
                module.load_verified(path)

    def test_reports_negative_opd_effect_and_ptq_improvement_without_selecting(self):
        texts = module.render(fixture())
        self.assertIn("+6.25", texts["paired_effects.md"])
        self.assertIn("-2.50", texts["paired_effects.md"])
        self.assertIn("100/160", texts["main_table.md"])
        self.assertIn("96/160", texts["main_table.md"])
        self.assertEqual(texts["per_task.md"].count("`task_"), 10)
        self.assertNotIn("显著提升", texts["summary.md"])

    def test_incomplete_control_or_contrasts_rejected(self):
        for field in ("control", "contrast"):
            data = fixture()
            if field == "control":
                data["control"] = {}
            else:
                data["uncertainty"]["contrasts"].pop("opd_vs_continued_qad")
            with self.assertRaises(ValueError):
                module.render(data)

    def test_zero_discordance_is_not_reported_as_equivalence(self):
        data = fixture()
        data["uncertainty"]["contrasts"]["opd_vs_qad"]["zero_discordance_warning"] = True
        self.assertIn("不代表总体差值", module.render(data)["paired_effects.md"])

    def test_writes_separate_inserts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "synthetic-inserts"
            with patch.object(module, "load_verified", return_value=(copy.deepcopy(fixture()), [])):
                module.build("unused-synthetic-fixture", output)
                self.assertEqual(len(list(output.iterdir())), 5)
                with self.assertRaisesRegex(ValueError, "refusing existing"):
                    module.build("unused-synthetic-fixture", output)


if __name__ == "__main__":
    unittest.main()
