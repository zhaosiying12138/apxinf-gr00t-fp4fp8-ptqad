"""CPU tests of frozen supplementary statistics and evidence guards."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval import compare_gptq_reference as comparison
from exp import bake_gptq_reference_category as launcher


PLAN = Path(__file__).resolve().parents[1] / "exp/gptq_reference_protocol_v12.json"


def fixture():
    plan = json.loads(PLAN.read_text())
    plan["comparison"]["confidence_interval"]["replicates"] = 200
    success_counts = {"bf16": 12, "ptq": 6, "gptq": 9, "qad": 11, "continued_qad": 10, "qad_opd": 12}
    arms = {arm: [{"task": task, "episode_index": i, "seed": 970000 + ti * 1000 + i,
                   "init_state_index": plan["execution"]["eval_partition"]["init_state_indices"][i],
                   "initial_state_sha256": hashlib.sha256(f"{ti}/{i}".encode()).hexdigest(),
                   "success": i < successes}
                  for ti, task in enumerate(comparison.TASKS) for i in range(16)]
            for arm, successes in success_counts.items()}
    return plan, arms


class SupplementComparisonTests(unittest.TestCase):
    def test_category_launcher_preserves_venv_and_fixed_damping(self):
        plan, _ = fixture()
        root = Path("fixture").absolute()
        python = root / "venv/bin/python"
        command = launcher.command_for(plan, python, root)
        self.assertEqual(command[0], str(python))
        self.assertEqual(command[command.index("--gptq-damp") + 1], "0.01")
        self.assertEqual(command[command.index("--expected-windows") + 1], "148")
        self.assertEqual(command[command.index("--method") + 1], "gptq_active")

    def test_category_launcher_cannot_start_while_main_training_is_incomplete(self):
        plan, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / plan["main_run"]
            run.mkdir(parents=True)
            (run / "run_manifest.json").write_text(json.dumps({"status": "initialized", "python": "/venv/bin/python"}))
            with patch.object(launcher, "ROOT", root), patch.object(launcher, "preflight", return_value=(plan, {})), \
                    patch.object(launcher.sys, "argv", ["bake_gptq_reference_category.py"]), \
                    patch.object(launcher.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "Main five-arm run is incomplete"):
                    launcher.main()
                execute.assert_not_called()
            self.assertFalse((root / plan["execution"]["output_root"]).exists())

    def test_main_bf16_must_match_calibration_teacher_bytes_and_metadata(self):
        root = Path("teacher").resolve()
        weight = {"bytes": 123, "sha256": "frozen_weight"}
        files = {str(root / "model.safetensors"): weight,
                 str(root / "processor_config.json"): {"bytes": 42, "sha256": "frozen_processor"}}
        plan = {"teacher_weights": {"model.safetensors": weight},
                "teacher_metadata": {"processor_config.json": "frozen_processor"}}
        comparison.verify_bf16_identity(root, files, plan)
        for name in ("model.safetensors", "processor_config.json"):
            with self.subTest(name=name):
                changed = copy.deepcopy(files)
                changed[str(root / name)]["sha256"] = "different"
                with self.assertRaises(ValueError):
                    comparison.verify_bf16_identity(root, changed, plan)

    def test_fixed_contrast_directions_counts_and_seed(self):
        plan, arms = fixture()
        with patch.object(comparison, "paired_effect", wraps=comparison.paired_effect) as effect:
            report = comparison.summarize(arms, plan)
        self.assertEqual(effect.call_count, 3)
        for call in effect.call_args_list:
            self.assertEqual(call.kwargs, {"replicates": 200, "seed": 20261007, "level": .95})
        self.assertEqual(report["arms"]["gptq"]["successes"], 90)
        self.assertEqual(report["arms"]["gptq"]["episodes"], 160)
        self.assertEqual(report["arms"]["gptq"]["macro_success_rate"], 9 / 16)
        expected = {"gptq_vs_rtn": 18.75, "qad_vs_gptq": 12.5, "opd_vs_gptq": 18.75}
        self.assertEqual({k: v["difference_pp"] for k, v in report["contrasts"].items()}, expected)
        raw = {k: v["exact_mcnemar_two_sided_p"] for k, v in report["contrasts"].items()}
        self.assertEqual({k: v["holm_adjusted_p"] for k, v in report["contrasts"].items()},
                         comparison.holm_adjust(raw))

    def test_descriptive_comparisons_carry_no_inferential_claims(self):
        plan, arms = fixture()
        rows = comparison.summarize(arms, plan)["descriptive_comparisons"]
        self.assertEqual([(r["baseline"], r["treatment"]) for r in rows],
                         [("bf16", "gptq"), ("gptq", "continued_qad")])
        self.assertEqual(rows[0]["difference_pp"], -18.75)
        for row in rows:
            self.assertNotIn("exact_mcnemar_two_sided_p", row)
            self.assertNotIn("holm_adjusted_p", row)
            self.assertNotIn("pointwise_ci_pp", row)

    def test_negative_opd_result_is_retained(self):
        plan, arms = fixture()
        for row in arms["qad_opd"]:
            row["success"] = row["episode_index"] < 3
        result = comparison.summarize(arms, plan)
        self.assertEqual(result["contrasts"]["opd_vs_gptq"]["difference_pp"], -37.5)
        self.assertEqual(len(result["contrasts"]), 3)
        self.assertEqual(len(result["arms"]), 6)

    def test_missing_arm_episode_or_boolean_outcome_fails(self):
        for mutation in ("missing_arm", "missing_episode", "integer_success"):
            with self.subTest(mutation=mutation):
                plan, arms = fixture()
                if mutation == "missing_arm":
                    arms.pop("gptq")
                elif mutation == "missing_episode":
                    arms["gptq"].pop()
                else:
                    arms["gptq"][0]["success"] = 1
                with self.assertRaises(ValueError):
                    comparison.summarize(arms, plan)

    def test_changed_reset_or_duplicate_pair_fails(self):
        for field in ("seed", "init_state_index", "initial_state_sha256", "duplicate"):
            with self.subTest(field=field):
                plan, arms = fixture()
                if field == "duplicate":
                    for rows in arms.values():
                        rows[1] = copy.deepcopy(rows[0])
                else:
                    arms["gptq"][0][field] = "different"
                with self.assertRaises(ValueError):
                    comparison.summarize(arms, plan)

    def test_plan_or_source_byte_change_fails_before_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = root / "main.json"
            main.write_text(json.dumps({"partitions": {"heldout": {"seed": 970000}}, "evaluation_contract": {}}))
            capture = root / "inputs.json"
            capture.write_text("{}")
            source = root / "numerical.py"
            source.write_text("# frozen numerical implementation\n")
            previous = root / "previous.json"
            previous.write_text("{}")
            def record(path):
                return comparison.identity(path) | {"path": path.name}
            plan = {"format": "w4a4_gptq_supplement_protocol_v1", "main_protocol": record(main),
                    "capture_manifest": record(capture), "preparation_source_files": {source.name: record(source)},
                    "amendment": {"previous_protocol_path": previous.name,
                                  "previous_protocol_sha256": comparison.identity(previous)["sha256"]},
                    "execution": {"eval_partition": {"seed": 970000}, "eval_contract": {}}}
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with patch.object(comparison, "PLAN_SHA256", comparison.identity(path)["sha256"]):
                _, records = comparison.preflight(path, root)
                self.assertEqual(len(records), 3)
                source.write_text("# changed numerical implementation\n")
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    comparison.preflight(path, root)
                path.write_text(json.dumps(plan) + "\n")
                with self.assertRaisesRegex(ValueError, "Supplement protocol changed"):
                    comparison.preflight(path, root)


if __name__ == "__main__":
    unittest.main()
