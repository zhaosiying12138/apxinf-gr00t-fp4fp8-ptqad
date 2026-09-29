"""CPU-only tests for the five-arm heldout evidence gate."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "eval"))

from compare_recovery import ARMS, compare_round  # noqa: E402
from run_recovery_eval import TASKS  # noqa: E402


OUTCOMES = {
    "bf16": [True] * 9 + [False],
    "ptq": [True] * 4 + [False] * 6,
    "qad": [True] * 7 + [False] * 3,
    "continued_qad": [True] * 6 + [False] * 4,
    "qad_opd": [True] * 8 + [False] * 2,
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def state_hash(task_index, episode, field):
    return hashlib.sha256(f"{task_index}/{episode}/{field}".encode()).hexdigest()


def make_round(root):
    """Write a complete generator-shaped versioned five-arm fixture."""
    protocol = root / "frozen" / "recovery_protocol_v3_high_fp4.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text('{"version":"v3-test"}\n')
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    for arm in ARMS:
        folder = root / f"heldout_{arm}"
        folder.mkdir()
        manifest = {
            "purpose": "heldout",
            "tasks": TASKS,
            "task_count": 10,
            "episodes": 10,
            "seed": 660000,
            "init_state_indices": list(range(30, 40)),
            "initial_state_protocol": "libero10_official_bank_v1",
            "n_envs": 1,
            "task_seed_stride": 1000,
            "episode_seed_stride": 1,
            "server_seed_offset": 10000000,
            "n_action_steps": 8,
            "max_episode_steps": 720,
            "settle_steps": 10,
            "protocol_file": str(protocol),
            "protocol_sha256": protocol_sha,
        }
        write_json(folder / "eval_manifest.json", manifest)
        results = {}
        for task_index, task in enumerate(TASKS):
            outcomes = list(OUTCOMES[arm])
            resets = [
                {
                    "episode_index": episode,
                    "seed": manifest["seed"] + 1000 * task_index + episode,
                    "init_state_index": 30 + episode,
                    "settle_steps": 10,
                    "initial_state_sha256": state_hash(task_index, episode, "initial"),
                    "restored_state_sha256": state_hash(task_index, episode, "restored"),
                    "init_state_bank_sha256": state_hash(task_index, 0, "bank"),
                }
                for episode in range(10)
            ]
            # The real rollout currently emits two post-score reset records.
            resets.extend(dict(resets[-1]) for _ in range(2))
            results[task] = {
                "results": outcomes,
                "successes": sum(outcomes),
                "episodes": len(outcomes),
                "success_rate": sum(outcomes) / len(outcomes),
                "resets": resets,
                "returncode": 0,
                "seed": manifest["seed"] + 1000 * task_index,
            }
        write_json(folder / "task_results.json", results)
        write_json(folder / "summary.json", {
            "tasks_complete": 10,
            "total_successes": sum(sum(row["results"]) for row in results.values()),
            "total_episodes": 100,
            "macro_success_rate": sum(row["success_rate"] for row in results.values()) / 10,
            "purpose": "heldout",
            "seed": manifest["seed"],
        })


class ComparisonEvidenceGateTests(unittest.TestCase):
    def test_reports_full_recovery_ladder_without_inferential_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_round(root)
            report = compare_round(root)

        self.assertTrue(report["environment_pairing_verified"])
        self.assertTrue(report["protocol_consistency_verified"])
        self.assertTrue(report["source_accounting_verified"])
        self.assertEqual({arm: report["arms"][arm]["successes"] for arm in ARMS}, {
            "bf16": 90, "ptq": 40, "qad": 70,
            "continued_qad": 60, "qad_opd": 80,
        })
        self.assertAlmostEqual(
            report["ptq_vs_bf16"]["success_rate_difference_treatment_minus_baseline"], -.5)
        self.assertAlmostEqual(
            report["qad_vs_ptq"]["success_rate_difference_treatment_minus_baseline"], .3)
        self.assertAlmostEqual(
            report["opd_vs_ptq"]["success_rate_difference_treatment_minus_baseline"], .4)
        self.assertAlmostEqual(
            report["opd_vs_qad"]["success_rate_difference_opd_minus_baseline"], .1)
        recovery = report["recovery_relative_to_bf16"]
        self.assertAlmostEqual(recovery["bf16_minus_ptq_success_rate_gap"], .5)
        self.assertAlmostEqual(
            recovery["methods"]["qad"]["fraction_of_bf16_minus_ptq_gap_recovered"], .6)
        self.assertAlmostEqual(
            recovery["methods"]["qad_opd"]["fraction_of_bf16_minus_ptq_gap_recovered"], .8)
        self.assertIn("not an inferential statistic", recovery["interpretation"])
        self.assertIn("No automatic significance decision", report["statistical_note"])

    def test_rejects_summary_and_raw_result_accounting_mismatches(self):
        cases = (
            ("summary total", "summary.json",
             lambda value: value.__setitem__("total_successes", 41), "summary total_successes"),
            ("summary macro", "summary.json",
             lambda value: value.__setitem__("macro_success_rate", .99), "summary macro_success_rate"),
            ("raw successes", "task_results.json",
             lambda value: value[TASKS[0]].__setitem__("successes", 5), "raw-result successes"),
            ("raw rate", "task_results.json",
             lambda value: value[TASKS[0]].__setitem__("success_rate", .5), "raw-result success_rate"),
            ("raw seed", "task_results.json",
             lambda value: value[TASKS[0]].__setitem__("seed", 1), "raw-result seed"),
        )
        for label, filename, mutation, error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_round(root)
                mutate_json(root / "heldout_ptq" / filename, mutation)
                with self.assertRaisesRegex(ValueError, error):
                    compare_round(root)

    def test_rejects_non_boolean_outcome_and_protocol_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_round(root)
            path = root / "heldout_qad" / "task_results.json"
            mutate_json(path, lambda value: value[TASKS[0]]["results"].__setitem__(0, 1))
            with self.assertRaisesRegex(ValueError, "outcomes must be Boolean"):
                compare_round(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_round(root)
            path = root / "heldout_qad" / "eval_manifest.json"
            mutate_json(path, lambda value: value.__setitem__("n_action_steps", 4))
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                compare_round(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_round(root)
            path = root / "heldout_qad" / "eval_manifest.json"
            mutate_json(path, lambda value: value.__setitem__("protocol_sha256", "not-a-sha"))
            with self.assertRaisesRegex(ValueError, "invalid execution protocol"):
                compare_round(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_round(root)
            path = root / "heldout_qad" / "eval_manifest.json"
            mutate_json(path, lambda value: value.__setitem__("protocol_sha256", "f" * 64))
            with self.assertRaisesRegex(ValueError, "protocol file SHA256"):
                compare_round(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_round(root)
            path = root / "heldout_qad" / "eval_manifest.json"
            mutate_json(path, lambda value: value.__setitem__("protocol_file", str(root / "missing.json")))
            with self.assertRaisesRegex(ValueError, "protocol file is missing"):
                compare_round(root)

    def test_rejects_any_paired_state_hash_mismatch(self):
        fields = ("initial_state_sha256", "restored_state_sha256", "init_state_bank_sha256")
        for field in fields:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_round(root)
                path = root / "heldout_qad_opd" / "task_results.json"
                mutate_json(
                    path,
                    lambda value, field=field: value[TASKS[3]]["resets"][4].__setitem__(field, "f" * 64),
                )
                with self.assertRaisesRegex(ValueError, "does not pair"):
                    compare_round(root)

    def test_legacy_input_is_readable_but_not_certified_as_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_round(root)
            for arm in ARMS:
                path = root / f"heldout_{arm}" / "eval_manifest.json"
                mutate_json(path, lambda value: (value.pop("protocol_file"), value.pop("protocol_sha256")))
            report = compare_round(root)
        self.assertTrue(report["environment_pairing_verified"])
        self.assertFalse(report["protocol_consistency_verified"])
        self.assertTrue(report["source_accounting_verified"])


if __name__ == "__main__":
    unittest.main()
