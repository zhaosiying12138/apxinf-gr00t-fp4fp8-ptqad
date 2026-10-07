"""CPU-only final-reference tests using complete raw logs and real file hashes."""
import json
from pathlib import Path
import tempfile
import unittest

from exp import independent_action_protocol as diagnostic
from exp.run_high_fp4_v3 import model_id
from eval.compare_recovery import compare_round
from eval.run_recovery_eval import TASKS, parse_log


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def mutate(path, operation):
    data = diagnostic.read(path)
    operation(data)
    write(path, data)


def fixture(root):
    """A complete tiny five-arm run; model shards are inert CPU fixture bytes."""
    protocol = root / "recovery_protocol_v12_rtn_w4a4.json"
    protocol.write_bytes((diagnostic.ROOT / "exp" / protocol.name).read_bytes())
    p = diagnostic.read(protocol)
    part = p["partitions"]["heldout"]
    models = {}
    for arm in diagnostic.ARMS:
        folder = root / "models" / arm
        folder.mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(("test only " + arm).encode())
        write(folder / "config.json", {"fixture_only": True})
        write(folder / "statistics.json", {"normalization": "same fixture"})
        if arm == "ptq":
            write(folder / "ptq_recipe.json", {"fixture_only": True})
        elif arm not in ("bf16", "ptq"):
            training = root / "training" / arm
            training.mkdir(parents=True)
            (training / "model.safetensors").write_bytes(("test adapter " + arm).encode())
            def weights(path):
                return {key: value for key, value in diagnostic.identity(path / "model.safetensors").items()
                        if key != "path"}
            write(folder / "merge_manifest.json", {"status": "complete", "base": str(root / "models/ptq"),
                  "training_checkpoint": str(training),
                  "base_weights": {"model.safetensors": weights(root / "models/ptq")},
                  "training_weights": {"model.safetensors": weights(training)}})
        models[arm] = model_id(folder)
    selection = root / "selection.json"
    write(selection, {"protocol_sha256": diagnostic.file_sha256(protocol), "selection_uses_heldout": False,
                      "selected_recipe": "rtn_w4a4_category",
                      "arms": {"bf16": {"model_identity": models["bf16"]},
                               "rtn_w4a4_category": {"model_identity": models["ptq"]}}})
    round_dir = root / "heldout"
    for arm in diagnostic.ARMS:
        folder = round_dir / ("heldout_" + arm)
        folder.mkdir(parents=True)
        manifest = {key: p["evaluation_contract"][key] for key in diagnostic.CONTRACT_FIELDS}
        manifest.update(purpose="heldout", tasks=TASKS, seed=part["seed"], episodes=16,
                        init_state_indices=part["init_state_indices"], protocol_file=str(protocol),
                        protocol_sha256=diagnostic.file_sha256(protocol), checkpoint=models[arm]["path"])
        write(folder / "eval_manifest.json", manifest)
        rows = {}
        for ti, task in enumerate(TASKS):
            resets = [{"episode_index": ei, "seed": part["seed"] + ti * 1000 + ei,
                       "init_state_index": bank, "settle_steps": 10,
                       "initial_state_sha256": f"{ti * 100 + ei:064x}",
                       "restored_state_sha256": f"{ti * 100 + ei + 1000:064x}",
                       "init_state_bank_sha256": f"{ti + 3000:064x}"}
                      for ei, bank in enumerate(part["init_state_indices"])]
            log = folder / (task + ".log")
            log.write_text("\n".join("FP4VLA_EPISODE_RESET " + json.dumps(row) for row in resets) +
                           "\nresults: ('fixture', " + repr([True, False] * 8) + ")\n")
            (folder / (task + ".server.log")).write_text("fixture only server output\n")
            rows[task] = {**parse_log(log), "seed": part["seed"] + ti * 1000, "returncode": 0}
        write(folder / "task_results.json", rows)
        write(folder / "summary.json", {"tasks_complete": 10, "total_successes": 80, "total_episodes": 160,
                                        "macro_success_rate": .5, "purpose": "heldout", "seed": part["seed"]})
    comparison = round_dir / "paired_comparison.json"
    write(comparison, compare_round(round_dir))
    final = root / "final_manifest.json"
    write(final, {"format": "w4a4_recovery_v12_final_manifest", "protocol_file": str(protocol),
                  "protocol_sha256": diagnostic.file_sha256(protocol), "selection_uses_heldout": False,
                  "required_arms": list(diagnostic.ARMS), "selected_pressure_recipe": "rtn_w4a4_category",
                  "selected_ptq_checkpoint": models["ptq"]["path"], "selection_file": str(selection),
                  "selection_sha256": diagnostic.file_sha256(selection), "heldout_round": str(round_dir),
                  "heldout_comparison": diagnostic.identity(comparison),
                  "selected_qad_model_identity": models["qad"],
                  "selected_continued_model_identity": models["continued_qad"],
                  "selected_opd_model_identity": models["qad_opd"]})
    write(root / "run_manifest.json", {"status": "complete", "output_layout": "stable_paths_v2",
                                      "protocol_sha256": diagnostic.file_sha256(protocol)})
    return final


class IndependentActionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.final = fixture(self.root)
        self.output = self.root / "independent/protocol.json"

    def prepare(self):
        return diagnostic.prepare(diagnostic.DRAFT, self.final, self.output)

    def test_real_final_audit_freezes_both_protocols_and_all_five_weights(self):
        original = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        prepared = self.prepare()
        self.assertEqual(prepared["status"], "prepared_not_executed")
        report = diagnostic.validate(self.output, self.final, self.root / "models/bf16")
        self.assertEqual(report["status"], "verified")
        self.assertEqual(set(report["arms"]), set(diagnostic.ARMS))
        self.assertNotEqual(report["diagnostic_protocol_sha256"], report["original_model_protocol_sha256"])
        self.assertEqual(report["source_final"], diagnostic.identity(self.final))
        self.assertEqual(report["settings"]["noise_seeds"], [2026100700, 2026100701])
        self.assertEqual(report["data"]["partitions"]["diagnostics"]["init_state_indices"], [29])
        self.assertEqual(report["raw_heldout_identity"]["file_count"], 100)
        self.assertTrue(all(path.read_bytes() == value for path, value in original.items()))

    def test_draft_is_not_executable_and_contains_no_fabricated_hash(self):
        data = diagnostic.read(diagnostic.DRAFT)
        self.assertIsNone(data["reference"])
        self.assertNotIn("sha256", json.dumps(data))
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            diagnostic.validate(diagnostic.DRAFT)

    def test_incomplete_run_cannot_prepare_and_creates_no_output(self):
        mutate(self.root / "run_manifest.json", lambda d: d.update(status="running"))
        with self.assertRaisesRegex(ValueError, "not complete"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_final_cannot_prepare(self):
        self.final.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.output.parent.exists())

    def test_no_overwrite(self):
        self.prepare()
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.prepare()
        self.assertEqual(self.output.read_bytes(), before)

    def test_only_original_bf16_can_collect_independent_observations(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "BF16 diagnostic reference"):
            diagnostic.validate(self.output, checkpoint=self.root / "models/qad")

    def test_rejects_another_final_path_even_with_identical_bytes(self):
        self.prepare()
        alternate = self.root / "other_final.json"
        alternate.write_bytes(self.final.read_bytes())
        with self.assertRaisesRegex(ValueError, "Requested final manifest"):
            diagnostic.validate(self.output, alternate)

    def test_changed_final_bytes_rejected(self):
        self.prepare()
        self.final.write_bytes(self.final.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen final manifest changed"):
            diagnostic.validate(self.output)

    def test_changed_original_protocol_is_not_diagnostic_protocol_relabeling(self):
        self.prepare()
        protocol = self.root / "recovery_protocol_v12_rtn_w4a4.json"
        protocol.write_bytes(protocol.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "protocol SHA-256 differs"):
            diagnostic.validate(self.output)

    def test_altered_settings_cannot_enable_selection_training_or_outcome_filtering(self):
        self.prepare()
        original = self.output.read_bytes()
        for key, value in (("model_selection_allowed", True), ("training_allowed", True),
                           ("success_only", True), ("capture_max_per_episode", 4),
                           ("noise_seeds", [1, 2]), ("windows_per_episode", 8)):
            with self.subTest(key=key):
                self.output.write_bytes(original)
                mutate(self.output, lambda d: d["diagnostics"].update({key: value}))
                with self.assertRaisesRegex(ValueError, "sampling or allowed use"):
                    diagnostic.validate(self.output)

    def test_original_heldout_bank_cannot_be_used_for_diagnostics(self):
        self.prepare()
        mutate(self.output, lambda d: d["partitions"]["diagnostics"].update(init_state_indices=[28]))
        with self.assertRaisesRegex(ValueError, "bank 29"):
            diagnostic.validate(self.output)

    def test_changed_selected_or_deployment_weights_rejected(self):
        self.prepare()
        for relative in ("models/bf16", "models/ptq", "models/qad", "training/qad_opd"):
            path = self.root / relative / "model.safetensors"
            before = path.read_bytes()
            with self.subTest(relative=relative):
                path.write_bytes(b"swapped weight file")
                with self.assertRaisesRegex(ValueError, "weights changed|weight identity changed"):
                    diagnostic.validate(self.output)
                path.write_bytes(before)

    def test_changed_raw_rollout_rejected_even_if_comparison_unchanged(self):
        self.prepare()
        path = self.root / "heldout/heldout_qad" / (TASKS[0] + ".log")
        path.write_text(path.read_text() + "tampering\n")
        with self.assertRaisesRegex(ValueError, "Raw rollout log hash"):
            diagnostic.validate(self.output)

    def test_swapped_final_arm_evaluation_rejected_before_prepare(self):
        path = self.root / "heldout/heldout_qad/eval_manifest.json"
        mutate(path, lambda d: d.update(checkpoint=str(self.root / "models/qad_opd")))
        comparison = self.root / "heldout/paired_comparison.json"
        write(comparison, compare_round(comparison.parent))
        mutate(self.final, lambda d: d.update(heldout_comparison=diagnostic.identity(comparison)))
        with self.assertRaisesRegex(ValueError, "checkpoint differs from selected qad"):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
