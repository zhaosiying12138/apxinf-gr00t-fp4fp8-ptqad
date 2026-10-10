"""CPU-only release capture tests. All evidence below is explicitly synthetic."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
import verify_completed_capture as capture
import prepare_w4a4_captures as prepare
import readme_v12_provenance as provenance


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")


def fixture(root):
    """No real weights, training, evaluations or desktop captures are used."""
    evidence = root / "paper/evidence"
    run = root / "run"
    protocol = run / "protocol.json"
    write(protocol, {"version": 12, "id": prepare.PROTOCOL_ID, "w4a4": True})
    protocol_sha = capture.identity(protocol)["sha256"]
    comparison = {"arms": dict.fromkeys(prepare.ARMS, {}), "environment_pairing_verified": True,
                  "protocol_consistency_verified": True}
    pair = evidence / "paired_comparison.json"
    write(pair, comparison)
    final = {"format": capture.FINAL_FORMAT, "required_arms": list(prepare.ARMS),
             "protocol_file": str(protocol), "protocol_sha256": protocol_sha,
             "selection_uses_heldout": False,
             "heldout_comparison": {"path": str(pair), **capture.identity(pair)}}
    final_path = run / "final_manifest.json"
    write(final_path, final)
    write(evidence / "final_manifest.json", final)
    run_manifest = {"status": "complete", "output_layout": "stable_paths_v2", "protocol_sha256": protocol_sha}
    write(run / "run_manifest.json", run_manifest)
    write(evidence / "training/orchestration/run_manifest.json", run_manifest)
    write(evidence / "training/orchestration/final_manifest.json", final)
    source = {"final_manifest": {"path": str(final_path), **capture.identity(final_path)},
              "heldout_comparison": final["heldout_comparison"]}
    write(evidence / "final_results.json", {"format": "publication_final_results_v1", "status": "complete", "source": source})
    write(evidence / "recipe_inventory.json", {"recipe": "rtn_all", "recipes": {"rtn_all": {
        "nvfp4_params": 64, "linear_params": 64}}, "recovery_residual": {"target_bytes": 16}})
    for name in ("category_ptq_recipe.json", "category_bake_manifest.json"):
        write(evidence / "selected_recipe" / name, {"synthetic_fixture": True})
    dataset = {"root": "/synthetic/teacher", "sample_count": 4,
               "files": [{"path": "sample_0.pt", "bytes": 1, "sha256": "a" * 64}]}
    dataset_sha = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
    mask = {"horizon": 16, "dimensions": 7}
    for arm in ("qad", "qad_opd", "continued_qad"):
        prefix = evidence / "training/stages" / arm
        write(prefix / "recovery_manifest.json", {"w4a4_enabled": True, "scope": "all_ordinary_linear", "rank": 32,
              "alpha": 64, "probe_weight": 1 if arm == "qad_opd" else 0, "trainable_parameters": 32,
              "execution_mode": "W4A4 numerical QDQ + BF16 LoRA residual", "capture_dataset_sha256": dataset_sha,
              "capture_dataset_samples": 4, "probe_cache_sha256": "a" * 64, "probe_action_mask": mask})
        write(prefix / "runtime_metrics.json", {"status": "completed", "error": None, "global_steps": 2, "wall_seconds": 3})
        write(prefix / "checkpoint/trainer_state.json", {"global_step": 2, "max_steps": 2})
        write(prefix / "train.log", "synthetic original log, not a real training run\n")
    write(evidence / "training/stages/qad/orchestrator_training_request.json", {"capture_dataset_identity": dataset})
    write(evidence / "training/collection/eval_manifest.json", {"purpose": "collection", "protocol_sha256": protocol_sha,
          "checkpoint": "/synthetic/qad"})
    write(evidence / "training/collection/task.log", "synthetic original collection log\n")
    write(evidence / "training/teacher/teacher_probes.json", {"count": 1, "source_observation_files": [{}],
          "source_kind": "student_rollout", "action_mask": mask, "objective": "masked velocity MSE",
          "student_checkpoint": "/synthetic/qad"})
    write(evidence / "training/costs.json", {"status": "complete", "protocol_sha256": protocol_sha,
          "training": {arm: {"optimizer_steps": 2} for arm in ("qad", "continued_qad", "qad_opd")},
          "collection": {"episodes": 1}, "teacher_labeling": {"cache_identity": {"sha256": "a" * 64, "bytes": 1}}})
    write(evidence / "heldout_raw_logs.json", {"synthetic_fixture": True})
    for arm in prepare.ARMS:
        write(evidence / ("heldout_" + arm) / "eval_manifest.json", {"purpose": "heldout", "protocol_sha256": protocol_sha})
        for task in range(10):
            write(evidence / ("heldout_" + arm) / (f"task{task}.server.log"), "synthetic original server log\n")
    source_manifest = evidence / "source_bundle_manifest.json"
    write(source_manifest, {"source_final_manifest": source})
    mapping = {"format": "installed_final_evidence_v1", "status": "complete",
               "source_bundle_manifest": capture.identity(source_manifest),
               "files": [{"published_path": path.relative_to(evidence).as_posix(), "source": capture.identity(path)}
                         for path in sorted(evidence.rglob("*")) if path.is_file() and path != source_manifest]}
    write(evidence / "evidence_manifest.json", mapping)
    return evidence, prepare.validate_final(final_path)


class CompletedCaptureTests(unittest.TestCase):
    def test_all_seven_commands_only_read_original_files_and_emit_verifiable_receipts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); evidence, bundle = fixture(root); out = root / "prepared"
            plan = prepare.render_completed_scripts(out, bundle, evidence)
            self.assertEqual(set(plan["execution_order"]), set(capture.FIGURES))
            self.assertEqual(len(plan["script_files"]), 9)
            self.assertEqual((out / "plan.json").read_bytes(), (out / "plan_v12_completed.json").read_bytes())
            before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            for figure in capture.FIGURES:
                text = (out / (figure + ".sh")).read_text()
                for forbidden in ("lora_qad.py", "serve_recovery.py", "rollout_seeded.py", "opd_probe_cache.py", "nvidia-smi"):
                    self.assertNotIn(forbidden, text)
                result = subprocess.run(["bash", str(out / (figure + ".sh"))], text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                receipt = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(receipt["capture_mode"], "verify-completed")
                self.assertEqual(receipt["figure"], figure)
                self.assertEqual(receipt["final_manifest_sha256"], plan["final_manifest_sha256"])
            after = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_modified_original_log_rejected_after_preparation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); evidence, bundle = fixture(root)
            plan = prepare.render_completed_scripts(root / "prepared", bundle, evidence)
            (evidence / "training/stages/qad/train.log").write_text("changed")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                capture.audit(plan, "shot_qad")

    def test_incomplete_or_other_release_rejected_before_any_commands_written(self):
        for mutation in ("old_final", "incomplete_training", "missing_server_log"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw); evidence, bundle = fixture(root); out = root / "prepared"
                if mutation == "old_final":
                    data = json.loads((evidence / "final_manifest.json").read_text()); data["format"] = "v11"
                    write(evidence / "final_manifest.json", data)
                elif mutation == "incomplete_training":
                    write(evidence / "training/stages/qad/runtime_metrics.json", {"status": "running"})
                else:
                    (evidence / "heldout_bf16/task0.server.log").unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    prepare.render_completed_scripts(out, bundle, evidence)
                self.assertFalse(out.exists())

    def test_default_cli_does_not_enter_legacy_input_or_renderer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); evidence, bundle = fixture(root)
            with patch.object(prepare, "validate_inputs", side_effect=AssertionError("legacy path invoked")), \
                 patch.object(prepare, "render_scripts", side_effect=AssertionError("legacy renderer invoked")):
                self.assertEqual(prepare.main(["--final-manifest", str(bundle["final_path"]), "--evidence-root", str(evidence),
                                               "--out", str(root / "prepared")]), 0)

    def test_missing_final_manifest_cannot_prepare(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaises(FileNotFoundError):
                prepare.main(["--final-manifest", str(root / "final_manifest.json"), "--out", str(root / "out")])
            self.assertFalse((root / "out").exists())

    def test_changed_helper_refuses_to_display_verified_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); evidence, bundle = fixture(root); out = root / "prepared"
            prepare.render_completed_scripts(out, bundle, evidence)
            helper = out / "verify_completed_capture.py"
            helper.write_text(helper.read_text() + "\n# changed after preparation\n")
            result = subprocess.run(["bash", str(out / "shot_qad.sh")], text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("helper differs", result.stderr)
            self.assertNotIn("completed_evidence_verified", result.stdout)

    def test_publication_rechecks_sources_and_exact_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); evidence, bundle = fixture(root); out = evidence / "captures"
            plan = prepare.render_completed_scripts(out, bundle, evidence)
            figure = "shot_qad"; script = out / (figure + ".sh")
            result = subprocess.run(["bash", str(script)], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            log = out / "shot_qad.log"; log.write_text(result.stdout)
            sidecar = out / "shot_qad.capture.json"; write(sidecar, {"script": str(script)})
            paper = root / "paper"
            row = {"figure": figure, "capture_sidecar": sidecar.relative_to(paper).as_posix(),
                   "script": script.relative_to(paper).as_posix(), "script_sha256": capture.identity(script)["sha256"],
                   "raw_log": log.relative_to(paper).as_posix()}
            support = {name: out / name for name in ("common_v12.sh", "verify_completed_capture.py", "plan_v12_completed.json")}
            with patch.object(provenance, "PROTOCOL_SHA256", plan["protocol_sha256"]):
                provenance._release_binding(row, paper, plan["final_manifest_sha256"], support)
                log.write_text(result.stdout + result.stdout)
                with self.assertRaisesRegex(RuntimeError, "one matching"):
                    provenance._release_binding(row, paper, plan["final_manifest_sha256"], support)
                log.write_text("completed_evidence_verified\n")
                with self.assertRaisesRegex(RuntimeError, "one matching"):
                    provenance._release_binding(row, paper, plan["final_manifest_sha256"], support)
                log.write_text(result.stdout)
                plan_path = support["plan_v12_completed.json"]
                original_plan = plan_path.read_text()
                legacy = json.loads(original_plan); legacy["capture_mode"] = "legacy-smoke"
                write(plan_path, legacy)
                with self.assertRaisesRegex(RuntimeError, "not experiment replays"):
                    provenance._release_binding(row, paper, plan["final_manifest_sha256"], support)
                plan_path.write_text(original_plan)
                (evidence / "training/stages/qad/train.log").write_text("tampered original bytes")
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    provenance._release_binding(row, paper, plan["final_manifest_sha256"], support)


if __name__ == "__main__":
    unittest.main()
