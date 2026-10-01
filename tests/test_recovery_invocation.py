from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
import recovery_invocation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import collect_training_costs


class RecoveryInvocationTest(unittest.TestCase):
    def test_provenance_summary_preserves_unrecorded_initial_stages(self):
        state = {
            "protocol_sha256": "p", "implementation_sha256": "a" * 64,
            "stages": {"dev_qad": {}, "train_qad": {}, "heldout": {}},
        }
        summary = collect_training_costs.recovery_provenance_summary(
            state, [{"completed_stages_before": ["dev_qad"],
                     "completed_stages_after": ["dev_qad", "train_qad"],
                     "newly_completed_stages": ["train_qad"],
                     "attempt": "mixed-recovery-attempt-0001", "status": "completed",
                     "returncode": 0, "invocation_identity": {"bytes": 1, "sha256": "x"},
                     "result_identity": {"bytes": 1, "sha256": "y"},
                     "sources_unchanged_at_completion": True}],
            "exp/run_high_fp4_v3_producer.py")
        self.assertEqual(summary["launch_snapshot_coverage"], "partial")
        self.assertEqual(summary["stages_without_launch_snapshot"], ["dev_qad", "heldout"])
        self.assertEqual(summary["stages_completed_during_recorded_invocations"], ["train_qad"])

    def test_snapshot_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative, text in (
                ("exp/run_mixed_pressure_study.py", "study"),
                ("exp/run_mixed_pressure_recovery.py", "recovery"),
                ("exp/run_high_fp4_v3.py", "driver"),
                ("exp/recovery_invocation.py", "provenance"),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            run = root / "run"
            (run / "recovery/stages").mkdir(parents=True)
            (run / "recovery/stages/dev.json").write_text(json.dumps({"status": "complete"}))
            selection = root / "selection.json"
            selection.write_text("{}")
            log = run / "recovery.log"
            log.write_text("RETURN_CODE 0\n")
            folder = recovery_invocation.begin(
                root, run / "recovery", "mixed-recovery-attempt-0001",
                ["python", "--until", "all"], "protocol-sha", selection)
            recovery_invocation.finish(folder, root, run / "recovery", log)
            recovery_invocation.verify(folder)
            (folder / "source/exp/run_high_fp4_v3.py").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "source snapshot hash mismatch"):
                recovery_invocation.verify(folder)


if __name__ == "__main__":
    unittest.main()
