"""CPU-only checks for the v2 action archive's independent trace contract."""
from __future__ import annotations

import unittest
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from paper.collect_action_diagnostics import ROOT, comparator, collect, identity, read, write
from exp.independent_action_inputs import freeze_diagnostic_inputs
from tests.test_independent_action_inputs import capture_fixture
from rl.checkpoint_identity import checkpoint_files


SPEC = {
    "horizon": 4,
    "dimensions": 8,
    "keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    "key_dimensions": [1, 1, 1, 1, 1, 1, 2],
}
SCOPE = "Each policy's own integration states and velocities; not shared-state field error or robot rollout drift"


def payload(arm: str, *, trace: bool = True):
    noise = torch.arange(32, dtype=torch.bfloat16).reshape(1, 4, 8) / 16
    action = torch.zeros(1, 4, 8, dtype=torch.float32)
    row = {
        "sample": {"task": "task", "path": "candidate.pt", "sha256": "a" * 64, "seed": 7},
        "initial_noise": noise,
        "action_pred": action,
        "integration_steps": 4,
        "decoded": {key: torch.zeros(1, 4, width) for key, width in zip(SPEC["keys"], SPEC["key_dimensions"])},
    }
    if trace:
        row["integration_trace"] = {
            "states": noise.expand(4, -1, -1, -1).clone(),
            "velocities": torch.zeros(4, 1, 4, 8),
            "time_buckets": torch.arange(4, dtype=torch.int64).reshape(4, 1),
            "scope": SCOPE,
        }
    return {
        "format": "gr00t_action_outputs_v1", "arm": arm,
        "inputs_sha256": "b" * 64, "final_manifest_sha256": "c" * 64,
        "source_files": {"producer": "d" * 64},
        "input_manifest": {"format": "gr00t_action_inputs_v2",
                           "interpretation": "independent_observation_diagnostic_not_training_or_success_rate",
                           "samples": [row["sample"]]},
        "action_spec": SPEC, "records": [row],
    }


class V2ActionTraceTests(unittest.TestCase):
    def test_trace_replays_with_own_path_scope(self):
        with comparator(ROOT) as module:
            result = module.compare(payload("bf16"), payload("qad"))
        self.assertTrue(result["same_actual_noise_verified"])
        metrics = result["per_sample"][0]["metrics"]
        self.assertIn("integration_state_step_1_mse", metrics)
        self.assertIn("integration_own_path_velocity_step_4_mse", metrics)
        self.assertEqual(result["interpretation"], "independent_observation_diagnostic_not_training_or_success_rate")

    def test_trace_scope_tamper_is_rejected(self):
        reference, candidate = payload("bf16"), payload("qad")
        candidate["records"][0]["integration_trace"]["scope"] = "shared state error"
        with self.assertRaisesRegex(ValueError, "different measurement scope"):
            with comparator(ROOT) as module:
                module.compare(reference, candidate)

    def test_trace_shape_tamper_is_rejected(self):
        reference, candidate = payload("bf16"), payload("qad")
        candidate["records"][0]["integration_trace"]["velocities"] = torch.zeros(3, 1, 4, 8)
        with self.assertRaisesRegex(ValueError, "Malformed or nonfinite integration trace"):
            with comparator(ROOT) as module:
                module.compare(reference, candidate)

    def test_actual_noise_mismatch_stays_fatal_with_trace(self):
        reference, candidate = payload("bf16"), payload("qad")
        candidate["records"][0]["initial_noise"] = candidate["records"][0]["initial_noise"].clone()
        candidate["records"][0]["initial_noise"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Actual initial noise"):
            with comparator(ROOT) as module:
                module.compare(reference, candidate)

    def test_independent_archive_collects_and_replays_without_models(self):
        """Exercise the v2 dual-protocol archive with the production input fixture."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, protocol, final = capture_fixture(root)
            final_data = read(final)
            inputs = freeze_diagnostic_inputs(capture, protocol)
            source = root / "diagnostic_outputs"
            final_sha = identity(final)["sha256"]
            source_files = {
                str(ROOT / name): hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("exp/action_chunk_diagnostics.py", "exp/independent_action_inputs.py",
                             "exp/independent_action_protocol.py", "rl/capture_sampling.py",
                             "rl/gr00t_runtime.py", "rl/probe_distill.py", "rl/checkpoint_identity.py")
            }
            for arm_index, arm in enumerate(("bf16", "ptq", "qad", "continued_qad", "qad_opd")):
                folder = source / arm
                folder.mkdir(parents=True)
                records = []
                for sample_index, sample in enumerate(inputs["samples"]):
                    noise = (torch.arange(32, dtype=torch.bfloat16).reshape(1, 4, 8)
                             + sample_index) / 16
                    trace = {
                        "states": noise.expand(4, -1, -1, -1).clone(),
                        "velocities": torch.full((4, 1, 4, 8), arm_index / 16),
                        "time_buckets": torch.arange(4, dtype=torch.int64).reshape(4, 1),
                        "scope": SCOPE,
                    }
                    records.append({"sample": sample, "initial_noise": noise,
                                    "action_pred": torch.full((1, 4, 8), arm_index / 16),
                                    "integration_steps": 4, "integration_trace": trace,
                                    "decoded": {key: torch.full((1, 4, width), arm_index / 16)
                                                for key, width in zip(SPEC["keys"], SPEC["key_dimensions"])}})
                payload = {"format": "gr00t_action_outputs_v1", "arm": arm,
                           "inputs_sha256": "",
                           "input_manifest": inputs, "final_manifest_sha256": final_sha,
                           "source_files": source_files, "action_spec": SPEC, "records": records}
                # The live collector stores the input manifest in this source
                # directory; create it before calculating the payload identity.
                source_inputs = source / "inputs.json"
                if not source_inputs.exists():
                    write(source_inputs, inputs)
                payload["inputs_sha256"] = identity(source_inputs)["sha256"]
                torch.save(payload, folder / "actions.pt")
                checkpoint = final_data["selected_ptq_checkpoint"] if arm == "ptq" else (
                    final_data[{"qad": "selected_qad_model_identity", "continued_qad": "selected_continued_model_identity",
                           "qad_opd": "selected_opd_model_identity"}[arm]]["path"] if arm != "bf16" else
                    read(final_data["selection_file"])["arms"]["bf16"]["model_identity"]["path"])
                write(folder / "manifest.json", {"format": "gr00t_action_run_v1", "status": "complete", "arm": arm,
                    "checkpoint": checkpoint, "checkpoint_files": checkpoint_files(checkpoint),
                    "final_manifest_sha256": final_sha, "inputs_sha256": payload["inputs_sha256"],
                    "actions_sha256": identity(folder / "actions.pt")["sha256"], "source_files": source_files,
                    "interpretation": inputs["interpretation"], "sample_count": len(records)})
                (source / f"{arm}.log").write_text("\n".join(
                    f"[action-diagnostic] {arm} {i}/{len(records)}" for i in range(1, len(records) + 1)) + "\n")
            # collect() expects its input manifest beside the run's arm folders;
            # it was created before the first arm above.
            archive = root / "archive"
            report = collect(source, final, archive)
            self.assertEqual(report["sample_count"], len(inputs["samples"]))
            moved = root / "moved"
            import shutil
            shutil.copytree(archive, moved)
            self.assertEqual(report["comparisons"], __import__("paper.collect_action_diagnostics", fromlist=["verify"]).verify(moved)["comparisons"])


if __name__ == "__main__":
    unittest.main()
