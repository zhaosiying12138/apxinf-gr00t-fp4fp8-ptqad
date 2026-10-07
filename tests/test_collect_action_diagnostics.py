"""Portable metric replay from small real CPU torch/safetensors artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from exp.action_chunk_diagnostics import checkpoint_files, freeze_inputs
from paper.collect_action_diagnostics import ARMS, ROOT, _publish_new, collect, decode, encode, identity, read, verify


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def fixture(root):
    """Use the real frozen protocol and existing completion/identity validators."""
    protocol = root / "recovery_protocol_v12_rtn_w4a4.json"
    protocol.write_bytes((ROOT / "exp" / protocol.name).read_bytes())
    protocol_sha = identity(protocol)["sha256"]
    checkpoints = {}
    for arm in ARMS:
        folder = root / "models" / arm
        folder.mkdir(parents=True)
        save_file({"weight": torch.ones(2, 2)}, folder / "model.safetensors")
        put(folder / "statistics.json", {"fixture": "normalization"})
        checkpoints[arm] = {"path": str(folder), "metadata": {"statistics.json": identity(folder / "statistics.json")["sha256"]},
                            "shards": [{"name": "model.safetensors", **identity(folder / "model.safetensors")}]}
    selection = root / "selection.json"
    put(selection, {"protocol_sha256": protocol_sha,
                    "arms": {"bf16": {"model_identity": checkpoints["bf16"]},
                             "rtn_w4a4_category": {"model_identity": checkpoints["ptq"]}}})
    arms = {}
    for arm in ARMS:
        rows = [{"task": f"task_{t}", "episode_index": i, "success": i < 8} for t in range(10) for i in range(16)]
        arms[arm] = {"successes": 80, "count": 160, "macro_success_rate": 0.5,
                     "per_task": {f"task_{t}": {"successes": 8, "episodes": 16, "success_rate": 0.5} for t in range(10)},
                     "episodes": rows}
    comparison = root / "heldout" / "paired_comparison.json"
    put(comparison, {"environment_pairing_verified": True, "protocol_consistency_verified": True,
                     "source_accounting_verified": True, "arms": arms})
    final = root / "final_manifest.json"
    put(final, {"format": "w4a4_recovery_v12_final_manifest", "protocol_file": str(protocol),
                "protocol_sha256": protocol_sha, "selection_uses_heldout": False,
                "selected_pressure_recipe": "rtn_w4a4_category", "selected_ptq_checkpoint": checkpoints["ptq"]["path"],
                "heldout_round": str(comparison.parent), "heldout_comparison": {"path": str(comparison), **identity(comparison)},
                "required_arms": list(ARMS), "selection_file": str(selection), "selection_sha256": identity(selection)["sha256"],
                "selected_qad_model_identity": checkpoints["qad"],
                "selected_continued_model_identity": checkpoints["continued_qad"],
                "selected_opd_model_identity": checkpoints["qad_opd"]})
    capture_root = root / "captures" / "observations"
    captured = capture_root / "task_0"
    captured.mkdir(parents=True)
    put(capture_root.parent / "eval_manifest.json", {"purpose": "teacher_supervision", "protocol_sha256": protocol_sha,
          "init_state_indices": [20, 21, 22, 23], "tasks": ["task_0"]})
    put(captured / "capture_manifest.json", {"finalized": True, "protocol_sha256": protocol_sha, "task_name": "task_0"})
    torch.save({"task_name": "task_0", "init_state_index": 20, "reset_identity": {"init_state_index": 20},
                "student_statistics_sha256": checkpoints["bf16"]["metadata"]["statistics.json"],
                "episode_index": 0, "server_call": 1, "inputs": {"camera": torch.zeros(1, 3, 2, 2)}},
               captured / "sample_000000.pt")
    source = root / "diagnostics"
    inputs = freeze_inputs(capture_root, protocol, 1, 2026100600)
    put(source / "inputs.json", inputs)
    external = root / "external_decoder.py"
    external.write_text("# Recorded inference source; not imported for CPU metric replay.\n")
    source_map = {str(p): identity(p)["sha256"] for p in (ROOT / "exp/action_chunk_diagnostics.py",
                  ROOT / "rl/gr00t_runtime.py", ROOT / "rl/probe_distill.py",
                  ROOT / "rl/checkpoint_identity.py", external)}
    spec = {"horizon": 2, "dimensions": 7, "keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"], "key_dimensions": [1] * 7}
    for i, arm in enumerate(ARMS):
        folder = source / arm
        folder.mkdir()
        payload = {"format": "gr00t_action_outputs_v1", "arm": arm,
                   "inputs_sha256": identity(source / "inputs.json")["sha256"], "input_manifest": inputs,
                   "final_manifest_sha256": identity(final)["sha256"], "source_files": source_map,
                   "action_spec": spec, "records": [{"sample": inputs["samples"][0],
                   "initial_noise": torch.arange(32, dtype=torch.bfloat16).reshape(1, 4, 8) / 16,
                   "action_pred": torch.full((1, 4, 8), i / 16, dtype=torch.float32), "integration_steps": 4,
                   "decoded": {key: torch.full((1, 2, 1), 0.5 + i / 16) for key in spec["keys"]}}]}
        torch.save(payload, folder / "actions.pt")
        put(folder / "manifest.json", {"format": "gr00t_action_run_v1", "status": "complete", "arm": arm,
            "checkpoint": checkpoints[arm]["path"], "checkpoint_files": checkpoint_files(checkpoints[arm]["path"]),
            "final_manifest_sha256": identity(final)["sha256"], "inputs_sha256": identity(source / "inputs.json")["sha256"],
            "actions_sha256": identity(folder / "actions.pt")["sha256"], "source_files": source_map,
            "interpretation": inputs["interpretation"], "sample_count": 1})
        (source / f"{arm}.log").write_text(f"fixture loader stdout\n[action-diagnostic] {arm} 1/1\n", encoding="utf-8")
    return source, final


def reseal(folder, relative):
    manifest = read(folder / "manifest.json")
    manifest["files"][relative] = identity(folder / relative)
    put(folder / "manifest.json", manifest)


class PortableActionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = tempfile.TemporaryDirectory()
        root = Path(cls.master.name)
        originals = root / "originals"
        originals.mkdir()
        cls.source, cls.final = fixture(originals)
        cls.archive = root / "archive"
        cls.report = collect(cls.source, cls.final, cls.archive)

    @classmethod
    def tearDownClass(cls):
        cls.master.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archived(self):
        folder = self.root / "moved-archive"
        shutil.copytree(self.archive, folder)
        return folder

    def original(self):
        return fixture(self.root)

    def test_roundtrip_includes_all_tensor_values_and_provenance(self):
        report = verify(self.archive, self.final)
        self.assertEqual(set(report["comparisons"]), set(ARMS[1:]))
        self.assertEqual(report["action_spec"]["dimensions"], 7)
        self.assertEqual(report["integration_steps"], 4)
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["task_count"], 1)
        for arm in ARMS:
            raw = torch.load(self.source / arm / "actions.pt", weights_only=True)
            restored = decode(read(self.archive / "runs" / arm / "outputs.json"))
            torch.testing.assert_close(restored["records"][0]["initial_noise"], raw["records"][0]["initial_noise"], rtol=0, atol=0)
            self.assertEqual(restored["records"][0]["initial_noise"].dtype, torch.bfloat16)
            self.assertEqual(report["manifest"]["arms"][arm]["raw_actions"]["sha256"], identity(self.source / arm / "actions.pt")["sha256"])
        self.assertFalse(any(p.suffix in (".pt", ".safetensors") for p in report["files"]))

    def test_moved_archive_replays_without_original_paths_or_repository(self):
        folder = self.archived()
        # Subprocess starts outside the project and invokes only the archived
        # exporter; delete this fresh fixture's models/cameras/original logs.
        source, final = self.original()
        independent = self.root / "independent"
        independent_report = collect(source, final, independent)
        for path in list(self.root.iterdir()):
            if path not in (independent, folder):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        script = independent / "sources/recompute/paper/collect_action_diagnostics.py"
        completed = subprocess.run([sys.executable, "-B", str(script), "verify", "--folder", str(independent)],
                                   cwd=self.root, capture_output=True, text=True, timeout=45)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertEqual(json.loads(completed.stdout)["status"], "verified")
        self.assertEqual(verify(independent)["comparisons"], independent_report["comparisons"])

    def test_finite_dtypes_and_signed_zero_roundtrip_exactly(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            original = torch.tensor([-0.0, 0.0, 0.125, -2.5], dtype=dtype)
            packed = json.loads(json.dumps(encode(original)))
            result = decode(packed)
            self.assertTrue(torch.equal(original.view(torch.uint8), result.view(torch.uint8)))

    def test_rejects_nonfinite_tensor_and_changed_dtype_shape_values(self):
        for number in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                encode(torch.tensor([number]))
        for change in ({"dtype": "float64"}, {"shape": [2]}, {"values": [0.25]}):
            record = encode(torch.tensor([0.125]))
            record.update(change)
            with self.assertRaises(ValueError):
                decode(record)

    def test_file_tamper_rejected(self):
        folder = self.archived()
        with (folder / "runs/qad/outputs.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "file identity"):
            verify(folder)

    def test_resealed_summary_must_still_match_recomputed_metrics(self):
        folder = self.archived()
        summary = read(folder / "summary.json")
        summary["comparisons"]["qad_opd"]["task_macro"]["normalized_mse"] = 0
        put(folder / "summary.json", summary)
        reseal(folder, "summary.json")
        with self.assertRaisesRegex(ValueError, "disagree with tensor replay"):
            verify(folder)

    def test_resealed_actual_noise_mismatch_rejected(self):
        folder = self.archived()
        relative = "runs/qad_opd/outputs.json"
        payload = decode(read(folder / relative))
        payload["records"][0]["initial_noise"][0, 0, 0] += 1
        put(folder / relative, encode(payload))
        reseal(folder, relative)
        with self.assertRaisesRegex(ValueError, "Actual initial noise"):
            verify(folder)

    def test_installed_main_final_mismatch_rejected(self):
        final = self.root / "different.json"
        final.write_bytes(self.final.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Installed final manifest"):
            verify(self.archive, final)

    def test_resealed_source_tamper_rejected_by_original_receipt(self):
        folder = self.archived()
        manifest = read(folder / "manifest.json")
        origin = next(iter(manifest["sources"]))
        relative = manifest["sources"][origin]["path"]
        with (folder / relative).open("a") as stream:
            stream.write("\n# mutation\n")
        reseal(folder, relative)
        with self.assertRaisesRegex(ValueError, "Recorded inference source"):
            verify(folder)

    def test_resealed_checkpoint_identity_mismatch_rejected(self):
        folder = self.archived()
        relative = "runs/ptq/manifest.json"
        receipt = read(folder / relative)
        key = next(p for p in receipt["checkpoint_files"] if p.endswith(".safetensors"))
        receipt["checkpoint_files"][key]["sha256"] = "0" * 64
        put(folder / relative, receipt)
        reseal(folder, relative)
        with self.assertRaisesRegex(ValueError, "weight identity"):
            verify(folder)

    def test_missing_arm_and_extra_file_are_rejected(self):
        folder = self.archived()
        (folder / "unexpected.pt").write_bytes(b"do not publish")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify(folder)
        (folder / "unexpected.pt").unlink()
        manifest = read(folder / "manifest.json")
        del manifest["arms"]["continued_qad"]
        put(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "five diagnostic arms"):
            verify(folder)

    def test_symlink_and_escaping_inventory_rejected(self):
        folder = self.archived()
        link = folder / "external-link"
        link.symlink_to(self.final)
        manifest = read(folder / "manifest.json")
        manifest["files"]["external-link"] = identity(self.final)
        put(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            verify(folder)
        link.unlink()
        manifest["files"].pop("external-link")
        manifest["files"]["../escape"] = identity(self.final)
        put(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify(folder)

    def test_changed_input_capture_rejected_without_creating_archive(self):
        source, final = self.original()
        inputs = read(source / "inputs.json")
        capture = Path(inputs["samples"][0]["path"])
        raw = torch.load(capture, weights_only=True)
        raw["inputs"]["camera"] += 1
        torch.save(raw, capture)
        with self.assertRaisesRegex(ValueError, "Frozen input sources"):
            collect(source, final, self.root / "rejected")
        self.assertFalse((self.root / "rejected").exists())

    def test_existing_target_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            collect(self.source, self.final, self.archive)
        verify(self.archive)

    def test_atomic_publish_refuses_concurrently_created_empty_target(self):
        stage, out = self.root / "stage", self.root / "destination"
        stage.mkdir()
        out.mkdir()
        (stage / "payload").write_text("complete staging data")
        with self.assertRaises(FileExistsError):
            _publish_new(stage, out)
        self.assertTrue((stage / "payload").exists())
        self.assertEqual(list(out.iterdir()), [])

    def test_collect_rejects_incomplete_main_before_creating_archive(self):
        source, final = self.original()
        value = read(final)
        value["required_arms"].remove("qad_opd")
        put(final, value)
        target = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "complete five-arm"):
            collect(source, final, target)
        self.assertFalse(target.exists())

    def test_collect_rejects_other_completed_final_identity(self):
        source, final = self.original()
        value = read(final)
        value["selected_opd_weight"] = 100
        put(final, value)
        with self.assertRaisesRegex(ValueError, "final manifest identity"):
            collect(source, final, self.root / "rejected")

    def test_collect_rejects_modified_checkpoint_and_raw_output(self):
        source, final = self.original()
        model = self.root / "models/bf16/model.safetensors"
        original = model.read_bytes()
        model.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "weights changed"):
            collect(source, final, self.root / "rejected")
        model.write_bytes(original)
        with (source / "bf16/actions.pt").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Action output changed"):
            collect(source, final, self.root / "rejected")

    def test_collect_rejects_incomplete_log_and_duplicate_log_location(self):
        source, final = self.original()
        (source / "bf16.log").write_text("loading only\n")
        with self.assertRaisesRegex(ValueError, "Incomplete/duplicated action log"):
            collect(source, final, self.root / "rejected")
        (source / "bf16/collect.log").write_text("[action-diagnostic] bf16 1/1\n")
        with self.assertRaisesRegex(ValueError, "exactly one original collect log"):
            collect(source, final, self.root / "rejected")


if __name__ == "__main__":
    unittest.main()
