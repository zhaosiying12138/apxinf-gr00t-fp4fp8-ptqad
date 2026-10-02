"""CPU-only operations tests; no driver, training or evaluation is launched."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("continue_w4a4", ROOT / "exp/continue_w4a4_run.py")
WRAPPER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WRAPPER)


class ContinuationOperationsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.protocol = self.root / "protocol.json"
        self.selection = self.root / "selection.json"
        self.protocol.write_text('{"w4a4": true}\n')
        self.selection.write_text('{"status": "complete"}\n')
        self.manifest = {
            "w4a4": True, "stages": {},
            "protocol_file": str(self.protocol), "protocol_sha256": WRAPPER.digest(self.protocol),
            "selection_file": str(self.selection), "selection_sha256": WRAPPER.digest(self.selection),
            "base": str(self.root / "base"), "gr00t_repo": str(self.root / "gr00t"),
            "python": str(self.root / "gr00t/.venv/bin/python"),
            "rollout_python": str(self.root / "libero/.venv/bin/python"),
            "dataset": str(self.root / "demos"),
            "capture_dataset_identity": {"root": str(self.root / "captures")},
        }
        self.write_manifest()

    def write_manifest(self):
        (self.run / "run_manifest.json").write_text(json.dumps(self.manifest))

    def process_fixture(self, pid=1234, run=None, starttime="400", script=None):
        proc = self.root / "proc"
        folder = proc / str(pid)
        folder.mkdir(parents=True, exist_ok=True)
        argv = ["python", str(script or ROOT / "exp/run_w4a4_recovery.py"),
                "--run-dir", str(run or self.run), "--port-base", "6780"]
        (folder / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
        # /proc stat fields following comm: state (#3), then through starttime (#22).
        (folder / "stat").write_text(f"{pid} (python worker) S " + " ".join(["0"] * 18 + [starttime]))
        cwd = folder / "cwd"
        if not cwd.exists():
            cwd.symlink_to(ROOT, target_is_directory=True)
        return proc

    def failure_fixture(self, error=None):
        output = self.run / "artifacts" / WRAPPER.STAGE
        output.mkdir(parents=True)
        log = self.run / "logs" / (WRAPPER.STAGE + ".log")
        log.parent.mkdir()
        command = [self.manifest["python"], str(ROOT / "eval/run_recovery_eval.py"),
                   "--out", str(output), "--purpose", "development",
                   "--protocol-file", str(self.protocol)]
        log.write_text("UTC 2026-10-02T00:00:00Z\nCWD " + str(ROOT) + "\nCOMMAND "
                       + shlex.join(command) + "\nTraceback (most recent call last):\n"
                       + (error or WRAPPER.FFMPEG_ERROR) + "\nRETURN_CODE 1\n")
        return output, log

    def test_manifest_reloads_stage_changes_but_verifies_source_hashes(self):
        self.assertEqual(WRAPPER.load_manifest(self.run)["stages"], {})
        self.manifest["stages"] = {"train": {"status": "complete"}}
        self.write_manifest()
        self.assertIn("train", WRAPPER.load_manifest(self.run)["stages"])
        self.protocol.write_text("changed")
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "protocol_sha256"):
            WRAPPER.load_manifest(self.run)

    def test_manifest_rejects_changed_selection_and_absent_run(self):
        self.selection.write_text("changed")
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "selection_sha256"):
            WRAPPER.load_manifest(self.run)
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "already exist"):
            WRAPPER.load_manifest(self.root / "missing")

    def test_manifest_rejects_invalid_state(self):
        (self.run / "run_manifest.json").write_text("[]")
        with self.assertRaises(WRAPPER.ContinuationError):
            WRAPPER.load_manifest(self.run)

    def test_command_uses_all_manifest_paths_and_preserves_venv_launcher(self):
        command = WRAPPER.continuation_command(self.run, self.manifest, 6780, True)
        self.assertEqual(command[0], self.manifest["python"])
        for flag, field in (("--protocol-file", "protocol_file"), ("--ptq-selection", "selection_file"),
                            ("--base", "base"), ("--gr00t-repo", "gr00t_repo"), ("--python", "python"),
                            ("--rollout-python", "rollout_python"), ("--dataset", "dataset")):
            self.assertEqual(WRAPPER.option(command, flag), self.manifest[field])
        self.assertEqual(WRAPPER.option(command, "--capture-dataset"), self.manifest["capture_dataset_identity"]["root"])
        self.assertEqual(WRAPPER.option(command, "--until"), "all")
        self.assertEqual(WRAPPER.option(command, "--port-base"), "6780")
        for flag in ("--adopt-complete", "--cleanup-duplicates", "--allow-opd-nonimprovement"):
            self.assertIn(flag, command)
        self.manifest["w4a4"] = False
        with self.assertRaises(WRAPPER.ContinuationError):
            WRAPPER.continuation_command(self.run, self.manifest, 6780)

    def test_environment_cleans_recovery_inheritance_and_forces_offline_media(self):
        env = WRAPPER.continuation_environment(Path("/media/lib"), {
            "QAD_INIT_ADAPTER": "stale", "QAD_OUT": "stale", "OPD_CACHE_PATH": "stale",
            "GR00T_BASE_CKPT": "stale", "TRAIN_SEED": "stale", "PROTOCOL_FILE": "stale",
            "PTQAD_PROTOCOL_FILE": "stale", "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0",
            "PTQAD_MEDIA_BIN": "wrong", "LD_LIBRARY_PATH": "/cuda/lib", "PATH": "/bin"})
        self.assertFalse(any(key.startswith(("QAD_", "OPD_")) for key in env))
        self.assertNotIn("TRAIN_SEED", env)
        self.assertNotIn("GR00T_BASE_CKPT", env)
        self.assertNotIn("PROTOCOL_FILE", env)
        self.assertNotIn("PTQAD_PROTOCOL_FILE", env)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NO_ALBUMENTATIONS_UPDATE", "PTQAD_LOCAL_HF_METADATA"):
            self.assertEqual(env[key], "1")
        self.assertEqual(env["PTQAD_MEDIA_LIB"], "/media/lib")
        self.assertEqual(env["PTQAD_MEDIA_BIN"], "/media/bin")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/media/lib:/cuda/lib")

    def test_bind_process_checks_starttime_script_and_run(self):
        proc = self.process_fixture()
        bound = WRAPPER.bind_process(1234, self.run, proc)
        self.assertEqual(bound["starttime"], "400")
        self.assertEqual(bound["port_base"], 6780)
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "different run-dir"):
            WRAPPER.bind_process(1234, self.root / "wrong", proc)
        self.process_fixture(script=ROOT / "exp/unrelated.py")
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "not this repository"):
            WRAPPER.bind_process(1234, self.run, proc)
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "absent/exited"):
            WRAPPER.bind_process(5678, self.run, proc)

    def test_pid_reuse_is_refused(self):
        proc = self.process_fixture()
        bound = WRAPPER.bind_process(1234, self.run, proc)
        self.process_fixture(starttime="401")
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "reused"):
            WRAPPER.wait_for_process(bound, proc)

    def test_lock_excludes_second_wrapper(self):
        with WRAPPER.run_lock(self.run):
            with self.assertRaisesRegex(WRAPPER.ContinuationError, "another W4A4"):
                with WRAPPER.run_lock(self.run):
                    self.fail("second lock must fail")

    def test_exact_empty_preflight_is_archived_with_hash_receipt(self):
        output, log = self.failure_fixture()
        original = log.read_bytes()
        with patch.object(WRAPPER, "require_idle_gpu") as idle:
            receipt_path = WRAPPER.recover_ffmpeg_preflight(self.run, self.manifest)
        idle.assert_called_once()
        receipt = WRAPPER.read_json(receipt_path)
        self.assertEqual(receipt["status"], "complete")
        archived = Path(receipt["archived_log"])
        self.assertEqual(archived.read_bytes(), original)
        self.assertEqual(WRAPPER.digest(archived), receipt["log_sha256"])
        self.assertFalse(output.exists())
        self.assertFalse(log.exists())

    def test_nonempty_directory_and_eval_manifest_are_never_removed(self):
        output, log = self.failure_fixture()
        (output / "eval_manifest.json").write_text("{}")
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "nonempty"):
            WRAPPER.recover_ffmpeg_preflight(self.run, self.manifest)
        self.assertTrue(log.exists())
        self.assertTrue((output / "eval_manifest.json").exists())
        self.assertFalse((self.run / "operations").exists())

    def test_other_failure_is_refused_unchanged(self):
        output, log = self.failure_fixture("RuntimeError: another problem")
        before = log.read_bytes()
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "exact ffmpeg"):
            WRAPPER.recover_ffmpeg_preflight(self.run, self.manifest)
        self.assertEqual(log.read_bytes(), before)
        self.assertTrue(output.is_dir())

    def test_matching_substring_and_wrong_command_are_refused(self):
        output, log = self.failure_fixture()
        good = log.read_text()
        log.write_text(good.replace(WRAPPER.FFMPEG_ERROR, "quoted: " + WRAPPER.FFMPEG_ERROR))
        with self.assertRaises(WRAPPER.ContinuationError):
            WRAPPER.recover_ffmpeg_preflight(self.run, self.manifest)
        log.write_text(good.replace(str(output), str(self.root / "wrong")))
        with self.assertRaisesRegex(WRAPPER.ContinuationError, "another evaluation"):
            WRAPPER.recover_ffmpeg_preflight(self.run, self.manifest)

    def test_active_gpu_refuses_before_mutating_failure(self):
        output, log = self.failure_fixture()
        with patch.object(WRAPPER, "require_idle_gpu", side_effect=WRAPPER.ContinuationError("GPU busy")):
            with self.assertRaisesRegex(WRAPPER.ContinuationError, "GPU busy"):
                WRAPPER.recover_ffmpeg_preflight(self.run, self.manifest)
        self.assertTrue(log.exists())
        self.assertTrue(output.exists())
        self.assertFalse((self.run / "operations").exists())

    def test_gpu_query_fails_closed(self):
        for stdout in ("1234\n", "[Not Supported]\n"):
            with patch.object(WRAPPER.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout, "")):
                with self.assertRaises(WRAPPER.ContinuationError):
                    WRAPPER.require_idle_gpu(self.root)
        with patch.object(WRAPPER.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            WRAPPER.require_idle_gpu(self.root)

    def test_empty_wsl_gpu_query_still_refuses_other_recovery_process(self):
        proc = self.process_fixture()
        with patch.object(WRAPPER.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            with self.assertRaisesRegex(WRAPPER.ContinuationError, "another recovery process"):
                WRAPPER.require_idle_gpu(proc)

    def test_completed_stage_is_left_to_driver_and_partial_training_is_untouched(self):
        output, log = self.failure_fixture()
        self.manifest["stages"][WRAPPER.STAGE] = {"status": "complete"}
        partial = self.run / "artifacts/train_qad_lr_5e-05"
        partial.mkdir()
        (partial / "checkpoint-500").mkdir()
        self.assertIsNone(WRAPPER.recover_ffmpeg_preflight(self.run, self.manifest))
        self.assertTrue(log.exists())
        self.assertTrue(output.exists())
        self.assertTrue((partial / "checkpoint-500").exists())


if __name__ == "__main__":
    unittest.main()
