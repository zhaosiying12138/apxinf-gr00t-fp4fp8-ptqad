"""CPU-only checkpoint byte identities and opt-in capture completion guards."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from eval import run_recovery_eval as runner
from rl.checkpoint_identity import checkpoint_files, file_sha256


ROOT = Path(__file__).resolve().parents[1]


class CheckpointIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def checkpoint(self, name):
        root = self.root / name
        root.mkdir()
        (root / "model.safetensors").write_bytes((name + " weights").encode())
        (root / "config.json").write_text(json.dumps({"name": name}))
        return root

    def adapter(self):
        base, training, export = [self.checkpoint(name) for name in ("base", "training", "export")]
        (export / "merge_manifest.json").write_text(json.dumps({
            "status": "complete", "base": str(base), "training_checkpoint": str(training)}))
        return base, training, export

    def test_plain_checkpoint_hashes_all_weights_and_json_only(self):
        root = self.checkpoint("base")
        (root / "model-2.safetensors").write_bytes(b"second shard")
        (root / "notes.txt").write_text("not loaded by the checkpoint loader")
        result = checkpoint_files(root)
        expected = [root / "config.json", root / "model.safetensors", root / "model-2.safetensors"]
        self.assertEqual(set(result), {str(path) for path in expected})
        for path in expected:
            self.assertEqual(result[str(path)], {
                "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def test_adapter_identity_tracks_referenced_base_and_training_weights(self):
        base, training, export = self.adapter()
        previous = checkpoint_files(export)
        for root in (base, training, export):
            self.assertIn(str(root / "model.safetensors"), previous)
        for root in (base, training):
            path = root / "model.safetensors"
            path.write_bytes(b"changed reference weights")
            current = checkpoint_files(export)
            self.assertNotEqual(current, previous)
            self.assertNotEqual(current[str(path)], previous[str(path)])
            previous = current

    def test_incomplete_manifest_and_missing_referenced_shards_are_rejected(self):
        base, training, export = self.adapter()
        path = export / "merge_manifest.json"
        manifest = json.loads(path.read_text())
        for status in ("running", None):
            path.write_text(json.dumps({**manifest, "status": status}))
            with self.assertRaisesRegex(ValueError, "Incomplete deployment manifest"):
                checkpoint_files(export)
        path.write_text(json.dumps(manifest))
        (training / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "No weight shards"):
            checkpoint_files(export)
        with self.assertRaisesRegex(ValueError, "No weight shards"):
            checkpoint_files(self.root / "missing")

    def test_sha_reads_bounded_blocks(self):
        path = self.root / "multiple_blocks.bin"
        content = b"12345678" * (1024 * 1024)
        path.write_bytes(content)
        self.assertEqual(file_sha256(path), hashlib.sha256(content).hexdigest())
        # Path.read_bytes must not be used by the shared streaming implementation.
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")):
            self.assertEqual(file_sha256(path), hashlib.sha256(content).hexdigest())

    def test_imports_and_actual_hashing_work_without_site_packages_or_torch(self):
        checkpoint = self.checkpoint("stdlib")
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from rl.checkpoint_identity import checkpoint_files\n"
            "from eval.run_recovery_eval import record_capture_checkpoint, verify_capture_checkpoint\n"
            f"manifest={{'checkpoint': {str(checkpoint)!r}, 'capture_sampling': {{'mode':'full_trajectory_candidates'}}}}\n"
            "record_capture_checkpoint(manifest)\n"
            "summary={}\n"
            "verify_capture_checkpoint(manifest, summary)\n"
            "assert summary['checkpoint_files_verified_unchanged'] is True\n"
            "assert not any(x == 'torch' or x.startswith('torch.') for x in sys.modules)\n"
            "print('stdlib-only checkpoint verification passed')\n"
        )
        completed = subprocess.run([sys.executable, "-S", "-B", "-c", code],
                                   check=True, capture_output=True, text=True)
        self.assertIn("stdlib-only checkpoint verification passed", completed.stdout)

    def test_begin_end_guard_rejects_mutated_adapter_reference(self):
        base, _, export = self.adapter()
        manifest = {"checkpoint": str(export), "capture_sampling": {"mode": "full_trajectory_candidates"}}
        runner.record_capture_checkpoint(manifest)
        self.assertEqual(manifest["checkpoint_files"], checkpoint_files(export))
        summary = {}
        runner.verify_capture_checkpoint(manifest, summary)
        self.assertIs(summary["checkpoint_files_verified_unchanged"], True)
        (base / "model.safetensors").write_bytes(b"changed during capture")
        failed_summary = {}
        with self.assertRaisesRegex(ValueError, "changed during full candidate capture"):
            runner.verify_capture_checkpoint(manifest, failed_summary)
        self.assertEqual(failed_summary, {})
        with self.assertRaisesRegex(ValueError, "lacks initial checkpoint identity"):
            runner.verify_capture_checkpoint({"checkpoint": str(export), "capture_sampling": {}}, {})

    def test_prefix_default_never_hashes_or_adds_identity_fields(self):
        manifest = {"checkpoint": "/nonexistent/legacy/checkpoint"}
        summary = {"total_episodes": 40}
        before_manifest, before_summary = copy.deepcopy(manifest), copy.deepcopy(summary)
        with patch.object(runner, "_capture_checkpoint_files", side_effect=AssertionError("legacy hash")):
            runner.record_capture_checkpoint(manifest)
            runner.verify_capture_checkpoint(manifest, summary)
        self.assertEqual(manifest, before_manifest)
        self.assertEqual(summary, before_summary)

    def run_fake_capture(self, mutate_checkpoint=False):
        checkpoint = self.checkpoint("teacher")
        protocol = self.root / "protocol.json"
        capture_sampling = {"mode": "full_trajectory_candidates", "every_server_calls": 4,
                            "safety_candidates_per_episode": 192}
        protocol.write_text(json.dumps({"partitions": {
            "teacher_supervision": {"seed": 950000, "episodes_per_task": 1,
                                    "init_state_indices": [20], "capture_sampling": capture_sampling},
            "development": {"seed": 940000, "episodes_per_task": 1, "init_state_indices": [4]},
            "heldout": {"seed": 970000, "episodes_per_task": 1, "init_state_indices": [9]}}}))
        output = self.root / "capture"
        argv = ["run_recovery_eval.py", "--checkpoint", str(checkpoint), "--out", str(output),
                "--protocol-file", str(protocol), "--purpose", "teacher_supervision", "--seed", "950000",
                "--gr00t", str(self.root)]
        process = Mock()
        process.poll.return_value = None
        begun = []

        def start_server(*args, **kwargs):
            manifest = json.loads((output / "eval_manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_files"], begun[0] if begun else checkpoint_files(checkpoint))
            begun.append(manifest["checkpoint_files"])
            if mutate_checkpoint and len(begun) == 1:
                (checkpoint / "model.safetensors").write_bytes(b"changed after server launch")
            return process

        raw = {"results": [True], "successes": 1, "episodes": 1, "success_rate": 1.0, "resets": []}
        with patch.object(sys, "argv", argv), patch.object(runner.shutil, "which", return_value="/fake/ffmpeg"), \
             patch.object(runner.socket, "socket"), patch.object(runner.socket, "create_connection"), \
             patch.object(runner.subprocess, "Popen", side_effect=start_server), \
             patch.object(runner.subprocess, "run", return_value=Mock(returncode=0)), \
             patch.object(runner, "parse_log", side_effect=lambda path: copy.deepcopy(raw)), \
             patch.object(runner, "validate_resets"), patch.object(runner, "finalize_capture_samples", return_value={}), \
             patch.object(runner, "stop"):
            if mutate_checkpoint:
                with self.assertRaisesRegex(ValueError, "changed during full candidate capture"):
                    runner.main()
            else:
                runner.main()
        self.assertEqual(len(begun), 10)
        return output

    def test_runner_records_before_first_server_and_marks_unchanged_only_after_all_tasks(self):
        output = self.run_fake_capture()
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["tasks_complete"], 10)
        self.assertIs(summary["checkpoint_files_verified_unchanged"], True)

    def test_runner_changed_checkpoint_cannot_publish_complete_summary(self):
        output = self.run_fake_capture(mutate_checkpoint=True)
        self.assertTrue((output / "eval_manifest.json").is_file())
        self.assertTrue((output / "task_results.json").is_file())
        self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
