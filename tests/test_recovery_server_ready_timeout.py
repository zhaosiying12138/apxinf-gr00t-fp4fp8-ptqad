"""CPU-only readiness budget checks: all sockets, processes, and sleeps are mocked."""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from eval import run_recovery_eval as runner


class RecoveryServerReadyTimeoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "evaluation"
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps({"partitions": {
            "smoke": {"init_state_indices": [0], "episodes_per_task": 1, "seed": 1}}}))
        self.argv = ["run_recovery_eval.py", "--checkpoint", str(self.root / "checkpoint"),
                     "--out", str(self.output), "--purpose", "smoke", "--seed", "1",
                     "--task-count", "1", "--protocol-file", str(self.protocol),
                     "--gr00t", str(self.root / "gr00t"), "--timeout", "1234"]

    def mocked_runtime(self, timeout=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        environment = {"CUDA_VISIBLE_DEVICES": ""}
        if timeout is not None:
            environment["PTQAD_SERVER_READY_TIMEOUT_S"] = timeout
        stack.enter_context(patch.dict(runner.os.environ, environment, clear=True))
        stack.enter_context(patch("sys.argv", self.argv))
        stack.enter_context(patch.object(runner.shutil, "which", return_value="/mock/ffmpeg"))
        self.socket = stack.enter_context(patch.object(runner.socket, "socket"))
        self.connect = stack.enter_context(patch.object(
            runner.socket, "create_connection", side_effect=ConnectionRefusedError))
        self.server = Mock()
        self.server.poll.return_value = None
        self.launch = stack.enter_context(patch.object(
            runner.subprocess, "Popen", return_value=self.server))
        self.rollout = stack.enter_context(patch.object(runner.subprocess, "run"))
        self.sleep = stack.enter_context(patch.object(runner.time, "sleep"))
        self.stop = stack.enter_context(patch.object(runner, "stop"))
        return stack

    def assert_exhausted_budget(self, configured, expected):
        with self.mocked_runtime(configured):
            with self.assertRaisesRegex(TimeoutError, "Server did not become ready"):
                runner.main()
            self.assertEqual(self.connect.call_count, expected)
            self.assertEqual(self.sleep.call_count, expected)
            self.sleep.assert_called_with(1)
            self.launch.assert_called_once()
            self.rollout.assert_not_called()
            self.stop.assert_called_once_with(self.server)
            manifest = json.loads((self.output / "eval_manifest.json").read_text())
            self.assertEqual(manifest["server_ready_timeout_s"], expected)
            self.assertEqual(manifest["timeout"], 1234)
            self.assertFalse((self.output / "summary.json").exists())

    def test_default_retains_180_readiness_attempts(self):
        self.assert_exhausted_budget(None, 180)

    def test_positive_override_controls_attempts_and_is_recorded(self):
        self.assert_exhausted_budget("3", 3)

    def test_smallest_positive_override_is_accepted(self):
        self.assert_exhausted_budget("1", 1)

    def test_invalid_values_fail_before_checkpoint_network_or_output(self):
        for value in ("", "abc", "1.5", "0", "-1"):
            with self.subTest(value=value), self.mocked_runtime(value):
                stderr = io.StringIO()
                with patch.object(runner, "validate_recovery_checkpoint") as validate, \
                     patch("sys.stderr", stderr), self.assertRaises(SystemExit) as raised:
                    runner.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("PTQAD_SERVER_READY_TIMEOUT_S", stderr.getvalue())
                self.assertIn("must be positive" if value in ("0", "-1") else
                              "must be an integer", stderr.getvalue())
                validate.assert_not_called()
                self.socket.assert_not_called()
                self.launch.assert_not_called()
                self.rollout.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_readiness_stops_waiting_and_preserves_rollout_timeout(self):
        with self.mocked_runtime("3"):
            self.connect.side_effect = [ConnectionRefusedError, Mock(
                __enter__=Mock(), __exit__=Mock(return_value=False))]
            self.rollout.side_effect = RuntimeError("mock rollout reached")
            with self.assertRaisesRegex(RuntimeError, "mock rollout reached"):
                runner.main()
            self.assertEqual(self.connect.call_count, 2)
            self.sleep.assert_called_once_with(1)
            self.rollout.assert_called_once()
            self.assertEqual(self.rollout.call_args.kwargs["timeout"], 1234)
            self.stop.assert_called_once_with(self.server)

    def test_server_exit_fails_without_consuming_readiness_budget(self):
        with self.mocked_runtime("3"):
            self.server.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "Server exited"):
                runner.main()
            self.connect.assert_not_called()
            self.sleep.assert_not_called()
            self.rollout.assert_not_called()
            self.stop.assert_called_once_with(self.server)


if __name__ == "__main__":
    unittest.main()
