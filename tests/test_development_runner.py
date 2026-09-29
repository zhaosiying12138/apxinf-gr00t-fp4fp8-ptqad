"""CPU guards for ordered development selection; no model or subprocess execution."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("development_runner", ROOT / "exp/run_development.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def checkpoint(path):
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"fixture weights, not a loadable model")
    for name in ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json"):
        write(path / name, {"fixture": name})
    write(path / "model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
    return runner.checkpoint_identity(path)


def bake(path, recipe, base):
    current = checkpoint(path)
    write(path / "bake_manifest.json", {
        "status": "complete", "base": base["path"],
        "source_weight_files": base["weights"], "output_weight_files": current["weights"]})
    write(path / "ptq_recipe.json", {"recipe": recipe, "base": base["path"],
        "calibration_mode": "required", "calibration_provenance": {"metadata": {
            "status": "complete", "windows_consumed": 128, "windows_requested": 128, "recipe_targets": "calib",
            "base_config_sha256": base["metadata"]["config.json"]["sha256"],
            "base_statistics_sha256": base["metadata"]["statistics.json"]["sha256"]}}})


class DevelopmentRunnerTest(unittest.TestCase):
    def test_completed_bake_rejects_changed_weights_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = checkpoint(root / "base")
            out = root / "fp8"
            bake(out, "fp8", base)
            runner.validate_bake(out, "fp8", base)
            write(out / "statistics.json", {"wrong": True})
            with self.assertRaisesRegex(ValueError, "changed base metadata"):
                runner.validate_bake(out, "fp8", base)
            (out / "statistics.json").write_bytes((root / "base/statistics.json").read_bytes())
            (out / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                runner.validate_bake(out, "fp8", base)

    def test_incomplete_bake_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = checkpoint(root / "base")
            out = root / "fp8"
            bake(out, "fp8", base)
            record = runner.load(out / "bake_manifest.json")
            record["status"] = "running"
            write(out / "bake_manifest.json", record)
            with self.assertRaisesRegex(ValueError, "Incomplete/failed"):
                runner.validate_bake(out, "fp8", base)

    def test_promoted_bake_requires_calibration_base_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = checkpoint(root / "base")
            out = root / "head_ffn"
            bake(out, "head_ffn", base)
            allocation = runner.load(out / "ptq_recipe.json")
            allocation["calibration_provenance"] = {}
            write(out / "ptq_recipe.json", allocation)
            with self.assertRaisesRegex(ValueError, "requires complete|required|lacks base metadata"):
                runner.validate_bake(out, "head_ffn", base)

    def test_reused_smoke_calibration_cannot_be_formal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = checkpoint(root / "base")
            out = root / "head_ffn"
            bake(out, "head_ffn", base)
            allocation = runner.load(out / "ptq_recipe.json")
            allocation["calibration_provenance"]["metadata"]["windows_consumed"] = 2
            write(out / "ptq_recipe.json", allocation)
            with self.assertRaisesRegex(ValueError, "128-window"):
                runner.validate_bake(out, "head_ffn", base)

    def args(self, root, dry=False):
        return argparse.Namespace(run_dir=str(root / "run"), development_root=str(root / "dev"),
            base=str(root / "base"), gr00t=str(root / "gr00t"), server_python=sys.executable,
            rollout_python=sys.executable, port=5610, dry_run=dry)

    def test_new_output_and_pretraining_order_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dev").mkdir()
            with self.assertRaisesRegex(ValueError, "new development root"):
                runner.execute(self.args(root))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run/train_qad").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "before recovery training"):
                runner.execute(self.args(root))

    def test_first_selected_stops_before_any_higher_bake_or_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = checkpoint(root / "base")
            write(root / "run/calibration/calib_meta.json", {
                "status": "complete", "windows_consumed": 128, "recipe_targets": "calib"})
            (root / "run/calibration/calib.pt").write_bytes(b"fixture")
            visited = []
            def run(command, log, cwd, env):
                if command[0] == "bash":
                    arm = command[-1]
                    visited.append(("bake", arm))
                    bake(root / "run" / arm, arm, base)
                else:
                    arm = Path(command[command.index("--out") + 1]).name
                    visited.append(("eval", arm))
            def audit(folder, arms):
                return {"selected_recipe": "head_ffn" if arms[-1] == "head_ffn" else None}
            with patch.object(runner, "run_logged", side_effect=run), \
                 patch.object(runner, "verify_result"), \
                 patch.object(runner, "audit_development", side_effect=audit), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.execute(self.args(root))
            self.assertEqual(visited, [("eval", "bf16"), ("bake", "fp8"), ("eval", "fp8"),
                                       ("bake", "head_ffn"), ("eval", "head_ffn")])
            self.assertEqual(runner.load(root / "dev/selection.json")["selected_recipe"], "head_ffn")
            self.assertFalse((root / "run/head_lang").exists())

    def test_dry_run_never_creates_output_or_hashes_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(runner, "checkpoint_identity", side_effect=AssertionError("must not hash")), \
                 patch.object(runner, "run_logged", side_effect=AssertionError("must not launch")), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                runner.execute(self.args(root, dry=True))
            self.assertFalse((root / "dev").exists())
            self.assertFalse(json.loads(output.getvalue())["resource_identities_verified"])


if __name__ == "__main__":
    unittest.main()
