"""CPU-only contract fixtures for ``run_high_fp4_v3.py``.

These tests validate protocol/selection provenance and the no-overwrite resume
guard. They deliberately create tiny fake checkpoints and never import torch,
launch a simulator, or start a GPU worker.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from exp import run_high_fp4_v3 as driver


class HighFp4V3CpuFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.protocol = self.root / "exp/recovery_protocol_v3_high_fp4.json"
        self.tmp = Path(tempfile.mkdtemp(prefix="fp4vla-v3-cpu-"))
        self.base = self.tmp / "bf16"
        self.ptq = self.tmp / "ptq"
        for path in (self.base, self.ptq):
            path.mkdir()
            (path / "model-00001.safetensors").write_bytes(b"fixture-weight")
            (path / "config.json").write_text("{}\n")
            (path / "statistics.json").write_text("{}\n")
        (self.ptq / "ptq_recipe.json").write_text("{}\n")
        protocol_sha = driver.sha(self.protocol)
        self.selection = self.tmp / "selection.json"
        arms = {}
        for name, successes, checkpoint in (
            ("bf16", 46, self.base),
            ("head_lang_vision", 25, self.ptq),
            ("calib", 15, self.ptq),
        ):
            arms[name] = {
                "successes": successes,
                "episodes": 50,
                "checkpoint": str(checkpoint),
                "environment_pairing_verified": True,
            }
        self.selection.write_text(json.dumps({
            "protocol_sha256": protocol_sha,
            "selection_uses_heldout": False,
            "selected_recipe": "calib",
            "pressure_candidates": ["head_lang_vision", "calib"],
            "arms": arms,
        }))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_validate_only_records_protocol_and_selection(self):
        run_dir = self.tmp / "run"
        result = driver.main([
            "--run-dir", str(run_dir), "--protocol-file", str(self.protocol),
            "--ptq-selection", str(self.selection), "--base", str(self.base),
            "--validate-only",
        ])
        self.assertEqual(result, 0)
        state = json.loads((run_dir / "run_manifest.json").read_text())
        self.assertEqual(state["protocol_sha256"], driver.sha(self.protocol))
        self.assertFalse(state["selection_uses_heldout"])

    def test_selection_change_is_rejected_on_resume(self):
        run_dir = self.tmp / "run"
        args = ["--run-dir", str(run_dir), "--protocol-file", str(self.protocol),
                "--ptq-selection", str(self.selection), "--base", str(self.base),
                "--validate-only"]
        self.assertEqual(driver.main(args), 0)
        changed = json.loads(self.selection.read_text())
        changed["selected_recipe"] = "head_lang_vision"
        self.selection.write_text(json.dumps(changed))
        self.assertEqual(driver.main(args), 2)


if __name__ == "__main__":
    unittest.main()
