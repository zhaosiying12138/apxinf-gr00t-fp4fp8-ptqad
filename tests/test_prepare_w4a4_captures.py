"""CPU checks for the v11 screenshot command preparer."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_captures", ROOT / "paper/prepare_w4a4_captures.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def fake_values(root):
    project = root / "repo"
    for name in ("paper", "rl", "eval", "exp"):
        (project / name).mkdir(parents=True, exist_ok=True)
    protocol = project / "exp/protocol.json"
    protocol.write_text("{}")
    paths = {}
    for name in ("groot", "teacher", "pressure", "q_model", "op_model", "q_adapter", "dataset", "capture"):
        paths[name] = root / name
        paths[name].mkdir()
    (paths["pressure"] / "category_ptq_recipe.json").write_text("{}")
    (paths["pressure"] / "category_bake_manifest.json").write_text("{}")
    for folder in (paths["q_model"], paths["op_model"]):
        (folder / "merge_manifest.json").write_text("{}")
    for folder in (paths["teacher"], paths["pressure"], paths["q_model"]):
        (folder / "config.json").write_text("{}")
        (folder / "statistics.json").write_text("{}")
    fake_python = root / "venv/bin/python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/sh\nexit 0\n")
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    return project, protocol, paths, fake_python


class PrepareW4A4Captures(unittest.TestCase):
    def test_executable_preserves_venv_symlink_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "system-python"
            target.write_text("#!/bin/sh\n")
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
            link = root / "venv/bin/python"
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            self.assertEqual(prepare.executable(link, "python"), link.absolute())
            self.assertNotEqual(prepare.executable(link, "python"), target)

    def test_rollout_capture_environment_precedes_fake_server_and_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, protocol_path, paths, fake_python = fake_values(root)
            values = {
                "groot": paths["groot"], "server_python": fake_python, "rollout_python": fake_python,
                "teacher": paths["teacher"], "pressure": paths["pressure"], "q_model": paths["q_model"],
                "q_adapter": paths["q_adapter"], "op_model": paths["op_model"], "dataset": paths["dataset"],
                "capture_dataset": paths["capture"], "capture_sha": "a" * 64, "protocol_path": protocol_path,
                "backbone": "local", "state_seed": 1, "task_seed": 1,
                "q_rec": {"scope": "all_ordinary_linear", "rank": 32, "alpha": 64,
                          "f16_activation_saturation": False},
                "op_rec": {"f16_activation_saturation": True},
            }
            bundle = {"final_path": project / "results/run/final_manifest.json", "protocol_path": protocol_path,
                      "final": {"selected_qad_learning_rate": .0001, "selected_opd_weight": .25},
                      "protocol": {"selection": {"opd_every": 4}}}
            out = root / "prepared"
            final_path = project / "results/run/final_manifest.json"
            final_path.parent.mkdir(parents=True)
            final_path.write_text("{}")
            prepare.render_scripts(out, bundle, values)
            plan = json.loads((out / "plan.json").read_text())
            self.assertEqual(plan["final_manifest_sha256"], prepare.digest(final_path))
            self.assertEqual(plan["protocol_sha256"], prepare.digest(protocol_path))
            rollout = (out / "shot_rollout.sh").read_text()
            self.assertLess(rollout.index("FP4VLA_CAPTURE_EVENT_FILE"),
                            rollout.index("serve_recovery.py"))
            self.assertIn('> >(tee "$OUT/server.raw.log") 2>&1 &', rollout)
            self.assertIn('"$QAD_MODEL"', rollout)
            self.assertIn('FP4VLA_SATURATE_F16_ACTIVATIONS=0', rollout)
            evalserver = (out / "shot_evalserver.sh").read_text()
            self.assertIn('"$OPD_MODEL"', evalserver)
            self.assertIn('FP4VLA_SATURATE_F16_ACTIVATIONS=1', evalserver)
            opdcache = (out / "shot_opdcache.sh").read_text()
            self.assertIn("export FP4VLA_QUANT=0 FP4VLA_W4A4=0 FP4VLA_W4A4_ADAPTER=0 FP4VLA_SATURATE_F16_ACTIVATIONS=0", opdcache)
            self.assertLess(opdcache.index("FP4VLA_W4A4=0"), opdcache.index("opd_probe_cache.py"))
            self.assertIn("unset QAD_INIT_ADAPTER QAD_MAX_GRAD_NORM OPD_CACHE_PATH", (out / "shot_qad.sh").read_text())
            # Execute only the generated server preamble with a fake process.
            fake = root / "fake-server.py"
            fake.write_text("""#!/usr/bin/env python3
import json, os, socket, sys, time
print('fake server stdout', flush=True)
json.dump({k: os.environ.get(k) for k in ('OPD_CAPTURE_DIR','FP4VLA_CAPTURE_EVENT_FILE','FP4VLA_W4A4')}, open(os.environ['FAKE_ENV'],'w'))
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1',int(sys.argv[1]))); s.listen(1)
time.sleep(4)
""")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            marker = root / "fake-env.json"
            fake_nvidia = root / "nvidia-smi"
            fake_nvidia.write_text("#!/bin/sh\nexit 0\n")
            fake_nvidia.chmod(fake_nvidia.stat().st_mode | stat.S_IXUSR)
            script = rollout.split('"$LIBERO_PYTHON"', 1)[0]
            launch = '"$PTQAD_PYTHON" -u "$PROJECT/eval/serve_recovery.py" --model-path "$QAD_MODEL" --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --host 127.0.0.1 --port "$PORT" > >(tee "$OUT/server.raw.log") 2>&1 &'
            replacement = '"$FAKE_SERVER" "$PORT" > >(tee "$OUT/server.raw.log") 2>&1 &'
            script = script.replace(launch, replacement)
            script += '\nkill -TERM "$server_pid" 2>/dev/null || true\nwait "$server_pid" || true\n'
            run_script = out / "fake-rollout.sh"
            run_script.write_text(script)
            run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR)
            env = {**os.environ, "FAKE_SERVER": str(fake), "FAKE_ENV": str(marker),
                   "PATH": str(root) + os.pathsep + os.environ.get("PATH", "")}
            result = subprocess.run([str(run_script)], env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = json.loads(marker.read_text())
            self.assertTrue(values["OPD_CAPTURE_DIR"].endswith("shot_rollout/observations"))
            self.assertTrue(values["FP4VLA_CAPTURE_EVENT_FILE"].endswith("reset_events.jsonl"))
            self.assertEqual(values["FP4VLA_W4A4"], "1")
            self.assertIn("fake server stdout", result.stdout)
            self.assertIn("fake server stdout", (out / "scratch/shot_rollout/server.raw.log").read_text())


if __name__ == "__main__":
    unittest.main()
