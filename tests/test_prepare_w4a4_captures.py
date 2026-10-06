"""CPU checks for the W4A4 screenshot command preparer; no real GPU runs."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_captures", ROOT / "paper/prepare_w4a4_captures.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def fake_values(root, local_protocol=False):
    project = root / "repo"
    for name in ("paper", "rl", "eval", "exp"):
        (project / name).mkdir(parents=True, exist_ok=True)
    protocol = project / ("results/ptqad_v11/recovery_protocol_v11_w4a4_category.local.json"
                          if local_protocol else "exp/protocol.json")
    protocol.parent.mkdir(parents=True, exist_ok=True)
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


def fake_evaluation(bundle, arm, model, saturation="1", update=None):
    """Build a hash-bound receipt; no model or evaluation is executed."""
    round_dir = bundle["final_path"].parent / "heldout_round"
    bundle["final"].update(heldout_round=str(round_dir), protocol_sha256="a" * 64)
    bundle["comparison_path"] = round_dir / "paired_comparison.json"
    manifest = {"checkpoint": str(model), "purpose": "heldout", "protocol_sha256": "a" * 64,
                "environment_summary": {"variables": {
                    "FP4VLA_QUANT": "0", "FP4VLA_W4A4": "1", "FP4VLA_W4A4_ADAPTER": "1",
                    "FP4VLA_SATURATE_F16_ACTIVATIONS": saturation}}}
    if update:
        update(manifest)
    relative = f"heldout_{arm}/eval_manifest.json"
    path = round_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    bundle.setdefault("comparison", {}).setdefault("source_files", {})[relative] = {
        "bytes": path.stat().st_size, "sha256": prepare.digest(path)}
    return path


class PrepareW4A4Captures(unittest.TestCase):
    def test_inference_saturation_is_read_from_bound_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = {"final_path": root / "final_manifest.json", "final": {}}
            for arm, saturation in (("qad", "1"), ("qad_opd", "0")):
                with self.subTest(arm=arm):
                    model = root / arm
                    path = fake_evaluation(bundle, arm, model, saturation)
                    contract = prepare.inference_contract(bundle, arm, model)
                    self.assertEqual(contract["environment"]["FP4VLA_SATURATE_F16_ACTIVATIONS"], saturation)
                    self.assertEqual(contract["eval_manifest"]["sha256"], prepare.digest(path))
                    self.assertEqual(contract["checkpoint"], str(model))

    def test_modified_evaluation_receipt_is_rejected_before_reading_switches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = {"final_path": root / "final_manifest.json", "final": {}}
            path = fake_evaluation(bundle, "qad", root / "qad")
            data = json.loads(path.read_text())
            data["environment_summary"]["variables"]["FP4VLA_SATURATE_F16_ACTIVATIONS"] = "0"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                prepare.inference_contract(bundle, "qad", root / "qad")

    def test_missing_comparison_source_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = {"final_path": root / "final_manifest.json", "final": {}}
            fake_evaluation(bundle, "qad", root / "qad")
            bundle["comparison"]["source_files"].clear()
            with self.assertRaisesRegex(ValueError, "identity is missing"):
                prepare.inference_contract(bundle, "qad", root / "qad")

    def test_wrong_checkpoint_protocol_or_partition_is_rejected(self):
        changes = (("checkpoint", "other-model", "checkpoint differs"),
                   ("protocol_sha256", "b" * 64, "heldout protocol"),
                   ("purpose", "development", "heldout protocol"))
        for key, value, message in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = {"final_path": root / "final_manifest.json", "final": {}}
                fake_evaluation(bundle, "qad", root / "qad", update=lambda data: data.update({key: value}))
                with self.assertRaisesRegex(ValueError, message):
                    prepare.inference_contract(bundle, "qad", root / "qad")

    def test_missing_or_nonbinary_saturation_cannot_fall_back_to_training(self):
        for value in (None, True, 1, "true", ""):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = {"final_path": root / "final_manifest.json", "final": {}}
                fake_evaluation(bundle, "qad", root / "qad", saturation=value)
                with self.assertRaisesRegex(ValueError, "explicit binary inference"):
                    prepare.inference_contract(bundle, "qad", root / "qad")

    def test_non_adapter_or_double_quantized_inference_is_rejected(self):
        for key, value in (("FP4VLA_QUANT", "1"), ("FP4VLA_W4A4", "0"),
                           ("FP4VLA_W4A4_ADAPTER", "0")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = {"final_path": root / "final_manifest.json", "final": {}}
                fake_evaluation(bundle, "qad", root / "qad", update=lambda data:
                                data["environment_summary"]["variables"].update({key: value}))
                with self.assertRaisesRegex(ValueError, "W4A4 adapter inference"):
                    prepare.inference_contract(bundle, "qad", root / "qad")

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
        self._check_rollout_capture_environment(local_protocol=False)

    def test_readme_protocol_copy_under_results_uses_generator_repository(self):
        self._check_rollout_capture_environment(local_protocol=True)

    def test_v12_bake_audit_reads_checkpoint_as_directory_not_json(self):
        self._check_rollout_capture_environment(local_protocol=False, version=12)

    def _check_rollout_capture_environment(self, local_protocol, version=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, protocol_path, paths, fake_python = fake_values(root, local_protocol)
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
            if version == 12:
                bundle["protocol"].update(version=12, w4a4=True, id="synthetic-v12")
                protocol_path.write_text(json.dumps(bundle["protocol"]))
            out = root / "prepared"
            final_path = project / "results/run/final_manifest.json"
            final_path.parent.mkdir(parents=True)
            final_path.write_text("{}")
            values["inference"] = {}
            for arm, model in (("qad", paths["q_model"]), ("qad_opd", paths["op_model"])):
                fake_evaluation(bundle, arm, model)
                values["inference"][arm] = prepare.inference_contract(bundle, arm, model)
            protocol_before = protocol_path.read_bytes()
            with patch.object(prepare, "ROOT", project):
                prepare.render_scripts(out, bundle, values)
            self.assertEqual(protocol_path.read_bytes(), protocol_before)
            common_name = "common_v12.sh" if version == 12 else "common_v11.sh"
            common = (out / common_name).read_text()
            self.assertIn(f"PROJECT={prepare.q(project)}\n", common)
            self.assertIn(f"PROTOCOL_FILE={prepare.q(protocol_path)}\n", common)
            if local_protocol:
                self.assertEqual(protocol_path.parents[1], project / "results")
                self.assertNotIn(f"PROJECT={prepare.q(project / 'results')}\n", common)
            plan = json.loads((out / "plan.json").read_text())
            self.assertEqual(plan["final_manifest_sha256"], prepare.digest(final_path))
            self.assertEqual(plan["protocol_sha256"], prepare.digest(protocol_path))
            rollout = (out / "shot_rollout.sh").read_text()
            self.assertLess(rollout.index("FP4VLA_CAPTURE_EVENT_FILE"),
                            rollout.index("serve_recovery.py"))
            self.assertIn('> >(tee "$OUT/server.raw.log") 2>&1 &', rollout)
            self.assertIn('"$QAD_MODEL"', rollout)
            self.assertIn('FP4VLA_SATURATE_F16_ACTIVATIONS=1', rollout)
            evalserver = (out / "shot_evalserver.sh").read_text()
            self.assertIn('"$OPD_MODEL"', evalserver)
            self.assertIn('FP4VLA_SATURATE_F16_ACTIVATIONS=1', evalserver)
            opdcache = (out / "shot_opdcache.sh").read_text()
            self.assertIn("export FP4VLA_QUANT=0 FP4VLA_W4A4=0 FP4VLA_W4A4_ADAPTER=0 FP4VLA_SATURATE_F16_ACTIVATIONS=0", opdcache)
            self.assertLess(opdcache.index("FP4VLA_W4A4=0"), opdcache.index("opd_probe_cache.py"))
            self.assertIn("unset QAD_INIT_ADAPTER QAD_MAX_GRAD_NORM OPD_CACHE_PATH", (out / "shot_qad.sh").read_text())
            qad_train = (out / "shot_qad.sh").read_text()
            opd_train = (out / "shot_opd.sh").read_text()
            self.assertIn("QAD_STEPS=2", qad_train)
            self.assertIn("FP4VLA_SATURATE_F16_ACTIVATIONS=0", qad_train)
            self.assertIn("QAD_STEPS=4", opd_train)
            self.assertIn("FP4VLA_SATURATE_F16_ACTIVATIONS=1", opd_train)
            self.assertEqual(plan["training_activation_saturation"], {"qad": "0", "qad_opd": "1"})
            self.assertEqual(plan["rollout_activation_saturation"], "1")
            self.assertEqual(plan["inference_contracts"], values["inference"])
            self.assertEqual(len(plan["script_files"]), 8 if version == 12 else 6)
            self.assertEqual(plan["script_files"][common_name]["sha256"], prepare.digest(out / common_name))
            self.assertIn(common_name, plan["script_dependencies"]["all_entrypoints"])
            self.assertIn(f"Retain `{common_name}` and `plan.json`", (out / "README.md").read_text())
            if version == 12:
                bake = (out / "shot_bake.sh").read_text()
                inventory = root / "synthetic-inventory.json"
                inventory.write_text(json.dumps({"recipe": "rtn_all", "recipes": {"rtn_all": {
                    "nvfp4_params": 64, "linear_params": 64, "fp8_params": 0, "bf16_params": 0}},
                    "recovery_residual": {"target_bytes": 32}}))
                # Run the actual generated Python audit against a directory.
                # No quantization, model execution or terminal capture occurs.
                code = bake.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
                audit = subprocess.run([sys.executable, "-", str(inventory), str(protocol_path),
                                        str(paths["pressure"])], input=code, text=True,
                                       capture_output=True, timeout=10)
                self.assertEqual(audit.returncode, 0, audit.stderr)
                self.assertEqual(json.loads(audit.stdout)["ptq_base"], str(paths["pressure"]))
                self.assertEqual(json.loads(audit.stdout)["nvfp4_params"], 64)
            for script_path in out.glob("*.sh"):
                syntax = subprocess.run(["bash", "-n", str(script_path)], text=True, capture_output=True)
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
            # Execute only the generated server preamble with a fake process.
            fake = root / "fake-server.py"
            fake.write_text("""#!/usr/bin/env python3
import json, os, socket, sys, time
print('fake server stdout', flush=True)
json.dump({k: os.environ.get(k) for k in ('OPD_CAPTURE_DIR','FP4VLA_CAPTURE_EVENT_FILE','FP4VLA_W4A4','FP4VLA_SATURATE_F16_ACTIVATIONS')}, open(os.environ['FAKE_ENV'],'w'))
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
            self.assertEqual(values["FP4VLA_SATURATE_F16_ACTIVATIONS"], "1")
            self.assertIn("fake server stdout", result.stdout)
            self.assertIn("fake server stdout", (out / "scratch/shot_rollout/server.raw.log").read_text())


if __name__ == "__main__":
    unittest.main()
