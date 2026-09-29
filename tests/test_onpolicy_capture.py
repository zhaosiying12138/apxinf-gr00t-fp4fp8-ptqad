"""CPU-only capture shape/provenance and evaluation accounting checks."""
import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "rl"))
sys.path.insert(0, str(PROJECT / "eval"))
from capture_onpolicy import install_capture
from run_recovery_eval import main as eval_main, parse_log, validate_resets
from rollout_seeded import install_bank_resets
from compare_recovery import ARMS, compare_round, exact_mcnemar_p, paired_summary
from run_recovery_eval import TASKS


class CaptureTests(unittest.TestCase):
    def test_each_observation_is_recollated_instead_of_slicing_image_patches(self):
        class FakeCollator:
            def __call__(self, features):
                patches = sum(f["patches"] for f in features)
                return {"inputs": {"state": torch.full((len(features), 1, 4), 1.003),
                                   "pixel_values": torch.ones(patches, 6)}}

        class FakeModel:
            def get_action(self, inputs):
                return {"action_pred": torch.arange(24.).reshape(2, 3, 4)}

        model_module = ModuleType("gr00t.model.gr00t_n1d7.gr00t_n1d7")
        model_module.Gr00tN1d7 = FakeModel
        processor_module = ModuleType("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
        processor_module.Gr00tN1d7DataCollator = FakeCollator
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            base.mkdir()
            (base / "config.json").write_text("{}")
            (base / "statistics.json").write_text(json.dumps({"libero_sim": {"action": {"x": {"mean": [0, 0]}}}}))
            (base / "processor_config.json").write_text(json.dumps({"processor_kwargs": {
                "modality_configs": {"libero_sim": {"action": {"delta_indices": [0, 1], "modality_keys": ["x"]}}}}}))
            out = Path(directory) / "capture"
            with patch.dict(sys.modules, {model_module.__name__: model_module,
                                          processor_module.__name__: processor_module}):
                install_capture(out, base, every=1)
                features = [{"patches": 7, "vlm_content": {"text": "task"}},
                            {"patches": 11, "vlm_content": {"text": "task"}}]
                with torch.inference_mode():
                    batch = FakeCollator()(features)
                    result = FakeModel().get_action(batch["inputs"])
                sample = torch.load(out / "sample_000000.pt", weights_only=True)
                self.assertEqual(sample["source_kind"], "student_rollout")
                self.assertEqual(sample["inputs"]["pixel_values"].shape[0], 7)
                self.assertEqual(sample["inputs"]["action"].shape[0], 1)
                self.assertTrue(torch.equal(sample["inputs"]["action"], result["action_pred"][:1]))
                self.assertEqual(result["action_pred"].shape, (2, 3, 4))
                self.assertEqual(sample["inputs"]["state"].dtype, torch.bfloat16)
                self.assertEqual(sample["inputs"]["state"][0, 0, 0].item(), torch.tensor(1.003).bfloat16().item())
                self.assertEqual(sample["inputs"]["action_mask"].sum().item(), 4)

    def test_actual_denominators_and_missing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "task.log"
            log.write_text('FP4VLA_EPISODE_RESET {"episode_index": 0, "seed": 12, "initial_state_sha256": "abc"}\n'
                           "results: ('libero_sim/task', [True, False, True])\n")
            result = parse_log(log)
            self.assertEqual((result["successes"], result["episodes"]), (2, 3))
            self.assertEqual(result["resets"][0]["seed"], 12)
            log.write_text("TimeoutError: server did not reply\n")
            self.assertIsNone(parse_log(log)["success_rate"])

    def test_heldout_overlap_is_rejected_before_process_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "collection.json"
            manifest.write_text(json.dumps({"purpose": "collection", "seed": 100, "episodes": 2,
                "initial_state_protocol": "libero10_official_bank_v1", "init_state_indices": [2, 3]}))
            argv = ["run_recovery_eval.py", "--checkpoint", "unused", "--out", str(Path(directory) / "out"),
                    "--seed", "101", "--purpose", "heldout", "--collection-manifest", str(manifest)]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "overlap"):
                eval_main()

    def test_reset_records_are_required_and_verified(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_resets({"resets": []}, 100, [0, 1])
        records = [{"episode_index": i, "seed": 100+i, "init_state_index": i,
                    "settle_steps": 10, "initial_state_sha256": "a"*64,
                    "restored_state_sha256": "b"*64, "init_state_bank_sha256": "c"*64}
                   for i in range(3)]
        self.assertEqual(len(validate_resets({"resets": records}, 100, [0, 1])), 2)
        records[1]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_resets({"resets": records}, 100, [0, 1])

    def test_bank_restore_uses_raw_zero_settling_and_declared_partition(self):
        import numpy as np
        class Raw:
            def __init__(self):
                self.sim = self
                self.state = np.zeros(3)
                self.zero_steps = 0
            def get_state(self):
                return self.state.copy()
            def set_init_state(self, value):
                self.state = value.copy()
                return {"state": self.state.copy()}
            def step(self, action):
                np.testing.assert_array_equal(action, np.zeros(7))
                self.zero_steps += 1
                self.state[0] += .01
                return {"state": self.state.copy()}, 0, False, {}
            def check_success(self):
                return False
        class Env:
            def __init__(self):
                self._env = Raw()
            def reset(self, seed=None, options=None):
                self._env.state[:] = seed
                return {}, {}
            def _process_observation(self, raw):
                return raw
        task = SimpleNamespace(name="task", problem_folder="bank", init_states_file="states.pt")
        suite = SimpleNamespace(get_num_tasks=lambda: 1, get_task=lambda i: task,
            get_task_init_states=lambda i: np.arange(150.).reshape(50,3))
        package = ModuleType("libero.libero")
        package.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": lambda: suite})
        utils = ModuleType("libero.libero.utils")
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bank"
            bank.mkdir(); (bank / "states.pt").write_bytes(b"test-bank")
            utils.get_libero_path = lambda key: directory
            with patch.dict(sys.modules, {package.__name__: package, utils.__name__: utils}):
                meta = install_bank_resets(Env, "task", [2,3])
                env = Env(); output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    first, _ = env.reset(seed=100)
                    env.reset(); env.reset()
                lines = [json.loads(x.split("FP4VLA_EPISODE_RESET ")[1]) for x in output.getvalue().splitlines()]
                self.assertEqual([x["init_state_index"] for x in lines], [2,3,2])
                self.assertEqual([x["seed"] for x in lines], [100,101,102])
                self.assertEqual(env._env.zero_steps, 30)
                self.assertEqual(meta["bank_size"], 50)
                self.assertAlmostEqual(first["state"][0], 6.1)

    def test_comparison_rejects_unpaired_initial_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ARMS:
                folder = root / f"heldout_{arm}"; folder.mkdir()
                manifest = {"purpose": "heldout", "tasks": TASKS, "episodes": 10,
                            "seed": 220000, "init_state_indices": list(range(10,20)),
                            "initial_state_protocol": "libero10_official_bank_v1"}
                (folder / "eval_manifest.json").write_text(json.dumps(manifest))
                (folder / "summary.json").write_text(json.dumps({"tasks_complete": 10}))
                records = {}
                for i,task in enumerate(TASKS):
                    records[task] = {"results": [True]*10, "returncode": 0,
                        "resets": [{"episode_index": j, "seed": 220000+1000*i+j,
                                    "init_state_index": 10+j, "settle_steps": 10,
                                    "initial_state_sha256": "a"*64,
                                    "restored_state_sha256": "b"*64,
                                    "init_state_bank_sha256": "c"*64} for j in range(10)]}
                (folder / "task_results.json").write_text(json.dumps(records))
            report = compare_round(root)
            self.assertTrue(report["environment_pairing_verified"])
            self.assertEqual(report["arms"]["qad"]["per_task"][TASKS[0]]["successes"],10)
            self.assertEqual(len(report["source_files"]),15)
            self.assertEqual(report["opd_vs_qad"]["both_success"],100)
            path = root / "heldout_qad_opd/task_results.json"
            records = json.loads(path.read_text())
            records[TASKS[0]]["resets"][0]["initial_state_sha256"] = "d"*64
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, "does not pair"):
                compare_round(root)

    def test_exact_mcnemar_and_directional_table(self):
        self.assertEqual(exact_mcnemar_p(0,0),1.)
        self.assertEqual(exact_mcnemar_p(3,3),1.)
        self.assertEqual(exact_mcnemar_p(0,6),.03125)
        result = paired_summary([{"success": x} for x in [True,True,False,False]],
                                [{"success": x} for x in [True,False,True,False]])
        self.assertEqual(result["table_rows_baseline_success_fail_columns_opd_success_fail"],[[1,1],[1,1]])
        self.assertEqual(result["success_rate_difference_opd_minus_baseline"],0.)


if __name__ == "__main__":
    unittest.main()
