"""Declared ladder order, exact 0.10 boundary, and second-episode pairing."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"eval"))
from compare_development import audit_development
from run_recovery_eval import TASKS


class DevelopmentSelectionTests(unittest.TestCase):
    def test_exact_boundary_and_second_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm,successes in (("bf16",19),("fp8",19),("head_ffn",17)):
                folder = root/arm; folder.mkdir()
                manifest = {"purpose":"development","seed":330000,"episodes":2,
                    "init_state_indices":[0,1],"n_envs":1,"tasks":TASKS,
                    "initial_state_protocol":"libero10_official_bank_v1","protocol_sha256":"a"*64,
                    "n_action_steps":8,"max_episode_steps":720,"settle_steps":10,"checkpoint":arm}
                tasks = {}
                for i,task in enumerate(TASKS):
                    tasks[task] = {"returncode":0,"results":[2*i+j<successes for j in range(2)],
                        "resets":[{"episode_index":j,"seed":330000+1000*i+j,"init_state_index":j,
                            "settle_steps":10,"initial_state_sha256":"a"*64,
                            "restored_state_sha256":"b"*64,"init_state_bank_sha256":"c"*64} for j in range(2)]}
                for name,data in (("eval_manifest.json",manifest),("task_results.json",tasks),
                    ("summary.json",{"tasks_complete":10,"wall_seconds_including_server_loads":1.})):
                    (folder/name).write_text(json.dumps(data))
            result = audit_development(root,["bf16","fp8","head_ffn"])
            self.assertEqual(result["selected_recipe"],"head_ffn")
            self.assertEqual(result["arms"]["bf16"]["successes"],19)
            path = root/"head_ffn/task_results.json"
            records = json.loads(path.read_text())
            records[TASKS[0]]["resets"][1]["initial_state_sha256"] = "d"*64
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError,"paired slots.*1"):
                audit_development(root,["bf16","fp8","head_ffn"])

    def test_v3_uses_declared_partition_and_highest_qualifying_pressure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({
                "version": 3,
                "partitions": {"development": {"seed": 440000,
                    "init_state_indices": [4, 5, 6, 7, 8], "episodes_per_task": 5}},
                "selection": {"pressure_candidates": ["head_lang_vision", "calib"],
                    "pressure_rule": {"min_drop_from_bf16": 0.20,
                        "min_absolute_success": 0.30, "choose": "highest_fp4"},
                    "rule": "Evaluate both candidates. Freeze the highest-FP4 candidate whose development macro success is at least 0.20 below BF16 and at least 0.30 absolute."}}))
            protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
            for arm, successes in (("bf16", 47), ("head_lang_vision", 32), ("calib", 20)):
                folder = root / arm
                folder.mkdir()
                manifest = {"purpose": "development", "seed": 440000, "episodes": 5,
                    "init_state_indices": [4, 5, 6, 7, 8], "n_envs": 1, "tasks": TASKS,
                    "initial_state_protocol": "libero10_official_bank_v1",
                    "protocol_sha256": protocol_sha, "n_action_steps": 8,
                    "max_episode_steps": 720, "settle_steps": 10, "checkpoint": arm}
                tasks = {}
                for i, task in enumerate(TASKS):
                    tasks[task] = {"returncode": 0,
                        "results": [5*i+j < successes for j in range(5)],
                        "resets": [{"episode_index": j, "seed": 440000+1000*i+j,
                            "init_state_index": 4+j, "settle_steps": 10,
                            "initial_state_sha256": "a"*64,
                            "restored_state_sha256": "b"*64,
                            "init_state_bank_sha256": "c"*64} for j in range(5)]}
                for name, data in (("eval_manifest.json", manifest), ("task_results.json", tasks),
                        ("summary.json", {"tasks_complete": 10,
                                         "wall_seconds_including_server_loads": 1.})):
                    (folder / name).write_text(json.dumps(data))
            partial = audit_development(root, ["bf16", "head_lang_vision"], protocol)
            self.assertIsNone(partial["selected_recipe"])
            result = audit_development(root, ["bf16", "head_lang_vision", "calib"], protocol)
            self.assertEqual(result["selected_recipe"], "calib")
            self.assertEqual(result["arms"]["calib"]["episodes"], 50)
            self.assertEqual(result["development_partition"]["init_state_indices"], [4,5,6,7,8])


if __name__ == "__main__":
    unittest.main()
