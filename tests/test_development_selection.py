"""Declared ladder order, exact 0.10 boundary, and second-episode pairing."""
import json
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


if __name__ == "__main__":
    unittest.main()
