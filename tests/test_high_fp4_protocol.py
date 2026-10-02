"""Protocol normalization checks for the W4A4 recovery driver."""
from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "high_fp4_driver", ROOT / "exp" / "run_high_fp4_v3.py"
)
DRIVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DRIVER)


class HighFp4ProtocolTest(unittest.TestCase):
    def test_v11_exposes_teacher_supervision_partition(self):
        # Wrappers need the normalized teacher partition to construct replay
        # commands; preserving it must not depend on a retired v9 JSON file.
        protocol = DRIVER.load_protocol(ROOT / "exp" / "recovery_protocol_v11_w4a4_category.json")
        self.assertEqual(protocol["partitions"]["teacher_supervision"]["episodes_per_task"], 4)
        self.assertEqual(protocol["partitions"]["teacher_supervision"]["init_state_indices"], [20, 21, 22, 23])
        self.assertIs(protocol["data"]["w4a4"], True)
        self.assertEqual(protocol["data"]["version"], 11)
        heldout = protocol["partitions"]["heldout"]
        self.assertEqual(heldout["episodes_per_task"], 16)
        self.assertEqual(len(heldout["init_state_indices"]), 16)
        self.assertEqual(heldout["episodes_per_task"] * protocol["data"]["evaluation_contract"]["task_count"], 160)
        self.assertEqual(protocol["selection"]["same_budget_control"], "continued_qad")
        self.assertEqual(protocol["selection"]["qad_optimizer_steps"], 2000)
        self.assertEqual(protocol["selection"]["continuation_optimizer_steps"], 2000)


if __name__ == "__main__":
    unittest.main()
