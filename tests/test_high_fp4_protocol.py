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
    def test_v9_exposes_teacher_supervision_partition(self):
        protocol = DRIVER.load_protocol(ROOT / "exp" / "recovery_protocol_v9_w4a4.json")
        self.assertEqual(protocol["partitions"]["teacher_supervision"]["episodes_per_task"], 4)
        self.assertEqual(protocol["partitions"]["teacher_supervision"]["init_state_indices"], [20, 21, 22, 23])


if __name__ == "__main__":
    unittest.main()
