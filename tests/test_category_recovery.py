"""CPU contracts for the category-aware recovery boundary."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import run_category_recovery as category


class CategoryRecoveryTest(unittest.TestCase):
    def test_frozen_v4_protocol_normalizes_v3_runner_contract(self):
        protocol = category.category_protocol(ROOT / "exp/recovery_protocol_v4_category_fp4.json")
        self.assertEqual(protocol["data"]["version"], 4)
        self.assertEqual(protocol["partitions"]["collection"]["episodes_per_task"], 4)
        self.assertEqual(protocol["selection"]["pressure_candidates"], list(category.CANDIDATES))

    def test_v3_protocol_is_rejected_at_category_boundary(self):
        with self.assertRaises(Exception):
            category.category_protocol(ROOT / "exp/recovery_protocol_v3_high_fp4.json")


if __name__ == "__main__":
    unittest.main()
