"""CPU-only checks for the preregistered v4 category development driver."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "category_development_driver", ROOT / "exp" / "run_category_development.py"
)
DRIVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DRIVER)


class CategoryDevelopmentProtocolTest(unittest.TestCase):
    def test_protocol_declares_v4_split_and_category_ladder(self):
        protocol = DRIVER.load_protocol(ROOT / "exp" / "recovery_protocol_v4_category_fp4.json")
        self.assertEqual(protocol["data"]["version"], 4)
        self.assertEqual(protocol["partitions"]["development"]["episodes_per_task"], 5)
        self.assertEqual(protocol["partitions"]["collection"]["episodes_per_task"], 4)
        self.assertEqual(protocol["partitions"]["heldout"]["episodes_per_task"], 10)
        self.assertEqual(protocol["selection"]["pressure_candidates"], list(DRIVER.CANDIDATES))
        self.assertEqual(protocol["category"]["active_category"], 2)
        self.assertEqual(protocol["category"]["categories"], 32)

    def test_pressure_selection_requires_drop_and_floor(self):
        pressure = {"min_drop_from_bf16": 0.20, "min_absolute_success": 0.30}
        audits = {
            "bf16": {"successes": 46, "episodes": 50},
            "head_lang_vision_category": {"successes": 36, "episodes": 50},
            "calib_category": {"successes": 35, "episodes": 50},
        }
        selected, qualifying = DRIVER.choose_pressure(audits, DRIVER.CANDIDATES, pressure)
        self.assertEqual(qualifying, list(DRIVER.CANDIDATES))
        self.assertEqual(selected, "calib_category")

        no_drop = dict(audits)
        no_drop["head_lang_vision_category"] = {"successes": 40, "episodes": 50}
        no_drop["calib_category"] = {"successes": 40, "episodes": 50}
        selected, qualifying = DRIVER.choose_pressure(no_drop, DRIVER.CANDIDATES, pressure)
        self.assertIsNone(selected)
        self.assertEqual(qualifying, [])

    def test_pressure_rejects_below_absolute_floor(self):
        pressure = {"min_drop_from_bf16": 0.20, "min_absolute_success": 0.30}
        audits = {
            "bf16": {"successes": 46, "episodes": 50},
            "head_lang_vision_category": {"successes": 14, "episodes": 50},
            "calib_category": {"successes": 12, "episodes": 50},
        }
        selected, qualifying = DRIVER.choose_pressure(audits, DRIVER.CANDIDATES, pressure)
        self.assertIsNone(selected)
        self.assertEqual(qualifying, [])


if __name__ == "__main__":
    unittest.main()
