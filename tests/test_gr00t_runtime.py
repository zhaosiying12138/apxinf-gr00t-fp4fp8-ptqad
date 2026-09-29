"""CPU regression checks for checkpoint data/normalization contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from gr00t_runtime import configure_libero_data, verify_libero_statistics, libero_action_spec, action_mask


class RuntimeTests(unittest.TestCase):
    def test_scope_keeps_checkpoint_horizon_statistics_mapping_and_valid_actions(self):
        action = {"delta_indices": list(range(16)), "modality_keys": ["xyz", "rotation", "gripper"]}
        modalities = {"libero_sim": {"action": action},
                      "other_saved": {"action": {"delta_indices": list(range(40))}}}
        stats = {"libero_sim": {"action": {key: {"mean": [0]*dim}
                  for key,dim in (("xyz",3),("rotation",3),("gripper",1))}}, "other_saved": {}}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "processor_config.json").write_text(json.dumps({"processor_kwargs": {"modality_configs": modalities}}))
            (base / "statistics.json").write_text(json.dumps(stats))
            (base / "embodiment_id.json").write_text(json.dumps({"libero_sim": 9, "other_saved": 2}))
            config = SimpleNamespace(model=SimpleNamespace(action_horizon=40, model_name="unused"),
                                     data=SimpleNamespace(modality_configs={"unitree": "invalid"}, override_pretraining_statistics=True))
            utility = ModuleType("gr00t.data.utils")
            utility.parse_modality_configs = deepcopy
            with patch.dict(sys.modules, {utility.__name__: utility}):
                configure_libero_data(config, base)
            self.assertEqual(list(config.data.modality_configs), ["libero_sim"])
            self.assertEqual(config.model.action_horizon, 40)
            self.assertFalse(config.data.override_pretraining_statistics)
            processor = SimpleNamespace(statistics={}, state_action_processor=SimpleNamespace(statistics=stats),
                                        embodiment_id_mapping={"libero_sim":9, "other_saved":2, "unused_new":3})
            verify_libero_statistics(processor, base)
            self.assertEqual(processor.statistics, stats)
            self.assertEqual(processor.embodiment_id_mapping, {"libero_sim":9, "other_saved":2})
            spec = libero_action_spec(base)
            self.assertEqual((spec["horizon"],spec["dimensions"]),(16,7))
            mask = action_mask(torch.zeros(1,40,132), spec)
            self.assertEqual(mask.sum().item(),112)
            with self.assertRaises(ValueError):
                action_mask(torch.zeros(1,8,132),spec)


if __name__ == "__main__":
    unittest.main()
