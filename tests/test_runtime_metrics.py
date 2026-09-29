"""Metrics are observations, not device initialization or estimated GPU peaks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from runtime_metrics import cuda_memory_peaks, parameter_storage


class RuntimeMetricTests(unittest.TestCase):
    def test_cpu_snapshot_does_not_initialize_cuda(self):
        with patch.object(torch.cuda, "is_initialized", return_value=False), \
             patch.object(torch.cuda, "device_count", side_effect=AssertionError("must not query")):
            self.assertEqual(cuda_memory_peaks(), [])
        model = nn.Module()
        model.base = nn.Parameter(torch.zeros(3,4,dtype=torch.bfloat16),requires_grad=False)
        model.adapter = nn.Parameter(torch.zeros(2,4,dtype=torch.float32))
        self.assertEqual(parameter_storage(model),{
            "bfloat16":{"tensors":1,"parameters":12,"bytes":24,"trainable_parameters":0},
            "float32":{"tensors":1,"parameters":8,"bytes":32,"trainable_parameters":8}})

    def test_peaks_report_direct_allocator_values(self):
        with patch.object(torch.cuda,"is_initialized",return_value=True), \
             patch.object(torch.cuda,"device_count",return_value=1), \
             patch.object(torch.cuda,"memory_stats",return_value={"measured":1}), \
             patch.object(torch.cuda,"get_device_name",return_value="test-device"), \
             patch.object(torch.cuda,"max_memory_allocated",return_value=123), \
             patch.object(torch.cuda,"max_memory_reserved",return_value=456), \
             patch.object(torch.cuda,"memory_allocated",return_value=12), \
             patch.object(torch.cuda,"memory_reserved",return_value=45):
            actual = cuda_memory_peaks()[0]
            self.assertEqual(actual["max_memory_allocated_bytes"],123)
            self.assertEqual(actual["max_memory_reserved_bytes"],456)
            self.assertEqual(actual["memory_allocated_bytes_at_end"],12)


if __name__ == "__main__":
    unittest.main()
