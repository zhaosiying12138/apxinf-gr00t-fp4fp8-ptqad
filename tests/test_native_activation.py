"""CPU arithmetic-contract tests; no GPU/native-parity claim or CUDA calls.

Expected boundary codes are literal; Torch FP8 and a scalar nearest-code
oracle independently check the scale and data arithmetic over broader input.
"""
import struct
import unittest

import numpy as np
import torch

from quant.native_activation import encode_native_activation, native_activation_qdq


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class NativeActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_e2m1_midpoints_signs_and_literal_packed_bytes(self):
        x = torch.tensor([[6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5,
                           -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5, -0.]])
        encoded = encode_native_activation(x)
        self.assertEqual(encoded.scales.tolist(), [[0x38]])  # E4M3 1.0
        self.assertEqual(encoded.packed.tolist(), [[0x07, 0x22, 0x44, 0x66,
                                                   0xA8, 0xCA, 0xEC, 0x0E]])
        expected = torch.tensor([[6, 0, 1, 1, 2, 2, 4, 4, -0., -1, -1, -2, -2, -4, -4, 0.]])
        torch.testing.assert_close(encoded.dequantized, expected, rtol=0, atol=0)
        self.assertTrue(torch.signbit(encoded.dequantized[0, 8]))
        self.assertFalse(torch.signbit(encoded.dequantized[0, 15]))

    def test_e2m1_half_neighbors_choose_opposite_sides(self):
        mids = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float16)
        values = np.concatenate([np.nextafter(mids, np.float16(-np.inf)),
                                 np.nextafter(mids, np.float16(np.inf)),
                                 np.array([6, 0], dtype=np.float16)])
        actual = encode_native_activation(torch.from_numpy(values)).dequantized
        torch.testing.assert_close(actual, torch.tensor([0, .5, 1, 1.5, 2, 3, 4,
                                                        .5, 1, 1.5, 2, 3, 4, 6, 6, 0]),
                                   rtol=0, atol=0)

    def test_scale_midpoints_and_subnormal_zero(self):
        maxima = torch.tensor([6.375, 7.125, 6 * 2.0**-10, 18 * 2.0**-10])
        x = maxima[:, None].expand(-1, 16)
        encoded = encode_native_activation(x)
        self.assertEqual(encoded.scales.flatten().tolist(), [0x38, 0x3A, 0, 2])
        self.assertEqual(encoded.zero_scale_blocks, 1)
        self.assertTrue(torch.equal(encoded.dequantized[2], torch.zeros(16)))

    def test_scale_oracle_for_every_positive_finite_half(self):
        values = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
        maxima = torch.from_numpy(values)
        x = torch.zeros(len(values), 16)
        x[:, 0] = maxima
        encoded = encode_native_activation(x)
        requested = maxima * torch.tensor(1.0 / 6.0, dtype=torch.float32)
        oracle = requested.clamp(max=448).to(torch.float8_e4m3fn).view(torch.uint8)
        torch.testing.assert_close(encoded.scales[:, 0], oracle, rtol=0, atol=0)

    def test_input_is_cast_to_f16_before_code_selection(self):
        x = torch.zeros(1, 16, dtype=torch.float32)
        x[0, 0], x[0, 1] = 6, .25001
        self.assertGreater(x[0, 1].item(), .25)
        encoded = encode_native_activation(x)
        self.assertEqual(encoded.dequantized[0, 1].item(), 0)
        self.assertEqual(encoded.packed[0, 0].item(), 0x07)

    def test_fixed_scale_saturation_and_tiny_values(self):
        x = torch.zeros(3, 16)
        x[0, :2] = torch.tensor([65504., -65504.])
        x[1] = 2.0**-24
        x[2, 0] = 6
        encoded = encode_native_activation(x)
        self.assertEqual(encoded.tensor_scale, 1.0)
        self.assertEqual(encoded.scales.flatten().tolist(), [0x7E, 0, 0x38])
        self.assertEqual(encoded.dequantized[0, :2].tolist(), [2688., -2688.])
        self.assertTrue(torch.equal(encoded.dequantized[1], torch.zeros(16)))
        self.assertEqual(encoded.saturated_values, 2)
        self.assertEqual(encoded.dequantized[2, 0].item(), 6)

    def test_padding_zero_and_noncontiguous_leading_shape(self):
        x = torch.full((1, 48, 3), 6.).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        encoded = encode_native_activation(x)
        self.assertEqual(tuple(encoded.dequantized.shape), (1, 3, 48))
        self.assertEqual(tuple(encoded.packed.shape), (3, 24))
        expected = torch.zeros(512, dtype=torch.uint8)
        expected[[0, 1, 2, 16, 17, 18, 32, 33, 34]] = 0x38
        torch.testing.assert_close(encoded.swizzled_scales, expected, rtol=0, atol=0)

    def test_scale_swizzle_crosses_row_and_k_tile_boundaries(self):
        encoded = encode_native_activation(torch.full((129, 80), 6.))
        # 2 row tiles * 2 K panels * 512 bytes; last row's last block.
        self.assertEqual(encoded.swizzled_scales.numel(), 2048)
        for offset in (0, 127, 496, 512, 1024, 1536):
            self.assertEqual(encoded.swizzled_scales[offset].item(), 0x38)
        self.assertEqual(int((encoded.swizzled_scales != 0).sum()), 129 * 5)

    def test_scalar_oracle_for_data_division_and_nearest_even(self):
        generator = torch.Generator().manual_seed(801)
        x = torch.randn(2, 3, 48, generator=generator) * .47
        actual = encode_native_activation(x).dequantized
        half_values = x.half().float().reshape(-1, 16)
        grid = (0., .5, 1., 1.5, 2., 3., 4., 6.)
        expected = []
        for block in half_values.tolist():
            requested = f32(max(abs(v) for v in block) * f32(1 / 6))
            scale = torch.tensor(min(requested, 448.)).to(torch.float8_e4m3fn).float().item()
            divisor = scale if scale > 0 else 1.
            for value in block:
                normalized = abs(f32(value / divisor))
                code = min(range(8), key=lambda k: (abs(normalized - grid[k]), k % 2))
                signed = -grid[code] if value < 0 else grid[code]
                expected.append(f32(signed * scale))
        torch.testing.assert_close(actual, torch.tensor(expected).reshape_as(x), rtol=0, atol=0)

    def test_qdq_dtype_ste_and_original_residual_input(self):
        x = torch.zeros(1, 16)
        x[0, :2] = torch.tensor([6., .25])
        x.requires_grad_()
        original = x.detach().clone()
        q = native_activation_qdq(x, ste=True)
        weights = torch.arange(16.).reshape_as(x)
        (q * weights).sum().backward()
        torch.testing.assert_close(x.grad, weights, rtol=0, atol=0)
        torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)
        self.assertFalse(native_activation_qdq(x).requires_grad)
        self.assertEqual(native_activation_qdq(x.bfloat16()).dtype, torch.bfloat16)
        # A branch selecting column 1 sees .25 from x but zero from qdq(x).
        a = torch.zeros(1, 16)
        a[0, 1] = 1
        self.assertEqual(torch.nn.functional.linear(x, a).item(), .25)
        self.assertEqual(torch.nn.functional.linear(q, a).item(), 0)

    def test_rejects_nonfinite_input_f16_overflow_and_invalid_shapes(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonfinite'):
                encode_native_activation(torch.full((1, 16), value))
        for value in (1e6, -1e6):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'overflows F16'):
                encode_native_activation(torch.full((1, 16), value))
        for shape in ((), (0, 16), (1, 15)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'final K'):
                encode_native_activation(torch.zeros(shape))
        with self.assertRaises(TypeError):
            encode_native_activation(torch.zeros(1, 16, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, 'CPU-only'):
            encode_native_activation(torch.empty(1, 16, device='meta'))


if __name__ == '__main__':
    unittest.main()
