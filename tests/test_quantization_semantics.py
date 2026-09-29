"""CPU regression tests against byte encoders and explicit matrix outputs.

Run: python -m unittest discover -s tests -p test_quantization_semantics.py -v
No CUDA context is created. Torch's native FP8 conversion provides a second
oracle independent of the repository's NumPy grid implementation.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quant"))
sys.path.insert(0, str(ROOT / "quant" / "ptq"))
from fp4_quant import (encode_e2m1, decode_e2m1, encode_e4m3, decode_e4m3,
                       nvfp4_quantize, nvfp4_dequantize)
from torch_fp4 import _quant_e2m1, _quant_e4m3, fake_quant_nvfp4_torch
from quantizers import nvfp4_dequant, fp8_e4m3_dequant, gptq_nvfp4, layer_mse_tr
from awq import search_site


class QuantizationSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_e2m1_all_codes_midpoints_and_neighbors(self):
        grid = decode_e2m1(np.arange(8, dtype=np.int8))
        mids = (grid[:-1] + grid[1:]) / 2
        positive = np.concatenate([grid, mids, np.nextafter(mids, -np.inf),
                                   np.nextafter(mids, np.inf), [100.0]])
        values = np.concatenate([positive, -positive]).astype(np.float64)
        expected = decode_e2m1(encode_e2m1(values))
        actual = _quant_e2m1(torch.from_numpy(values)).numpy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(_quant_e2m1(torch.from_numpy(mids)).numpy(),
                                      [0, 1, 1, 2, 2, 4, 4])
        self.assertEqual(actual[7], 6.0)
        self.assertTrue(np.signbit(_quant_e2m1(torch.tensor(-0.0)).item()))

    def test_e4m3_all_codes_midpoints_native_fp8_oracle(self):
        grid = decode_e4m3(np.arange(127, dtype=np.uint8))
        mids = (grid[:-1] + grid[1:]) / 2
        positive = np.concatenate([grid, mids, np.nextafter(mids, -np.inf),
                                   np.nextafter(mids, np.inf), [1e-8, 464, 1e5]])
        values = np.concatenate([positive, -positive]).astype(np.float32)
        # Clamp implements satfinite; unclamped native cast deliberately emits
        # NaN beyond its overflow boundary rather than saturating.
        native = torch.from_numpy(values).clamp(-448, 448).to(torch.float8_e4m3fn).float().numpy()
        reference = decode_e4m3(encode_e4m3(values))
        actual = _quant_e4m3(torch.from_numpy(values)).numpy()
        np.testing.assert_array_equal(reference, native)
        np.testing.assert_array_equal(actual, native)
        np.testing.assert_array_equal(_quant_e4m3(torch.tensor([0., 1., 1.125, 448.])).numpy(),
                                      [0, 1, 1.125, 448])
        self.assertEqual(encode_e4m3(np.array([2.0**-10]))[0], 0)

    def test_rtn_packed_reference_and_matrix_outputs(self):
        rng = np.random.default_rng(20260929)
        for shape in [(3, 16), (64, 128), (8, 64)]:
            W = rng.normal(0, .05, shape).astype(np.float32)
            W[0] = 0
            if shape[1] > 16:
                W[1, :16] *= 1e-7  # include block-scale underflow
            packed, scales, ts = nvfp4_quantize(W)
            reference = nvfp4_dequantize(packed, scales, ts)
            ptq = nvfp4_dequant(torch.from_numpy(W)).numpy()
            fake = fake_quant_nvfp4_torch(torch.from_numpy(W)).numpy()
            np.testing.assert_array_equal(ptq, reference)
            np.testing.assert_array_equal(fake, reference)
            X = rng.normal(size=(11, shape[1])).astype(np.float32)
            np.testing.assert_allclose(torch.from_numpy(X).mm(torch.from_numpy(ptq).T).numpy(),
                                       X @ reference.T, atol=1e-6, rtol=1e-5)
            # A chunk may use the tensor scale, never estimate a private one.
            chunked = torch.cat([nvfp4_dequant(torch.from_numpy(w), tscale=float(ts))
                                 for w in np.array_split(W, 2)])
            np.testing.assert_array_equal(chunked.numpy(), reference)

    def test_zero_tiny_weights_and_zero_physical_scale(self):
        for value in [0.0, np.nextafter(np.float32(0), np.float32(1))]:
            W = np.full((2, 32), value, dtype=np.float32)
            ref = nvfp4_dequantize(*nvfp4_quantize(W))
            np.testing.assert_array_equal(nvfp4_dequant(torch.from_numpy(W)).numpy(), ref)
            self.assertTrue(torch.isfinite(fp8_e4m3_dequant(torch.from_numpy(W))).all())
        # A physical zero scale must zero its block, regardless of data bits.
        decoded = nvfp4_dequantize(np.full((1, 8), 0x77, np.uint8),
                                  np.zeros((1, 1), np.uint8), np.float32(2))
        np.testing.assert_array_equal(decoded, np.zeros((1, 16)))
        zero = torch.zeros(2, 32)
        self.assertTrue(torch.equal(gptq_nvfp4(zero, torch.eye(32)), zero))

    def test_ste_value_and_identity_gradient(self):
        W = torch.linspace(-3, 3, 64).reshape(2, 32).requires_grad_()
        loss_weights = torch.linspace(.1, 1, 64).reshape(2, 32)
        out = fake_quant_nvfp4_torch(W)
        torch.testing.assert_close(out, nvfp4_dequant(W), rtol=0, atol=0)
        (out * loss_weights).sum().backward()
        torch.testing.assert_close(W.grad, loss_weights, rtol=0, atol=0)
        bf16 = W.detach().bfloat16()
        self.assertEqual(fake_quant_nvfp4_torch(bf16).dtype, torch.bfloat16)

    def test_fp8_row_scale_reference(self):
        rng = np.random.default_rng(19)
        W = rng.normal(size=(5, 32)).astype(np.float32)
        W[0] = 0
        scale = (np.abs(W).max(axis=1, keepdims=True).astype(np.float64) / 448).astype(np.float32)
        scale[scale == 0] = 1
        ref = decode_e4m3(encode_e4m3(W.astype(np.float64) / scale)) * scale
        np.testing.assert_array_equal(fp8_e4m3_dequant(torch.from_numpy(W)).numpy(), ref)

    def test_gptq_diagonal_equals_rtn_one_tensor_scale(self):
        torch.manual_seed(8)
        W = torch.randn(5, 48)
        W[:, :16] *= 0.02
        W[:, 32:] *= 20
        for clip in [1.0, 0.8]:
            result, meta = gptq_nvfp4(W, torch.eye(48), clip=clip, return_metadata=True)
            torch.testing.assert_close(result, nvfp4_dequant(W, clip=clip), rtol=0, atol=0)
            expected_ts = np.float32(W.abs().max().item() * clip / 448)
            self.assertEqual(meta["tscale"].item(), float(expected_ts))

    def test_gptq_representability_and_matrix_objective(self):
        torch.manual_seed(37)
        X = torch.randn(96, 32)
        X[:, 16:] += .9 * X[:, :16]  # compensation must cross a block boundary
        W = torch.randn(7, 32) * torch.linspace(.1, 3, 32)
        H = X.T @ X
        result, meta = gptq_nvfp4(W, H, return_metadata=True)
        scales = meta["block_scales"].numpy()
        codes = encode_e4m3(scales)
        np.testing.assert_array_equal(decode_e4m3(codes), scales)
        effective = scales * np.float32(meta["tscale"].item())
        q = encode_e2m1(result.numpy().reshape(7, 2, 16) / effective[:, :, None])
        packed = ((q[:, :, 0::2] & 15) | ((q[:, :, 1::2] & 15) << 4)).astype(np.uint8).reshape(7, 16)
        decoded = nvfp4_dequantize(packed, codes, np.float32(meta["tscale"].item()))
        np.testing.assert_array_equal(result.numpy(), decoded)
        error = ((X @ result.T) - (X @ W.T)).square().sum()
        torch.testing.assert_close(layer_mse_tr(result-W, H), error, rtol=2e-6, atol=1e-5)
        self.assertLess(error, ((X @ nvfp4_dequant(W).T) - (X @ W.T)).square().sum())

    def test_awq_norm_objective_equals_explicit_outputs(self):
        torch.manual_seed(4)
        W = torch.randn(6, 32)
        X = torch.randn(20, 32) * torch.linspace(.1, 4, 32)
        calib = {"w": {"H": X.T @ X, "abs": X.abs().sum(0), "n": len(X)}}
        result = search_site({"kind": "norm", "members": ["w"]}, lambda _: W,
                             calib, alphas=(.5,))
        s = result["s"]
        explicit = ((X*s) @ nvfp4_dequant(W/s).T - X @ W.T).square().sum().item()
        self.assertAlmostEqual(result["err"], explicit, delta=explicit*2e-6)
        # Zero observed channels must not create non-invertible folds.
        calib["w"]["abs"].zero_()
        zero_result = search_site({"kind": "norm", "members": ["w"]}, lambda _: W,
                                  calib, alphas=(1.0,))
        self.assertTrue(torch.isfinite(zero_result["s"]).all())
        self.assertTrue((zero_result["s"] > 0).all())

    def test_awq_gqa_mapping_and_producer_consumer_objective(self):
        torch.manual_seed(5)
        Wp, Wc = torch.randn(16, 16), torch.randn(6, 32)
        Xp, Xc = torch.randn(20, 16), torch.randn(20, 32)
        activations = torch.arange(1., 33.)
        weights = {"p": Wp, "c": Wc}
        calib = {"p": {"H": Xp.T @ Xp},
                 "c": {"H": Xc.T @ Xc, "abs": activations, "n": 1}}
        site = {"kind": "producer_cols", "producer": "p", "consumer": "c",
                "gqa_groups": 2, "head_dim": 8}
        result = search_site(site, weights.__getitem__, calib, alphas=(1.0,))
        expected = torch.cat([activations[8:16], activations[24:32]])
        expected /= expected.mean()
        torch.testing.assert_close(result["s"], expected)
        s = result["s"]
        sb = s.reshape(2, 8).repeat_interleave(2, dim=0).flatten()
        scaled_p = Wp * s[:, None]
        ep = (Xp @ (nvfp4_dequant(scaled_p)-scaled_p).T).square().sum()
        ec = ((Xc*sb) @ nvfp4_dequant(Wc/sb).T-Xc @ Wc.T).square().sum()
        self.assertAlmostEqual(result["err"], (ep+ec).item(), delta=(ep+ec).item()*2e-6)


if __name__ == "__main__":
    unittest.main()
