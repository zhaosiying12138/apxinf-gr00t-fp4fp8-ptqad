"""NVFP4 / FP8-E4M3 weight quantizers and calibrated GPTQ.

NVFP4 uses the same RNE grid and single FP32 tensor scale as fp4_quant.py.
Returned values are dequantized FP32 weights, not packed artifacts. GPTQ
updates E4M3 block scales after error compensation but keeps the tensor scale
fixed, so every output block belongs to one representable NVFP4 tensor.
"""
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_fp4 import (
    _quant_e2m1, _quant_e4m3, _nvfp4_dequant, _nvfp4_tensor_scale,
    _nvfp4_block_scales, _validate_nvfp4,
)


@torch.no_grad()
def nvfp4_dequant(W, block=16, clip=1.0, tscale=None):
    """(N,K) -> NVFP4-dequantized FP32 values; optional shared chunk scale."""
    return _nvfp4_dequant(W, block=block, clip=clip, tscale=tscale)


@torch.no_grad()
def fp8_e4m3_dequant(W):
    """Weight-only FP8 E4M3 with FP32 per-output-channel (row) scales."""
    w = W.float()
    peak = w.abs().amax(dim=1, keepdim=True)
    s = (peak.double() / 448.0).clamp_min(2.0 ** -149).float()
    s = torch.where(peak > 0, s, torch.ones_like(s))
    return _quant_e4m3(w.double() / s.double()).float() * s


@torch.no_grad()
def layer_mse_tr(dW, H):
    """sum_i ||x_i dW^T||^2 = tr(dW H dW^T); H = sum x x^T (f32)."""
    return torch.einsum("nk,kl,nl->", dW.float(), H, dW.float())


def rtnc_best_clip(W, H, grid=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)):
    """Per-layer clip search using RTN candidates, before GPTQ compensation."""
    best, best_err = 1.0, None
    for c in grid:
        err = layer_mse_tr(nvfp4_dequant(W, clip=c) - W.float(), H).item()
        if best_err is None or err < best_err:
            best, best_err = c, err
    return best, best_err


@torch.no_grad()
def gptq_nvfp4(W, H, block=16, clip=1.0, damp=0.01, return_metadata=False):
    """GPTQ with adaptive per-row block scales and one fixed tensor scale.

    The tensor scale is chosen from the original tensor (including clip),
    before dead-channel handling or compensation. At each 16-column boundary,
    a new E4M3 scale is chosen using the compensated block's maximum. Overflow
    saturates at the E4M3 rail; a later block never changes the shared scale.

    Default return: dequantized (N,K) FP32. With return_metadata=True, return
    (weights, {tscale, block_scales}); block_scales are decoded E4M3 values,
    sufficient to verify representability without re-estimating scales.
    """
    _validate_nvfp4(W, block, clip)
    ts = _nvfp4_tensor_scale(W, clip)
    W = W.float().clone()
    N, K = W.shape
    if H.shape != (K, K):
        raise ValueError("H must have shape (K,K) matching the weight input dimension")
    H = H.to(device=W.device, dtype=torch.float32).clone()
    dead = torch.diag(H) == 0
    if dead.any():
        W[:, dead] = 0
        H[dead, dead] = 1.0
    H += damp * H.diagonal().mean() * torch.eye(K, device=H.device, dtype=H.dtype)
    L = torch.linalg.cholesky(H)
    Hi = torch.cholesky_inverse(L)
    Hi = torch.linalg.cholesky(Hi, upper=True)
    stored_scales = torch.empty((N, K // block), device=W.device) if return_metadata else None
    for i in range(K):
        if i % block == 0:
            amax = W[:, i:i + block].abs().amax(1).double() * clip
            scales = _nvfp4_block_scales(amax, ts)
            if stored_scales is not None:
                stored_scales[:, i // block] = scales
            exact_scale = scales.double() * ts.double()
            divisor = torch.where(exact_scale > 0, exact_scale, torch.ones_like(exact_scale))
            s_row = scales * ts
        q = _quant_e2m1(W[:, i].double() / divisor).float() * s_row
        d = Hi[i, i]
        err = (W[:, i] - q) / d
        if i + 1 < K:
            W[:, i + 1:] -= err.unsqueeze(1) * Hi[i, i + 1:].unsqueeze(0)
        W[:, i] = q
    if return_metadata:
        return W, {"tscale": ts, "block_scales": stored_scales}
    return W
