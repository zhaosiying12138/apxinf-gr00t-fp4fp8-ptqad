"""NVFP4 / FP8-E4M3 weight quantizers + GPTQ (Hessian rounding) for GR00T PTQ.

NVFP4 semantics mirror quant/torch_fp4.py (proven equivalent to the APXInf
engine artifact path, gold_check4 corr=1.0): per-16-col block, E2M1 values on
a per-block E4M3 scale, per-tensor secondary scale tscale=amax/448.
'clip' (<1) is the learned-clipping knob: the scale uses amax*clip so the
block max maps above the E2M1 rail 6 and gets clamped (Choi et al. 2018).
"""
import torch, sys
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
from torch_fp4 import _quant_e4m3, _E2M1, _E2M1_MIDS

@torch.no_grad()
def nvfp4_dequant(W, block=16, clip=1.0):
    """(N,K) -> NVFP4-dequantized values (float32)."""
    N, K = W.shape
    w = W.float().view(N, K // block, block)
    amax = w.abs().amax(-1) * clip
    ts = (amax.max() / 448.0).clamp_min(1e-30)
    s = _quant_e4m3(amax / 6.0 / ts)
    s = torch.where(s > 0, s, torch.ones_like(s)) * ts
    x = w / s.unsqueeze(-1)
    grid = _E2M1.to(W.device); mids = _E2M1_MIDS.to(W.device)
    idx = torch.clamp(torch.searchsorted(mids, x.abs()), 0, 6)
    q = grid[idx] * torch.sign(x)
    return (q * s.unsqueeze(-1)).reshape(N, K)

@torch.no_grad()
def fp8_e4m3_dequant(W):
    """Weight-only FP8 E4M3 with per-output-channel (row) scales."""
    s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    q = _quant_e4m3((W / s).abs()) * torch.sign(W / s)
    return q * s

@torch.no_grad()
def layer_mse_tr(dW, H):
    """sum_i ||x_i dW^T||^2 = tr(dW H dW^T); H = sum x x^T (f32)."""
    return torch.einsum("nk,kl,nl->", dW.float(), H, dW.float())

def rtnc_best_clip(W, H, grid=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)):
    """Per-layer learned clip search for RTN (max vs MSE calibration)."""
    best, best_err = 1.0, None
    for c in grid:
        err = layer_mse_tr(nvfp4_dequant(W, clip=c) - W.float(), H).item()
        if best_err is None or err < best_err:
            best, best_err = c, err
    return best, best_err

@torch.no_grad()
def gptq_nvfp4(W, H, block=16, clip=1.0, damp=0.01):
    """GPTQ (Frantar et al. 2023) adapted to NVFP4 per-16-block scales.

    Columns processed left->right; the per-row block scale is recomputed from
    the CURRENT (error-compensated) block values each time a new block starts.
    Returns dequantized (N,K) f32.
    """
    W = W.float().clone()
    N, K = W.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    if dead.any():
        W[:, dead] = 0
        H[dead, dead] = 1.0
    H += damp * H.diagonal().mean() * torch.eye(K, device=H.device, dtype=H.dtype)
    L = torch.linalg.cholesky(H)
    Hi = torch.cholesky_inverse(L)
    Hi = torch.linalg.cholesky(Hi, upper=True)
    grid = _E2M1.to(W.device); mids = _E2M1_MIDS.to(W.device)
    s_row = None
    for i in range(K):
        if i % block == 0:
            blk = W[:, i:i + block]
            amax = blk.abs().amax(1) * clip
            ts = (amax.max() / 448.0).clamp_min(1e-30)
            s_row = _quant_e4m3(amax / 6.0 / ts)
            s_row = torch.where(s_row > 0, s_row, torch.ones_like(s_row)) * ts
        x = W[:, i] / s_row
        idx = torch.clamp(torch.searchsorted(mids, x.abs()), 0, 6)
        q = grid[idx] * torch.sign(x) * s_row
        d = Hi[i, i]
        err = (W[:, i] - q) / d
        if i + 1 < K:
            W[:, i + 1:] -= err.unsqueeze(1) * Hi[i, i + 1:].unsqueeze(0)
        W[:, i] = q
    return W
