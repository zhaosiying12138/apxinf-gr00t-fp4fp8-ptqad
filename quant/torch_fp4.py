"""NVFP4 numerical emulation with an identity straight-through gradient.

E2M1 and finite E4M3 use round-to-nearest, ties-to-even. A tensor has one
FP32 secondary scale (amax/448, matching fp4_quant.py) and one E4M3 scale
per 16 values. The returned tensors are ordinary floating-point values,
not packed storage or native low-precision execution.
"""
from __future__ import annotations
import math
import torch

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MIDS = (_E2M1[:-1] + _E2M1[1:]) / 2
# Positive finite codes 0x00..0x7e, including zero and excluding NaN 0x7f.
_E4M3 = torch.tensor(
    [m * 2.0 ** -9 for m in range(8)]
    + [(1 + m / 8.0) * 2.0 ** (e - 7)
       for e in range(1, 16) for m in range(8) if (e, m) != (15, 7)],
    dtype=torch.float64,
)


def _round_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Signed nearest value; grid indices are magnitude codes, so parity is RNE."""
    g = grid.to(device=x.device, dtype=torch.float64)
    mids = (g[:-1] + g[1:]) / 2
    a = x.double().abs().contiguous()
    index = torch.searchsorted(mids, a)
    # searchsorted selects the lower neighbor at a midpoint. Only odd lower
    # codes need to be advanced to choose an even code.
    at_mid = a == mids[index.clamp_max(len(mids) - 1)]
    index = index + (at_mid & ((index & 1) != 0)).long()
    out = torch.copysign(g[index], x.double()).to(x.dtype)
    return torch.where(torch.isnan(x), x, out)


@torch.no_grad()
def _quant_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to signed E2M1, saturating at +/-6 (all eight magnitude codes)."""
    return _round_grid(x, _E2M1)


@torch.no_grad()
def _quant_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to signed E4M3FN, including zero/subnormals; satfinite +/-448."""
    return _round_grid(x, _E4M3)


def _validate_nvfp4(W: torch.Tensor, block: int, clip: float = 1.0):
    if block != 16 or W.ndim != 2 or W.shape[1] % block or W.numel() == 0:
        raise ValueError("NVFP4 requires a nonempty 2D matrix with K divisible by block=16")
    if not math.isfinite(clip) or not 0 < clip <= 1:
        raise ValueError("clip must be finite and in (0, 1]")


@torch.no_grad()
def _nvfp4_tensor_scale(W: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """The single FP32 scale shared by every block, including chunked conversion."""
    peak = W.float().abs().amax().double() * clip
    # Keep the scale representable even for a tensor of FP32 subnormals.
    ts = (peak / 448.0).clamp_min(2.0 ** -149).float()
    return torch.where(peak > 0, ts, torch.ones_like(ts))


@torch.no_grad()
def _nvfp4_block_scales(amax: torch.Tensor, tscale: torch.Tensor) -> torch.Tensor:
    """Decoded E4M3 scales; a zero code remains zero in the physical contract."""
    return _quant_e4m3(amax.double() / 6.0 / tscale.double()).float()


@torch.no_grad()
def _nvfp4_dequant(W: torch.Tensor, block: int = 16, clip: float = 1.0,
                   tscale: torch.Tensor | float | None = None) -> torch.Tensor:
    _validate_nvfp4(W, block, clip)
    w = W.float().reshape(W.shape[0], -1, block)
    if tscale is None:
        ts = _nvfp4_tensor_scale(W, clip)
    else:
        ts = torch.as_tensor(tscale, dtype=torch.float32, device=W.device)
        if ts.numel() != 1 or not bool(torch.isfinite(ts).all() & (ts > 0).all()):
            raise ValueError("tscale must be one finite positive FP32 value")
    scales = _nvfp4_block_scales(w.abs().amax(-1).double() * clip, ts)
    # Code selection uses the exact product of the stored FP32 scale and
    # E4M3 value, as the NumPy encoder does. Decoding uses FP32 arithmetic.
    divisor = scales.double() * ts.double()
    safe_divisor = torch.where(divisor > 0, divisor, torch.ones_like(divisor))
    q = _quant_e2m1(w.double() / safe_divisor.unsqueeze(-1)).float()
    return (q * (scales * ts).unsqueeze(-1)).reshape(W.shape)


def fake_quant_nvfp4_torch(W: torch.Tensor, block: int = 16) -> torch.Tensor:
    """NVFP4-dequantized values in W.dtype, with exactly an identity STE to W."""
    dequant = _nvfp4_dequant(W, block=block).to(W.dtype)
    # Parenthesize the zero-valued gradient path: subtracting W from dequant
    # first would introduce cancellation error in the forward value.
    return dequant + (W - W.detach())


def install_global_linear_fakequant():
    """Patch nn.Linear.forward to NVFP4-emulate weights (smoke harness)."""
    import torch.nn as nn
    orig = nn.Linear.forward
    def patched(self, x):
        if self.weight.ndim == 2 and self.weight.shape[1] % 16 == 0 and self.weight.is_cuda:
            w = fake_quant_nvfp4_torch(self.weight)
            return nn.functional.linear(x, w, self.bias)
        return orig(self, x)
    nn.Linear.forward = patched
    return orig
