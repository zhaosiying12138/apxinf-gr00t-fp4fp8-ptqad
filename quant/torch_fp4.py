"""GPU-side NVFP4 fake-quant (torch) — fast enough for per-forward emulation.

E2M1 grid via bucketize; E4M3 block scales via round-to-nearest on the fp8
log-lattice (matches fp4_quant.py bit-exact semantics within float tolerance);
STE gradient passthrough for the future QAD training path (smoke only needs fwd+bwd to run).

Run env: sister's Isaac-GR00T/.venv or any torch>=2 env.
"""
from __future__ import annotations
import torch

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

@torch.no_grad()
def _quant_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to E4M3 grid (positives; satfinite 448)."""
    grid, codes = [], []
    for m in range(1, 8):
        grid.append(m * 2.0 ** -9); codes.append(m)
    for e in range(1, 16):
        for m in range(8):
            grid.append((1 + m / 8.0) * 2.0 ** (e - 7)); codes.append((e << 3) | m)
    g = torch.tensor(grid, device=x.device, dtype=torch.float64)
    idx = torch.clamp(torch.searchsorted(g, x.double()), 1, len(g)) - 1
    lo, hi = g[idx], g[torch.clamp(idx + 1, max=len(g) - 1)]
    pick_hi = (x.double() - lo) > (hi - lo)
    out = torch.where(pick_hi, hi, lo)
    return torch.clamp(out, max=448.0).to(x.dtype)

def fake_quant_nvfp4_torch(W: torch.Tensor, block: int = 16) -> torch.Tensor:
    """W: (rows, K) on GPU. Returns NVFP4-emulated weights (float, STE-grad to W)."""
    rows, K = W.shape
    KB = K // block
    w = W.float().view(rows, KB, block)
    amax = w.abs().amax(dim=2)
    tscale = amax.max() / 448.0
    tscale = torch.where(tscale > 0, tscale, torch.ones_like(tscale))
    scale = _quant_e4m3(amax / 6.0 / tscale)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)) * tscale
    # E2M1 round-to-nearest with STE: derivative of rounding treated as 1
    with torch.no_grad():
        grid = _E2M1.to(W.device)
        mids = _E2M1_MIDS.to(W.device)
        normalized_abs = (w / scale.unsqueeze(2)).abs()
        idx = torch.clamp(torch.searchsorted(mids, normalized_abs), 0, 6)
        q = grid[idx] * (w / scale.unsqueeze(2)).sign()   # symmetric E2M1
        q = torch.clamp(q, -6.0, 6.0)
    # STE: out = q + (w/s - w/s).detach() pattern
    normalized = w / scale.unsqueeze(2)
    deq = q * scale.unsqueeze(2) + (normalized - normalized.detach()) * scale.unsqueeze(2) * 0  # grad path via scale? keep identity-free:
    # simpler STE: pass gradient straight through quantization
    deq = normalized + (q - normalized).detach()
    return (deq * scale.unsqueeze(2)).reshape(rows, K).to(W.dtype)

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
