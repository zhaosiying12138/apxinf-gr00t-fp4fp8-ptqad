"""PyTorch NVFP4 fake-quant (bit-exact mirror of quant/fp4_quant.py + spike).

Used for:
  - E0 parity: fake-quant linear layers vs engine NVFP4 kernels
  - L1 recovery: learnable per-block scales (STE)
  - L2 recovery: PPO fine-tuning with quantization simulated in the loop

No torch at import time in fp4_quant.py; this module imports torch lazily so the
numpy core stays dependency-free.
"""
from __future__ import annotations
import numpy as np
import torch
from fp4_quant import encode_e2m1, decode_e2m1, encode_e4m3, decode_e4m3

def _np(x):  return x.detach().cpu().numpy()
def _th(x, like): return torch.as_tensor(x, dtype=like.dtype, device=like.device)

def fake_quant_nvfp4(W: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Return dequantized NVFP4 weights (per-16 E4M3 scales, per-tensor FP32)."""
    rows, K = W.shape
    KB = K // block
    w = W.detach().to(torch.float64)
    wblk = w.view(rows, KB, block)
    amax = wblk.abs().amax(dim=2)
    tscale = torch.tensor(float(np.float32((_np(amax).max()) / 448.0)) or 1.0)
    scale_f = _th(_np(amax) / 6.0 / float(tscale), w)
    scales = torch.as_tensor(encode_e4m3(_np(scale_f)), device=W.device)
    sd = torch.as_tensor(decode_e4m3(_np(scales)), device=W.device, dtype=torch.float64) * float(tscale)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    q = torch.as_tensor(encode_e2m1(_np(wblk / sd.unsqueeze(2))), device=W.device, dtype=torch.int8)
    deq = torch.as_tensor(decode_e2m1(_np(q)), device=W.device, dtype=torch.float64)
    return (deq * sd.unsqueeze(2)).view(rows, K).to(W.dtype)

class NVFP4Linear(torch.nn.Module):
    """Drop-in Linear with NVFP4-simulated weights (activations stay BF16 here;
    W4A4 activation path is exercised by the engine, not this module)."""
    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = lin.weight.data.clone()   # kept BF16 master copy
        self.bias = lin.bias.data.clone() if lin.bias is not None else None

    def forward(self, x):
        wq = fake_quant_nvfp4(self.weight.to(torch.float32)).to(x.dtype)
        return torch.nn.functional.linear(x, wq, self.bias)

def swap_linear_nvfp4(module: torch.nn.Module, filter=None) -> int:
    """Recursively replace nn.Linear by NVFP4Linear (optionally filtered by name)."""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and (filter is None or filter(name)):
            setattr(module, name, NVFP4Linear(child)); n += 1
        else:
            n += swap_linear_nvfp4(child, filter)
    return n

if __name__ == "__main__":  # quick self-test vs numpy core
    W = torch.randn(64, 128, dtype=torch.float32) * 0.05
    from fp4_quant import nvfp4_quantize, nvfp4_dequantize
    packed, scales, ts = nvfp4_quantize(_np(W))
    ref = torch.as_tensor(nvfp4_dequantize(packed, scales, ts), dtype=torch.float32)
    fq = fake_quant_nvfp4(W)
    diff = (fq - ref).abs().max().item()
    print(f"fake_quant vs numpy core: max|diff|={diff:.2e} (expect 0)")
    lin = torch.nn.Linear(128, 64)
    x = torch.randn(4, 128)
    y_ref = torch.nn.functional.linear(x, fake_quant_nvfp4(lin.weight.data), lin.bias)
    nl = NVFP4Linear(lin); y_t = nl(x)
    print("linear path max|diff|:", (y_ref - y_t).abs().max().item())
