"""Scoped NVFP4 quantize-once: REPLACE in-scope Linear weights with their
NVFP4-dequantized values at model-load time (bf16 dtype). Forward then runs
at native BF16 speed while weights carry exactly the quantization noise a
real fp4 engine would consume. Numerically identical to per-forward
fake-quant, but ~50x faster in closed-loop rollouts.

Modes (FP4VLA_SCOPE): all | head | backbone
  all      - every eligible nn.Linear
  head     - action_head.* only (GR00T) / gemma_expert (pi05 naming)
  backbone - everything except the action head
"""
import torch
import torch.nn as nn

_ORIG = nn.Linear.forward
_MODE = "all"


def install_scoped_fakequant(mode: str):
    global _MODE
    _MODE = mode
    print(f"[fp4vla] scoped quantize-once armed: mode={mode}", flush=True)


def mark_scope(model):
    """Quantize-once: rewrite in-scope Linear weights in place."""
    import sys
    sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
    from torch_fp4 import fake_quant_nvfp4_torch
    mode = _MODE
    n_in = n_out = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                headish = ("action_head" in name) or ("gemma_expert" in name)
                if mode == "all":
                    inscope = True
                elif mode == "head":
                    inscope = headish
                elif mode == "backbone":
                    inscope = not headish
                else:
                    inscope = True
                w = mod.weight
                if inscope and w.ndim == 2 and w.shape[1] % 16 == 0 and w.is_cuda:
                    mod.weight.data = fake_quant_nvfp4_torch(w).to(w.dtype)
                    n_in += 1
                else:
                    n_out += 1
    nn.Linear.forward = _ORIG  # native speed; weights already quantized
    print(f"[fp4vla] scope={mode}: {n_in} weights QUANTIZED-ONCE / {n_out} bf16",
          flush=True)
