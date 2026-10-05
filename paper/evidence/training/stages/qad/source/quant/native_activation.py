"""CPU reference for the APXInf native NVFP4 activation encoding contract.

This reproduces the *specified arithmetic* of
``apxinf-cuda/adapters/cublaslt_fp4_adapter.cu``: cast the input to F16,
compute block maxima and ``amax * float32(1/6)`` in FP32, encode an E4M3
scale for every 16 K-elements, then divide in FP32 and round to E2M1.
The secondary tensor scale is fixed at 1. No dynamic scale or clipping
policy is introduced here. CPU tests do not establish GPU/native parity.

Nonfinite inputs, including overflow during the F16 cast, are rejected.
The CUDA kernel does not define a useful model-level NaN/Inf contract; this
reference deliberately fails closed instead of treating those as evidence
of valid quantization. Finite F16 inputs may still saturate at +/-2688 or
underflow through a zero E4M3 scale, exactly as this fixed-scale policy allows.

``native_activation_qdq(x)`` is a CPU-only building block for a future
W4A4 base branch. Keep the original ``x`` for any high-precision LoRA branch:
``linear(qdq(x), Wq) + scale * linear(linear(x, A), B)``. This module does
not simulate GEMM accumulation or its F16-output/BF16-output casts, implement
weight quantization, or enable a model/service hook.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:  # package import (tests) and direct quant/ path import (GR00T runtime)
    from .fp4_quant import (
        decode_e2m1, decode_e4m3, encode_e2m1, encode_e4m3, swizzle_scales,
    )
except ImportError:  # pragma: no cover - exercised by external runtime loader
    from fp4_quant import (
        decode_e2m1, decode_e4m3, encode_e2m1, encode_e4m3, swizzle_scales,
    )


@torch.no_grad()
def _torch_native_activation_qdq(x: torch.Tensor) -> torch.Tensor:
    """Torch implementation of the fixed-secondary-scale native contract.

    The CPU encoder above is the audit oracle.  This function keeps the same
    order of operations on CPU or CUDA so training and service evaluation can
    exercise W4A4 without copying activations to the host.  It intentionally
    returns only the dequantized value; packing and the APXInf GEMM are owned
    by the native runtime.
    """
    try:
        from .torch_fp4 import _quant_e2m1, _quant_e4m3
    except ImportError:  # pragma: no cover - external runtime loader
        from torch_fp4 import _quant_e2m1, _quant_e4m3

    if not isinstance(x, torch.Tensor):
        raise TypeError("activation must be a torch.Tensor")
    if x.ndim == 0 or x.numel() == 0 or x.shape[-1] % 16:
        raise ValueError("activation must be nonempty with final K divisible by 16")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("activation must have float16, bfloat16, float32 or float64 dtype")
    # The CUDA contract casts to F16 before finding block maxima.  Fail closed
    # on overflow: silently propagating infinities would make an invalid
    # episode look like a valid quantized rollout.
    half = x.detach().to(torch.float16)
    if not bool(torch.isfinite(half).all()):
        raise ValueError("activation overflows F16 or contains nonfinite values")
    width = x.shape[-1]
    blocks = half.float().reshape(-1, width // 16, 16)
    amax = blocks.abs().amax(dim=-1)
    # float32(1/6) is intentional; the native adapter does not use a dynamic
    # tensor scale for activations.
    requested = amax * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)
    scales = _quant_e4m3(requested)
    divisor = torch.where(scales > 0, scales, torch.ones_like(scales))
    normalized = blocks / divisor.unsqueeze(-1)
    q = _quant_e2m1(normalized)
    decoded = (q * scales.unsqueeze(-1)).reshape(x.shape)
    return decoded.to(x.dtype)


def native_activation_qdq_torch(x: torch.Tensor, *, ste: bool = False) -> torch.Tensor:
    """Device-preserving W4A4 activation QDQ for training and evaluation.

    ``ste=True`` supplies an identity straight-through derivative for QAD and
    OPD.  The low-rank residual must call this function's input *before* QDQ;
    see :mod:`rl.w4a4_lora` for the enforced branch structure.
    """
    decoded = _torch_native_activation_qdq(x)
    return decoded + (x - x.detach()) if ste else decoded


@dataclass(frozen=True)
class NativeActivationEncoding:
    """CPU tensors; rows flatten all input dimensions except the final K.

    ``dequantized`` is FP32 in the original input shape. ``packed`` is
    uint8 [rows,K/2], ``scales`` is uint8 [rows,K/16], and
    ``swizzled_scales`` is a zero-padded uint8 physical VEC16 scale buffer.
    These are activation reference artifacts, not a deployable checkpoint.
    """

    dequantized: torch.Tensor
    packed: torch.Tensor
    scales: torch.Tensor
    swizzled_scales: torch.Tensor
    zero_scale_blocks: int
    saturated_values: int
    tensor_scale: float = 1.0


@torch.no_grad()
def encode_native_activation(x: torch.Tensor) -> NativeActivationEncoding:
    """Encode finite CPU floating tensors with nonempty K divisible by 16.

    Leading dimensions, including a vector with one implicit row, are
    flattened only for encoding. No K-padding is applied: the native GEMM
    contract requires K divisible by 16. Only the physical scale tile is
    padded, and every padding byte is initialized to zero.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("activation must be a torch.Tensor")
    if x.device.type != "cpu":
        raise ValueError("native activation reference is CPU-only; GPU parity is unverified")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("activation must have float16, bfloat16, float32 or float64 dtype")
    if x.ndim == 0 or x.numel() == 0 or x.shape[-1] % 16:
        raise ValueError("activation must be nonempty with final K divisible by 16")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("nonfinite activation input is outside the supported native contract")
    x_half = x.detach().to(torch.float16)
    if not bool(torch.isfinite(x_half).all()):
        raise ValueError("activation overflows F16 before native quantization")

    width = x.shape[-1]
    values = x_half.float().contiguous().numpy().reshape(-1, width // 16, 16)
    maxima = np.abs(values).max(axis=-1)
    # CUDA evaluates a float multiplication, not a double-precision /6.
    requested_scales = maxima * np.float32(1.0 / 6.0)
    scale_codes = encode_e4m3(requested_scales)
    decoded_scales = decode_e4m3(scale_codes)
    # The safe divisor affects code selection only; zero physical scales
    # still decode their entire block to zero.
    divisors = np.where(decoded_scales > 0, decoded_scales, np.float32(1.0))
    normalized = np.divide(values, divisors[..., None], dtype=np.float32)
    codes = encode_e2m1(normalized).astype(np.uint8)
    # CUDA uses `vals[j] < 0`, so -0.0 loses its sign, while a negative
    # nonzero value rounded to magnitude zero keeps its negative code.
    codes = (codes & np.uint8(7)) | np.where(values < 0, 8, 0).astype(np.uint8)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(-1, width // 2)
    dequantized = (decode_e2m1(codes) * decoded_scales[..., None]).reshape(tuple(x.shape))
    return NativeActivationEncoding(
        dequantized=torch.from_numpy(dequantized),
        packed=torch.from_numpy(packed),
        scales=torch.from_numpy(scale_codes),
        swizzled_scales=torch.from_numpy(swizzle_scales(scale_codes)),
        zero_scale_blocks=int(np.count_nonzero(scale_codes == 0)),
        saturated_values=int(np.count_nonzero(np.abs(normalized) > np.float32(6.0))),
    )


def native_activation_qdq(x: torch.Tensor, *, ste: bool = False) -> torch.Tensor:
    """Return CPU reference QDQ values in x.dtype without modifying x.

    ``ste=True`` explicitly chooses an identity straight-through derivative
    for later recovery experiments. It is an estimator, not the derivative
    of native rounding, and does not make this CPU reference a GPU kernel.
    The ordinary reference result is detached from autograd.
    """
    decoded = encode_native_activation(x).dequantized.to(x.dtype)
    return decoded + (x - x.detach()) if ste else decoded
