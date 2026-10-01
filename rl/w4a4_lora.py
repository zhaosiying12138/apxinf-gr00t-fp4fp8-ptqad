"""Explicit W4A4 base + raw-input LoRA forward contract.

The recovery training driver imports this module for its W4A4 path. It is also
the smallest safe integration seam for a future native APXInf operator: a
caller supplies an activation-QDQ callable and, optionally, a native base
operator. The resulting layer computes

    base_operator(module, QDQ(x)) + (B(A(x)) * alpha / rank)

The residual branch receives the original BF16 ``x``. Passing ``QDQ(x)`` to
both branches would silently define a different model and invalidates the
QAD/OPD checkpoint. ``base_operator`` can later dispatch packed NVFP4 weight
GEMM; its default is a dense ``F.linear`` and therefore is only a numerical
contract reference, not a native kernel implementation.

The returned patch object restores every original module forward exactly. No
global ``nn.Linear.forward`` monkey patch is used, and installing this helper
without an explicit call has no effect on existing W4A16 evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


ActivationQDQ = Callable[[torch.Tensor], torch.Tensor]
BaseOperator = Callable[[nn.Linear, torch.Tensor], torch.Tensor]
ScopePredicate = Callable[[str, nn.Linear], bool]


def _dense_base(module: nn.Linear, x_qdq: torch.Tensor) -> torch.Tensor:
    return F.linear(x_qdq, module.weight, module.bias)


@dataclass
class W4A4LoRAPatch:
    """A reversible set of module-local forward replacements."""

    _originals: tuple[tuple[nn.Module, Callable], ...]

    @property
    def count(self) -> int:
        return len(self._originals)

    def restore(self) -> None:
        for module, original in self._originals:
            module.forward = original
        self._originals = ()


def install_w4a4_lora(
    model: nn.Module,
    activation_qdq: ActivationQDQ,
    *,
    rank: int,
    alpha: float,
    scope: ScopePredicate | None = None,
    base_operator: BaseOperator | None = None,
) -> W4A4LoRAPatch:
    """Install explicit QDQ/base and raw-input LoRA branches on LoRA modules.

    Only modules that already expose ``lora_A`` and ``lora_B`` are touched;
    this function never invents adapters or changes ordinary BF16 layers. A
    caller must supply ``activation_qdq`` so an accidental W4A16 fallback is
    impossible. ``base_operator`` is useful for a future native packed GEMM;
    the dense default exists solely for CPU algebra tests.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not callable(activation_qdq):
        raise TypeError("activation_qdq must be callable")
    if rank <= 0 or alpha <= 0:
        raise ValueError("rank and alpha must be positive")
    if base_operator is None:
        base_operator = _dense_base
    if not callable(base_operator):
        raise TypeError("base_operator must be callable")

    originals: list[tuple[nn.Module, Callable]] = []
    scale = float(alpha) / float(rank)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if scope is not None and not scope(name, module):
            continue
        A, B = module.lora_A, module.lora_B
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != module.in_features or B.shape[0] != module.out_features:
            raise ValueError(f"invalid LoRA shapes at {name}: A={tuple(A.shape)} B={tuple(B.shape)}")
        if A.shape[0] != B.shape[1] or A.shape[0] != rank:
            raise ValueError(f"LoRA rank mismatch at {name}: expected {rank}, A={tuple(A.shape)}, B={tuple(B.shape)}")

        original = module.forward

        def forward(x, *, _module=module, _original=original):
            # Keep the callback explicit and local: native packed GEMM can be
            # supplied as _base_operator without changing residual semantics.
            x_qdq = activation_qdq(x)
            if not isinstance(x_qdq, torch.Tensor) or tuple(x_qdq.shape) != tuple(x.shape):
                raise ValueError(f"activation_qdq changed shape at {_module}: {getattr(x_qdq, 'shape', None)}")
            y = base_operator(_module, x_qdq)
            z = F.linear(x, _module.lora_A)
            z = F.linear(z, _module.lora_B)
            return y + (z * scale).to(y.dtype)

        module.forward = forward
        originals.append((module, original))
    return W4A4LoRAPatch(tuple(originals))
