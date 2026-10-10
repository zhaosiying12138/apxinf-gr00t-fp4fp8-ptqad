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
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

_ORIG = nn.Linear.forward
_MODE = "all"
_ACTIVATION_W4A4 = False
_ACTIVATION_STE = False
_RECIPE_METHODS = None
_CATEGORY_METHODS = None


@dataclass
class ActivationInstallReport:
    """Audit record for one activation-only installation.

    The report deliberately separates ordinary ``nn.Linear`` modules from
    category banks. A category layer is reported as BF16 unless its active bank
    is explicitly present in ``category_ptq_recipe.json`` and carries an NVFP4
    method; merely seeing a ``CategorySpecificLinear`` is not evidence that
    its weights were quantized.
    """

    mode: str
    ste: bool
    ordinary_total: int = 0
    ordinary_w4a4: int = 0
    ordinary_bf16: int = 0
    category_total: int = 0
    category_w4a4: int = 0
    category_bf16: int = 0
    padded_modules: int = 0
    records: list[dict] = field(default_factory=list)
    _originals: list[tuple[nn.Module, object]] = field(default_factory=list,
                                                       repr=False)

    @property
    def count(self) -> int:
        return self.ordinary_w4a4 + self.category_w4a4

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ste": self.ste,
            "ordinary_total": self.ordinary_total,
            "ordinary_w4a4": self.ordinary_w4a4,
            "ordinary_bf16": self.ordinary_bf16,
            "category_total": self.category_total,
            "category_w4a4": self.category_w4a4,
            "category_bf16": self.category_bf16,
            "padded_modules": self.padded_modules,
            "records": list(self.records),
        }

    def restore(self) -> None:
        """Restore every module forward captured by this installation."""
        for module, original in self._originals:
            module.forward = original
            for attribute in ("_fp4vla_activation_format",
                              "_fp4vla_w4a4_padded", "_fp4vla_w4a4"):
                if hasattr(module, attribute):
                    delattr(module, attribute)
        self._originals.clear()


def install_scoped_fakequant(mode: str):
    global _MODE, _ACTIVATION_W4A4, _ACTIVATION_STE
    _MODE = mode
    _ACTIVATION_W4A4 = os.environ.get("FP4VLA_W4A4", "0") == "1"
    _ACTIVATION_STE = False
    print(f"[fp4vla] scoped quantize-once armed: mode={mode}", flush=True)


def configure_quant_recipe(folder):
    """Load ordinary and category PTQ gates for activation-only wrapping.

    Ordinary records use module names without ``.weight``. Category records
    use the checkpoint key ending in ``.W`` and select the active LIBERO bank.
    Missing category provenance is intentionally treated as BF16 rather than
    inferred from the presence of a custom module.
    """
    global _RECIPE_METHODS, _CATEGORY_METHODS
    root = Path(folder)
    path = root / "ptq_recipe.json"
    if not path.is_file():
        _RECIPE_METHODS = None
    else:
        recipe = json.loads(path.read_text())
        layers = recipe.get("layers", {})
        if not isinstance(layers, dict):
            raise ValueError("ptq_recipe.layers must be an object")
        methods = {}
        for name, record in layers.items():
            method = record.get("actual_method", record.get("method", record.get("requested_method")))
            if isinstance(method, str):
                methods[name] = method
        _RECIPE_METHODS = methods

    category_path = root / "category_ptq_recipe.json"
    _CATEGORY_METHODS = {}
    if category_path.is_file():
        category = json.loads(category_path.read_text())
        active = category.get("active_libero_bank")
        categories = category.get("categories", {})
        if not isinstance(active, int) or not isinstance(categories, dict):
            raise ValueError("category_ptq_recipe has invalid active bank/categories")
        for key, record in categories.items():
            banks = record.get("banks", []) if isinstance(record, dict) else []
            if 0 <= active < len(banks) and isinstance(banks[active], dict):
                method = banks[active].get("method")
                if isinstance(method, str):
                    _CATEGORY_METHODS[key] = method

    ordinary_count = len(_RECIPE_METHODS or {})
    ordinary_nvfp4 = sum(str(v).startswith("nvfp4") for v in (_RECIPE_METHODS or {}).values())
    category_nvfp4 = sum(str(v).startswith("nvfp4") for v in _CATEGORY_METHODS.values())
    print(f"[fp4vla] recipe gate loaded: {ordinary_nvfp4}/{ordinary_count} ordinary layers use NVFP4; "
          f"{category_nvfp4}/{len(_CATEGORY_METHODS)} active category banks use NVFP4", flush=True)


def _is_nvfp4(name):
    return _RECIPE_METHODS is None or _RECIPE_METHODS.get(name, "").startswith("nvfp4")


def _is_category_nvfp4(name):
    """Return true only for a category layer with explicit active-bank proof."""
    method = (_CATEGORY_METHODS or {}).get(name + ".W")
    return isinstance(method, str) and method.startswith("nvfp4")


def _native_activation_qdq(x, *, ste=False):
    try:
        from native_activation import native_activation_qdq_torch
    except ImportError:  # repository tests import this module as rl.scoped_quant
        from quant.native_activation import native_activation_qdq_torch
    return native_activation_qdq_torch(x, ste=ste)


def _pad_qdq_slice(x, qdq):
    """Run a K-blocked QDQ on arbitrary K and return the original width.

    The native contract consumes 16-element blocks. Model projections whose K
    is not a multiple of 16 are padded with zeros only for the QDQ call, then
    sliced back before the dense matmul. This keeps the module interface and
    bias shape unchanged while making the boundary explicit in the audit
    report.
    """
    if not isinstance(x, torch.Tensor) or x.ndim == 0:
        raise TypeError("activation must be a non-scalar torch.Tensor")
    width = int(x.shape[-1])
    if width <= 0:
        raise ValueError("activation final dimension must be positive")
    padded_width = (width + 15) // 16 * 16
    if padded_width == width:
        return qdq(x), False
    padded = F.pad(x, (0, padded_width - width))
    result = qdq(padded)
    if not isinstance(result, torch.Tensor) or tuple(result.shape) != tuple(padded.shape):
        raise ValueError("activation_qdq changed padded activation shape")
    return result[..., :width], True


def _activation_qdq(x, *, ste=False):
    return _pad_qdq_slice(x, lambda value: _native_activation_qdq(value, ste=ste))[0]


def mark_scope(model):
    """Quantize-once: rewrite in-scope Linear weights in place."""
    import sys
    sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
    from torch_fp4 import fake_quant_nvfp4_torch
    mode = _MODE
    n_in = n_out = n_act = n_category = 0
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
                eligible = (inscope and _is_nvfp4(name) and w.ndim == 2 and
                            w.shape[1] % 16 == 0)
                if eligible and w.is_cuda:
                    mod.weight.data = fake_quant_nvfp4_torch(w).to(w.dtype)
                    n_in += 1
                else:
                    n_out += 1
                if _ACTIVATION_W4A4 and eligible:
                    original = mod.forward

                    def w4a4_forward(x, *, _mod=mod):
                        # Only the base matmul sees Qa(x).  A separate LoRA
                        # branch is installed by rl.w4a4_lora during QAD/OPD
                        # training; this service path handles plain PTQ.
                        return torch.nn.functional.linear(
                            _activation_qdq(x), _mod.weight, _mod.bias)

                    mod.forward = w4a4_forward
                    mod._fp4vla_w4a4_original_forward = original
                    n_act += 1
            # CategorySpecificLinear stores [category,K,N] weights and is not
            # an nn.Linear.  Its activation contract is still K-blocked, so
            # wrap its bmm when W4A4 is explicitly requested.
            if (_ACTIVATION_W4A4 and type(mod).__name__ == "CategorySpecificLinear"
                    and _is_category_nvfp4(name) and hasattr(mod, "W")
                    and mod.W.ndim == 3):
                def category_forward(x, cat_ids, *, _mod=mod):
                    qx = _activation_qdq(x)
                    return torch.bmm(qx, _mod.W[cat_ids]) + _mod.b[cat_ids].unsqueeze(1)
                mod.forward = category_forward
                mod._fp4vla_activation_format = "NVFP4"
                mod._fp4vla_w4a4 = True
                n_category += 1
            elif (_ACTIVATION_W4A4 and type(mod).__name__ == "CategorySpecificLinear"):
                mod._fp4vla_activation_format = "BF16"
                mod._fp4vla_w4a4 = False
    nn.Linear.forward = _ORIG  # native speed; weights already quantized
    print(f"[fp4vla] scope={mode}: {n_in} weights QUANTIZED-ONCE / {n_out} bf16; "
          f"activation W4A4={'on' if _ACTIVATION_W4A4 else 'off'} ({n_act} Linear, {n_category} category)",
          flush=True)


def install_activation_only(
    model: nn.Module,
    *,
    mode: str = "all",
    ste: bool = False,
    activation_qdq=None,
) -> ActivationInstallReport:
    """Install the single activation-QDQ contract without touching weights.

    ``activation_qdq`` is an optional one-argument callback used by tests or a
    native executor. When omitted, the fixed NVFP4 activation reference is
    used, with ``ste`` forwarded to its straight-through path. Ordinary
    modules are gated by ``ptq_recipe.json``; category modules require the
    active-bank record from ``category_ptq_recipe.json``. Every skipped layer is
    recorded as BF16 so callers cannot infer W4A4 from a module type alone.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if mode not in ("all", "head", "backbone"):
        raise ValueError("mode must be all, head or backbone")
    if activation_qdq is None:
        activation_qdq = lambda value: _native_activation_qdq(value, ste=ste)
    if not callable(activation_qdq):
        raise TypeError("activation_qdq must be callable")

    report = ActivationInstallReport(mode=mode, ste=bool(ste))

    def in_scope(name: str) -> bool:
        headish = ("action_head" in name) or ("gemma_expert" in name)
        return mode == "all" or (mode == "head" and headish) or (mode == "backbone" and not headish)

    def apply_qdq(value):
        return _pad_qdq_slice(value, activation_qdq)

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if not in_scope(name):
                continue
            report.ordinary_total += 1
            allowed = _is_nvfp4(name)
            if not allowed:
                report.ordinary_bf16 += 1
                report.records.append({"name": name, "format": "BF16", "reason": "recipe_not_nvfp4"})
                setattr(mod, "_fp4vla_activation_format", "BF16")
                setattr(mod, "_fp4vla_w4a4", False)
                report._originals.append((mod, mod.forward))
                continue
            original = mod.forward

            def linear_forward(value, *, _mod=mod):
                qvalue, padded = apply_qdq(value)
                return F.linear(qvalue, _mod.weight, _mod.bias)

            mod.forward = linear_forward
            mod._fp4vla_w4a4_original_forward = original
            mod._fp4vla_activation_format = "NVFP4"
            mod._fp4vla_w4a4 = True
            report.ordinary_w4a4 += 1
            padded = int(mod.in_features) % 16 != 0
            report.padded_modules += int(padded)
            report.records.append({"name": name, "format": "W4A4", "padded_k": padded})
            report._originals.append((mod, original))
            continue

        if type(mod).__name__ != "CategorySpecificLinear":
            continue
        if not in_scope(name):
            continue
        report.category_total += 1
        allowed = (_is_category_nvfp4(name) and hasattr(mod, "W") and
                   isinstance(mod.W, torch.Tensor) and mod.W.ndim == 3)
        if not allowed:
            report.category_bf16 += 1
            reason = "category_recipe_not_nvfp4" if _CATEGORY_METHODS else "category_recipe_missing"
            report.records.append({"name": name, "format": "BF16", "reason": reason})
            setattr(mod, "_fp4vla_activation_format", "BF16")
            setattr(mod, "_fp4vla_w4a4", False)
            report._originals.append((mod, mod.forward))
            continue
        original = mod.forward

        def category_forward(value, cat_ids, *, _mod=mod):
            qvalue, _ = apply_qdq(value)
            return torch.bmm(qvalue, _mod.W[cat_ids]) + _mod.b[cat_ids].unsqueeze(1)

        mod.forward = category_forward
        mod._fp4vla_w4a4_original_forward = original
        mod._fp4vla_activation_format = "NVFP4"
        mod._fp4vla_w4a4 = True
        report.category_w4a4 += 1
        padded = int(mod.W.shape[1]) % 16 != 0
        report.padded_modules += int(padded)
        report.records.append({"name": name, "format": "W4A4", "padded_k": padded})
        report._originals.append((mod, original))

    print("[fp4vla] activation-only " + json.dumps(report.as_dict(), ensure_ascii=False), flush=True)
    return report


def install_scoped_activation(mode: str = "all", *, ste: bool = False):
    """Arm activation-only W4A4 wrapping for an already baked base.

    Recovery evaluation must not run ``mark_scope`` because a merged adapter
    is no longer a valid quantized weight.  This entry point wraps only the
    base matmul; a subsequent ``install_w4a4_lora`` call can replace adapter
    modules with the raw-input residual branch.
    """
    global _MODE, _ACTIVATION_W4A4, _ACTIVATION_STE
    _MODE = mode
    _ACTIVATION_W4A4 = True
    _ACTIVATION_STE = bool(ste)


def mark_activation_scope(model, *, ste: bool | None = None):
    """Install W4A4 activation QDQ without changing any weight tensor."""
    if not _ACTIVATION_W4A4:
        raise RuntimeError("call install_scoped_activation before mark_activation_scope")
    return install_activation_only(
        model, mode=_MODE, ste=_ACTIVATION_STE if ste is None else bool(ste))
