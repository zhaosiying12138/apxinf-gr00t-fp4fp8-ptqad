"""Load a QAD/OPD adapter without merging it into the NVFP4 base.

The existing merge exporter is intentionally retained for W4A16 diagnostics.
Formal W4A4 evaluation needs the frozen PTQ weights and the BF16 LoRA branch
to stay separate, otherwise activation QDQ would also quantize the recovery
residual.  This module loads only ``lora_A``/``lora_B`` tensors and installs
the explicit raw-input branch from :mod:`w4a4_lora`.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

try:
    from .w4a4_lora import W4A4LoRAPatch, install_w4a4_lora
except ImportError:  # GR00T runtime puts rl/ directly on sys.path
    from w4a4_lora import W4A4LoRAPatch, install_w4a4_lora


def _shards(checkpoint: Path) -> list[Path]:
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        return [checkpoint / name for name in sorted(set(mapping.values()))]
    single = checkpoint / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"adapter checkpoint has no safetensors index: {checkpoint}")


def install_saved_w4a4_adapter(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    activation_qdq,
    rank: int,
    alpha: float,
    scope=None,
) -> W4A4LoRAPatch:
    """Attach frozen A/B tensors from a Trainer checkpoint and install W4A4.

    The checkpoint is expected to contain the exact adapter keys produced by
    ``rl/lora_qad.py``.  All loaded tensors are copied to the model's device
    and dtype; no base weight is changed or requantized.
    """
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "recovery_manifest.json"
    if not manifest_path.is_file() and (root.parent / "recovery_manifest.json").is_file():
        manifest_path = root.parent / "recovery_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest.get("rank") is not None and int(manifest["rank"]) != int(rank):
        raise ValueError("adapter rank differs from recovery manifest")
    if manifest.get("alpha") is not None and float(manifest["alpha"]) != float(alpha):
        raise ValueError("adapter alpha differs from recovery manifest")

    modules = dict(model.named_modules())
    seen: set[str] = set()
    with torch.no_grad():
        for shard in _shards(root):
            with safe_open(str(shard), framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if not key.endswith((".lora_A", ".lora_B")):
                        continue
                    stem = key.rsplit(".lora_", 1)[0]
                    suffix = key.rsplit(".", 1)[1]
                    module = modules.get(stem)
                    if not isinstance(module, nn.Linear):
                        raise ValueError(f"adapter key has no Linear target: {key}")
                    if key in seen:
                        raise ValueError(f"duplicate adapter key: {key}")
                    tensor = reader.get_tensor(key).to(device=module.weight.device, dtype=module.weight.dtype)
                    expected = (rank, module.in_features) if suffix == "lora_A" else (module.out_features, rank)
                    if tuple(tensor.shape) != expected:
                        raise ValueError(f"adapter shape mismatch at {key}: {tuple(tensor.shape)} != {expected}")
                    module.register_parameter(suffix, nn.Parameter(tensor, requires_grad=False))
                    seen.add(key)
    if not seen:
        raise ValueError(f"no LoRA tensors found in {root}")
    stems = {key.rsplit(".lora_", 1)[0] for key in seen}
    for stem in stems:
        if {f"{stem}.lora_A", f"{stem}.lora_B"} - seen:
            raise ValueError(f"unpaired LoRA tensors for {stem}")
    patch = install_w4a4_lora(model, activation_qdq, rank=rank, alpha=alpha, scope=scope)
    if patch.count != len(stems):
        raise ValueError(f"installed {patch.count} adapters but loaded {len(stems)}")
    return patch
