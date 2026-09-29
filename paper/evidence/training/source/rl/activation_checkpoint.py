"""Optional non-reentrant block recomputation, including eval-mode LoRA.

No parameter dtype, module names, loss, batch size or dropout mode changes.
Unlike native training-only checkpoint gates, this also works for the eval-mode
teacher probe's differentiable student pass. Frozen teacher no-grad bypasses it.
"""
import functools
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_block(module):
    if getattr(module.forward, "_recovery_checkpoint", False):
        raise ValueError("Block checkpoint wrapper already installed")
    original = module.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        if not torch.is_grad_enabled():
            return original(*args, **kwargs)
        return checkpoint(original, *args, use_reentrant=False,
                          preserve_rng_state=True, **kwargs)

    forward._recovery_checkpoint = True
    module.forward = forward


def install_activation_checkpointing(model):
    """Wrap active language/DiT/VL blocks without changing checkpoint keys."""
    backbone = model.backbone.model
    language = backbone.language_model
    # Cache mutation during backward recomputation is invalid. Training never
    # reuses a generation cache, so disable it without changing token outputs.
    for module in (backbone, language):
        if hasattr(module, "config"):
            module.config.use_cache = False
            if hasattr(module.config, "text_config"):
                module.config.text_config.use_cache = False
    groups = {"language": language.layers,
              "dit": model.action_head.model.transformer_blocks,
              "vl": model.action_head.vl_self_attention.transformer_blocks}
    counts = {}
    for name, blocks in groups.items():
        active = [block for block in blocks if any(p.requires_grad for p in block.parameters())]
        for block in active:
            checkpoint_block(block)
        counts[name] = len(active)
    if not sum(counts.values()):
        raise ValueError("No trainable blocks available for activation checkpointing")
    return {"implementation": "torch.non_reentrant_per_block", "blocks": counts,
            "preserve_rng_state": True, "eval_mode_gradients_supported": True,
            "kv_cache": False, "parameter_dtypes_changed": False}
