"""Shared module-name selection for runtime injection and checkpoint inventory."""
SCOPES = ("head", "head+lang", "head+lang_all", "all_ordinary_linear")

# GR00T keeps a language-model ``lm_head`` in the checkpoint, but the LIBERO
# flow-matching action path never calls it.  Attaching a recovery adapter there
# creates a parameter with no backward path and makes the gradient audit fail;
# it remains part of the frozen W4A4 base and is intentionally excluded from
# the trainable recovery scope.
INACTIVE_RECOVERY_LINEARS = frozenset({"backbone.model.lm_head"})


def in_scope(name, scope):
    if scope not in SCOPES:
        raise ValueError(f"Unsupported LoRA scope: {scope}")
    # This scope is structural: lora_qad.install_lora still filters to
    # two-dimensional nn.Linear weights with K divisible by 16. Custom
    # CategorySpecificLinear banks therefore remain frozen.
    if scope == "all_ordinary_linear":
        return name not in INACTIVE_RECOVERY_LINEARS
    if name.startswith("action_head."):
        return True
    if scope != "head" and ".language_model.layers." in name:
        attention = (".q_proj", ".k_proj", ".v_proj", ".o_proj")
        mlp = (".gate_proj", ".up_proj", ".down_proj") if scope == "head+lang_all" else ()
        return name.endswith(attention + mlp)
    return False
