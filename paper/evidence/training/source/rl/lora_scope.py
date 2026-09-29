"""Shared module-name selection for runtime injection and checkpoint inventory."""
SCOPES = ("head", "head+lang", "head+lang_all")


def in_scope(name, scope):
    if scope not in SCOPES:
        raise ValueError(f"Unsupported LoRA scope: {scope}")
    if name.startswith("action_head."):
        return True
    if scope != "head" and ".language_model.layers." in name:
        attention = (".q_proj", ".k_proj", ".v_proj", ".o_proj")
        mlp = (".gate_proj", ".up_proj", ".down_proj") if scope == "head+lang_all" else ()
        return name.endswith(attention + mlp)
    return False
