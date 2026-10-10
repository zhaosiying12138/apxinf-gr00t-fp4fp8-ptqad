"""Checkpoint-faithful LIBERO data configuration shared by recovery/calibration.

The upstream processor merges modality overrides into its saved modalities.
Passing the default all-robot data registry adds an unrelated 50-step robot to
a 40-step checkpoint. Restrict the *override* to saved LIBERO definitions;
retain the checkpoint's complete processor metadata and statistics on disk.
"""
from copy import deepcopy
import json
import os
from pathlib import Path


def restore_checkpoint_model_config(config, checkpoint):
    """Keep architectural dimensions from the base, plus explicit train knobs.

launch_finetune starts from a generic model config although AutoModel loads the
full checkpoint. Synchronize the pipeline's config before model/dataset setup,
otherwise its processor can pad to a different action horizon than the model.
"""
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    runtime = config.model
    full = Gr00tN1d7Config.from_pretrained(checkpoint, local_files_only=True)
    for field in ("tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model",
                  "tune_vlln", "state_dropout_prob", "load_bf16",
                  "backbone_trainable_params_fp32", "random_rotation_angle",
                  "color_jitter_params", "use_relative_action"):
        if hasattr(runtime, field):
            setattr(full, field, getattr(runtime, field))
    full.model_name = os.environ.get("GR00T_BACKBONE_MODEL", full.model_name)
    config.model = full
    configure_libero_data(config, checkpoint)
    return config


def configure_libero_data(config, checkpoint):
    from gr00t.data.utils import parse_modality_configs
    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint / "processor_config.json").read_text())
    modalities = saved["processor_kwargs"]["modality_configs"]
    if "libero_sim" not in modalities:
        raise ValueError("Checkpoint has no LIBERO processor definition")
    config.data.modality_configs = parse_modality_configs(
        {"libero_sim": deepcopy(modalities["libero_sim"])})
    config.data.override_pretraining_statistics = False
    # Keep the literal model path: the upstream factory dispatches on the
    # 'nvidia/Cosmos-Reason2' substring, which symlink resolution would erase.
    if os.environ.get("GR00T_BACKBONE_MODEL"):
        config.model.model_name = os.environ["GR00T_BACKBONE_MODEL"]
    horizon = config.model.action_horizon
    for name, modality in modalities.items():
        if len(modality["action"]["delta_indices"]) > horizon:
            raise ValueError(f"Saved processor {name} exceeds model horizon {horizon}")
    return config


def verify_libero_statistics(processor, checkpoint):
    expected = json.loads((Path(checkpoint) / "statistics.json").read_text())
    # The processor's serialization cache starts empty in this upstream
    # revision; the StateActionProcessor owns the actual normalization.
    actual = processor.state_action_processor.statistics
    if actual.get("libero_sim") != expected.get("libero_sim") or "libero_sim" not in expected:
        raise ValueError("Effective LIBERO normalization differs from the base checkpoint")
    processor.statistics = deepcopy(actual)
    mapping = json.loads((Path(checkpoint) / "embodiment_id.json").read_text())
    if any(processor.embodiment_id_mapping.get(k) != v for k, v in mapping.items()):
        raise ValueError("A saved embodiment ID changed during processor construction")
    # Upstream appends newly registered unused embodiments at construction.
    # Preserve the source mapping verbatim for deterministic checkpoint export.
    processor.embodiment_id_mapping = mapping


def libero_action_spec(checkpoint):
    """Read the unpadded action extent from the actual processor/statistics."""
    checkpoint = Path(checkpoint)
    processor = json.loads((checkpoint / "processor_config.json").read_text())["processor_kwargs"]
    action = processor["modality_configs"]["libero_sim"]["action"]
    stats = json.loads((checkpoint / "statistics.json").read_text())["libero_sim"]["action"]
    keys = action["modality_keys"]
    dims = [len(stats[key]["mean"]) for key in keys]
    if not dims or min(dims) < 1 or not action["delta_indices"]:
        raise ValueError("Empty LIBERO action dimensions/horizon")
    return {"embodiment": "libero_sim", "horizon": len(action["delta_indices"]),
            "dimensions": sum(dims), "keys": keys, "key_dimensions": dims,
            "delta_indices": action["delta_indices"],
            "policy": "prefix horizon and concatenated processor action keys; exclude padding"}


def action_mask(action, spec):
    import torch
    if action.ndim != 3 or not (0 < spec["horizon"] <= action.shape[1] and
                                0 < spec["dimensions"] <= action.shape[2]):
        raise ValueError(f"Action shape {tuple(action.shape)} cannot contain {spec}")
    mask = torch.zeros_like(action, dtype=torch.float32)
    mask[:, :spec["horizon"], :spec["dimensions"]] = 1
    return mask
