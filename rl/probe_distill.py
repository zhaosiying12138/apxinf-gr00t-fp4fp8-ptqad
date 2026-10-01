"""Offline-labelled, differentiable velocity matching (no GPU work on import).

The caller keeps replay_context open through backward, so checkpoint recomputation
sees the same eval flags. RNG state is restored even when a forward fails.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch
CACHE_VERSION = 3


def masked_velocity_mse(student, teacher, mask):
    if mask.shape != student.shape or teacher.shape != student.shape:
        raise ValueError("Velocity/mask shapes must match exactly")
    mask = mask.float()
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all() or mask.sum() <= 0:
        raise ValueError("Velocity mask must be binary and contain valid action elements")
    return ((student.float() - teacher.float()).square() * mask).sum() / mask.sum()


def tensor_tree(value, device="cpu", clone=True):
    if torch.is_tensor(value):
        value = value.detach().to(device)
        return value.clone() if clone else value
    if isinstance(value, Mapping):
        return {k: tensor_tree(v, device, clone) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_tree(v, device, clone) for v in value)
    return value


def prediction(outputs):
    """Accept GR00T dict/ModelOutput and Trainer-style (loss, outputs)."""
    if isinstance(outputs, Mapping) and "pred_actions" in outputs:
        return outputs["pred_actions"]
    if hasattr(outputs, "pred_actions"):
        return outputs.pred_actions
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        if torch.is_tensor(outputs[1]) and outputs[1].ndim >= 2:
            return outputs[1]
        return prediction(outputs[1])
    raise TypeError("Probe forward must expose pred_actions; no action/loss fallback")


def architecture(model):
    model = getattr(model, "module", model)
    return {
        "language_layers": len(model.backbone.model.language_model.layers),
        "dit_layers": len(model.action_head.model.transformer_blocks),
        "vl_layers": len(model.action_head.vl_self_attention.transformer_blocks),
    }


def require_full_model(model):
    actual = architecture(model)
    expected = {"language_layers": 16, "dit_layers": 32, "vl_layers": 4}
    if actual != expected:
        raise ValueError(f"Expected complete GR00T LIBERO architecture {expected}, got {actual}")
    return actual


def flow_signature(model):
    model = getattr(model, "module", model)
    config = getattr(model, "config", None)
    return {name: getattr(config, name, None) for name in (
        "noise_beta_alpha", "noise_beta_beta", "noise_s", "num_timestep_buckets",
        "action_horizon", "max_action_dim", "state_history_length")}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def replay_context(model, seed, autocast_dtype="bfloat16"):
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    flags = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            # manual_seed() would seed every visible GPU, including uncaptured
            # generators. Seed only the CPU and this model's CUDA generators.
            torch.random.default_generator.manual_seed(int(seed))
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(int(seed))
            with torch.autocast(device_type=device.type,
                                dtype=getattr(torch, autocast_dtype) if autocast_dtype != "none" else None,
                                enabled=autocast_dtype != "none"):
                yield
    finally:
        # Preserve mixed train/eval states, not just the top-level flag.
        for module, flag in flags:
            module.training = flag


class ProbeAnchor:
    def __init__(self, path):
        self.path = Path(path)
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache.get("version") != CACHE_VERSION:
            raise ValueError("Legacy teacher cache rejected; regenerate with opd_probe_cache.py")
        self.meta = cache["metadata"]
        self.samples = cache["samples"]
        if not self.samples:
            raise ValueError("Teacher cache is empty")
        self._validated = False

    def validate_model(self, model):
        if self._validated:
            return
        if require_full_model(model) != self.meta["architecture"]:
            raise ValueError("Teacher/student architecture mismatch")
        dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        if dtype != self.meta["model_dtype"]:
            raise ValueError(f"Teacher noise dtype {self.meta['model_dtype']} != student {dtype}; rebuild cache")
        if flow_signature(model) != self.meta["flow_config"]:
            raise ValueError("Teacher/student flow/noise schedule differs")
        self._validated = True

    def loss(self, model, index):
        """Caller owns replay_context; exactly one cached observation per graph."""
        sample = self.samples[index % len(self.samples)]
        device = next(model.parameters()).device
        inputs = tensor_tree(sample["inputs"], device)
        if inputs["action"].shape[0] != 1:
            raise ValueError("Probe microbatch must contain exactly one observation")
        outputs = model(inputs)
        student = prediction(outputs)
        teacher = sample["pred"].to(device=device, dtype=torch.float32)
        if student.shape != teacher.shape:
            raise ValueError(f"Teacher/student prediction shapes differ: {teacher.shape} / {student.shape}")
        # Shared full endpoint/noise stays unchanged; only real LIBERO action
        # elements contribute to the loss (padding is not a robot action).
        mse = masked_velocity_mse(student, teacher, inputs["action_mask"])
        if not mse.requires_grad:
            raise RuntimeError("Teacher MSE has no gradient; refusing a silent no-op")
        return mse


def install_sequential_probe(trainer_cls, cache_path, weight, every=4):
    """Accumulate auxiliary gradients AFTER the main backward, before optimizer.

    Each scheduled optimizer update gets one probe per training microbatch:
      mean_j demo_loss_j + weight * mean_j probe_MSE_j.
    Other updates contain demo_loss only; the long-run coefficient is weight/every.
    This adds forward/backward compute but never stacks its graph with demo_loss.
    Trainer.compute_loss and its return_outputs protocol are left untouched.
    """
    if weight <= 0 or every < 1:
        raise ValueError("weight and every must be positive")
    original = trainer_cls.training_step
    if getattr(original, "_sequential_probe", False):
        raise RuntimeError("Sequential probe hook already installed")
    anchor = ProbeAnchor(cache_path)  # fail before a lengthy training run

    def training_step(self, model, inputs, *args, **kwargs):
        # The integration has only been audited for HF Trainer + Accelerator on
        # one device. Do not silently invent scaling for FSDP/DeepSpeed/Apex.
        accelerator_accumulation = int(getattr(self.accelerator,
                                               "gradient_accumulation_steps", 1) or 1)
        trainer_accumulation = int(getattr(self.args,
                                           "gradient_accumulation_steps", 1) or 1)
        # Transformers versions differ in where accumulation is represented:
        # some leave Accelerator at one and let Trainer accumulate, while
        # others pass the configured value through Accelerator.  Divide here
        # only when Trainer owns accumulation; Accelerator.backward() already
        # applies the divisor when it owns it.
        if (getattr(self, "use_apex", False) or
            getattr(self.accelerator, "num_processes", 1) != 1 or
            accelerator_accumulation not in (1, trainer_accumulation) or
            getattr(self, "is_deepspeed_enabled", False) or
            getattr(self, "is_fsdp_enabled", False)):
            raise ValueError("Probe hook supports single-device HF Trainer accumulation only")
        anchor.validate_model(model)
        result = original(self, model, inputs, *args, **kwargs)
        # original has already backward()ed and freed the main activations.
        step = int(self.state.global_step)
        if (step + 1) % every:
            return result
        micro = getattr(self, "_probe_micro_step", 0)
        sample = anchor.samples[micro % len(anchor.samples)]
        accumulation = int(getattr(self, "current_gradient_accumulation_steps",
                                   trainer_accumulation) or trainer_accumulation)
        if accumulation not in (1, trainer_accumulation):
            raise ValueError("Probe hook saw an unexpected gradient accumulation count")
        with replay_context(model, sample["seed"], anchor.meta["autocast_dtype"]):
            mse = anchor.loss(model, micro)
            backward_divisor = accumulation if accelerator_accumulation == 1 else 1
            scaled = weight * mse / backward_divisor
            self.accelerator.backward(scaled)
            extra = scaled.detach()
        self._probe_micro_step = micro + 1
        if step % max(every, 20) == every - 1:
            print(f"[opd] step={step + 1} probe={micro % len(anchor.samples)} "
                  f"mse={mse.detach().item():.6g} weight={weight} microbatch=1", flush=True)
        # Stock Trainer returns a detached tensor; tolerate custom wrappers
        # without corrupting tuple/dict contracts.
        if isinstance(result, dict):
            return {**result, "loss": result["loss"] + extra}
        if isinstance(result, tuple):
            return (result[0] + extra, *result[1:])
        return result + extra

    training_step._sequential_probe = True
    trainer_cls.training_step = training_step
    return anchor
