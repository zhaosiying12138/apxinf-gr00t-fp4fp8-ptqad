"""LoRA QAD / OPD: supervised recovery with optional teacher velocity MSE.
training for GR00T N1.7 — the 24GB-safe successor to full_head_qad.py.

v2 additive semantics: every in-scope Linear computes
    y = x @ W_baked^T + (x @ A^T @ B^T) * (alpha/r)
where W_baked are the PTQ-baked quantization values (frozen, never
requantized) and the low-rank residual is an exact-gradient BF16 branch
(SVDQuant/EoRA family: quantized base + full-precision low-rank correction).
Deploy adds (B@A)*s to W_baked without requantization. BF16 export introduces
rounding, so algebraic equivalence is not a claim of bitwise equality.

Env knobs:
  QAD_LORA_R (32)  rank; QAD_LORA_ALPHA (64); QAD_LORA_SCOPE
    head       action_head.* Linears (default)
    head+lang  + language attention q/k/v/o
    head+lang_all  + language attention and MLP gate/up/down
    all_ordinary_linear  + every eligible two-dimensional Linear, including
      the visual tower; CategorySpecificLinear banks remain frozen
  QAD_STEPS (1000), QAD_MICRO_BATCH (1), QAD_GLOBAL_BATCH (16)
  QAD_ACCUM_STEPS must equal GLOBAL/MICRO if also supplied (single GPU)
  QAD_OPD_MSE_W (0: off; QAD_OPD_KL_W is a deprecated alias)
  OPD_CACHE_PATH, OPD_EVERY (4), QAD_INIT_ADAPTER (optional prior checkpoint)
  TRAIN_SEED (from protocol; historical default 42), PROTOCOL_FILE
  QAD_ACTIVATION_CHECKPOINTING (0): optional non-reentrant block recomputation
  On selected optimizer updates, teacher backward follows the main backward on
  every accumulation microbatch; the two forward graphs never stack.
Run from ~/codebase/groot-fsdp2/Isaac-GR00T with .venv python.
"""
import json, os, sys, time
from glob import glob
from pathlib import Path
METRICS_STARTED = time.perf_counter()

os.environ.setdefault("GR00T_BASE_CKPT",
    "/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LD_LIBRARY_PATH",
    os.path.expanduser("~/miniforge3/envs/media7/lib:") + os.environ.get("LD_LIBRARY_PATH", ""))

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "quant"))
sys.path.insert(0, str(PROJECT / "rl"))
sys.path.insert(0, os.getcwd())
import torch, torch.nn as nn
from probe_distill import file_sha256, install_sequential_probe, require_full_model
from lora_scope import SCOPES, in_scope as module_in_scope
from recovery_batch import resolve_batch
from derived_replay_guard import prepare_derived_replay
from endpoint_cache_guard import prepare_endpoint_training
from gr00t_runtime import configure_libero_data, verify_libero_statistics, libero_action_spec, restore_checkpoint_model_config
from runtime_metrics import cuda_memory_peaks, parameter_storage

T0 = time.time()
R = int(os.environ.get("QAD_LORA_R", "32"))
ALPHA = float(os.environ.get("QAD_LORA_ALPHA", "64"))
SCOPE = os.environ.get("QAD_LORA_SCOPE", "head")
if SCOPE not in SCOPES or R < 1:
    raise ValueError(f"QAD_LORA_SCOPE must be one of {SCOPES}, rank must be positive")
STEPS = os.environ.get("QAD_STEPS", "1000")
MICRO_BATCH, ACCUM_STEPS, GLOBAL_BATCH = resolve_batch(os.environ)
MSE_W = float(os.environ.get("QAD_OPD_MSE_W", os.environ.get("QAD_OPD_KL_W", "0")))
OUT = os.environ.get("QAD_OUT", "/mnt/c/fq_lora_out")
_max_grad_norm_env = os.environ.get("QAD_MAX_GRAD_NORM")
if _max_grad_norm_env is None:
    MAX_GRAD_NORM = None
else:
    MAX_GRAD_NORM = float(_max_grad_norm_env)
    if not (MAX_GRAD_NORM > 0 and torch.isfinite(torch.tensor(MAX_GRAD_NORM))):
        raise ValueError("QAD_MAX_GRAD_NORM must be a finite positive value")
PROTOCOL_FILE = Path(os.environ.get("PROTOCOL_FILE", os.environ.get(
    "PTQAD_PROTOCOL_FILE", PROJECT / "exp/recovery_protocol.json"))).resolve()
if not PROTOCOL_FILE.is_file():
    raise FileNotFoundError(f"PROTOCOL_FILE does not exist: {PROTOCOL_FILE}")
PROTOCOL = json.loads(PROTOCOL_FILE.read_text())
PROTOCOL_VERSION = int(PROTOCOL.get("version", 1))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", str(
    PROTOCOL.get("selection", {}).get("train_seed", 42))))
if TRAIN_SEED < 0:
    raise ValueError("TRAIN_SEED must be non-negative")

_lora_count = [0]
_runtime_trainer = [None]
_w4a4_patch = [None]
_endpoint_cache_audit = [None]
W4A4 = os.environ.get("QAD_W4A4", "0") == "1"
_W4A4_NVFP4_NAMES = None


def _w4a4_nvfp4(name, base):
    """Return whether a frozen recipe assigns NVFP4 to this Linear."""
    global _W4A4_NVFP4_NAMES
    if _W4A4_NVFP4_NAMES is None:
        recipe_path = Path(base) / "ptq_recipe.json"
        if recipe_path.is_file():
            payload = json.loads(recipe_path.read_text())
            layers = payload.get("layers", {})
            _W4A4_NVFP4_NAMES = {
                layer for layer, record in layers.items()
                if str(record.get("actual_method", record.get("method", ""))).startswith("nvfp4")
            }
        else:
            _W4A4_NVFP4_NAMES = set()
    return name in _W4A4_NVFP4_NAMES


class CapturedStateDataset(torch.utils.data.Dataset):
    """Replay already-collated teacher snapshots as a training dataset.

    ``capture_onpolicy.py`` stores the exact normalized model inputs and the
    action endpoint used by the flow-matching loss.  Replaying these tensors
    avoids silently converting a rollout into a different image/state
    representation.  The recovery trainer is intentionally single-device and
    micro-batch one, so each item retains the original one-sample shapes.
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.derived_guard = prepare_derived_replay(self.root, PROTOCOL_FILE)
        self.paths = (self.derived_guard.paths if self.derived_guard is not None
                      else sorted(self.root.rglob("sample_*.pt")))
        if not self.paths:
            raise ValueError(f"QAD_CAPTURE_DATASET has no sample_*.pt files: {self.root}")
        self.samples = []
        for path in self.paths:
            sample = (self.derived_guard.load_sample(path) if self.derived_guard is not None
                      else torch.load(path, map_location="cpu", weights_only=True))
            inputs = sample.get("inputs")
            if not isinstance(inputs, dict):
                raise ValueError(f"Captured sample lacks inputs: {path}")
            required = {"embodiment_id", "state", "input_ids", "attention_mask",
                        "pixel_values", "image_grid_thw", "action", "action_mask"}
            if set(inputs) != required:
                raise ValueError(f"Captured sample keys differ at {path}: {sorted(inputs)}")
            for key, value in inputs.items():
                if not torch.is_tensor(value):
                    raise ValueError(f"Captured input is not a tensor: {path} {key}")
            # Captures are saved as a one-sample inference batch.  Store the
            # sample without that leading batch axis; the Trainer collator
            # adds it back exactly once.
            for key, value in list(inputs.items()):
                if key not in {"pixel_values", "image_grid_thw"} and value.ndim >= 1 and value.shape[0] == 1:
                    inputs[key] = value[0].contiguous()
            self.samples.append({"path": path, "inputs": inputs,
                                 "task_text": sample.get("task_text", "")})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]["inputs"]


class CapturedStateCollator:
    """Collate one captured sample while preserving VLM patch dimensions."""

    _non_batch = {"pixel_values", "image_grid_thw"}

    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("Captured recovery dataset requires micro-batch one")
        sample = features[0]
        batch = {}
        for key, value in sample.items():
            if key in self._non_batch:
                batch[key] = value
            else:
                batch[key] = value.unsqueeze(0)
        return {"inputs": batch}


def in_scope(name):
    return module_in_scope(name, SCOPE)


def install_lora(model):
    import torch.nn.init as init
    baked_base = os.path.exists(os.path.join(os.environ["GR00T_BASE_CKPT"], "ptq_recipe.json"))
    if not baked_base:
        raise ValueError("Recovery requires an explicit PTQ-baked base with ptq_recipe.json")
    frozen = trainable = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.ndim == 2:
            if in_scope(name) and mod.weight.shape[1] % 16 == 0:
                W = mod.weight
                mod.lora_A = nn.Parameter(torch.empty(R, W.shape[1], dtype=W.dtype, device=W.device))
                mod.lora_B = nn.Parameter(torch.zeros(W.shape[0], R, dtype=W.dtype, device=W.device))
                init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)
                scale = ALPHA / R

                def make_fwd(m, s):
                    # v2 additive semantics: quantized base (baked in m.weight,
                    # values never change) + exact-grad low-rank BF16 residual.
                    # Deploy = lora_merge_bake (W_saved + (B@A)*s, no requant).
                    def fwd(x):
                        y = torch.nn.functional.linear(x, m.weight, m.bias)
                        z = torch.nn.functional.linear(x, m.lora_A)
                        z = torch.nn.functional.linear(z, m.lora_B)
                        return y + (z * s).to(y.dtype)
                    return fwd
                # W4A4 installs the two branches after every adapter exists,
                # so the base receives Qa(x) while B(Ax) keeps raw BF16 x.
                # The legacy branch remains available for registered W4A16
                # protocols and is never silently mixed with W4A4.
                if not W4A4 or not _w4a4_nvfp4(name, os.environ["GR00T_BASE_CKPT"]):
                    mod.forward = make_fwd(mod, scale)
                _lora_count[0] += 1
    if W4A4:
        from native_activation import native_activation_qdq_torch
        from scoped_quant import configure_quant_recipe, install_activation_only
        from w4a4_lora import install_w4a4_lora
        # Install the activation contract on the complete frozen base first.
        # The adapter patch below replaces only LoRA-bearing Linear modules so
        # their residual branch can keep the original BF16 input.  Without
        # this first pass, non-LoRA vision/DiT layers would silently remain
        # BF16 during QAD/OPD while the PTQ service ran W4A4 everywhere.
        configure_quant_recipe(os.environ["GR00T_BASE_CKPT"])
        base_report = install_activation_only(model, mode="all", ste=True)
        print(f"[lora-qad] W4A4 base activation contract installed on "
              f"{base_report.count} modules (padded={base_report.padded_modules}); "
              "LoRA residual keeps raw BF16 input", flush=True)
        _w4a4_patch[0] = install_w4a4_lora(
            model,
            lambda value: native_activation_qdq_torch(value, ste=True),
            rank=R,
            alpha=ALPHA,
            scope=lambda name, module: in_scope(name) and _w4a4_nvfp4(name, os.environ["GR00T_BASE_CKPT"]),
        )
        print(f"[lora-qad] W4A4 activation contract installed on {_w4a4_patch[0].count} LoRA Linears; "
              "residual input remains raw BF16", flush=True)
    for pname, p in model.named_parameters():
        p.requires_grad = "lora_" in pname.split(".")[-1]
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    print(f"[lora-qad] LoRA on {_lora_count[0]} Linears (r={R}, scope={SCOPE}); "
          f"trainable={trainable/1e6:.1f}M frozen={frozen/1e6:.0f}M", flush=True)


def load_initial_adapter(model, checkpoint):
    """Continue from exact A/B on the same frozen base; reset Adam in both arms."""
    from safetensors import safe_open
    path = Path(checkpoint)
    manifest_path = path / "recovery_manifest.json"
    if not manifest_path.exists():
        manifest_path = path.parent / "recovery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    base = Path(os.environ["GR00T_BASE_CKPT"]).resolve()
    if (Path(manifest["base"]).resolve() != base or manifest["rank"] != R or
        manifest["alpha"] != ALPHA or manifest["scope"] != SCOPE):
        raise ValueError("Continuation base/rank/alpha/scope differs from its QAD source")
    if PROTOCOL_VERSION >= 3:
        if (manifest.get("train_seed") != TRAIN_SEED or
                manifest.get("protocol_sha256") != file_sha256(PROTOCOL_FILE)):
            raise ValueError("Continuation training seed/protocol differs from its QAD source")
    for field, filename in (("base_config_sha256", "config.json"),
                            ("base_statistics_sha256", "statistics.json"),
                            ("base_recipe_sha256", "ptq_recipe.json")):
        if manifest[field] != file_sha256(base / filename):
            raise ValueError(f"Continuation base changed since QAD: {filename}")
    targets = {n: p for n, p in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))}
    seen = set()
    with torch.no_grad():
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if not name.endswith((".lora_A", ".lora_B")):
                        continue
                    if name not in targets or name in seen:
                        raise ValueError(f"Unexpected/duplicate LoRA key {name}")
                    source = f.get_tensor(name)
                    if source.shape != targets[name].shape:
                        raise ValueError(f"LoRA shape mismatch: {name}")
                    targets[name].copy_(source)
                    seen.add(name)
    if seen != set(targets):
        raise ValueError(f"Missing {len(set(targets)-seen)} LoRA tensors in {checkpoint}")
    print(f"[lora-qad] loaded {len(seen)} exact A/B tensors from {path}; fresh optimizer", flush=True)


def install_gradient_audit(trainer_cls):
    original = trainer_cls.training_step

    # GR00T's LIBERO action policy never evaluates the language-model logits
    # head.  ``all_ordinary_linear`` still installs an adapter there so the
    # checkpoint inventory remains complete, but its B matrix is intentionally
    # inactive and cannot receive a gradient from the action loss.  Keep this
    # explicit allow-list separate from the three active branches so a missing
    # gradient in an actually used module still fails closed.
    inactive_prefixes = ("backbone.model.lm_head.",)

    def audited_step(self, model, *args, **kwargs):
        result = original(self, model, *args, **kwargs)
        if not getattr(self, "_lora_grad_verified", False):
            groups = {"head": [], "language": [], "vision": [], "other": []}
            expected_groups = set()
            for name, parameter in model.named_parameters():
                if not name.endswith(".lora_B"):
                    continue
                if name.startswith(inactive_prefixes):
                    continue
                if name.startswith("action_head."):
                    group = "head"
                elif ".language_model.layers." in name:
                    group = "language"
                elif ".visual." in name:
                    group = "vision"
                else:
                    group = "other"
                expected_groups.add(group)
                if parameter.grad is not None:
                    groups[group].append(parameter.grad.detach().float().square().sum())
            for group in sorted(expected_groups):
                squares = groups[group]
                if not squares:
                    raise RuntimeError(f"No {group} LoRA B gradients after first backward")
                norm = torch.stack(squares).sum().sqrt().item()
                if not (0 < norm < float("inf")):
                    raise RuntimeError(f"Invalid {group} LoRA B gradient norm: {norm}")
                print(f"[lora-grad-check] {group}: {len(squares)} B tensors, norm={norm:.6g}", flush=True)
            self._lora_grad_verified = True
        return result

    trainer_cls.training_step = audited_step


def install_trainer_hooks():
    """Inject LoRA and preserve checkpoint normalization without external edits."""
    _endpoint_cache_audit[0] = prepare_endpoint_training(PROTOCOL_FILE, MSE_W)
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from transformers import TrainerCallback
    original_model = Gr00tN1d7Pipeline._create_model
    original_dataset = Gr00tN1d7Pipeline._create_dataset
    original_collator = Gr00tN1d7Pipeline._create_collator

    def full_checkpoint_model(self, *args, **kwargs):
        restore_checkpoint_model_config(self.config, os.environ["GR00T_BASE_CKPT"])
        return original_model(self, *args, **kwargs)

    Gr00tN1d7Pipeline._create_model = full_checkpoint_model

    def fixed_statistics(self, *args, **kwargs):
        configure_libero_data(self.config, os.environ["GR00T_BASE_CKPT"])
        result = original_dataset(self, *args, **kwargs)
        verify_libero_statistics(self.processor, os.environ["GR00T_BASE_CKPT"])
        capture_root = os.environ.get("QAD_CAPTURE_DATASET")
        if capture_root:
            replay = CapturedStateDataset(capture_root)
            self._capture_dataset = replay
            print(f"[lora-qad] replaying {len(replay)} captured teacher snapshots from {replay.root}", flush=True)
            return replay, None
        return result

    Gr00tN1d7Pipeline._create_dataset = fixed_statistics

    def fixed_collator(self, *args, **kwargs):
        if hasattr(self, "_capture_dataset"):
            return CapturedStateCollator()
        return original_collator(self, *args, **kwargs)

    Gr00tN1d7Pipeline._create_collator = fixed_collator
    orig_init = Gr00tTrainer.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        _runtime_trainer[0] = self
        model = getattr(self, "model", None) or getattr(self, "policy", None)
        if (self.args.per_device_train_batch_size != MICRO_BATCH or
                self.args.gradient_accumulation_steps != ACCUM_STEPS):
            raise ValueError("Upstream Trainer batch semantics differ from the audited single-GPU contract")
        if self.args.seed != TRAIN_SEED:
            raise ValueError(f"Upstream Trainer seed {self.args.seed} differs from TRAIN_SEED {TRAIN_SEED}")
        core = getattr(model, "module", model)
        arch = require_full_model(core)
        install_lora(core)
        initial = os.environ.get("QAD_INIT_ADAPTER")
        if initial:
            load_initial_adapter(core, initial)
        activation_checkpointing = None
        if os.environ.get("QAD_ACTIVATION_CHECKPOINTING", "0") == "1":
            from activation_checkpoint import install_activation_checkpointing
            if self.args.gradient_checkpointing:
                raise ValueError("Do not combine native and recovery block checkpointing")
            activation_checkpointing = install_activation_checkpointing(core)
            print(f"[activation-checkpoint] {activation_checkpointing}", flush=True)
        base = Path(os.environ["GR00T_BASE_CKPT"]).resolve()
        manifest = {"version": 1, "base": str(base), "rank": R, "alpha": ALPHA,
                    "scope": SCOPE, "architecture": arch,
                    "parameter_dtype": str(next(core.parameters()).dtype),
                    "compute_dtype": "bfloat16" if self.args.bf16 else "float32",
                    "base_config_sha256": file_sha256(base / "config.json"),
                    "base_statistics_sha256": file_sha256(base / "statistics.json"),
                    "base_recipe_sha256": file_sha256(base / "ptq_recipe.json"),
                    "base_category_recipe_sha256": (file_sha256(base / "category_ptq_recipe.json")
                                                     if (base / "category_ptq_recipe.json").is_file() else None),
                    "base_category_manifest_sha256": (file_sha256(base / "category_bake_manifest.json")
                                                       if (base / "category_bake_manifest.json").is_file() else None),
                    "initial_adapter": initial, "optimizer_resumed": False,
                    "probe_weight": MSE_W, "probe_every": int(os.environ.get("OPD_EVERY", "4")),
                    "probe_cache": os.environ.get("OPD_CACHE_PATH") if MSE_W else None,
                    "probe_cache_sha256": file_sha256(os.environ["OPD_CACHE_PATH"]) if MSE_W else None,
                    "probe_action_mask": libero_action_spec(base) if MSE_W else None,
                    "objective": "demo flow loss + scheduled sequential teacher velocity MSE" if MSE_W else "demo flow loss",
                    "activation_format": "NVFP4",
                    "execution_mode": "W4A4 numerical QDQ + BF16 LoRA residual" if W4A4 else "W4A16 weight-only",
                    "w4a4_enabled": W4A4,
                    "normalization": "frozen base statistics", "seed": self.args.seed,
                    "max_grad_norm": float(self.args.max_grad_norm),
                    "f16_activation_saturation": os.environ.get(
                        "FP4VLA_SATURATE_F16_ACTIVATIONS", "0") == "1",
                    # Keep a secret-free copy of all numerical environment
                    # switches in the recovery receipt.  The orchestrator
                    # request records the same summary before launch; this
                    # second copy proves what Trainer actually observed.
                    "environment_summary": {"variables": {
                        key: os.environ.get(key) for key in (
                            "FP4VLA_QUANT", "FP4VLA_W4A4", "FP4VLA_W4A4_ADAPTER",
                            "FP4VLA_SATURATE_F16_ACTIVATIONS", "QAD_W4A4",
                            "QAD_MAX_GRAD_NORM", "QAD_LORA_SCOPE", "QAD_LORA_R",
                            "QAD_LORA_ALPHA", "QAD_LR", "QAD_STEPS", "QAD_OPD_MSE_W",
                            "OPD_EVERY", "TRAIN_SEED", "QAD_ACTIVATION_CHECKPOINTING")}},
                    "train_seed": TRAIN_SEED,
                    "protocol_file": str(PROTOCOL_FILE),
                    "protocol_sha256": file_sha256(PROTOCOL_FILE),
                    "capture_dataset": os.environ.get("QAD_CAPTURE_DATASET"),
                    "capture_dataset_sha256": os.environ.get("QAD_CAPTURE_DATASET_SHA256"),
                    "capture_dataset_samples": (len(list(Path(os.environ["QAD_CAPTURE_DATASET"]).expanduser().rglob("sample_*.pt")))
                                                 if os.environ.get("QAD_CAPTURE_DATASET") else None)}
        manifest["activation_checkpointing"] = activation_checkpointing
        if _endpoint_cache_audit[0] is not None:
            manifest["endpoint_training_identity"] = _endpoint_cache_audit[0]
        recovery_sources = [
            PROJECT / "rl" / filename for filename in (
                "lora_qad.py", "probe_distill.py", "lora_scope.py", "recovery_batch.py",
                "derived_replay_guard.py", "endpoint_cache_guard.py",
                "gr00t_runtime.py", "activation_checkpoint.py", "runtime_metrics.py",
                "w4a4_lora.py")]
        if W4A4:
            recovery_sources.append(PROJECT / "quant" / "native_activation.py")
        manifest["recovery_source_sha256"] = {
            str(path.relative_to(PROJECT)): file_sha256(path) for path in recovery_sources}
        manifest.update({"micro_batch": MICRO_BATCH, "gradient_accumulation_steps": ACCUM_STEPS,
                         "effective_global_batch": GLOBAL_BATCH,
                         "upstream_global_batch_size_cli": MICRO_BATCH,
                         "trainable_parameters": sum(p.numel() for p in core.parameters() if p.requires_grad),
                         "lora_linear_modules": sum(hasattr(m, "lora_A") for m in core.modules()),
                         "budget_note": "Equal continuation demo examples/optimizer steps; OPD adds probe forward/backward compute"})
        destination = Path(self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        text = json.dumps(manifest, indent=2) + "\n"
        (destination / "recovery_manifest.json").write_text(text)

        class SaveRecoveryManifest(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "recovery_manifest.json").write_text(text)
        self.add_callback(SaveRecoveryManifest())
        print(f"[lora-qad] injected at trainer init ({time.time()-T0:.0f}s)", flush=True)
    Gr00tTrainer.__init__ = patched_init

    if MSE_W > 0:
        cache_path = os.environ.get("OPD_CACHE_PATH")
        if not cache_path:
            raise ValueError("Set OPD_CACHE_PATH to a version-3 masked teacher cache")
        every = int(os.environ.get("OPD_EVERY", "4"))
        cache_sha = (_endpoint_cache_audit[0]["cache_sha256"] if _endpoint_cache_audit[0] is not None else None)
        anchor = install_sequential_probe(Gr00tTrainer, cache_path, MSE_W, every, cache_sha256=cache_sha)
        print(f"[opd] sequential velocity MSE weight={MSE_W} every={every} "
              f"source={anchor.meta['source_kind']} probes={len(anchor.samples)}", flush=True)
    install_gradient_audit(Gr00tTrainer)

if __name__ == "__main__":
    # The upstream fine-tune CLI has no seed flag.  Its experiment.run reads
    # config.data.seed, so wrap that entry point before launch_finetune imports
    # it and keep TrainingArguments, dataset sharding, and set_seed aligned.
    import gr00t.experiment.experiment as _experiment
    _original_run = _experiment.run

    def _seeded_run(config):
        config.data.seed = TRAIN_SEED
        if MAX_GRAD_NORM is not None:
            config.training.max_grad_norm = MAX_GRAD_NORM
            print(f"[lora-qad] max_grad_norm override={MAX_GRAD_NORM:g}", flush=True)
        # The recovery orchestrator controls checkpoint retention explicitly.
        # Respect it here so long W4A4 runs do not accumulate full-model shards
        # at every save interval and exhaust the experiment volume.
        save_limit = os.environ.get("QAD_SAVE_TOTAL_LIMIT")
        if save_limit is not None:
            config.training.save_total_limit = int(save_limit)
        return _original_run(config)

    _experiment.run = _seeded_run
    install_trainer_hooks()
    sys.argv = ["launch_finetune.py",
        "--base-model-path", os.environ["GR00T_BASE_CKPT"],
        "--dataset-path", os.environ.get("QAD_DATASET", "./demo_data/libero_demo"),
        "--embodiment-tag", "LIBERO_PANDA",
        "--num-gpus", "1",
        "--output-dir", OUT,
        "--save-steps", os.environ.get("QAD_SAVE_STEPS", "100"),
        "--max-steps", STEPS,
        # This upstream CLI calls the batch BEFORE accumulation "global".
        "--global-batch-size", str(MICRO_BATCH),
        "--gradient-accumulation-steps", str(ACCUM_STEPS),
        "--learning-rate", os.environ.get("QAD_LR", "1e-4"),
        "--save-only-model",
        "--num-shards-per-epoch", "4",
        "--dataloader-num-workers", "1"]

    import runpy
    print(f"[lora-qad] launching: r={R} scope={SCOPE} steps={STEPS} mse_w={MSE_W} "
          f"micro_batch={MICRO_BATCH} accumulation={ACCUM_STEPS} effective_global_batch={GLOBAL_BATCH}", flush=True)
    runtime_status, runtime_error = "completed", None
    try:
        runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
        print(f"[lora-qad] COMPLETED in {(time.time()-T0)/3600:.2f}h")
    except KeyboardInterrupt:
        runtime_status, runtime_error = "interrupted", "KeyboardInterrupt"
        raise
    except SystemExit as e:
        if e.code not in (None, 0):
            runtime_status, runtime_error = "failed", f"SystemExit: {e.code}"
        print(f"[lora-qad] exit {e} after {(time.time()-T0)/3600:.2f}h")
        raise
    except Exception as e:
        runtime_status, runtime_error = "failed", f"{type(e).__name__}: {e}"
        import traceback; traceback.print_exc()
        print(f"[lora-qad] FAILED after {(time.time()-T0)/3600:.2f}h: {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        trainer = _runtime_trainer[0]
        model = getattr(trainer, "model", None) if trainer is not None else None
        metrics = {"status": runtime_status, "error": runtime_error,
                   "wall_seconds": time.perf_counter() - METRICS_STARTED,
                   "timing_scope": "script import through training and checkpoint serialization",
                   "global_steps": int(trainer.state.global_step) if trainer is not None else None,
                   "requested_optimizer_steps": int(STEPS),
                   "cuda_peak_memory": cuda_memory_peaks(),
                   "cuda_peak_scope": "PyTorch allocator peaks for this training process; excludes other processes and driver-only allocations",
                   "parameter_storage_by_dtype": parameter_storage(model),
                   "compute_dtype": "bfloat16" if trainer is not None and trainer.args.bf16 else None,
                   "micro_batch": MICRO_BATCH, "effective_global_batch": GLOBAL_BATCH,
                   "gradient_accumulation_steps": ACCUM_STEPS,
                   "activation_checkpointing_enabled": os.environ.get("QAD_ACTIVATION_CHECKPOINTING", "0") == "1"}
        destination = Path(OUT)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "runtime_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"[runtime-metrics] saved {destination / 'runtime_metrics.json'}", flush=True)
