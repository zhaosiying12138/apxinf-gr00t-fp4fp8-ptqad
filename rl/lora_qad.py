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
  QAD_STEPS (1000), QAD_MICRO_BATCH (1), QAD_GLOBAL_BATCH (16)
  QAD_ACCUM_STEPS must equal GLOBAL/MICRO if also supplied (single GPU)
  QAD_OPD_MSE_W (0: off; QAD_OPD_KL_W is a deprecated alias)
  OPD_CACHE_PATH, OPD_EVERY (4), QAD_INIT_ADAPTER (optional prior checkpoint)
  QAD_ACTIVATION_CHECKPOINTING (0): optional non-reentrant block recomputation
  Teacher backward follows the main backward on microbatch 1; graphs never stack.
Run from ~/codebase/groot-fsdp2/Isaac-GR00T with .venv python.
"""
import json, os, sys, time
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

_lora_count = [0]
_runtime_trainer = [None]


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
                mod.forward = make_fwd(mod, scale)
                _lora_count[0] += 1
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

    def audited_step(self, model, *args, **kwargs):
        result = original(self, model, *args, **kwargs)
        if not getattr(self, "_lora_grad_verified", False):
            groups = {"head": []}
            if SCOPE != "head":
                groups["language"] = []
            for name, parameter in model.named_parameters():
                if not name.endswith(".lora_B"):
                    continue
                group = "language" if ".language_model.layers." in name else "head"
                if group in groups and parameter.grad is not None:
                    groups[group].append(parameter.grad.detach().float().square().sum())
            for group, squares in groups.items():
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
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from transformers import TrainerCallback
    original_model = Gr00tN1d7Pipeline._create_model
    original_dataset = Gr00tN1d7Pipeline._create_dataset

    def full_checkpoint_model(self, *args, **kwargs):
        restore_checkpoint_model_config(self.config, os.environ["GR00T_BASE_CKPT"])
        return original_model(self, *args, **kwargs)

    Gr00tN1d7Pipeline._create_model = full_checkpoint_model

    def fixed_statistics(self, *args, **kwargs):
        configure_libero_data(self.config, os.environ["GR00T_BASE_CKPT"])
        result = original_dataset(self, *args, **kwargs)
        verify_libero_statistics(self.processor, os.environ["GR00T_BASE_CKPT"])
        return result

    Gr00tN1d7Pipeline._create_dataset = fixed_statistics
    orig_init = Gr00tTrainer.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        _runtime_trainer[0] = self
        model = getattr(self, "model", None) or getattr(self, "policy", None)
        if (self.args.per_device_train_batch_size != MICRO_BATCH or
                self.args.gradient_accumulation_steps != ACCUM_STEPS):
            raise ValueError("Upstream Trainer batch semantics differ from the audited single-GPU contract")
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
                    "initial_adapter": initial, "optimizer_resumed": False,
                    "probe_weight": MSE_W, "probe_every": int(os.environ.get("OPD_EVERY", "4")),
                    "probe_cache": os.environ.get("OPD_CACHE_PATH") if MSE_W else None,
                    "probe_action_mask": libero_action_spec(base) if MSE_W else None,
                    "objective": "demo flow loss + scheduled sequential teacher velocity MSE" if MSE_W else "demo flow loss",
                    "normalization": "frozen base statistics", "seed": self.args.seed}
        manifest["activation_checkpointing"] = activation_checkpointing
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
        anchor = install_sequential_probe(Gr00tTrainer, cache_path, MSE_W, every)
        print(f"[opd] sequential velocity MSE weight={MSE_W} every={every} "
              f"source={anchor.meta['source_kind']} probes={len(anchor.samples)}", flush=True)
    install_gradient_audit(Gr00tTrainer)

if __name__ == "__main__":
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
