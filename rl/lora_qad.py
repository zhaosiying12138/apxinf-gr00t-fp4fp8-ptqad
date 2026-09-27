"""LoRA QAD / OPD: quantization-aware (optionally teacher-KL) LoRA recovery
training for GR00T N1.7 — the 24GB-safe successor to full_head_qad.py.

Semantics: every in-scope Linear computes  y = x @ Q(W + (alpha/r) B A)^T
per forward, with NVFP4 fake-quant STE through the merged weight. Base weights
frozen; only LoRA A/B train. Deploy = bake quant(W + delta) — training and
deployment see the SAME quantized function (no merge mismatch).

Env knobs:
  QAD_LORA_R (32)  rank; QAD_LORA_ALPHA (64); QAD_LORA_SCOPE
    head       action_head.* Linears (default)
    head+lang  + language attention q/k/v/o
  QAD_STEPS (1000), QAD_BSZ (16), QAD_ACC (2)
  QAD_OPD_KL_W (0 default: off; e.g. 1.0 enables probe-cached teacher-KL)
Run from ~/codebase/groot-fsdp2/Isaac-GR00T with .venv python.
"""
import os, sys, time

os.environ.setdefault("GR00T_BASE_CKPT",
    "/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LD_LIBRARY_PATH",
    os.path.expanduser("~/miniforge3/envs/media7/lib:") + os.environ.get("LD_LIBRARY_PATH", ""))

sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/rl")
sys.path.insert(0, os.getcwd())
import torch, torch.nn as nn
from torch_fp4 import fake_quant_nvfp4_torch

T0 = time.time()
R = int(os.environ.get("QAD_LORA_R", "32"))
ALPHA = float(os.environ.get("QAD_LORA_ALPHA", "64"))
SCOPE = os.environ.get("QAD_LORA_SCOPE", "head")
STEPS = os.environ.get("QAD_STEPS", "1000")
BSZ = os.environ.get("QAD_BSZ", "16")
ACC = os.environ.get("QAD_ACC", "2")
KL_W = float(os.environ.get("QAD_OPD_KL_W", "0"))
OUT = os.environ.get("QAD_OUT", "/mnt/c/fq_lora_out")

_lora_count = [0]


def in_scope(name):
    if SCOPE in ("head", "head+lang"):
        if name.startswith("action_head."):
            return True
    if SCOPE == "head+lang" and ".language_model.layers." in name:
        if name.endswith(("_proj.weight",)):
            return True
    return False


def install_lora(model):
    import torch.nn.init as init
    sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant/ptq")
    from bake import alloc_aggr
    from torch_fp4 import fake_quant_nvfp4_torch as fq4
    baked_base = os.path.exists(os.path.join(os.environ["GR00T_BASE_CKPT"], "ptq_recipe.json"))
    frozen = trainable = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.ndim == 2 and mod.weight.is_cuda:
            if in_scope(name) and mod.weight.shape[1] % 16 == 0:
                W = mod.weight
                mod.lora_A = nn.Parameter(torch.empty(R, W.shape[1], dtype=W.dtype, device=W.device))
                mod.lora_B = nn.Parameter(torch.zeros(W.shape[0], R, dtype=W.dtype, device=W.device))
                init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)
                scale = ALPHA / R

                def make_fwd(m, s):
                    def fwd(x):
                        Wq = fake_quant_nvfp4_torch(m.weight.float() + (m.lora_B.float() @ m.lora_A.float()) * s)
                        return torch.nn.functional.linear(x, Wq.to(x.dtype), m.bias)
                    return fwd
                mod.forward = make_fwd(mod, scale)
                _lora_count[0] += 1
            elif baked_base:
                pass   # weights already carry the baked PTQ values (incl GPTQ/FP8)
            else:
                kind = alloc_aggr(name, mod.weight)
                if kind == "bf16":
                    pass
                elif kind == "fp8":
                    with torch.no_grad():
                        s = mod.weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
                        import torch_fp4 as _tf
                        mod.weight.data = (_tf._quant_e4m3(mod.weight.float() / s)
                                           * torch.sign(mod.weight.float()) * s).to(mod.weight.dtype)
                else:
                    with torch.no_grad():
                        mod.weight.data = fq4(mod.weight).to(mod.weight.dtype)
    for pname, p in model.named_parameters():
        p.requires_grad = "lora_" in pname.split(".")[-1]
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    print(f"[lora-qad] LoRA on {_lora_count[0]} Linears (r={R}, scope={SCOPE}); "
          f"trainable={trainable/1e6:.1f}M frozen={frozen/1e6:.0f}M", flush=True)


def install_trainer_hooks():
    """Patch Gr00tTrainer: inject LoRA after model wiring; optional OPD-KL."""
    from gr00t.experiment.trainer import Gr00tTrainer
    orig_init = Gr00tTrainer.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        model = getattr(self, "model", None) or getattr(self, "policy", None)
        core = getattr(model, "module", model)
        install_lora(core)
        print(f"[lora-qad] injected at trainer init ({time.time()-T0:.0f}s)", flush=True)
    Gr00tTrainer.__init__ = patched_init

    if KL_W > 0:
        CACHE = "/mnt/c/fq_opd_probes/teacher_probes.pt"
        _probe = {"loaded": False, "inputs": None, "pred": None}

        def load_probe():
            if not _probe["loaded"]:
                d = torch.load(CACHE, map_location="cpu")
                _probe["inputs"] = d["inputs"]; _probe["pred"] = d["pred"]
                _probe["loaded"] = True
                print(f"[opd] probe cache loaded: pred {tuple(d['pred'].shape)}", flush=True)

        orig_loss = Gr00tTrainer.compute_loss

        def with_probe_kl(self, model, inputs, return_outputs=False, **kw):
            out = orig_loss(self, model, inputs, return_outputs=return_outputs, **kw)
            try:
                if model.training and self.state.global_step % 4 == 0:
                    load_probe()
                    tin = {k: (v.to(next(model.parameters()).device) if torch.is_tensor(v) else v)
                           for k, v in _probe["inputs"].items()}
                    tin.pop("labels", None)
                    torch.manual_seed(20260927)
                    s_pred = model(**tin, output_hidden_states=False).get("pred_actions")
                    if s_pred is not None:
                        t_pred = _probe["pred"].to(s_pred.device)
                        kl = torch.nn.functional.mse_loss(s_pred.float(), t_pred.float())
                        loss = out["loss"] if isinstance(out, dict) else out
                        loss = loss + KL_W * kl
                        if isinstance(out, dict):
                            out["loss"] = loss
                        else:
                            out = loss
                        if self.state.global_step % 20 == 0:
                            print(f"[opd] step {self.state.global_step} kl={float(kl):.6f}", flush=True)
            except Exception as e:
                print("[opd] probe-kl skip:", type(e).__name__, str(e)[:120], flush=True)
            return out
        Gr00tTrainer.compute_loss = with_probe_kl
        print(f"[opd] probe-cached teacher-KL hook installed (w={KL_W})", flush=True)


install_trainer_hooks()

sys.argv = ["launch_finetune.py",
    "--base-model-path", os.environ["GR00T_BASE_CKPT"],
    "--dataset-path", "./demo_data/libero_demo",
    "--embodiment-tag", "LIBERO_PANDA",
    "--num-gpus", "1",
    "--output-dir", OUT,
    "--save-steps", "100",
    "--max-steps", STEPS,
    "--global-batch-size", BSZ,
    "--gradient-accumulation-steps", ACC,
    "--save-only-model",
    "--num-shards-per-epoch", "4",
    "--dataloader-num-workers", "1"]

import runpy
print(f"[lora-qad] launching: r={R} scope={SCOPE} steps={STEPS} kl_w={KL_W}", flush=True)
try:
    runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
    print(f"[lora-qad] COMPLETED in {(time.time()-T0)/3600:.2f}h")
except SystemExit as e:
    print(f"[lora-qad] exit {e} after {(time.time()-T0)/3600:.2f}h")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[lora-qad] FAILED after {(time.time()-T0)/3600:.2f}h: {type(e).__name__}: {e}")
    sys.exit(1)
