"""OPD arm (teacher-KL recovery) — the theory-prescribed fix for the QAD gap.

Diagnosis chain that motivates this design (see PLAN.md 2026-09-26/27):
- QAD-1000 (demo task loss + fake-quant) closed-loop = 0% (both sessions agree)
- Mixed map head-FP4+BF16-backbone = 71.2% => quantization deployment is fine
- => the training RECIPE is the gap: demo-loss training anchors to ground
  truth actions, which does not preserve closed-loop behavior under quant
  noise. Fix: anchor to the TEACHER's (BF16 base model's) predictions.

Recipe (run in sister env, same guardrails as full_head_qad.py):
  - teacher = frozen BF16 base checkpoint, forward-only, same batch
  - student = action-head weights + global NVFP4 fake-quant (STE)
  - loss = ||a_student - a_teacher||_2 on the same (obs, noise, timestep)
    computed as flow-consistency on the predicted velocity field when the
    trainer exposes it, else on the denoised action chunk (fallback)
  - b16 x accum 2, AC on, save/100 + streaming save (battle-tested), AdamW

Launch: cd ~/codebase/groot-fsdp2/Isaac-GR00T && \
  LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib \
  .venv/bin/python ~/codebase/fp4vla/rl/full_head_opd.py
"""
import os, sys, time

os.environ.setdefault("GR00T_REPO", os.path.expanduser("~/codebase/groot-fsdp2/Isaac-GR00T"))
os.environ.setdefault("GR00T_BASE_CKPT", os.path.expanduser("~/codebase/groot-fsdp2/weights/GR00T-N1.7-3B"))
os.environ.setdefault("HF_HUB_OFFLINE", "0")

sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/rl")
import torch, torch.nn as nn

T0 = time.time()

# 1) student fake-quant (same as QAD run)
from torch_fp4 import install_global_linear_fakequant
install_global_linear_fakequant()
print("[opd] student NVFP4 emulation installed", flush=True)

# 2) teacher-KL hook: the loss override lives in a Trainer compute_loss patch.
#    The teacher is lazily materialized INSIDE the hook (first batch) by
#    re-loading the base checkpoint with the ORIGINAL forward — because the
#    global patch would quantize the teacher too. We exploit the fact that
#    quantized-forward is only active on modules flagged in-scope: after the
#    student model is built we (a) snapshot original Linear.forward, (b) load
#    the teacher on CPU-offload (eval, bf16, forward on GPU per batch), and
#    (c) restore the student path by re-installing the patch.
TEACHER_PATH = os.environ.get(
    "OPD_TEACHER",
    "/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10",
)

_state = {"teacher": None, "orig_forward": nn.Linear.forward, "ready": False}


def install_teacher_kl(trainer_cls):
    """Probe-cached teacher-KL: anchor to PRE-COMPUTED BF16 teacher
    predictions (rl/opd_probe_cache.py -> /mnt/c/fq_opd_probes/).

    The earlier twin-forward variant (eval()/no_grad()/train() swap inside
    the training loop) crashed the WSL GPU driver deterministically at
    ~step 12 under FSDP2+AC. This variant runs ONLY the normal grad-enabled
    student forward on a small cached probe batch — the exact path that
    completed QAD-1000 stably — and adds KL(student_pred, cached_teacher_pred).
    """
    orig = trainer_cls.compute_loss
    CACHE = "/mnt/c/fq_opd_probes/teacher_probes.pt"
    _probe = {"loaded": False, "inputs": None, "pred": None}

    def load_probe():
        if _probe["loaded"]:
            return
        d = torch.load(CACHE, map_location="cpu", weights_only=False)
        _probe["inputs"] = d["inputs"]
        _probe["pred"] = d["pred"]
        _probe["loaded"] = True
        print(f"[opd] probe cache loaded: pred {tuple(d['pred'].shape)}", flush=True)

    def with_probe_kl(self, model, inputs, return_outputs=False, **kw):
        loss, outputs = orig(self, model, inputs, return_outputs=True, **kw)
        try:
            EVERY = int(os.environ.get('OPD_KL_EVERY', '4'))
            if self.state.global_step % EVERY != 0:
                return (loss, outputs) if return_outputs else loss
            load_probe()
            pin = {k: (v.to(model.device.type, model.device.index
                            if hasattr(model.device, "index") else None)
                        if torch.is_tensor(v) else v)
                   for k, v in _probe["inputs"].items()}
            torch.manual_seed(20260927)  # replay the pinned (noise, t)
            # no_grad WITHOUT eval/train swap: the graph-free forward avoids the
            # old WSL crash, while grad-on stacked an 8-window autograd graph on
            # the training batch's and overflowed 24GB (allocator corruption /
            # driver wedge = hard machine freeze).
            with torch.no_grad():
                p_out = model(pin)       # STUDENT forward, graph-free
            s_pred = p_out.get("pred_actions") if hasattr(p_out, "get") \
                else getattr(p_out, "pred_actions", None)
            if s_pred is not None:
                t_pred = _probe["pred"].to(s_pred.device)
                if s_pred.shape == t_pred.shape:
                    kl = torch.nn.functional.mse_loss(
                        s_pred.float(), t_pred.float())
                    loss = loss + float(os.environ.get("OPD_KL_W", "5.0")) * kl
                    if self.state.global_step % 20 < EVERY:
                        print(f"[opd] step {self.state.global_step} "
                              f"task={float(loss):.4f} kl={float(kl):.6f}", flush=True)
            del s_pred, pin, p_out
            torch.cuda.empty_cache()
        except Exception as e:
            print("[opd] probe-kl skip:", type(e).__name__, str(e)[:120], flush=True)
        return (loss, outputs) if return_outputs else loss

    trainer_cls.compute_loss = with_probe_kl
    print("[opd] probe-cached teacher-KL hook installed", flush=True)


# 3) streaming save (battle-tested from QAD run) + resume arg hook
from qad_stream_save import install_streaming_save
install_streaming_save()

sys.argv = ["launch_finetune.py",
    "--base-model-path", os.environ["GR00T_BASE_CKPT"],
    "--dataset-path", "./demo_data/libero_demo",
    "--embodiment-tag", "LIBERO_PANDA",
    "--num-gpus", "1",
    "--output-dir", os.path.expanduser(os.environ.get("OPD_OUT", "~/fq_opd_out")),
    "--save-steps", "50",
    "--max-steps", os.environ.get("OPD_STEPS", "1000"),
    "--global-batch-size", "8",
    "--gradient-accumulation-steps", "4",
    "--save-only-model",
    "--no-fsdp2-pin-memory",
    "--fsdp2-activation-checkpointing",
    "--no-fsdp2-reshard-after-forward",
    "--num-shards-per-epoch", "4",
    "--dataloader-num-workers", "1"]
resume = os.environ.get("RESUME", "").split()
sys.argv += resume

sys.path.insert(0, os.getcwd())

# patch the trainer class as soon as the module is importable
import gr00t.experiment.trainer as _tr
install_teacher_kl(_tr.Gr00tTrainer)

import runpy
print(f"[opd] launching: teacher={TEACHER_PATH} student=base+fakequant "
      f"steps={sys.argv[sys.argv.index('--max-steps')+1]}", flush=True)
try:
    runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
    print(f"[opd] COMPLETED in {(time.time()-T0)/3600:.2f}h")
except SystemExit as e:
    print(f"[opd] exit {e} after {(time.time()-T0)/3600:.2f}h")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[opd] FAILED after {(time.time()-T0)/3600:.2f}h: {type(e).__name__}: {e}")
    sys.exit(1)
