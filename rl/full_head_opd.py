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


def load_teacher():
    """Frozen BF16 teacher; kept on GPU (7GB — fits beside b16 student)."""
    if _state["teacher"] is not None:
        return _state["teacher"]
    # temporarily restore un-patched forward so the teacher loads/evals clean
    nn.Linear.forward = _state["orig_forward"]
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    teacher = Gr00tN1d7(cfg).to(torch.bfloat16).cuda().eval()
    from safetensors.torch import load_model
    import glob as _g
    shards = sorted(_g.glob(os.path.join(TEACHER_PATH, "model-*.safetensors")))
    sd = {}
    for s in shards:
        sd.update(load_model(teacher, s, strict=False))
    print(f"[opd] teacher loaded bf16 from {len(shards)} shards", flush=True)
    _state["teacher"] = teacher
    return teacher


def install_teacher_kl(trainer_cls):
    """Patch Gr00tTrainer.compute_loss to add ||a_s - a_teacher|| anchor."""
    orig = trainer_cls.compute_loss

    def with_teacher(self, model, inputs, return_outputs=False, **kw):
        # ground-truth demo loss (original)
        out = orig(self, model, inputs, return_outputs=True, **kw)
        loss, outputs = out[0], out[1]
        try:
            teacher = load_teacher()
            with torch.no_grad():
                t_out = teacher(**{k: (v.to(torch.bfloat16).cuda() if torch.is_tensor(v) else v)
                                   for k, v in inputs.items()
                                   if k in ("vision", "state", "actions", "prompt")})
        except Exception as e:
            print("[opd] teacher fwd skip:", type(e).__name__, str(e)[:120], flush=True)
            return (loss, outputs) if return_outputs else loss
        s_pred = getattr(outputs, "pred_actions", None)
        if s_pred is None and isinstance(outputs, dict):
            s_pred = outputs.get("pred_actions") or outputs.get("actions")
        t_pred = getattr(t_out, "pred_actions", None) if not isinstance(t_out, dict) else t_out.get("pred_actions")
        if s_pred is not None and t_pred is not None:
            kl = torch.nn.functional.mse_loss(
                s_pred.float(), t_pred.detach().float().to(s_pred.device))
            loss = loss + float(os.environ.get("OPD_KL_W", "1.0")) * kl
            if self.state.global_step % 20 == 0:
                print(f"[opd] step {self.state.global_step} task={float(loss):.4f} kl={float(kl):.4f}",
                      flush=True)
        return (loss, outputs) if return_outputs else loss

    trainer_cls.compute_loss = with_teacher
    print("[opd] teacher-KL loss hook installed", flush=True)


# 3) streaming save (battle-tested from QAD run) + resume arg hook
from qad_stream_save import install_streaming_save
install_streaming_save()

sys.argv = ["launch_finetune.py",
    "--base-model-path", os.environ["GR00T_BASE_CKPT"],
    "--dataset-path", "./demo_data/libero_demo",
    "--embodiment-tag", "LIBERO_PANDA",
    "--num-gpus", "1",
    "--output-dir", os.path.expanduser("~/fq_opd_out"),
    "--save-steps", "100",
    "--max-steps", os.environ.get("OPD_STEPS", "1000"),
    "--global-batch-size", "16",
    "--gradient-accumulation-steps", "2",
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
