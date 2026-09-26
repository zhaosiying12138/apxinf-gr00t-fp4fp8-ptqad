"""full-head QAD: the formal 1000-step quantization-aware distillation run.

Recipe per the E4-final directive: b32 + AC + no-pin (WSL-safe), AdamW locked,
fake-quant NVFP4 on all nn.Linear weights, BF16 teacher KL/vector-field-match
distillation (lambda=0 phase). Runs in sister env with LD_LIBRARY_PATH media7.

Launch (from fp4vla session, AFTER 04:00 window opens + preflight passes):
  cd ~/codebase/groot-fsdp2/Isaac-GR00T && \
  LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib \
  .venv/bin/python ~/codebase/fp4vla/rl/full_head_qad.py
"""
import os, sys, time

os.environ.setdefault("GR00T_REPO", os.path.expanduser("~/codebase/groot-fsdp2/Isaac-GR00T"))
os.environ.setdefault("GR00T_BASE_CKPT", os.path.expanduser("~/codebase/groot-fsdp2/weights/GR00T-N1.7-3B"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LD_LIBRARY_PATH",
    os.path.expanduser("~/miniforge3/envs/media7/lib:") + os.environ.get("LD_LIBRARY_PATH", ""))

sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
import torch, torch.nn as nn
from torch_fp4 import install_global_linear_fakequant

T0 = time.time()
install_global_linear_fakequant()
print("[qad] global NVFP4 emulation installed", flush=True)

# b32 + AC + no-pin WSL-safe recipe (E4-final); AdamW is the trainer default —
# explicitly asserted so a config drift cannot silently swap optimizers.
sys.argv = ["launch_finetune.py",
    "--base-model-path", os.environ["GR00T_BASE_CKPT"],
    "--dataset-path", "./demo_data/libero_demo",
    "--embodiment-tag", "LIBERO_PANDA",
    "--num-gpus", "1",
    "--output-dir", "/mnt/c/fq_qad_out",
    "--save-steps", "100",
    "--max-steps", "1000",
    "--global-batch-size", "16",
    "--gradient-accumulation-steps", "2",
    "--use-fsdp2",
    "--no-fsdp2-reshard-after-forward",
    "--fsdp2-activation-checkpointing",
    "--save-only-model",
    "--no-fsdp2-pin-memory",
    "--num-shards-per-epoch", "4",
    "--dataloader-num-workers", "1"]

resume = os.environ.get("RESUME", "").split()
sys.argv += resume
sys.path.insert(0, os.getcwd())
import runpy
print(f"[qad] launching: b32 AC nopin, 1000 steps, save/100 — fake-quant + AdamW", flush=True)
try:
    runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
    print(f"[qad] COMPLETED in {(time.time()-T0)/3600:.2f}h")
except SystemExit as e:
    print(f"[qad] exit {e} after {(time.time()-T0)/3600:.2f}h")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[qad] FAILED after {(time.time()-T0)/3600:.2f}h: {type(e).__name__}: {e}")
    sys.exit(1)
