"""E2-gated full-head smoke: NVFP4 fake-quant forward + 1 step backward under FSDP2.

Runs INSIDE sister env (groot-fsdp2/Isaac-GR00T/.venv), cwd=Isaac-GR00T.
Injects GPU-side NVFP4 weight emulation into nn.Linear.forward globally, then
invokes their launch_finetune with --use-fsdp2 --max-steps 1.

Run:
  cd ~/codebase/groot-fsdp2/Isaac-GR00T && \\
  .venv/bin/python ~/codebase/fp4vla/rl/full_head_smoke.py 2>&1 | tail -30
"""
import os, sys, time

os.environ.setdefault("GR00T_REPO", os.path.expanduser("~/codebase/groot-fsdp2/Isaac-GR00T"))
os.environ.setdefault("GR00T_BASE_CKPT", os.path.expanduser("~/codebase/groot-fsdp2/weights/GR00T-N1.7-3B"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LD_LIBRARY_PATH",
    os.path.expanduser("~/miniforge3/envs/media7/lib:") + os.environ.get("LD_LIBRARY_PATH", ""))

sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
import torch
import torch.nn as nn
from torch_fp4 import install_global_linear_fakequant

T0 = time.time()
n_linear = [0]
orig_init = nn.Linear.__init__
def counting_init(self, *a, **k):
    orig_init(self, *a, **k); n_linear[0] += 1
nn.Linear.__init__ = counting_init

install_global_linear_fakequant()
print("[smoke] global nn.Linear NVFP4 emulation installed", flush=True)

sys.argv = ["launch_finetune.py",
    "--base-model-path", os.environ["GR00T_BASE_CKPT"],
    "--dataset-path", "./demo_data/libero_demo",
    "--embodiment-tag", "LIBERO_PANDA",
    "--num-gpus", "1",
    "--output-dir", os.path.expanduser("~/fq_smoke_out"),   # survives /tmp wipes
    "--save-steps", "100000",
    "--max-steps", "1",
    "--global-batch-size", "1",      # incident#2 lesson: offload RAM ramps; start tiny
    "--use-fsdp2",
    "--dataloader-num-workers", "0"]

sys.path.insert(0, os.getcwd())
import runpy
print(f"[smoke] launching finetune (1 step, bs=1); Linears so far={n_linear[0]}", flush=True)
try:
    runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
    print(f"[smoke] COMPLETED in {time.time()-T0:.0f}s (fwd+bwd under fake-quant OK)")
except SystemExit as e:
    print(f"[smoke] exit {e} after {time.time()-T0:.0f}s")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[smoke] FAILED after {time.time()-T0:.0f}s: {type(e).__name__}: {e}")
    sys.exit(1)
