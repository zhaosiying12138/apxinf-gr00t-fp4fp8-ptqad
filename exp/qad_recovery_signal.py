"""QAD recovery signal: PTQ vs QAD-deployed action correlation vs BF16 teacher.

QAD-deployed = Q(W_qad): the QAD-trained weights quantized to NVFP4 (what the
engine would run). PTQ = Q(W_orig). Teacher = BF16 base checkpoint actions.
Recovery = corr(QAD, teacher) - corr(PTQ, teacher) on identical observations.
GR00T N1.7 via the official PyTorch policy; fake-quant via torch_fp4.
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "0")
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/rl")
import numpy as np, torch

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from torch_fp4 import install_global_linear_fakequant, fake_quant_nvfp4_torch
import torch.nn as _nn
ORIG_LINEAR_FWD = _nn.Linear.forward

COSMOS = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561")
rng = np.random.default_rng(42)

def obs():
    rr = np.random.default_rng(42)
    st = rr.uniform(-0.3, 0.3, 8).astype(np.float32)
    g = {"x": st[0:1], "y": st[1:2], "z": st[2:3], "roll": st[3:4], "pitch": st[4:5],
         "yaw": st[5:6], "gripper": st[6:8]}
    return {
        "video": {
            "image": rr.integers(0, 255, (1, 1, 224, 224, 3)).astype(np.uint8),
            "wrist_image": rr.integers(0, 255, (1, 1, 224, 224, 3)).astype(np.uint8),
        },
        "state": {k: v.reshape(1, 1, -1) for k, v in g.items()},
        "language": {"annotation.human.action.task_description":
                     [["put the moka pot on the stove"]]},
    }

def acts(model_dir, quant_action_head=False, quant_all=False, seed=42):
    np.random.seed(seed)
    p = Gr00tPolicy(model_path=model_dir,
                    embodiment_tag=EmbodimentTag.LIBERO_PANDA, device="cuda:0", strict=False)
    p.model.eval()
    if quant_all or quant_action_head:
        import torch.nn as nn
        orig = nn.Linear.forward
        def patched(self, x):
            w = self.weight
            if w.ndim == 2 and w.shape[1] % 16 == 0 and w.is_cuda and w.abs().max() < 100:
                wq = fake_quant_nvfp4_torch(w)
                return nn.functional.linear(x, wq, self.bias)
            return ORIG_LINEAR_FWD(self, x)
        nn.Linear.forward = patched
    with torch.no_grad():
        out, _info = p.get_action(obs())   # ({modality_key: (B,T,D) ndarray}, info)
        a = np.asarray(next(iter(out.values())), dtype=np.float32)
    import torch.nn as nn
    nn.Linear.forward = ORIG_LINEAR_FWD
    del p; torch.cuda.empty_cache()
    return a

BASE = "/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10"
QAD  = "/home/zhaosiying/codebase/fp4vla/weights/gr00t.qad1000"

print("== teacher (BF16 base) ==", flush=True)
a_teacher = acts(BASE)
print("teacher actions:", a_teacher.shape, a_teacher.flatten()[:4].round(3), flush=True)

print("== PTQ: Q(W_orig) all-linear ==", flush=True)
a_ptq = acts(BASE, quant_all=True)

print("== QAD-deployed: Q(W_qad) all-linear ==", flush=True)
a_qad = acts(QAD, quant_all=True)

print("== QAD weights BF16 (upper sanity) ==", flush=True)
a_qadbf = acts(QAD)

def corr(x, y): return float(np.corrcoef(x.flatten(), y.flatten())[0, 1])
print(f"\ncorr(PTQ, teacher)        = {corr(a_ptq, a_teacher):.4f}")
print(f"corr(QAD-deployed, teacher) = {corr(a_qad, a_teacher):.4f}")
print(f"corr(QAD-bf16, teacher)     = {corr(a_qadbf, a_teacher):.4f}")
print(f"RECOVERY GAIN = {corr(a_qad, a_teacher) - corr(a_ptq, a_teacher):+.4f}")
