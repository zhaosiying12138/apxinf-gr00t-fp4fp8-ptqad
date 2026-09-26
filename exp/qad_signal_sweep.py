"""N-observation QAD recovery signal sweep (statistics for paper §2.6).

Loads each policy ONCE (teacher / PTQ / QAD-deployed / QAD-bf16) and runs a
seeded battery of observations through all four, reporting mean±std corr.
GPU ~10GB, ~15-20 min for N=16.
Run: cd ~/codebase/groot-fsdp2/Isaac-GR00T && LD_LIBRARY_PATH=$HOME/miniforge3/envs/media7/lib \
     .venv/bin/python ~/codebase/fp4vla/exp/qad_signal_sweep.py
"""
import os, sys, json, statistics as st
os.environ.setdefault("HF_HUB_OFFLINE", "0")
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
import numpy as np, torch
import torch.nn as nn
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from torch_fp4 import fake_quant_nvfp4_torch

ORIG = nn.Linear.forward
N = 16

def obs(seed):
    rr = np.random.default_rng(seed)
    s = rr.uniform(-0.3, 0.3, 8).astype(np.float32)
    g = {"x": s[0:1], "y": s[1:2], "z": s[2:3], "roll": s[3:4], "pitch": s[4:5],
         "yaw": s[5:6], "gripper": s[6:8]}
    return {
        "video": {
            "image": rr.integers(0, 255, (1, 1, 224, 224, 3)).astype(np.uint8),
            "wrist_image": rr.integers(0, 255, (1, 1, 224, 224, 3)).astype(np.uint8),
        },
        "state": {k: v.reshape(1, 1, -1) for k, v in g.items()},
        "language": {"annotation.human.action.task_description":
                     [["put the moka pot on the stove"]]},
    }

def patched(self, x):
    w = self.weight
    if w.ndim == 2 and w.shape[1] % 16 == 0 and w.is_cuda and w.abs().max() < 100:
        return nn.functional.linear(x, fake_quant_nvfp4_torch(w), self.bias)
    return ORIG(self, x)

def battery(model_dir, quant, seeds):
    p = Gr00tPolicy(model_path=model_dir, embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                    device="cuda:0", strict=False)
    p.model.eval()
    nn.Linear.forward = patched if quant else ORIG
    acts = []
    with torch.no_grad():
        for s in seeds:
            out, _ = p.get_action(obs(s))
            acts.append(np.asarray(next(iter(out.values())), dtype=np.float32))
    nn.Linear.forward = ORIG
    del p; torch.cuda.empty_cache()
    return acts

BASE = "/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10"
QAD  = "/home/zhaosiying/codebase/fp4vla/weights/gr00t.qad1000"
seeds = list(range(1000, 1000 + N))

def corr(x, y): return float(np.corrcoef(x.flatten(), y.flatten())[0, 1])

print(f"== teacher ({N} obs) ==", flush=True)
T = battery(BASE, False, seeds)
print("== PTQ ==", flush=True);      P = battery(BASE, True, seeds)
print("== QAD-deployed ==", flush=True); Q = battery(QAD, True, seeds)
print("== QAD-bf16 ==", flush=True);  B = battery(QAD, False, seeds)

r_ptq = [corr(p, t) for p, t in zip(P, T)]
r_qad = [corr(q, t) for q, t in zip(Q, T)]
r_qbf = [corr(b, t) for b, t in zip(B, T)]
Tf = np.concatenate([t.flatten() for t in T])
Pf = np.concatenate([p_.flatten() for p_ in P])
Qf = np.concatenate([q.flatten() for q in Q])
Bf = np.concatenate([b.flatten() for b in B])
pooled = {"ptq":   float(np.corrcoef(Pf, Tf)[0, 1]),
          "qad":   float(np.corrcoef(Qf, Tf)[0, 1]),
          "qadbf": float(np.corrcoef(Bf, Tf)[0, 1])}
l2 = {"ptq":   float(np.linalg.norm(Pf - Tf) / (np.linalg.norm(Tf) + 1e-9)),
      "qad":   float(np.linalg.norm(Qf - Tf) / (np.linalg.norm(Tf) + 1e-9)),
      "qadbf": float(np.linalg.norm(Bf - Tf) / (np.linalg.norm(Tf) + 1e-9))}
res = {"n": N, "seeds": seeds,
       "ptq":   {"mean": st.mean(r_ptq), "std": st.stdev(r_ptq), "all": r_ptq},
       "qad":   {"mean": st.mean(r_qad), "std": st.stdev(r_qad), "all": r_qad},
       "qadbf": {"mean": st.mean(r_qbf), "std": st.stdev(r_qbf), "all": r_qbf},
       "gain": st.mean(r_qad) - st.mean(r_ptq),
       "pooled_corr": pooled, "rel_l2": l2}
path = "/home/zhaosiying/codebase/fp4vla/results/r_qad_signal_sweep.json"
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(res, open(path, "w"), indent=1)
print(f"\nN={N} per-obs corr: PTQ {res['ptq']['mean']:+.4f}+/-{res['ptq']['std']:.4f} | "
      f"QAD {res['qad']['mean']:+.4f}+/-{res['qad']['std']:.4f} | "
      f"QAD-bf16 {res['qadbf']['mean']:+.4f}+/-{res['qadbf']['std']:.4f} | "
      f"GAIN {res['gain']:+.4f}")
print(f"POOLED corr (n={N * T[0].size}): PTQ {pooled['ptq']:+.4f} | "
      f"QAD {pooled['qad']:+.4f} | QAD-bf16 {pooled['qadbf']:+.4f}")
print(f"REL L2: PTQ {l2['ptq']:.4f} | QAD {l2['qad']:.4f} | QAD-bf16 {l2['qadbf']:.4f}")
