"""E1 baseline: NVIDIA official GR00T N1.7 PyTorch inference latency on RTX 5090 Laptop.

This is the number our ApxInf engine runs must beat. Measures end-to-end
policy.get_action() (processor + backbone + DiT action head + decode), batch 1,
LIBERO-10 embodiment (libero_sim, 8-dim state, 2x224x224 cameras).

Run:  baselines/.venv/bin/python baselines/bench_gr00t_pt.py
Out:  results/baselines/gr00t_n17_pt_5090.json
"""
from __future__ import annotations
import json, statistics, subprocess, threading, time, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT = ROOT / "weights" / "GR00T-N1.7-LIBERO" / "libero_10"

class PowerSampler:
    def __init__(self, interval=0.2):
        self.samples, self.stop, self.t = [], False, None
        self.interval = interval
    def _run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,clocks.sm",
                     "--format=csv,noheader,nounits"], text=True).strip().split("\n")[0]
                self.samples.append([float(x) for x in out.split(", ")])
            except Exception:
                pass
            time.sleep(self.interval)
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self.stop = True; self.t.join(timeout=2)

def main():
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    assert torch.cuda.is_available(), "CUDA required"
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    t0 = time.time()
    policy = Gr00tPolicy(
        model_path=str(CKPT),
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        device="cuda:0",
        strict=False,           # we feed synthetic obs; validation strictness off
    )
    policy.model.eval()   # Gr00tPolicy wraps model; .eval lives on the module
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s")

    rng = np.random.default_rng(7)
    # Gr00tPolicy batched observation contract (see gr00t_policy.py docstring):
    #   video: {key: uint8 (B,T,H,W,C)}, state: {key: f32 (B,T,D)},
    #   language: {key: [[str]] (B,T)}
    def make_obs():
        s = rng.uniform(-0.5, 0.5, 8).astype(np.float32)   # LIBERO 8-dim state order:
        groups = {"x": s[0:1], "y": s[1:2], "z": s[2:3],   # x y z roll pitch yaw gripper(2)
                  "roll": s[3:4], "pitch": s[4:5], "yaw": s[5:6], "gripper": s[6:8]}
        return {
            "video": {
                "image": rng.integers(0, 255, (1, 1, 224, 224, 3)).astype(np.uint8),
                "wrist_image": rng.integers(0, 255, (1, 1, 224, 224, 3)).astype(np.uint8),
            },
            "state": {k: v.reshape(1, 1, -1) for k, v in groups.items()},
            "language": {"annotation.human.action.task_description":
                         [["put the moka pot on the stove"]]},
        }
    obs = make_obs()
    with torch.no_grad():
        act = policy.get_action(obs)
    print("first action:", {k: np.asarray(v).shape for k, v in act.items()} if isinstance(act, dict) else np.asarray(act).shape)

    with torch.no_grad():
        for _ in range(10):
            policy.get_action(obs)
        torch.cuda.synchronize()

    N = 50
    lat = []
    with PowerSampler() as ps, torch.no_grad():
        for i in range(N):
            o = make_obs()
            torch.cuda.synchronize(); t1 = time.perf_counter()
            policy.get_action(o)
            torch.cuda.synchronize(); lat.append((time.perf_counter() - t1) * 1e3)

    lat.sort()
    p = lambda q: lat[min(N - 1, int(q * N))]
    power = [s[0] for s in ps.samples] or [0]
    util = [s[1] for s in ps.samples] or [0]
    res = {
        "model": "nvidia/GR00T-N1.7-LIBERO", "engine": "pytorch-official",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "load_s": round(load_s, 1),
        "n": N, "latency_ms_p50": round(p(0.5), 2), "latency_ms_p99": round(p(0.99), 2),
        "latency_ms_mean": round(statistics.mean(lat), 2),
        "hz_mean": round(1000.0 / statistics.mean(lat), 1),
        "vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "power_w_mean": round(statistics.mean(power), 1),
        "power_w_max": round(max(power), 1),
        "gpu_util_mean": round(statistics.mean(util), 0),
        "action_shape": list(np.asarray(act["action"]).shape) if isinstance(act, dict) else list(np.asarray(act).shape),
        "lat_all_ms": [round(x, 2) for x in lat],
    }
    out = ROOT / "results" / "baselines"; out.mkdir(parents=True, exist_ok=True)
    (out / "gr00t_n17_pt_5090.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "lat_all_ms"}, indent=1))

if __name__ == "__main__":
    main()
