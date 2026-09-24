"""E1 baseline: lerobot-native pi0.5 PyTorch inference latency on RTX 5090 Laptop.

Counterpart to results/engine/pi05_bf16_sm120_v2.json (ApxInf engine, same GPU).
Run:  baselines/.venv-pi05/bin/python baselines/bench_pi05_lerobot.py
Out:  results/baselines/pi05_lerobot_pt_5090.json
"""
from __future__ import annotations
import json, statistics, subprocess, threading, time, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT = ROOT / "weights" / "pi05_libero_base"

class PowerSampler:
    def __init__(self, interval=0.2):
        self.samples, self.stop = [], False
    def _run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], text=True).strip().split("\n")[0]
                self.samples.append([float(x) for x in out.split(", ")])
            except Exception:
                pass
            time.sleep(interval if False else 0.2)
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self.stop = True; self.t.join(timeout=2)

def main():
    assert torch.cuda.is_available()
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    t0 = time.time()
    policy = PI05Policy.from_pretrained(str(CKPT))
    policy.eval().to("cuda")
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s; action_dim={getattr(policy.config, 'action_dim', '?')}")

    rng = np.random.default_rng(7)
    # keys/shapes from policy.config.input_features of pi05_libero_base
    def img(hw):
        return torch.from_numpy(rng.integers(0, 255, (1, 3, hw, hw))).to("cuda", torch.float32) / 255.0
    batch = {
        "observation.images.image": img(256),
        "observation.images.image2": img(256),
        "observation.images.empty_camera_0": img(224),
        "observation.state": torch.from_numpy(rng.uniform(-0.3, 0.3, (1, 8))).to("cuda", torch.float32),
        "task": ["put the moka pot on the stove"],
    }
    b = batch
    with torch.no_grad():
        a = policy.select_action(b)
        print("first action:", tuple(a.shape))

        for _ in range(10):
            policy.select_action(b)
        torch.cuda.synchronize()

        N = 30
        lat = []
        with PowerSampler() as ps:
            for i in range(N):
                b["observation.images.image"] = img(256)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                policy.select_action(b)
                torch.cuda.synchronize(); lat.append((time.perf_counter() - t1) * 1e3)

    lat.sort()
    p = lambda q: lat[min(N - 1, int(q * N))]
    power = [s[0] for s in ps.samples] or [0]
    vram = max((s[2] for s in ps.samples), default=0)
    res = {
        "model": "lerobot/pi05_libero_base", "engine": "pytorch-lerobot",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "load_s": round(load_s, 1), "n": N,
        "latency_ms_p50": round(p(.5), 2), "latency_ms_p99": round(p(.99), 2),
        "latency_ms_mean": round(statistics.mean(lat), 2),
        "hz_mean": round(1000 / statistics.mean(lat), 1),
        "vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "vram_mb_process_peak": round(vram),
        "power_w_mean": round(statistics.mean(power), 1), "power_w_max": round(max(power), 1),
        "lat_all_ms": [round(x, 2) for x in lat],
    }
    out = ROOT / "results" / "baselines"; out.mkdir(parents=True, exist_ok=True)
    (out / "pi05_lerobot_pt_5090.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "lat_all_ms"}, indent=1))

if __name__ == "__main__":
    from lerobot.policies.pi05 import PI05Policy
    main()
