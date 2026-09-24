"""Engine L2 smoke/bench: AutoPolicy (engine main API) latency on sm_120.

Works for any AutoPolicy-supported model dir (pi05 now; gr00t after Cosmos token).
Observations are synthetic; measures end-to-end policy.infer() incl. preprocess.

Run (from ~, NOT from inside apxinf-robo, to avoid submodule shadowing):
  cd ~ && <robo-venv>/python ~/codebase/fp4vla/exp/bench_engine.py \
       --model-dir ~/codebase/fp4vla/weights/pi05_libero_base --variant bf16
Out: results/engine/<tag>.json
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, subprocess, threading, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

class PowerSampler:
    def __init__(self, interval=0.2):
        self.samples, self.stop = [], False
        self.interval = interval
    def _run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used",
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--variant", default="bf16")
    ap.add_argument("--extra-kwarg", action="append", default=[])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from apxinf import AutoPolicy
    kw = dict(kv.split("=", 1) for kv in args.extra_kwarg)
    for k, v in kw.items():
        if v.isdigit(): kw[k] = int(v)
        elif v.replace(".", "", 1).isdigit(): kw[k] = float(v)
    t0 = time.time()
    policy = AutoPolicy.from_pretrained(args.model_dir, model_variant=args.variant, **kw)
    load_s = time.time() - t0
    md = policy.metadata
    print(f"loaded {load_s:.1f}s | type={md.get('model_type')} action_dim={md.get('action_dim')} "
          f"horizon={md.get('action_horizon')} img_keys={md.get('image_keys')}")

    rng = np.random.default_rng(7)
    obs = {k: rng.integers(0, 255, (256, 256, 3)).astype(np.uint8) for k in md["image_keys"]}
    obs[md["prompt_key"]] = "put the moka pot on the stove"
    if sk := md.get("state_key"):
        obs[sk] = rng.uniform(-0.3, 0.3, md["state_dim"]).astype(np.float32)

    r = policy.infer(obs)
    a0 = r["actions"]; t0m = r.get("timing", {})
    print(f"first infer ok: actions {np.asarray(a0).shape}, timing {t0m}")

    for _ in range(args.warmup):
        policy.infer(obs)

    lat_m, lat_t = [], []
    with PowerSampler() as ps:
        for i in range(args.samples):
            o = dict(obs)
            for k in md["image_keys"]:
                o[k] = rng.integers(0, 255, (256, 256, 3)).astype(np.uint8)
            r = policy.infer(o)
            lat_m.append(r["timing"]["model_ms"]); lat_t.append(r["timing"]["total_ms"])
    lat_m.sort(); lat_t.sort()
    p = lambda x, q: x[min(len(x)-1, int(q*len(x)))]
    power = [s[0] for s in ps.samples] or [0]
    vram = max((s[2] for s in ps.samples), default=0)
    res = {
        "model_dir": str(args.model_dir), "variant": args.variant,
        "model_type": md.get("model_type"), "load_s": round(load_s, 1),
        "n": args.samples,
        "model_ms_p50": round(p(lat_m, .5), 2), "model_ms_p99": round(p(lat_m, .99), 2),
        "total_ms_p50": round(p(lat_t, .5), 2), "total_ms_p99": round(p(lat_t, .99), 2),
        "hz_p50": round(1000/p(lat_t, .5), 1),
        "power_w_mean": round(statistics.mean(power), 1), "power_w_max": round(max(power), 1),
        "vram_mb_peak": round(vram),
        "lat_model_ms": [round(x,2) for x in lat_m], "lat_total_ms": [round(x,2) for x in lat_t],
    }
    out = ROOT/"results"/"engine"; out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{res['model_type']}_{args.variant}"
    (out/f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("lat_")}, indent=1))
    policy.close()

if __name__ == "__main__":
    main()
