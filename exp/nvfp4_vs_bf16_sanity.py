"""Action-space sanity: same synthetic observation through BF16 vs nvfp4_static,
report per-step action deltas. Gate before LIBERO (expect correlated actions
with fp4-level noise, not divergent outputs)."""
import sys, numpy as np

def load(variant):
    from apxinf import AutoPolicy
    kw = {"norm_stats": "/home/zhaosiying/codebase/fp4vla/weights/pi05_libero_base/norm_stats.json",
         "model_variant": variant}
    return AutoPolicy.from_pretrained(
        "/home/zhaosiying/codebase/fp4vla/weights/pi05_libero_base", **kw)

rng = np.random.default_rng(7)

def obs(md):
    o = {k: rng2.integers(0, 255, (256, 256, 3)).astype(np.uint8) for k in md["image_keys"]}
    o[md["prompt_key"]] = "put the moka pot on the stove"
    if (sk := md.get("state_key")):
        o[sk] = np.full(md["state_dim"], 0.1, np.float32)
    return o

rng2 = np.random.default_rng(42)
p16 = load("bf16")
md = p16.metadata
o = obs(md)
a16 = np.asarray(p16.infer(o)["actions"])
rng2 = np.random.default_rng(42)
p4 = load("nvfp4_static")
o2 = obs(p4.metadata)
a4 = np.asarray(p4.infer(o2)["actions"])
p16.close(); p4.close()
print("bf16 actions:", a16.shape, "first:", a16[0][:3].round(4))
print("nvfp4 actions:", a4.shape, "first:", a4[0][:3].round(4))
d = np.linalg.norm(a16 - a4, axis=1)
scale = np.linalg.norm(a16, axis=1) + 1e-6
print(f"per-step L2 delta: mean={d.mean():.4f} max={d.max():.4f}")
print(f"relative: mean={(d/scale).mean():.3f} max={(d/scale).max():.3f}")
corr = np.corrcoef(a16.flatten(), a4.flatten())[0,1]
print(f"flatten corr: {corr:.4f}")
print("VERDICT:", "PASS" if corr > 0.9 else "INVESTIGATE")
