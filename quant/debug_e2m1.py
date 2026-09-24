import numpy as np
import sys
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
from fp4_quant import nvfp4_quantize, nvfp4_dequantize

rng = np.random.default_rng(7); W = rng.normal(0, 0.05, (256, 512)).astype(np.float32)
packed, scales, ts = nvfp4_quantize(W.astype(np.float64))
Wq = nvfp4_dequantize(packed, scales, ts)
print("numpy full-pipeline rel:", round(np.linalg.norm(Wq-W)/np.linalg.norm(W), 4))

w = W.reshape(256, 32, 16); amax = np.abs(w).max(2); s = amax / 6
grids = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6]); mids = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
norm = w / s[..., None]
idx = np.clip(np.searchsorted(mids, np.abs(norm)), 0, 6)
q = grids[idx] * np.sign(norm)
print("numpy pure-e2m1-perfect-scale rel:", round(np.linalg.norm(q*s[...,None]-w)/np.linalg.norm(w), 4))
print("normalized |v| quantiles:", np.quantile(np.abs(norm), [0.5, 0.9, 0.99, 1.0]).round(2))
