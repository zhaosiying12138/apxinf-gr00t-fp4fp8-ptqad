"""Single-site artifact verification: dequantize one packed language qkv and
compare matmul vs raw-checkpoint reference (with fold). Isolates artifact
correctness from Rust routing."""
import sys, numpy as np
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
from fp4_quant import nvfp4_dequantize
from safetensors import safe_open

ART = "/home/zhaosiying/codebase/fp4vla/weights/pi05.nvfp4.packed"
CKPT = "/home/zhaosiying/codebase/fp4vla/weights/pi05_libero_base/model.safetensors"
LAYER = 0
p = f"paligemma_with_expert.paligemma.model.language_model.layers.{LAYER}"

# dequantize packed artifact (rows x K packed -> [3H, D])
with open(f"{ART}/{p}.qkv.weight.packed.u8", "rb") as f:
    packed = np.frombuffer(f.read(), dtype=np.uint8)
with open(f"{ART}/{p}.qkv.weight.scale.u8", "rb") as f:
    swz = np.frombuffer(f.read(), dtype=np.uint8)
import json
meta = [t for t in json.load(open(f"{ART}/manifest.json"))["tensors"] if t["name"].endswith(f"layers.{LAYER}.qkv.weight")][0]
rows, K = meta["shape"]
KB = K // 16
# inverse swizzle -> logical scales [rows, KB]
PS = 512 * ((KB + 3) // 4)
logical = np.zeros((rows, KB), dtype=np.uint8)
for r in range(rows):
    for b in range(KB):
        logical[r, b] = swz[PS * (r // 128) + 512 * (b // 4) + 16 * (r % 32) + 4 * ((r // 32) % 4) + (b % 4)]
Wq = nvfp4_dequantize(packed.reshape(rows, K // 2), logical, np.float32(meta["tscale"]))
print("artifact dequant:", Wq.shape, "rel-frob:", meta["rel_frob"])

# reference: concat raw q,k,v with gamma fold along input channels
with safe_open(CKPT, framework="pt") as f:
    g = f.get_tensor(f"{p}.input_layernorm.weight").float().numpy()
    q = f.get_tensor(f"{p}.self_attn.q_proj.weight").float().numpy()
    k = f.get_tensor(f"{p}.self_attn.k_proj.weight").float().numpy()
    v = f.get_tensor(f"{p}.self_attn.v_proj.weight").float().numpy()
Wref = np.concatenate([(1 + g)[None, :] * q, (1 + g)[None, :] * k, (1 + g)[None, :] * v], axis=0)
print("reference:", Wref.shape)
rel = np.linalg.norm(Wq - Wref) / np.linalg.norm(Wref)
print(f"WEIGHT check: rel-frob vs ref = {rel:.4f} (expect ~0.095)")

# functional check: random activation through both
rng = np.random.default_rng(3)
x = rng.normal(0, 1, (64, K)).astype(np.float32)
y_art = x @ Wq.T
y_ref = x @ Wref.T
corr = np.corrcoef(y_art.flatten(), y_ref.flatten())[0, 1]
print(f"GEMM check: corr = {corr:.5f} (expect >0.999)")
print("ARTIFACT VERDICT:", "PASS" if corr > 0.999 else "FAIL -> bug is IN THE ARTIFACT")
