"""Offline NVFP4 checkpoint converter: pi05 safetensors -> FP4 artifact.

Produces, per eligible 2-D weight (out_features x in_features, K-major as the
engine consumes):
  <name>.packed.u8   (rows x K/2, E2M1 pairs, low nibble = even k)
  <name>.scale.u8    (physical cuBLASLt swizzle layout, per-16 E4M3)
  <name>.meta.json   (tscale fp32, shape, rel-Frobenius error, block=16)

Artifacts land in weights/<tag>.nvfp4/ with a top-level manifest.json.
Bit-exact with quant/fp4_quant.py (verified vs spike hardware semantics).

Run:  uv run --with numpy,safetensors python quant/nvfp4_convert.py \
        --ckpt weights/pi05_libero_base/model.safetensors --out weights/pi05.nvfp4 \
        [--limit 8 --only backbone]     # small runs for smoke tests
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fp4_quant import nvfp4_quantize, nvfp4_dequantize, swizzle_scales

def iter_safetensors(path: pathlib.Path):
    """Lazy per-tensor iterator (name, np.ndarray) — never holds the whole ckpt.
    bf16 checkpoints (GR00T) need the torch backend; numpy has no bfloat16."""
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework="numpy") as f:
            for name in sorted(f.keys()):
                yield name, f.get_tensor(name)
    except TypeError:
        import torch
        with safe_open(str(path), framework="pt") as f:
            for name in sorted(f.keys()):
                t = f.get_tensor(name)
                yield name, t.float().numpy()

def convert_one(W: np.ndarray, chunk_rows: int = 8192):
    """Chunked float32 quantization. tscale FORCED to 1.0: the hardware GEMM
    applies only the per-block E4M3 bytes, so double quantization (encoding
    amax/6/tscale) loses the tscale factor — root cause of the 2026-09-25
    action-decorrelation bug. amax/6 fits E4M3 dynamic range for VLA weights."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError("2-D only")
    rows, K = W.shape
    if K % 16 != 0:
        raise ValueError(f"K={K} not multiple of 16 (pad upstream or skip)")
    packed = np.empty((rows, K // 2), dtype=np.uint8)
    KB = K // 16
    scales = np.empty((rows, KB), dtype=np.uint8)
    sq_sum_w = 0.0; sq_sum_d = 0.0
    tscale = np.float32(1.0)
    for r0 in range(0, rows, chunk_rows):
        w = W[r0:r0 + chunk_rows].astype(np.float64)
        p, s, _ = nvfp4_quantize(w, block=16, tscale=tscale)
        packed[r0:r0 + chunk_rows] = p; scales[r0:r0 + chunk_rows] = s
        d = nvfp4_dequantize(p, s, tscale)
        sq_sum_w += float((w * w).sum()); sq_sum_d += float(((d - w) ** 2).sum())
    rel = float(np.sqrt(sq_sum_d) / (np.sqrt(sq_sum_w) + 1e-12))
    return packed, swizzle_scales(scales), tscale, rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--only", default=None, help="substring filter on tensor names")
    ap.add_argument("--limit", type=int, default=None, help="convert first N eligible")
    ap.add_argument("--exclude", default=None, help="exclude substring (e.g. norm/embed)")
    args = ap.parse_args()

    tensors = None  # lazy iteration below
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, done = [], 0
    total_rel, n_ok = [], 0
    for name, W in iter_safetensors(args.ckpt):
        if W.ndim != 2: continue
        if W.shape[-1] % 16: continue
        if args.only and args.only not in name: continue
        if args.exclude and args.exclude in name: continue
        if args.limit and done >= args.limit: break
        try:
            packed, phys, tscale, rel = convert_one(W)
        except Exception as e:
            manifest.append({"name": name, "error": str(e)[:100]}); continue
        stem = name.replace("/", ".")
        packed.tofile(args.out / f"{stem}.packed.u8")
        phys.tofile(args.out / f"{stem}.scale.u8")
        manifest.append({
            "name": name, "shape": list(W.shape), "block": 16,
            "tscale": float(tscale), "rel_frob": round(rel, 5),
            "packed_bytes": int(packed.nbytes), "scale_bytes": int(phys.nbytes),
            "orig_bytes": int(W.nbytes),
        })
        total_rel.append(rel); n_ok += 1; done += 1
        if done % 25 == 0:
            print(f"  {done} tensors, mean rel-err {np.mean(total_rel):.4f}")

    summary = {
        "ckpt": str(args.ckpt),
        "tensors": n_ok,
        "mean_rel_frob": round(float(np.mean(total_rel)), 5) if total_rel else None,
        "max_rel_frob": round(float(np.max(total_rel)), 5) if total_rel else None,
        "orig_bytes": sum(m.get("orig_bytes", 0) for m in manifest),
        "packed_bytes": sum(m.get("packed_bytes", 0) + m.get("scale_bytes", 0) for m in manifest),
    }
    summary["compression"] = round(summary["orig_bytes"] / max(1, summary["packed_bytes"]), 2)
    (args.out / "manifest.json").write_text(json.dumps(
        {"summary": summary, "tensors": manifest}, indent=1))
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
