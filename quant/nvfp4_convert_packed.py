"""Packed-name artifact generator: quantize engine-layout (PACKED) weights.

Why: the engine's Gemma executor packs qkv ([q;k;v] row-concat) and gate_up
([gate;up] row-concat). For the VLM backbone it also FOLDS the learned RMSNorm
multiplier (1+gamma) into consuming weights' input channels (host.rs language
path); the action expert (DiT) uses runtime ada-norm with NO folding.

Emitted names (engine-internal convention, for Fp4Blocks gemm_maybe_fp4):
  paligemma.model.language_model.layers.{i}.qkv.weight / .gate_up.weight  (folded)
  paligemma...vision...encoder.layers.{i}.qkv.weight                      (no folding)
  gemma_expert.model.layers.{i}.qkv.weight / .gate_up.weight              (no folding)

Run: .venv-pi05/bin/python quant/nvfp4_convert_packed.py \
       --ckpt weights/pi05_libero_base/model.safetensors \
       --out weights/pi05.nvfp4.packed
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nvfp4_convert import convert_one
from safetensors import safe_open

ROOT = "paligemma_with_expert"
LANG = f"{ROOT}.paligemma.model.language_model.layers"
ACT = f"{ROOT}.gemma_expert.model.layers"
VISION = f"{ROOT}.paligemma.model.vision_tower.vision_model.encoder.layers"

def fold_scale(W: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return (W.astype(np.float64) * (1.0 + gamma.astype(np.float64))[None, :]).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--lang-depth", type=int, default=27)
    ap.add_argument("--act-depth", type=int, default=32)
    ap.add_argument("--vision-depth", type=int, default=27)
    ap.add_argument("--chunk-rows", type=int, default=512,
                    help="CPU quantization workspace bound; does not change packed values")
    ap.add_argument("--scope", choices=["all","lang","act","vision","lang+act"],
                    default="all", help="sensitivity-grid exemption artifacts: which tree(s) to quantize")
    ap.add_argument("--skip-first", type=int, default=0, help="skip first N layers of each tree (exemption)")
    ap.add_argument("--skip-last", type=int, default=0, help="skip last N layers of each tree")
    args = ap.parse_args()

    if args.chunk_rows < 1 or min(args.lang_depth, args.act_depth, args.vision_depth) < 0:
        ap.error("chunk-rows must be positive and depths nonnegative")
    if args.out.exists():
        ap.error(f"refusing existing output: {args.out}")
    # This converter targets the native pi0.5 float32 source checkpoint.
    # Keep the safetensors map open and materialize only projections needed now.
    with safe_open(str(args.ckpt), framework="numpy") as source:
        class TensorView:
            def __getitem__(self, name):
                if name not in source.keys():
                    raise KeyError(name)
                return source.get_tensor(name)
        print(f"opened {len(source.keys())} source tensors (lazy CPU reads)", flush=True)
        convert_checkpoint(args, TensorView())


def convert_checkpoint(args, tensors):
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = []

    def emit(name: str, W: np.ndarray):
        packed, phys, tscale, rel = convert_one(W, chunk_rows=args.chunk_rows)
        stem = name.replace("/", ".")
        packed.tofile(args.out / f"{stem}.packed.u8")
        phys.tofile(args.out / f"{stem}.scale.u8")
        manifest.append({"name": name, "shape": list(W.shape), "block": 16,
                         "tscale": float(tscale), "rel_frob": round(rel, 5)})

    def in_scope(i, depth):
        if i < args.skip_first or i >= depth - args.skip_last:
            return False
        return args.scope in ("all", "lang", "lang+act")

    for i in range(args.lang_depth):
        if not in_scope(i, args.lang_depth):
            continue
        p = f"{LANG}.{i}"
        try:
            g_in = tensors[f"{p}.input_layernorm.weight"]
            g_post = tensors[f"{p}.post_attention_layernorm.weight"]
            q = tensors[f"{p}.self_attn.q_proj.weight"]; k = tensors[f"{p}.self_attn.k_proj.weight"]; v = tensors[f"{p}.self_attn.v_proj.weight"]
            emit(f"{p}.qkv.weight", np.concatenate(
                [fold_scale(q, g_in), fold_scale(k, g_in), fold_scale(v, g_in)], axis=0))
            gate = tensors[f"{p}.mlp.gate_proj.weight"]; up = tensors[f"{p}.mlp.up_proj.weight"]
            emit(f"{p}.gate_up.weight", np.concatenate(
                [fold_scale(gate, g_post), fold_scale(up, g_post)], axis=0))
            if i % 8 == 0: print(f"  lang {i}", flush=True)
        except KeyError as e:
            raise ValueError(f"missing required language layer {i}: {e}") from e

    for i in range(args.act_depth):
        if args.scope not in ("all", "act", "lang+act") or i < args.skip_first or i >= args.act_depth - args.skip_last:
            continue
        p = f"{ACT}.{i}"
        try:
            q = tensors[f"{p}.self_attn.q_proj.weight"]; k = tensors[f"{p}.self_attn.k_proj.weight"]; v = tensors[f"{p}.self_attn.v_proj.weight"]
            emit(f"{p}.qkv.weight", np.concatenate([q, k, v], axis=0))
            gate = tensors[f"{p}.mlp.gate_proj.weight"]; up = tensors[f"{p}.mlp.up_proj.weight"]
            emit(f"{p}.gate_up.weight", np.concatenate([gate, up], axis=0))
            if i % 8 == 0: print(f"  act {i}", flush=True)
        except KeyError as e:
            raise ValueError(f"missing required action layer {i}: {e}") from e

    for i in range(args.vision_depth):
        if args.scope not in ("all", "vision") or i < args.skip_first or i >= args.vision_depth - args.skip_last:
            continue
        p = f"{VISION}.{i}"
        try:
            q = tensors[f"{p}.self_attn.q_proj.weight"]; k = tensors[f"{p}.self_attn.k_proj.weight"]; v = tensors[f"{p}.self_attn.v_proj.weight"]
            emit(f"{p}.qkv.weight", np.concatenate([q, k, v], axis=0))
            if i % 9 == 0: print(f"  vision {i}", flush=True)
        except KeyError as e:
            raise ValueError(f"missing required vision layer {i}: {e}") from e

    total_packed = sum(m["shape"][0]*m["shape"][1]//2
                       + 512*(((m["shape"][1]+15)//16+3)//4)*((m["shape"][0]+127)//128)
                       for m in manifest)
    summary = {"tensors": len(manifest),
               "mean_rel_frob": round(float(np.mean([m["rel_frob"] for m in manifest])), 5),
               "orig_bytes": sum(m["shape"][0]*m["shape"][1]*4 for m in manifest),
               "packed_bytes": total_packed}
    (args.out / "manifest.json").write_text(json.dumps({"summary": summary, "tensors": manifest}, indent=1))
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
