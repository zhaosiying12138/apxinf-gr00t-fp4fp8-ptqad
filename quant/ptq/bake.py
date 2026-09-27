"""Bake a PTQ checkpoint for GR00T N1.7 (Qwen3-VL backbone) — safetensors
rewritten in place-value form (BF16 tensors carrying exactly the NVFP4 / FP8
quantization noise), so the existing closed-loop eval stack runs them at
native BF16 speed with no code changes (same principle as rl/scoped_quant.py
quantize-once, but calibration-aware and persistent).

Recipes (--recipe), all weight-only:
  rtn      all-NVFP4 round-to-nearest, max calibration (0% baseline parity)
  calib    all-NVFP4 + per-layer MSE clip search + GPTQ Hessian rounding
  fp8      backbone FP8-E4M3 per-channel, head NVFP4 RTN
  mixed    NVIDIA Jetson AI Lab mixed_nvfp4 allocation (their LIBERO Spatial:
           97.5%->97.3%): ViT linears FP8; LLM q/k/v/gate/up NVFP4(GPTQ);
           LLM o_proj/down_proj FP8; lm_head FP8; DiT FFN NVFP4(GPTQ); other
           DiT linears FP8; timestep/action-decoder kept BF16.

AWQ per-channel folding was implemented and searched (quant/ptq/awq.py):
every site selected alpha=0 (no gain) — with per-16-block E4M3 scales the
channel-outlier structure is already absorbed; documented as a negative
result, folds therefore disabled here.
"""
import os, sys, json, time, shutil, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")

import torch
from safetensors.torch import load_file, save_file
from quantizers import nvfp4_dequant, fp8_e4m3_dequant, gptq_nvfp4, rtnc_best_clip, layer_mse_tr

CLIP_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)


def alloc_mixed(name, W):
    """NVIDIA mixed_nvfp4 precision allocation for GR00T N1.7."""
    if name.startswith("backbone.model.model.visual."):
        return "fp8"
    if name == "backbone.lm_head":
        return "fp8"
    if ".language_model.layers." in name:
        if name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj"):
            return "fp8"
        return "nvfp4_gptq"
    if name.startswith("action_head."):
        base = name.split("action_head.model.")[-1]
        if base.startswith("ff.net.0.proj") or base == "ff.net.2":
            return "nvfp4_gptq"
        if base.startswith("transformer_blocks.") or base == "proj_out_1":
            return "fp8"
        return "bf16"   # timestep embedder, proj_out_2 (action decoder), misc
    return "fp8"


def alloc_aggr(name, W):
    """Aggressive: NVFP4 everywhere except the two proven killers
    (LLM o_proj/down_proj -> FP8; timestep/action-decoder -> BF16)."""
    if name == "backbone.lm_head":
        return "nvfp4_gptq"
    if ".language_model.layers." in name:
        if name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj"):
            return "fp8"
        return "nvfp4_gptq"
    if name.startswith("action_head."):
        base = name.split("action_head.model.")[-1]
        if base.startswith("ff.net.0.proj") or base == "ff.net.2":
            return "nvfp4_gptq"
        if base.startswith("transformer_blocks.") or base == "proj_out_1":
            return "nvfp4_rtn"
        return "bf16"
    return "nvfp4_gptq"   # vision tower


def alloc(name, W, recipe):
    if recipe == "rtn" or recipe == "calib":
        return "nvfp4_gptq" if recipe == "calib" else "nvfp4_rtn"
    if recipe == "fp8":
        return "fp8" if "action_head" not in name else "nvfp4_rtn"
    if recipe == "mixed":
        return alloc_mixed(name, W)
    if recipe == "aggr":
        return alloc_aggr(name, W)
    raise ValueError(recipe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="mixed",
                    choices=["rtn", "calib", "fp8", "mixed", "aggr"])
    ap.add_argument("--calib", default="/mnt/c/fq_ptq_calib/calib.pt")
    ap.add_argument("--gptq-damp", type=float, default=0.01)
    ap.add_argument("--head-bf16", action="store_true",
                    help="keep action_head BF16 (backbone-only quant)")
    args = ap.parse_args()
    t0 = time.time()

    need_calib = args.recipe in ("calib", "mixed")
    calib = torch.load(args.calib, map_location="cpu") if need_calib else {}
    idx = json.load(open(f"{args.base}/model.safetensors.index.json"))
    wmap = idx["weight_map"]
    sd = {}
    for shard in sorted(set(wmap.values())):
        sd.update(load_file(f"{args.base}/{shard}"))
    print(f"[bake] base loaded: {len(sd)} tensors ({time.time()-t0:.0f}s)", flush=True)

    lin_names = [k[:-len(".weight")] for k, v in sd.items()
                 if k.endswith(".weight") and v.ndim == 2 and v.shape[1] % 16 == 0]
    edits, records = {}, {}
    for name in lin_names:
        W = sd[name + ".weight"].to("cuda").float()
        kind = alloc(name, W, args.recipe)
        if args.head_bf16 and name.startswith("action_head."):
            kind = "bf16"
        if kind == "bf16":
            records[name] = {"method": "bf16", "rel_mse": 0.0}
            continue
        if kind == "nvfp4_rtn":
            Wq = nvfp4_dequant(W)
            clip = 1.0
        elif kind == "nvfp4_gptq":
            if name in calib:
                H = calib[name]["H"].to("cuda")
                clip, _ = rtnc_best_clip(W, H, CLIP_GRID)
                Wq = gptq_nvfp4(W, H, clip=clip, damp=args.gptq_damp)
                del H
                torch.cuda.empty_cache()
            else:   # head layers: no calibration collected -> plain RTN
                clip = 1.0
                Wq = nvfp4_dequant(W, clip=clip)
        elif kind == "fp8":
            Wq = fp8_e4m3_dequant(W)
            clip = 1.0
        err = float(((Wq - W) ** 2).mean())
        edits[name + ".weight"] = Wq.to(sd[name + ".weight"].dtype).cpu()
        records[name] = {"method": kind, "clip": clip, "wmse": err}

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.base):
        src, dst = f"{args.base}/{f}", f"{args.out}/{f}"
        if f.endswith(".safetensors") or not os.path.isfile(src):
            continue
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    for shard in sorted(set(wmap.values())):
        tensors = load_file(f"{args.base}/{shard}")
        for k in list(tensors):
            if k in edits:
                tensors[k] = edits[k].to(tensors[k].dtype)
        save_file(tensors, f"{args.out}/{shard}", metadata={"format": "pt"})
    assert os.path.exists(f"{args.out}/{sorted(set(wmap.values()))[0]}")

    nv4 = fp8 = bf16 = 0
    for name, r in records.items():
        v = sd.get(name + ".weight")
        if v is None:
            continue
        if r["method"].startswith("nvfp4"):
            nv4 += v.numel()
        elif r["method"] == "fp8":
            fp8 += v.numel()
        else:
            bf16 += v.numel()
    lin_n = nv4 + fp8 + bf16
    nv4_b = nv4 * 0.5625   # 4 bits + 1B E4M3 scale per 16 block
    fp8_b = fp8 * 1.0002
    bf16_b = bf16 * 2
    recipe = {"recipe": args.recipe, "base": args.base,
              "n_weights_edited": len(edits), "layers": records,
              "memory": {"linear_params": lin_n,
                         "nvfp4_params": nv4, "fp8_params": fp8, "bf16_params": bf16,
                         "bf16_GB": round((lin_n * 2) / 1e9, 3),
                         "quantized_GB": round((nv4_b + fp8_b + bf16_b) / 1e9, 3),
                         "linear_compression_x": round((lin_n * 2) / (nv4_b + fp8_b + bf16_b), 2)},
              "calib": args.calib}
    json.dump(recipe, open(f"{args.out}/ptq_recipe.json", "w"), indent=1)
    print(f"[bake] {args.recipe}: {len(edits)} weights edited | "
          f"nvfp4={nv4/1e6:.0f}M fp8={fp8/1e6:.0f}M bf16={bf16/1e6:.0f}M | "
          f"linear {lin_n*2/1e9:.2f}GB -> {(nv4_b+fp8_b+bf16_b)/1e9:.2f}GB "
          f"({(lin_n*2)/(nv4_b+fp8_b+bf16_b):.2f}x) | {time.time()-t0:.0f}s -> {args.out}", flush=True)


def _unused_head_H():
    pass


if __name__ == "__main__":
    main()
