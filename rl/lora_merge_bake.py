"""Merge a QAD/OPD-LoRA checkpoint into a deployable PTQ-baked checkpoint.

Training semantics (rl/lora_qad.py): LoRA Linears compute
    y = x @ Q(W + (alpha/r) B A)^T   per forward (NVFP4 fake-quant)
Non-LoRA in-scope Linears were quantize-once'd in-place, so their saved
weights already carry quantization values. This script rewrites each LoRA
layer to  W <- fake_quant(W + (alpha/r) B A)  once and drops the lora_*
parameters, producing a plain checkpoint the eval server loads unchanged.
"""
import os, sys, json, shutil, argparse

sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/quant")
import torch
from safetensors.torch import load_file, save_file
from torch_fp4 import fake_quant_nvfp4_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LoRA training checkpoint dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    args = ap.parse_args()

    idx_path = f"{args.ckpt}/model.safetensors.index.json"
    single = f"{args.ckpt}/model.safetensors"
    shards = ["model.safetensors"] if os.path.exists(single) else None
    if shards is None:
        idx = json.load(open(idx_path))
        wmap = idx["weight_map"]
        shards = sorted(set(wmap.values()))
    else:
        idx = None
        wmap = {}

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.ckpt):
        src, dst = f"{args.ckpt}/{f}", f"{args.out}/{f}"
        if f.endswith(".safetensors") or not os.path.isfile(src):
            continue
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # load ALL shards first: lora_A and its base weight may live in different
    # shards — merge globally, then write shards back per the original index.
    sd = {}
    shard_keys = {}
    for shard in shards:
        tensors = load_file(f"{args.ckpt}/{shard}")
        shard_keys[shard] = list(tensors.keys())
        sd.update(tensors)

    merged = skipped = 0
    for k in list(sd.keys()):
        if not k.endswith(".lora_A"):
            continue
        base = k[: -len(".lora_A")] + ".weight"
        kb = k[: -len(".lora_A")] + ".lora_B"
        assert kb in sd and base in sd, f"lora_B/base missing for {base}"
        # v2 additive merge: saved base weight already carries the baked
        # quantization values; the low-rank residual is added in BF16 and
        # NOT requantized (deploy = quantized base + BF16 low-rank branch).
        W = sd[base].float()
        A = sd[k].float()
        B = sd[kb].float()
        sd[base] = (W + (args.alpha / args.rank) * (B @ A)).to(sd[base].dtype)
        del sd[k], sd[kb]
        merged += 1
    for k in list(sd.keys()):
        if ".lora_" in k:
            del sd[k]
            skipped += 1

    for shard in shards:
        tensors = {}
        for k in shard_keys[shard]:
            if k in sd:
                tensors[k] = sd[k]
        save_file(tensors, f"{args.out}/{shard}", metadata={"format": "pt"})
    print(f"[merge] {merged} LoRA layers merged (additive, no requant), "
          f"{skipped} stragglers dropped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
