"""Load fp4vla streaming-save shards back into a single HF-format checkpoint.

Input: checkpoint dir with fp4vla-shard-*.safetensors + fp4vla_shard_index.json
       (params saved fp32; action-head trained, backbone frozen)
Output: HF-style model dir (model.safetensors bf16 + config copies) usable by
        the packed converter (quant/nvfp4_convert_packed.py) for requantization
        and by the GR00T PyTorch policy for QAD-vs-PTQ eval.

Usage:
  python rl/load_qad_shards.py --ckpt /mnt/c/fq_qad_out/checkpoint-1000 \
      --out weights/gr00t.qad1000 [--bf16]
"""
from __future__ import annotations
import argparse, json, pathlib, shutil
import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    index = json.loads((args.ckpt / "fp4vla_shard_index.json").read_text())
    shards = sorted(set(index.values()))
    sd = {}
    for shard in shards:
        part = load_file(str(args.ckpt / shard))
        sd.update(part)
    assert set(sd.keys()) == set(index.keys()), "shard index mismatch"

    if args.bf16:
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(args.out / "model.safetensors"),
              metadata={"format": "pt"})
    # carry over configs/processor assets from the source checkpoint
    for name in ["config.json", "embodiment_id.json", "statistics.json",
                 "processor_config.json", "preprocessor_config.json"]:
        src = args.ckpt / name
        if src.exists():
            shutil.copy(src, args.out / name)
    for sub in ["processor", "experiment_cfg"]:
        src = args.ckpt / sub
        if src.is_dir():
            shutil.copytree(src, args.out / sub, dirs_exist_ok=True)
    meta = json.loads((args.ckpt / "fp4vla_save_meta.json").read_text())
    print(f"exported {len(sd)} tensors ({meta['bytes']/1e9:.2f}GB fp32 -> "
          f"{'bf16' if args.bf16 else 'fp32'}) to {args.out}")


if __name__ == "__main__":
    main()
