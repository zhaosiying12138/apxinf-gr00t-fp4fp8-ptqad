"""Streaming replacement for gr00t fsdp2 save_fsdp2_model.

The original accumulates the full 6.9GB CPU state dict before saving
(monolithic gather = the VM-kill spike, 8 crashes). This version shards:
gather one parameter at a time, flush a safetensors shard every ~500MB.
Peak extra host memory = one shard (~500MB) instead of 6.9GB.

Injected from full_head_qad.py BEFORE runpy; the sibling trainer's
patch_trainer_save closure resolves save_fsdp2_model via module globals at
call time, so replacing the module attribute is sufficient.

Loader counterpart: rl/load_qad_shards.py
"""
import os
import json


def streaming_save_fsdp2_model(model, out_dir: str) -> None:
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    index = {}
    shard, shard_bytes, shard_id = {}, 0, 0
    total = 0

    def flush():
        nonlocal shard, shard_bytes, shard_id
        if shard:
            path = os.path.join(out_dir, f"fp4vla-shard-{shard_id:03d}.safetensors")
            save_file(shard, path)
            for k in shard:
                index[k] = os.path.basename(path)
            shard_id += 1
            shard, shard_bytes = {}, 0

    for name, p in model.named_parameters():
        t = p
        if hasattr(p, "full_tensor"):
            t = p.full_tensor()
        t = t.detach().to("cpu", copy=True).contiguous()
        shard[name] = t
        shard_bytes += t.numel() * t.element_size()
        total += t.numel() * t.element_size()
        if shard_bytes >= 500 * 1024 * 1024:
            flush()
    flush()
    with open(os.path.join(out_dir, "fp4vla_shard_index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(out_dir, "fp4vla_save_meta.json"), "w") as f:
        json.dump({"tensors": len(index), "bytes": total}, f)
    try:
        model.config.save_pretrained(out_dir)
    except Exception:
        pass
    print(f"[fp4vla-save] streamed {len(index)} tensors "
          f"({total/1e9:.2f}GB) in {shard_id} shards to {out_dir}", flush=True)


def install_streaming_save() -> None:
    import gr00t.experiment.fsdp2 as fsdp2_mod
    fsdp2_mod.save_fsdp2_model = streaming_save_fsdp2_model
    print("[fp4vla-save] streaming save installed over fsdp2.save_fsdp2_model", flush=True)
