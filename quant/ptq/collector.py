"""Calibration collector for GR00T N1.7 backbone PTQ (AWQ/GPTQ/MSE-calib).

Runs the BF16 teacher on libero_demo batches with forward-pre-hooks on every
backbone nn.Linear; accumulates EXACT per-layer Hessians H = sum(x x^T) in
fp32 on GPU (moved to CPU at end) plus per-input-channel sum(|x|) and row
count. Output-MSE evaluation elsewhere uses the identity
    sum_i ||x_i dW^T||^2 = tr(dW H dW^T)
so raw activations need not be stored (memory: only K x K + K per layer).

Head (action_head.*) is excluded: it is quantized by plain RTN at bake time
(head-only-FP4 is already known usable at 71.2% closed-loop).
"""
import os, sys, time, json

BASE = os.environ.get("PTQ_BASE", "/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10")
OUT = "/mnt/c/fq_ptq_calib"
N_BATCH = int(os.environ.get("PTQ_CAL_BATCHES", "16"))
BATCH = 8

def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, os.getcwd())
    t0 = time.time()
    import torch
    import torch.nn as nn
    from gr00t.configs.base_config import get_default_config
    from gr00t.model import MODEL_REGISTRY
    config = get_default_config().load_dict({
        "data": {"download_cache": False, "datasets": [{
            "dataset_paths": ["./demo_data/libero_demo"], "mix_ratio": 1.0,
            "embodiment_tag": "libero_sim"}]},
    })
    config.load_config_path = None
    config.model.model_path = BASE
    config.model.action_horizon = 64
    config.training.use_fsdp2 = False
    from pathlib import Path
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, Path(OUT))
    pipeline.setup()
    model = pipeline.return_model().to("cuda").to(torch.bfloat16).eval()

    acc = {}   # name -> {"H": gpu f32 (K,K) or None, "abs": gpu f32 (K,), "n": int}
    def mk_hook(name):
        def hook(mod, args):
            x = args[0]
            if not torch.is_tensor(x) or x.ndim == 0 or not x.is_floating_point():
                return
            K = mod.weight.shape[1]
            if K % 16 != 0:
                return
            xf = x.detach().reshape(-1, K).float()
            e = acc[name]
            if e["H"] is None:
                e["H"] = xf.t() @ xf
            else:
                e["H"] += xf.t() @ xf
            e["abs"] += xf.abs().sum(0)
            e["n"] += xf.shape[0]
        return hook
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "action_head" not in name            and mod.weight.shape[1] % 16 == 0:
            acc[name] = {"H": None, "abs": torch.zeros(mod.weight.shape[1], device="cuda"), "n": 0}
            hooks.append(mod.register_forward_pre_hook(mk_hook(name)))
    print(f"[calib] {len(hooks)} backbone Linears hooked; model load {time.time()-t0:.0f}s", flush=True)

    train_dataset, _ = pipeline.return_dataset()
    collator = pipeline.return_collator()
    it = iter(train_dataset)
    for bi in range(N_BATCH):
        raw = [next(it) for _ in range(BATCH)]
        batch = collator(raw)
        tin = batch["inputs"] if "inputs" in batch else batch
        tin = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in tin.items()}
        with torch.no_grad():
            model(tin)
        tot_n = sum(e["n"] for e in acc.values())
        print(f"[calib] batch {bi+1}/{N_BATCH} done ({time.time()-t0:.0f}s, rows(layer0)~{tot_n//max(len(acc),1)})", flush=True)

    for h in hooks: h.remove()
    out = {}
    for name, e in acc.items():
        if e["H"] is None:
            continue
        out[name] = {"H": e["H"].cpu(), "abs": e["abs"].cpu(), "n": e["n"]}
    torch.save(out, f"{OUT}/calib.pt")
    gb = sum(v["H"].numel() for v in out.values()) * 4 / 1e9
    meta = {"n_layers": len(out), "batches": N_BATCH, "batch": BATCH,
            "H_GB_f32": round(gb, 2), "rows_per_layer": {k: v["n"] for k, v in out.items()}}
    json.dump(meta, open(f"{OUT}/calib_meta.json", "w"))
    print(f"[calib] saved {len(out)} layers, H total {gb:.2f}GB -> {OUT}/calib.pt ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
