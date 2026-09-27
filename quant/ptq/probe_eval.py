"""Deterministic offline probe evaluation of baked PTQ checkpoints.

The DiT head samples noise and timesteps from the global RNG, so raw pred
comparisons across processes measure RNG drift, not quantization damage
(BF16 self-check vs the old cache gave corr 0.10 — the harness was the bug).
Protocol here: pin torch.randn to a dedicated generator (seed 20260927) and
pin sample_time to t=0.5, so the whole forward is deterministic. Reference
preds are produced ONCE from the BF16 base (--ckpt <base> --make-ref) and
cached; every arm is then compared under the identical protocol.
"""
import os, sys, time, argparse


def run(args):
    import torch
    os.makedirs(args.workdir, exist_ok=True)
    sys.path.insert(0, os.getcwd())
    ref_path = f"{args.workdir}/fixed_teacher.pt"
    t0 = time.time()

    _orig_randn = torch.randn
    _gen = {"g": None}

    def pinned_randn(*shape_or_size, **kw):
        if _gen["g"] is not None and "generator" not in kw:
            kw["generator"] = _gen["g"]
        return _orig_randn(*shape_or_size, **kw)
    torch.randn = pinned_randn

    from gr00t.configs.base_config import get_default_config
    from gr00t.model import MODEL_REGISTRY
    config = get_default_config().load_dict({
        "data": {"download_cache": False, "datasets": [{
            "dataset_paths": ["./demo_data/libero_demo"], "mix_ratio": 1.0,
            "embodiment_tag": "libero_sim"}]},
    })
    config.load_config_path = None
    config.model.model_path = args.ckpt
    config.model.action_horizon = 64
    config.training.use_fsdp2 = False
    from pathlib import Path
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, Path(args.workdir))
    pipeline.setup()
    model = pipeline.return_model().to("cuda").to(torch.bfloat16).eval()

    n_time = 0
    for m in model.modules():
        if hasattr(m, "sample_time") and callable(getattr(m, "sample_time", None)):
            def fixed_time(b, device=None, dtype=None, _m=None):
                return torch.full((b,), 0.5, device=device, dtype=dtype or torch.float32)
            m.sample_time = fixed_time
            n_time += 1

    d = torch.load(args.probe, map_location="cpu")
    tin = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in d["inputs"].items()}
    tin.pop("labels", None)

    _gen["g"] = torch.Generator(device="cuda")
    _gen["g"].manual_seed(20260927)
    with torch.no_grad():
        out = model(tin)
    pred = out["pred_actions"].float().cpu()
    torch.randn = _orig_randn

    tag = os.path.basename(args.ckpt.rstrip("/"))
    if args.make_ref:
        torch.save({"pred": pred}, ref_path)
        print(f"[probe] REF cached from {tag} ({time.time()-t0:.0f}s) -> {ref_path}", flush=True)
        return None
    assert os.path.exists(ref_path), "run with --make-ref on the BF16 base first"
    ref = torch.load(ref_path, map_location="cpu")["pred"].float()
    mse = float(((pred - ref) ** 2).mean())
    rel = mse / float(ref.var())
    pv, rv = pred.flatten(), ref.flatten()
    corr = float(((pv - pv.mean()) * (rv - rv.mean())).mean() /
                 (pv.std(unbiased=False) * rv.std(unbiased=False) + 1e-12))
    print(f"[probe] {tag}: relMSE={rel:.5f} pooledCorr={corr:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    return rel, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probe", default="/mnt/c/fq_opd_probes/teacher_probes.pt")
    ap.add_argument("--workdir", default="/mnt/c/fq_ptq_calib")
    ap.add_argument("--make-ref", action="store_true")
    args = ap.parse_args()
    import glob
    shards = glob.glob(f"{args.ckpt}/*.safetensors")
    assert shards, f"NO SAFETENSORS in {args.ckpt} — bake incomplete, refusing to eval"
    run(args)


if __name__ == "__main__":
    main()
