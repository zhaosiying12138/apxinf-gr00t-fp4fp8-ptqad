"""Sensitivity screening driver (two-stage protocol, exp/configs/sensitivity.json).

For each grid config: point model_dir/fp4 symlink at the right artifact, run a
short LIBERO eval via apxinf-robo CLI, collect success + on-policy action error.
Stage 1 (screen): 3 tasks x 20 episodes on the 3 tasks with largest PT4 drop.
Stage 2 (final): 10 tasks x 50 episodes for finalists.

Usage (GPU window):
  robo-venv python exp/run_sensitivity.py --stage screen [--configs pt4-full,pt4-bb-only,...]
Artifacts are produced by quant/nvfp4_convert_packed.py --scope/--skip-*.
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, shutil, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "weights" / "pi05_libero_base"
FP4_LINK = MODEL_DIR / "fp4"
VENV_PY = ROOT / "third_party" / "apxinf-robo" / ".venv" / "bin" / "python"

# config -> converter invocation producing the artifact (reproducibility manifest)
ARTIFACT_SPECS = {
    "pt4-full":    [],                                          # all trees (weights/pi05.nvfp4.packed)
    "pt4-bb-only": ["--scope", "lang"],                         # weights/pi05.nvfp4.lang
    "pt4-ah-only": ["--scope", "act"],                          # weights/pi05.nvfp4.act
    "pt4-mixed":   ["--scope", "lang"],                         # lang fp4 + act bf16 = same artifact as bb-only (action exempt)
    "pt4-skip1":   ["--scope", "all", "--skip-first", "1", "--skip-last", "1"],
}
ARTIFACT_DIRS = {
    "pt4-full":    ROOT / "weights/pi05.nvfp4.packed",
    "pt4-bb-only": ROOT / "weights/pi05.nvfp4.lang",
    "pt4-ah-only": ROOT / "weights/pi05.nvfp4.act",
    "pt4-mixed":   ROOT / "weights/pi05.nvfp4.lang",
    "pt4-skip1":   ROOT / "weights/pi05.nvfp4.skip1",   # generate on demand
}

def ensure_artifact(cfg: str) -> pathlib.Path:
    d = ARTIFACT_DIRS[cfg]
    if d.exists() and (d / "manifest.json").exists():
        return d
    if not d.exists() and cfg == "pt4-skip1":
        cmd = ["python", str(ROOT / "quant/nvfp4_convert_packed.py"),
               "--ckpt", str(MODEL_DIR / "model.safetensors"),
               "--out", str(d)] + ARTIFACT_SPECS[cfg]
        print("[sens] generating", cfg)
        subprocess.run(cmd, check=True, cwd=str(ROOT / "quant"))
    return d

def point_artifact(cfg: str | None):
    if FP4_LINK.exists() or FP4_LINK.is_symlink():
        FP4_LINK.unlink()
    if cfg is None:
        return
    target = ensure_artifact(cfg)
    FP4_LINK.symlink_to(target.resolve())

def run_eval(tag: str, variant: str, episodes: int, tasks: int, suite: str = "libero_10"):
    out_dir = ROOT / "results" / "sensitivity" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(VENV_PY), "-m", "apxinf_robo.cli", "eval-libero",
           "--backend", "in-process",
           "--model-dir", str(MODEL_DIR),
           "--precision", variant,
           "--suite", suite, "--trials-per-task", str(episodes), "--tasks", str(tasks),
           "--results-jsonl", str(out_dir / "results.jsonl"),
           "--summary-json", str(out_dir / "summary.json"),
           "--norm-stats", str(MODEL_DIR / "norm_stats.json")]
    print("[sens]", " ".join(cmd[:8]), "...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    (out_dir / "stderr.log").write_text(r.stderr[-20000:])
    if r.returncode != 0:
        print(f"[sens] {tag} FAILED rc={r.returncode}: {r.stderr[-500:]}")
    else:
        print(f"[sens] {tag} done in {time.time()-t0:.0f}s")
    return out_dir / "summary.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "final"], default="screen")
    ap.add_argument("--configs", default="bf16,pt4-full,pt4-bb-only,pt4-ah-only,pt4-skip1")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--tasks", type=int, default=None)
    args = ap.parse_args()
    cfgs = args.configs.split(",")

    if args.stage == "screen":
        episodes, tasks = args.episodes or 20, args.tasks or 3
    else:
        episodes, tasks = args.episodes or 50, args.tasks or 10

    ledger = []
    for cfg in cfgs:
        if cfg == "bf16":
            point_artifact(None)
            summary = run_eval(f"bf16", "bf16", episodes, tasks)
        else:
            point_artifact(cfg)
            summary = run_eval(cfg, "nvfp4_static", episodes, tasks)
        if summary.exists():
            ledger.append({"config": cfg, "summary": json.loads(summary.read_text())})
        else:
            ledger.append({"config": cfg, "error": "no summary"})

    out = ROOT / "results" / "sensitivity" / f"{args.stage}_ledger.json"
    out.write_text(json.dumps(ledger, indent=1))
    print(json.dumps(ledger, indent=1)[:3000])
    print("wrote", out)

if __name__ == "__main__":
    main()
