"""Optional task-weighted self-imitation research implementation.

Not included in the current five-arm experiment and not validated against its
40x132 model tensor / 16x7 effective-action protocol. This file is retained for
the appendix's source discussion, not as a supported reproduction entry point.
It has no current success-rate or performance result.

Per-task success rates weight all windows in tasks whose rate is positive;
there are no per-episode success labels, so these are not success-only episodes.
Task boundaries are inferred from log mtimes. Within each batched environment
slot, eight executed action prefixes form a 64-step window; episode boundaries
are unavailable and a window can cross them. This objective is weighted flow-
matching imitation, not PPO, a likelihood-ratio policy gradient, or teacher KL.
Before any future experiment, update the policy API, paths, episode boundaries,
action horizon, normalization checks and export protocol independently.
"""
import os, sys, time, json, glob, gzip, pickle, argparse, re

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("LD_LIBRARY_PATH",
    os.path.expanduser("~/miniforge3/envs/media7/lib:") + os.environ.get("LD_LIBRARY_PATH", ""))

DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_tasks(log_dir, results_glob):
    """Per-task contiguous server-call file ranges + success rates (mtime split)."""
    logs = [f for f in glob.glob(results_glob) if "server" not in f]
    tasks = []
    for f in logs:
        txt = open(f, errors="ignore").read()
        m = re.findall(r"results:\s+\('[^']+', \[([^\]]*)\],", txt)
        sr = re.search(r"success rate:\s*([0-9.]+)", txt)
        if m and sr:
            tasks.append({"log": f, "rate": float(sr.group(1)), "end": os.path.getmtime(f)})
    tasks.sort(key=lambda t: t["end"])
    files = sorted(glob.glob(f"{log_dir}/step_*.pkl.gz"), key=os.path.getmtime)
    assert files and tasks, f"no rollout files/logs ({len(files)} files, {len(tasks)} tasks)"
    bounds = [t["end"] for t in tasks]
    for t in tasks:
        t["files"] = []
    for f in files:
        mt = os.path.getmtime(f)
        for i, t in enumerate(tasks):
            if mt <= bounds[i]:
                t["files"].append(f)
                break
        else:
            tasks[-1]["files"].append(f)
    return tasks


def load_entry(path):
    import numpy as np
    ent = pickle.loads(gzip.decompress(open(path, "rb").read()))
    obs = ent["obs"]
    o = {"video": {}, "state": {}, "language": obs.get("language")}
    for k, v in obs["video"].items():
        o["video"][k] = np.array(v, dtype=np.uint8)
    for k, v in obs["state"].items():
        o["state"][k] = np.array(v, dtype=np.float32)
    return o, ent["action"]


def stitch(action_calls, t, b):
    """(64, 7) target from executed 8-prefixes of calls t..t+7, slot b."""
    import numpy as np
    rows = []
    for ac in action_calls:
        rows.append(np.stack([np.asarray(ac[d])[b, :8, 0] for d in DIMS], axis=1))
    return np.concatenate(rows, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--results-glob", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    args = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/rl")
    import numpy as np
    import torch

    tasks = load_tasks(args.rollout_dir, args.results_glob)
    for t in tasks:
        print(f"[rwr] task {os.path.basename(t['log'])[:40]}: rate={t['rate']} "
              f"calls={len(t['files'])}", flush=True)
    live = [t for t in tasks if t["rate"] > 0]
    assert live, "no task with any success"

    # training windows: (task, call_idx, slot), weighted by task success rate
    windows = []
    for ti, t in enumerate(live):
        n = len(t["files"])
        for c in range(n - 8):
            for b in range(8):
                windows.append((ti, c, b))
    print(f"[rwr] windows={len(windows)} over {len(live)} live tasks ({time.time()-t0:.0f}s)", flush=True)

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    policy = Gr00tPolicy(embodiment_tag="LIBERO_PANDA", model_path=args.base,
                         device="cuda", strict=True)
    model = policy.model
    proc = policy.processor
    groups = proc.modality_configs["libero_sim"]["action"].modality_keys
    print(f"[rwr] action groups (model order): {groups}", flush=True)

    os.environ.setdefault("GR00T_BASE_CKPT", args.base)
    import lora_qad as LQ
    LQ.R, LQ.ALPHA, LQ.SCOPE = args.rank, args.alpha, "head"
    LQ.install_lora(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    _inject = {"on": False}

    # libero occupies slots 0-6 of the unified 132-dim action space (verified
    # empirically from probe demo actions: var>0 exactly at 0..6, values in
    # normalized [-1,1] from q01/q99 min-max) — same order as DIMS.
    def inject_action(batch, target):
        a = torch.zeros(1, 64, 132, dtype=torch.bfloat16)
        mk = torch.zeros(1, 64, 132, dtype=torch.float32)
        a[0, :, :7] = torch.from_numpy(target).to(torch.bfloat16)
        mk[0, :, :7] = 1.0
        inner = dict(batch["inputs"]) if "inputs" in batch else dict(batch)
        inner["action"] = a
        inner["action_mask"] = mk
        return inner

    # preload action dicts per task (small)
    acts = {}
    for ti, t in enumerate(live):
        acts[ti] = [load_entry(f)[1] for f in t["files"]]

    def build_batch(idx):
        ti, c, b = windows[idx]
        obs_full, _ = load_entry(live[ti]["files"][c])
        target_phys = stitch(acts[ti][c:c + 8], 0, b)   # (64, 7) absolute, DIMS order
        unbatched = policy._unbatch_observation(obs_full)
        state_raw = unbatched[b]["state"]
        act_dict = {d: target_phys[:, i:i + 1] for i, d in enumerate(DIMS)}
        norm = proc.state_action_processor.apply_action(
            act_dict, "libero_sim", state={k: np.asarray(v, dtype=np.float32)
                                           for k, v in state_raw.items()})
        target = np.concatenate([norm[g] for g in groups], axis=1).astype(np.float32)
        sd = policy._to_vla_step_data(unbatched[b])
        from gr00t.data.types import MessageType
        msg = [{"type": MessageType.EPISODE_STEP.value, "content": sd}]
        processed = policy.processor(msg)
        batch = policy.collate_fn([processed])
        if _inject["on"]:
            batch = inject_action(batch, target)
        return batch, live[ti]["rate"]

    probe_batch, w0 = build_batch(0)
    _inject["on"] = True
    tb, _ = build_batch(0)
    print(f"[rwr] injected action shape={tuple(tb['action'].shape)} "
          f"z-range={float(tb['action'].abs().max()):.2f} (expect < ~2)", flush=True)

    model.train()
    for step in range(args.steps):
        idxs = np.random.randint(0, len(windows), args.batch)
        losses, wsum = [], 0.0
        for i in idxs:
            b, w = build_batch(int(i))
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b)
            losses.append(out["action_loss"].mean() * w)
            wsum += w
        loss = torch.stack(losses).sum() / max(wsum, 1e-6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"[rwr] step {step} loss={float(loss):.5f} ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(getattr(mod, "lora_A", None), torch.nn.Parameter):
                mod.weight.data += ((mod.lora_B.data.float() @ mod.lora_A.data.float())
                                    * (args.alpha / args.rank)).to(mod.weight.dtype)
                del mod.lora_A, mod.lora_B
                mod.forward = torch.nn.Linear.forward
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    import shutil
    for f in ["config.json", "embodiment_id.json", "processor_config.json", "statistics.json"]:
        src = f"{args.base}/{f}"
        if os.path.exists(src):
            shutil.copy(src, f"{args.out}/{f}")
    print(f"[rwr] DONE saved {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
