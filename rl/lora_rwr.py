"""RWR (reward-weighted self-imitation) — the PPO-family on-policy baseline.

Arms comparison (same 2.44x fp8-arm deployment base, same LoRA budget):
  QAD-LoRA : demo-loss (offline demonstrations)
  OPD-LoRA : + probe-cached BF16 teacher-KL
  RWR-LoRA : flow-matching BC on SUCCESSFUL on-policy rollouts only
             (sparse success reward -> weight w=1 on success, 0 otherwise;
             REINFORCE's reward-weighted likelihood reduces to this under
             terminal 0/1 reward; full PPO/DPPO cited as the exact-likelihood
             version, Ren et al. 2024)

Two-phase loop, cross-venv safe:
  Phase A (rollout): run_libero_eval_mini.sh + FP4VLA_LOG_DIR -> server dumps
             (obs, action_chunk) per policy call (JPEG-compressed);
             rollout log's results line gives per-episode success + lengths.
             Episode k spans ceil(len_k/8) consecutive server calls; the env
             executes the FIRST 8 of each returned 16-chunk.
  Phase B (train) : this script. Windows: obs at call t + target =
             concat(executed(t..t+7)) -> (64, D); normalize with the base
             checkpoint's statistics.json; replay obs through the policy's
             own preprocessing; overwrite the collated action target;
             model(batch) -> action_loss; AdamW on LoRA params.
"""
import os, sys, time, json, glob, gzip, pickle, argparse, re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LD_LIBRARY_PATH",
    os.path.expanduser("~/miniforge3/envs/media7/lib:") + os.environ.get("LD_LIBRARY_PATH", ""))


def load_rollout(log_dir, results_log):
    """Return list of episodes: {success, calls: [file paths in order]}."""
    txt = open(results_log, errors="ignore").read()
    m = re.findall(r"results:\s+\('[^']+', \[([^\]]*)\], \{'episode_lengths': \[([^\]]*)\]",
                   txt)
    assert m, "no results line in rollout log"
    succ = [s.strip() == "True" for s in m[-1][0].split(",")]
    lens = [int(x) for x in m[-1][1].split(",")]
    files = sorted(glob.glob(f"{log_dir}/step_*.pkl.gz"))
    n_calls = [ (L + 7) // 8 for L in lens ]
    assert sum(n_calls) == len(files), f"calls {sum(n_calls)} != files {len(files)}"
    episodes, i = [], 0
    for s, c in zip(succ, n_calls):
        episodes.append({"success": s, "calls": files[i:i + c]})
        i += c
    return episodes


def stitch_target(ep, t, horizon_chunks=8, exec_steps=8):
    """Concat the executed prefix of chunks t..t+7 -> (64, D) float list."""
    rows = []
    for f in ep["calls"][t:t + horizon_chunks]:
        ent = pickle.loads(gzip.decompress(open(f, "rb").read()))
        act = ent["action"]
        if isinstance(act, dict):
            act = next(iter(act.values()))
        rows.extend(act[0][:exec_steps])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--results-log", required=True)
    ap.add_argument("--base", required=True, help="merged deploy ckpt (fp8 arm or qad-lora)")
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

    episodes = load_rollout(args.rollout_dir, args.results_log)
    wins = [e for e in episodes if e["success"]]
    print(f"[rwr] episodes={len(episodes)} success={len(wins)} ({time.time()-t0:.0f}s)", flush=True)
    assert wins, "no successful episodes — nothing to imitate"

    # training windows from successful episodes
    windows = []
    for e in wins:
        for t in range(len(e["calls"]) - 8):
            windows.append((e, t))
    print(f"[rwr] windows={len(windows)}", flush=True)

    # load policy + model (Baked base, eval transforms)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    policy = Gr00tPolicy(embodiment_tag="LIBERO_PANDA", model_path=args.base,
                         device="cuda", strict=True)
    model = policy.model

    # action normalization stats from the BASE teacher's statistics
    stats = json.load(open(f"{args.base}/statistics.json"))
    key = next(k for k in stats if "action" in k.lower() and "libero" in k.lower()) \
        if any("libero" in k.lower() for k in stats) else next(k for k in stats if "action" in k.lower())
    astat = stats[key]
    mean = np.array(astat["mean"], dtype=np.float32)
    std = np.array(astat["std"], dtype=np.float32).clip(1e-6)

    # LoRA injection (reuse lora_qad machinery on this baked base)
    os.environ.setdefault("GR00T_BASE_CKPT", args.base)
    import lora_qad as LQ
    LQ.R, LQ.ALPHA, LQ.SCOPE = args.rank, args.alpha, "head"
    LQ.install_lora(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    _inject = {"slot": None}

    def inject_action(batch, target):
        d = batch
        for k in _inject["slot"][:-1]:
            d = d[k]
        d[_inject["slot"][-1]] = target.unsqueeze(0).to(torch.bfloat16)
        return batch

    def build_batch(idx):
        obs_ent, tgt = windows[idx]
        ent = pickle.loads(gzip.decompress(open(obs_ent["calls"][tgt], "rb").read()))
        obs = ent["obs"]
        from PIL import Image
        import io
        for k, v in list(obs.items()):
            if isinstance(v, dict) and "__jpeg__" in v:
                obs[k] = np.array(Image.open(io.BytesIO(v["__jpeg__"])))
        unbatched = policy._unbatch_observation(obs)
        sd = policy._to_vla_step_data(unbatched[0])
        from gr00t.data.message_type import MessageType
        msg = [{"type": MessageType.EPISODE_STEP.value, "content": sd}]
        processed = policy.processor(msg)
        batch = policy.collate_fn([processed])
        target = (np.array(stitch_target(obs_ent, tgt), dtype=np.float32) - mean) / std
        if _inject["slot"] is not None:
            batch = inject_action(batch, torch.from_numpy(target))
        return batch

    # locate the action key inside collated batch once (structure discovery)
    probe_batch = build_batch(0)
    def find_action_slot(d, path=()):
        if isinstance(d, dict):
            for k, v in d.items():
                if "action" in str(k).lower() and hasattr(v, "shape") and v.ndim == 3:
                    return path + (k,)
                r = find_action_slot(v, path + (k,))
                if r: return r
        return None
    slot = find_action_slot(probe_batch)
    print(f"[rwr] action slot = {slot}", flush=True)
    assert slot is not None, "action target slot not found in collated batch"
    _inject["slot"] = slot
    model.train()
    for step in range(args.steps):
        idxs = np.random.randint(0, len(windows), args.batch)
        losses = []
        for i in idxs:
            b = build_batch(int(i))
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b)
            losses.append(out["action_loss"].mean())
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"[rwr] step {step} loss={float(loss):.5f} ({time.time()-t0:.0f}s)", flush=True)

    # save: strip to state dict with LoRA merged via lora_merge_bake semantics
    os.makedirs(args.out, exist_ok=True)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(sd, f"{args.out}/rwr_state.pt")
    print(f"[rwr] DONE saved {args.out}/rwr_state.pt ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
