"""R0 drift-motivation experiment (paper §3.3 / H1 evidence) — two-pass design.

Pass 1 (teacher): roll out BF16 teacher with fixed seeds, recording per-step
states + teacher actions + episode end.
Pass 2 (student): roll out the quantized policy on IDENTICAL env seeds; at the
SAME wall-step, also query the teacher's action at the student's CURRENT state
(one extra teacher forward per replan — single process holds one policy at a
time; teacher actions at student states are collected by a third micro-pass
OR approximated by the same-state teacher from pass 1 only at t=0).

Practical protocol (fits 24GB with ONE policy resident):
  pass A: teacher rollouts (env seeded)  -> teacher trajectories + success
  pass B: student rollouts (same seeds)  -> student trajectories + success
  metric: horizon-truncated success from both passes; per-step action error
          is measured at t=0 (identical initial states) plus the divergence
          of state visitation histograms between passes (proprio TV).
This yields the R0 deliverables: success-vs-horizon, on/off-policy gap proxy,
state-visitation divergence — without two concurrent 3B policies.

Run (GPU window, ~40min for 10 tasks x 3 eps x 2 passes):
  cd ~ && <robo-venv>/python ~/codebase/fp4vla/exp/r0_drift.py \
    --pass teacher --variant bf16 ... ; then --pass student --variant nvfp4_static
"""
from __future__ import annotations
import argparse, json, pathlib, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "r0"

def build_env(bddl_root, task, camera=224):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib as pl
    bddl = pl.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(bddl_file_name=str(bddl),
                              camera_heights=camera, camera_widths=camera)

def obs_to_policy(obs, prompt, md):
    return {
        md["image_keys"][0]: obs["agentview_image"],
        md["image_keys"][1] if len(md["image_keys"]) > 1 else "left_wrist_0_rgb":
            obs.get("robot0_eye_in_hand_image", obs["agentview_image"]),
        md.get("state_key", "state"): np.concatenate([
            obs["robot0_eef_pos"], obs["robot0_eef_quat"],
            [obs["robot0_gripper_qpos"].mean()],
        ]).astype(np.float32),
        md["prompt_key"]: prompt,
    }

def proprio(obs):
    return np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                           [obs["robot0_gripper_qpos"].mean()]])

def run_pass(which: str, variant: str, args, seeds_by_task):
    from apxinf import AutoPolicy
    kw = {"norm_stats": str(ROOT / "weights/pi05_libero_base/norm_stats.json"),
          "model_variant": variant}
    policy = AutoPolicy.from_pretrained(str(ROOT / "weights/pi05_libero_base"), **kw)
    md = policy.metadata
    from libero.libero.benchmark import get_benchmark
    bench = get_benchmark("libero_10")()
    horizons = args.horizons
    out = {"pass": which, "variant": variant, "horizons": horizons, "tasks": []}

    for ti in range(args.tasks):
        task = bench.get_task(ti)
        prompt = task.language
        env = build_env(None, task)
        for ep in range(args.episodes):
            seed = seeds_by_task[ti][ep]
            obs = env.reset(seed=int(seed))
            states, dones, t0_actions = [], {}, []
            max_h = max(horizons)
            for step in range(max_h):
                po = obs_to_policy(obs, prompt, md)
                a = np.asarray(policy.infer(po)["actions"])[0]
                if step == 0:
                    t0_actions.append(a.tolist())
                obs, r, done, info = env.step(a.astype(np.float64))
                states.append(proprio(obs).tolist())
                if step + 1 in horizons:
                    dones[str(step + 1)] = bool(done)
                if done:
                    for h in horizons:
                        if str(h) not in dones:
                            dones[str(h)] = True
                    break
            for h in horizons:
                dones.setdefault(str(h), False)
            out["tasks"].append({"task": task.name, "ep": ep, "seed": int(seed),
                                 "horizon_success": dones,
                                 "t0_action": t0_actions[0] if t0_actions else None,
                                 "states_first64": states[:64]})
            env.close()
        print(f"[{which}] task {ti} done", flush=True)
    policy.close()
    return out

def analyze(teacher: dict, student: dict):
    horizons = teacher["horizons"]
    print("== success vs horizon ==")
    for h in horizons:
        t = np.mean([t["horizon_success"][str(h)] for t in teacher["tasks"]])
        s = np.mean([t["horizon_success"][str(h)] for t in student["tasks"]])
        print(f"  H={h:4d}: teacher={t:.3f} student={s:.3f}")
    # state-visitation TV over proprio histograms (first 64 steps)
    def flat(d):
        return np.array([v for t in d["tasks"] for v in t["states_first64"]]).ravel()
    a, b = flat(teacher), flat(student)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ha, _ = np.histogram(a, bins=64, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=64, range=(lo, hi), density=True)
    pa, pb = ha / ha.sum(), hb / hb.sum()
    tv = 0.5 * np.abs(pa - pb).sum()
    print(f"state-visitation TV (proprio, pooled): {tv:.4f}")
    # t=0 action divergence (identical seeds -> identical initial states)
    da = [np.linalg.norm(np.array(t["t0_action"]) - np.array(s["t0_action"]))
          for t, s in zip(teacher["tasks"], student["tasks"]) if t["t0_action"] and s["t0_action"]]
    print(f"t=0 action L2 (same initial states): mean={np.mean(da):.4f} max={np.max(da):.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", choices=["teacher", "student", "analyze"], required=True)
    ap.add_argument("--variant", default=None, help="bf16 | nvfp4_static")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--episodes-per-task", type=int, default=3)
    ap.add_argument("--horizons", default="90,180,360,720")
    ap.add_argument("--seed-base", type=int, default=7)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds_by_task = {ti: [args.seed_base + 1000 * ti + e for e in range(args.episodes_per_task)]
                     for ti in range(args.tasks)}

    if args.which == "analyze":
        teacher = json.loads((RESULTS / "teacher.json").read_text())
        student = json.loads((RESULTS / "student.json").read_text())
        analyze(teacher, student)
        return

    variant = args.variant or ("bf16" if args.which == "teacher" else "nvfp4_static")
    args.horizons = [int(h) for h in args.horizons.split(",")]
    out = run_pass(args.which, variant, args, seeds_by_task)
    (RESULTS / f"{args.which}.json").write_text(json.dumps(out, indent=1))
    print("wrote", RESULTS / f"{args.which}.json")

if __name__ == "__main__":
    main()
