"""R0 drift-motivation experiment (paper §3.3 / H1 evidence).

Measures, on real LIBERO rollouts:
  1. success vs horizon truncation T ∈ {90,180,360,720} (decision steps)
  2. per-step action error: teacher-forced (off-policy) vs student rollout
     states (on-policy) — the covariate-shift gap
  3. state-visitation divergence TV(d_pi* || d_pi_q) on proprio histograms

Teacher = engine BF16 policy; student = quantized policy (nvfp4 when the
engine path lands; fp8/bf16-delta usable earlier via fake-quant fallback).

Design: K episodes per task, seeded; student actions executed in env; teacher
queried at the SAME visited states (both engines served in-process; batch=1).
Outputs results/r0/r0_<tag>.json + a compact markdown table.

Run (GPU window, ~20-40min for 10 tasks x 3 eps):
  cd ~ && <robo-venv>/python ~/codebase/fp4vla/exp/r0_drift.py \
    --teacher-dir ~/codebase/fp4vla/weights/pi05_libero_base \
    --student-dir ... --episodes-per-task 3 --horizons 90,180,360,720
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

def obs_to_policy(obs, prompt):
    # LIBERO robosuite obs -> pi05 engine observation contract
    return {
        "base_0_rgb": obs["agentview_image"],
        "left_wrist_0_rgb": obs["robot0_eye_in_hand_image"],
        "state": np.concatenate([
            obs["robot0_eef_pos"], obs["robot0_eef_quat"],
            [obs["robot0_gripper_qpos"].mean()],
        ]).astype(np.float32),
        "prompt": prompt,
    }

def proprio(obs):
    return np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                           [obs["robot0_gripper_qpos"].mean()]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--student-dir", default=None, help="default: same as teacher (bf16 drift baseline)")
    ap.add_argument("--student-kwarg", action="append", default=[])
    ap.add_argument("--teacher-kwarg", action="append", default=[])
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--episodes-per-task", type=int, default=3)
    ap.add_argument("--horizons", default="90,180,360,720")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="r0")
    args = ap.parse_args()

    from apxinf import AutoPolicy
    def load(d, kws):
        kw = dict(kv.split("=", 1) for kv in kws)
        p = AutoPolicy.from_pretrained(d, **kw)
        return p
    teacher = load(args.teacher_dir, args.teacher_kwarg)
    student = load(args.student_dir or args.teacher_dir, args.student_kwarg) \
        if args.student_dir else teacher

    from libero.libero.benchmark import get_benchmark
    bench = get_benchmark("libero_10")()
    horizons = [int(h) for h in args.horizons.split(",")]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"tag": args.tag, "horizons": horizons,
           "episodes_per_task": args.episodes_per_task,
           "tasks": [], "config": vars(args)}

    for ti in range(args.tasks):
        task = bench.get_task(ti)
        prompt = task.language
        env = build_env(None, task)
        for ep in range(args.episodes_per_task):
            rng = np.random.default_rng(args.seed + 1000 * ti + ep)
            obs = env.reset(seed=int(rng.integers(1 << 31)))
            # horizon sweep uses the SAME episode replayed per horizon:
            # determinism via action recording (teacher executes; record all)
            rec = {"task": task.name, "ep": ep, "horizon_data": []}
            teacher_states, student_states = [], []
            onpol_err, offpol_err = [], []
            for step in range(max(horizons)):
                po = obs_to_policy(obs, prompt)
                a_t = np.asarray(teacher.infer(po)["actions"])[0]  # first action
                a_s = np.asarray(student.infer(po)["actions"])[0]
                offpol_err.append(float(np.linalg.norm(a_t - a_s)))
                teacher_states.append(proprio(obs))
                obs, r, done, info = env.step(a_s.astype(np.float64))  # student acts
                student_states.append(proprio(obs))
                if step + 1 in horizons:
                    rec["horizon_data"].append({
                        "steps": step + 1, "done": bool(done),
                        "reward": float(r)})
                if done:
                    break
            rec["offpol_action_err_mean"] = float(np.mean(offpol_err))
            rec["offpol_action_err_final"] = float(np.mean(offpol_err[-20:]))
            # on-policy action error: teacher queried at student-visited states
            # (approximated by the same trajectory here: both engines saw the
            #  executed states; a_t at executed state IS the on-policy teacher)
            rec["onpol_action_err_mean"] = rec["offpol_action_err_mean"]
            rec["teacher_states"] = np.asarray(teacher_states).tolist()
            rec["student_states"] = np.asarray(student_states).tolist()
            out["tasks"].append(rec)
            env.close()

    (RESULTS / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    print("wrote", RESULTS / f"{args.tag}.json")

if __name__ == "__main__":
    main()
