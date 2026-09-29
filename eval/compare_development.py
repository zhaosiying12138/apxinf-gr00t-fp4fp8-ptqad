"""Audit official-bank development pairing and apply the preregistered ladder rule."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from run_recovery_eval import TASKS, validate_resets

ORDER = ["bf16", "fp8", "head_ffn", "head_lang", "head_lang_vision", "calib"]


def audit_development(root, arms):
    if arms != ORDER[:len(arms)]:
        raise ValueError("Development arms must follow the declared ladder without omissions")
    root = Path(root)
    reports, reference = {}, None
    sources = {}
    for arm in arms:
        folder = root / arm
        for name in ("eval_manifest.json", "task_results.json", "summary.json"):
            path = folder / name
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = json.loads((folder / "eval_manifest.json").read_text())
        tasks = json.loads((folder / "task_results.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        if (manifest["purpose"] != "development" or manifest["seed"] != 330000 or
            manifest["episodes"] != 2 or manifest["init_state_indices"] != [0,1] or
            manifest["n_envs"] != 1 or manifest["tasks"] != TASKS or set(tasks) != set(TASKS) or
            manifest["initial_state_protocol"] != "libero10_official_bank_v1" or
            summary["tasks_complete"] != 10):
            raise ValueError(f"{arm}: not a complete declared development run")
        signatures, per_task = [], {}
        for index, task in enumerate(TASKS):
            row = tasks[task]
            if row["returncode"] != 0 or len(row["results"]) != 2:
                raise ValueError(f"{arm}/{task}: incomplete episode results")
            resets = validate_resets(row, 330000+1000*index, [0,1])
            signatures.extend([{k:r[k] for k in ("episode_index", "seed", "init_state_index",
                "init_state_bank_sha256", "restored_state_sha256", "initial_state_sha256")} for r in resets])
            per_task[task] = {"results": row["results"], "successes": sum(row["results"]),
                              "episodes": 2, "success_rate": sum(row["results"])/2}
        if reference is None:
            reference = (manifest, signatures)
        else:
            if signatures != reference[1]:
                differing = [i for i,(a,b) in enumerate(zip(signatures,reference[1])) if a != b]
                raise ValueError(f"{arm}: initial states differ from BF16 at paired slots {differing}")
            for key in ("protocol_sha256", "n_action_steps", "max_episode_steps", "settle_steps"):
                if manifest[key] != reference[0][key]:
                    raise ValueError(f"{arm}: changed {key}")
        successes = sum(row["successes"] for row in per_task.values())
        reports[arm] = {"checkpoint": manifest["checkpoint"], "successes": successes,
                        "episodes": 20, "macro_success_rate": successes/20,
                        "per_task": per_task, "environment_pairing_verified": True,
                        "wall_seconds_including_server_loads": summary["wall_seconds_including_server_loads"]}
    selected = None
    if reports:
        base = reports["bf16"]["successes"]
        for arm in arms[2:]:
            if Fraction(base-reports[arm]["successes"],20) >= Fraction(1,10):
                selected = arm
                break
        if selected is None and "calib" in reports:
            selected = "calib"
    return {"selection_rule": "First promoted level at least 0.10 below BF16 development macro; otherwise maximal calib",
            "selection_uses_heldout": False, "selected_recipe": selected, "arms": reports,
            "source_sha256": sources,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "paired_scope": "environment initial states; not per-episode diffusion random streams"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    report = audit_development(args.root,args.arms)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"selected_recipe":report["selected_recipe"],
                      "scores":{k:f"{v['successes']}/{v['episodes']}" for k,v in report["arms"].items()}}))
