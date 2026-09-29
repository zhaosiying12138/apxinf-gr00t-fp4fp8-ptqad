"""Verify heldout environment pairing before producing a five-arm comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from run_recovery_eval import TASKS, validate_resets

ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")


def exact_mcnemar_p(only_baseline, only_opd):
    """Two-sided exact conditional binomial test of discordant pairs, p=.5."""
    n = only_baseline + only_opd
    if n == 0:
        return 1.0
    lower = sum(math.comb(n, k) for k in range(min(only_baseline, only_opd) + 1))
    return min(1.0, math.ldexp(float(lower), 1-n))


def paired_summary(baseline, opd):
    if len(baseline) != len(opd):
        raise ValueError("Paired comparisons require equal numbers of outcomes")
    both = sum(a["success"] and b["success"] for a,b in zip(baseline,opd))
    only_opd = sum(not a["success"] and b["success"] for a,b in zip(baseline,opd))
    only_baseline = sum(a["success"] and not b["success"] for a,b in zip(baseline,opd))
    neither = sum(not a["success"] and not b["success"] for a,b in zip(baseline,opd))
    return {"both_success": both, "only_opd_success": only_opd,
            "only_baseline_success": only_baseline, "both_fail": neither,
            "table_rows_baseline_success_fail_columns_opd_success_fail": [[both,only_baseline],[only_opd,neither]],
            "paired_count": len(baseline), "discordant_count": only_baseline+only_opd,
            "success_rate_difference_opd_minus_baseline": (only_opd-only_baseline)/len(baseline),
            "exact_mcnemar_two_sided_p_exploratory": exact_mcnemar_p(only_baseline,only_opd)}


def compare_round(directory):
    directory = Path(directory)
    arms = {}
    reference = None
    source_files = {}
    fields = ("seed", "episodes", "tasks", "n_envs", "n_action_steps", "max_episode_steps",
              "initial_state_protocol", "init_state_indices", "settle_steps", "protocol_sha256")
    for arm in ARMS:
        folder = directory / f"heldout_{arm}"
        manifest = json.loads((folder / "eval_manifest.json").read_text())
        results = json.loads((folder / "task_results.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        for name in ("eval_manifest.json", "task_results.json", "summary.json"):
            path = folder / name
            source_files[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if (manifest["purpose"] != "heldout" or manifest["tasks"] != TASKS or
            set(results) != set(TASKS) or summary["tasks_complete"] != 10):
            raise ValueError(f"{arm} is not a complete ten-task heldout evaluation")
        if reference is None:
            reference = manifest
        if any(manifest.get(key) != reference.get(key) for key in fields):
            raise ValueError(f"{arm} evaluation protocol differs from BF16")
        pairs, successes, per_task = [], 0, {}
        for i, task in enumerate(TASKS):
            result = results[task]
            if result.get("returncode") != 0 or len(result["results"]) != manifest["episodes"]:
                raise ValueError(f"{arm}/{task} has incomplete results")
            resets = validate_resets(result, manifest["seed"] + 1000*i, manifest["init_state_indices"])
            for outcome, reset in zip(result["results"], resets):
                pairs.append({"task": task, "episode_index": reset["episode_index"],
                              "init_state_index": reset["init_state_index"],
                              "initial_state_sha256": reset["initial_state_sha256"],
                              "restored_state_sha256": reset["restored_state_sha256"],
                              "init_state_bank_sha256": reset["init_state_bank_sha256"],
                              "success": outcome})
            successes += sum(result["results"])
            per_task[task] = {"successes": sum(result["results"]), "episodes": len(result["results"]),
                              "success_rate": sum(result["results"])/len(result["results"])}
        if arm != "bf16":
            for actual, expected in zip(pairs, arms["bf16"]["episodes"]):
                if {k:v for k,v in actual.items() if k != "success"} != {k:v for k,v in expected.items() if k != "success"}:
                    raise ValueError(f"{arm} initial state does not pair with BF16: {actual['task']}/{actual['episode_index']}")
        arms[arm] = {"successes": successes, "count": len(pairs),
                     "macro_success_rate": sum(x["success_rate"] for x in per_task.values())/10,
                     "per_task": per_task, "episodes": pairs}
    comparisons = {}
    for baseline in ("qad", "continued_qad"):
        left, right = arms[baseline]["episodes"], arms["qad_opd"]["episodes"]
        comparisons[f"opd_vs_{baseline}"] = paired_summary(left,right) | {
            "baseline": baseline,
            "per_task": {task: paired_summary([x for x in left if x["task"]==task],
                                              [x for x in right if x["task"]==task]) for task in TASKS}}
    comparison = {"environment_pairing_verified": True,
                  "paired_scope": "official environment initial states; policy noise is seeded per task",
                  "arms": arms,
                  "source_files": source_files,
                  "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "statistical_note": "McNemar exact two-sided p-values are exploratory conditional binomial calculations on discordant paired episodes. Episodes share ten tasks and policies; independence across all episodes is not established. The two overall comparisons and per-task calculations have no multiple-testing adjustment and do not establish broader task generalization.",
                  **comparisons}
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    result = compare_round(args.round)
    output = Path(args.round) / "paired_comparison.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({arm: {k:v for k,v in value.items() if k != "episodes"}
                      for arm,value in result["arms"].items()}, indent=2))
