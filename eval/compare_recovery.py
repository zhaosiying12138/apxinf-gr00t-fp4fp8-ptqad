"""Verify heldout environment pairing before producing a five-arm comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from run_recovery_eval import TASKS, validate_resets

ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")
PROTOCOL_FIELDS = (
    "seed", "episodes", "tasks", "task_count", "n_envs", "task_seed_stride",
    "episode_seed_stride", "server_seed_offset", "n_action_steps",
    "max_episode_steps", "initial_state_protocol", "init_state_indices",
    "settle_steps", "protocol_file", "protocol_sha256",
)


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
    if not baseline:
        raise ValueError("Paired comparisons require at least one outcome")
    if any(type(row.get("success")) is not bool for row in baseline + opd):
        raise ValueError("Paired outcomes must be Boolean")
    both = sum(a["success"] and b["success"] for a,b in zip(baseline,opd))
    only_opd = sum(not a["success"] and b["success"] for a,b in zip(baseline,opd))
    only_baseline = sum(a["success"] and not b["success"] for a,b in zip(baseline,opd))
    neither = sum(not a["success"] and not b["success"] for a,b in zip(baseline,opd))
    difference = (only_opd-only_baseline)/len(baseline)
    return {"both_success": both,
            "only_treatment_success": only_opd,
            "only_opd_success": only_opd,
            "only_baseline_success": only_baseline, "both_fail": neither,
            "table_rows_baseline_success_fail_columns_treatment_success_fail": [[both,only_baseline],[only_opd,neither]],
            "table_rows_baseline_success_fail_columns_opd_success_fail": [[both,only_baseline],[only_opd,neither]],
            "paired_count": len(baseline), "discordant_count": only_baseline+only_opd,
            "baseline_success_rate": (both+only_baseline)/len(baseline),
            "treatment_success_rate": (both+only_opd)/len(baseline),
            "success_rate_difference_treatment_minus_baseline": difference,
            "success_rate_difference_opd_minus_baseline": difference,
            "exact_mcnemar_two_sided_p_exploratory": exact_mcnemar_p(only_baseline,only_opd)}


def _is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same_number(actual, expected):
    return _is_number(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)


def _validate_manifest(arm, manifest, protocol_paths=None):
    if not isinstance(manifest, dict):
        raise ValueError(f"{arm} manifest must be a JSON object")
    required = ("purpose", "tasks", "episodes", "seed", "init_state_indices",
                "initial_state_protocol")
    if any(key not in manifest for key in required):
        raise ValueError(f"{arm} manifest is missing a required protocol field")
    if (manifest["purpose"] != "heldout" or manifest["tasks"] != TASKS or
            type(manifest["seed"]) is not int or type(manifest["episodes"]) is not int or
            manifest["episodes"] <= 0 or
            manifest["initial_state_protocol"] != "libero10_official_bank_v1"):
        raise ValueError(f"{arm} is not a valid ten-task heldout manifest")
    indices = manifest["init_state_indices"]
    if (not isinstance(indices, list) or len(indices) != manifest["episodes"] or
            len(set(indices)) != len(indices) or any(type(x) is not int for x in indices) or
            any(x < 0 or x >= 50 for x in indices)):
        raise ValueError(f"{arm} manifest has invalid heldout initial-state indices")

    # Current versioned runs record both the protocol bytes' digest and their
    # source path.  Older manifests sometimes had only a digest; they remain
    # readable but cannot be certified as carrying the complete contract.
    versioned = "protocol_file" in manifest
    if versioned:
        missing = [key for key in PROTOCOL_FIELDS if key not in manifest]
        if missing:
            raise ValueError(f"{arm} versioned manifest is missing {missing}")
        if (type(manifest["task_count"]) is not int or manifest["task_count"] != 10 or
                type(manifest["n_envs"]) is not int or manifest["n_envs"] != 1 or
                type(manifest["task_seed_stride"]) is not int or manifest["task_seed_stride"] != 1000 or
                type(manifest["episode_seed_stride"]) is not int or manifest["episode_seed_stride"] != 1 or
                type(manifest["settle_steps"]) is not int or manifest["settle_steps"] != 10 or
                type(manifest["server_seed_offset"]) is not int or manifest["server_seed_offset"] != 10000000 or
                type(manifest["n_action_steps"]) is not int or manifest["n_action_steps"] <= 0 or
                type(manifest["max_episode_steps"]) is not int or manifest["max_episode_steps"] <= 0 or
                not isinstance(manifest["protocol_file"], str) or not manifest["protocol_file"] or
                not isinstance(manifest["protocol_sha256"], str) or
                len(manifest["protocol_sha256"]) != 64 or
                any(c not in "0123456789abcdef" for c in manifest["protocol_sha256"])):
            raise ValueError(f"{arm} versioned manifest has an invalid execution protocol")
        # A public byte-for-byte evidence copy can explicitly resolve the
        # original path to its archived protocol. Never guess a replacement.
        protocol_path = Path((protocol_paths or {}).get(
            manifest["protocol_file"], manifest["protocol_file"]))
        if not protocol_path.is_file():
            raise ValueError(f"{arm} protocol file is missing: {protocol_path}")
        actual_protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        if actual_protocol_sha != manifest["protocol_sha256"]:
            raise ValueError(f"{arm} protocol file SHA256 disagrees with its manifest")
    return versioned


def _validate_declared_accounting(arm, task, result, outcomes, expected_seed, strict):
    declarations = {
        "seed": expected_seed,
        "episodes": len(outcomes),
        "successes": sum(outcomes),
        "success_rate": sum(outcomes) / len(outcomes),
    }
    if strict:
        missing = [key for key in declarations if key not in result]
        if missing:
            raise ValueError(f"{arm}/{task} is missing raw-result accounting {missing}")
    for key, expected in declarations.items():
        if key not in result:
            continue
        actual = result[key]
        if key == "success_rate":
            matches = _same_number(actual, expected)
        else:
            matches = type(actual) is int and actual == expected
        if not matches:
            raise ValueError(f"{arm}/{task} raw-result {key} disagrees with Boolean outcomes")
    return all(key in result for key in declarations)


def _validate_summary(arm, summary, manifest, successes, episodes, macro, strict):
    expected = {
        "tasks_complete": 10,
        "total_successes": successes,
        "total_episodes": episodes,
        "macro_success_rate": macro,
        "purpose": "heldout",
        "seed": manifest["seed"],
    }
    if strict:
        missing = [key for key in expected if key not in summary]
        if missing:
            raise ValueError(f"{arm} summary is missing accounting fields {missing}")
    for key, value in expected.items():
        if key not in summary:
            continue
        actual = summary[key]
        if key == "macro_success_rate":
            matches = _same_number(actual, value)
        elif key in ("tasks_complete", "total_successes", "total_episodes", "seed"):
            matches = type(actual) is int and actual == value
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"{arm} summary {key} disagrees with raw outcomes or manifest")
    return all(key in summary for key in expected)


def _comparison(arms, baseline, treatment):
    left, right = arms[baseline]["episodes"], arms[treatment]["episodes"]
    return paired_summary(left, right) | {
        "baseline": baseline,
        "treatment": treatment,
        "per_task": {
            task: paired_summary(
                [x for x in left if x["task"] == task],
                [x for x in right if x["task"] == task],
            )
            for task in TASKS
        },
    }


def _recovery_relative_to_bf16(arms):
    rates = {name: arm["successes"] / arm["count"] for name, arm in arms.items()}
    gap = rates["bf16"] - rates["ptq"]
    methods = {}
    for name in ("qad", "continued_qad", "qad_opd"):
        gain = rates[name] - rates["ptq"]
        methods[name] = {
            "method_minus_ptq_success_rate": gain,
            "method_minus_bf16_success_rate": rates[name] - rates["bf16"],
            "fraction_of_bf16_minus_ptq_gap_recovered": gain / gap if gap > 0 else None,
        }
    return {
        "bf16_minus_ptq_success_rate_gap": gap,
        "methods": methods,
        "interpretation": (
            "Descriptive ratios on this paired heldout bank. The recovered fraction may exceed one; "
            "it is undefined when PTQ does not score below BF16 and is not an inferential statistic."
        ),
    }


def compare_round(directory, protocol_paths=None):
    directory = Path(directory)
    arms = {}
    reference = None
    source_files = {}
    all_versioned_protocols = True
    all_source_accounting_verified = True
    for arm in ARMS:
        folder = directory / f"heldout_{arm}"
        manifest = json.loads((folder / "eval_manifest.json").read_text())
        results = json.loads((folder / "task_results.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        for name in ("eval_manifest.json", "task_results.json", "summary.json"):
            path = folder / name
            source_files[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        versioned = _validate_manifest(arm, manifest, protocol_paths)
        all_versioned_protocols &= versioned
        if not isinstance(results, dict) or set(results) != set(TASKS):
            raise ValueError(f"{arm} is not a complete ten-task heldout evaluation")
        if not isinstance(summary, dict):
            raise ValueError(f"{arm} summary must be a JSON object")
        if reference is None:
            reference = manifest
        if any(manifest.get(key) != reference.get(key) for key in PROTOCOL_FIELDS):
            raise ValueError(f"{arm} evaluation protocol differs from BF16")
        pairs, successes, per_task = [], 0, {}
        for i, task in enumerate(TASKS):
            result = results[task]
            if not isinstance(result, dict) or type(result.get("returncode")) is not int or result["returncode"] != 0:
                raise ValueError(f"{arm}/{task} has incomplete results")
            outcomes = result.get("results")
            if not isinstance(outcomes, list) or len(outcomes) != manifest["episodes"]:
                raise ValueError(f"{arm}/{task} has incomplete results")
            if any(type(outcome) is not bool for outcome in outcomes):
                raise ValueError(f"{arm}/{task} outcomes must be Boolean")
            expected_seed = manifest["seed"] + 1000*i
            all_source_accounting_verified &= _validate_declared_accounting(
                arm, task, result, outcomes, expected_seed, versioned)
            resets = validate_resets(result, expected_seed, manifest["init_state_indices"])
            for outcome, reset in zip(outcomes, resets):
                pairs.append({"task": task, "episode_index": reset["episode_index"],
                              "init_state_index": reset["init_state_index"],
                              "initial_state_sha256": reset["initial_state_sha256"],
                              "restored_state_sha256": reset["restored_state_sha256"],
                              "init_state_bank_sha256": reset["init_state_bank_sha256"],
                              "success": outcome})
            task_successes = sum(outcomes)
            successes += task_successes
            per_task[task] = {"successes": task_successes, "episodes": len(outcomes),
                              "success_rate": task_successes/len(outcomes)}
        if arm != "bf16":
            for actual, expected in zip(pairs, arms["bf16"]["episodes"]):
                if {k:v for k,v in actual.items() if k != "success"} != {k:v for k,v in expected.items() if k != "success"}:
                    raise ValueError(f"{arm} initial state does not pair with BF16: {actual['task']}/{actual['episode_index']}")
        macro = sum(x["success_rate"] for x in per_task.values())/10
        all_source_accounting_verified &= _validate_summary(
            arm, summary, manifest, successes, len(pairs), macro, versioned)
        arms[arm] = {"successes": successes, "count": len(pairs),
                     "macro_success_rate": macro,
                     "per_task": per_task, "episodes": pairs}
    comparisons = {
        "ptq_vs_bf16": _comparison(arms, "bf16", "ptq"),
        "qad_vs_ptq": _comparison(arms, "ptq", "qad"),
        "continued_qad_vs_ptq": _comparison(arms, "ptq", "continued_qad"),
        "opd_vs_ptq": _comparison(arms, "ptq", "qad_opd"),
        "opd_vs_qad": _comparison(arms, "qad", "qad_opd"),
        "opd_vs_continued_qad": _comparison(arms, "continued_qad", "qad_opd"),
    }
    comparison = {"environment_pairing_verified": True,
                  "protocol_consistency_verified": all_versioned_protocols,
                  "common_manifest_fields_consistent": True,
                  "source_accounting_verified": all_source_accounting_verified,
                  "paired_scope": "official environment initial states; policy noise is seeded per task",
                  "arms": arms,
                  "source_files": source_files,
                  "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "statistical_note": "McNemar exact two-sided p-values are exploratory conditional binomial calculations on discordant paired episodes. Episodes share ten tasks and policies; independence across all episodes is not established. Overall and per-task calculations have no multiple-testing adjustment and do not establish broader task generalization. No automatic significance decision is made.",
                  "recovery_relative_to_bf16": _recovery_relative_to_bf16(arms),
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
