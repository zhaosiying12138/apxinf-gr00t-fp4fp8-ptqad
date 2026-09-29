"""Audit official-bank development pairing and apply the preregistered ladder rule."""
import argparse
from fractions import Fraction
import hashlib
import json
import re
from pathlib import Path

from run_recovery_eval import TASKS, validate_resets

ORDER = ["bf16", "fp8", "head_ffn", "head_lang", "head_lang_vision", "calib"]


def load_protocol(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    partitions = data.get("partitions", data)
    development = partitions.get("development")
    if not isinstance(development, dict):
        raise ValueError(f"Protocol {path} has no development partition")
    indices = development.get("init_state_indices")
    if (not isinstance(indices, list) or not indices or
            any(type(x) is not int for x in indices) or
            len(set(indices)) != len(indices) or any(x < 0 or x >= 50 for x in indices)):
        raise ValueError(f"Protocol {path} has invalid development.init_state_indices")
    seed = development.get("seed")
    episodes = development.get("episodes_per_task")
    if type(seed) is not int or type(episodes) is not int or episodes < 1:
        raise ValueError(f"Protocol {path} has invalid development seed/episodes")
    if episodes != len(indices):
        raise ValueError("development episodes_per_task must equal init_state_indices length")
    selection = data.get("selection", {})
    candidates = selection.get("pressure_candidates")
    pressure_rule = selection.get("pressure_rule")
    high_fp4_rule = candidates is not None and pressure_rule is not None
    if candidates is None:
        order = ORDER
    else:
        if (not isinstance(candidates, list) or not candidates or
                any(not isinstance(x, str) or not re.fullmatch(r"[a-z0-9_]+", x)
                    for x in candidates) or len(set(candidates)) != len(candidates)):
            raise ValueError(f"Protocol {path} has invalid selection.pressure_candidates")
        if any(x not in ORDER[1:] for x in candidates):
            raise ValueError(f"Protocol {path} declares an unknown pressure candidate")
        order = ["bf16", *candidates]
    if high_fp4_rule:
        if (not isinstance(pressure_rule, dict) or
                pressure_rule.get("choose") != "highest_fp4" or
                not isinstance(pressure_rule.get("min_drop_from_bf16"), (int, float)) or
                not isinstance(pressure_rule.get("min_absolute_success"), (int, float)) or
                not 0 <= pressure_rule["min_drop_from_bf16"] <= 1 or
                not 0 <= pressure_rule["min_absolute_success"] <= 1):
            raise ValueError(f"Protocol {path} has invalid selection.pressure_rule")
    return {"path": path, "data": data, "seed": seed, "episodes": episodes,
            "indices": indices, "order": order, "selection": selection,
            "high_fp4_rule": high_fp4_rule,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def development_order(protocol_file=None):
    """Return the declared development ladder for a protocol.

    Omitting ``protocol_file`` preserves the historical six-arm ladder.
    Versioned protocols may deliberately expose only the preregistered pressure
    candidates, so a heldout result cannot silently add a recipe.
    """
    return load_protocol(protocol_file)["order"] if protocol_file else ORDER


def audit_development(root, arms, protocol_file=None):
    protocol = load_protocol(protocol_file) if protocol_file else None
    expected_order = protocol["order"] if protocol else ORDER
    if arms != expected_order[:len(arms)]:
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
        expected_seed = protocol["seed"] if protocol else 330000
        expected_episodes = protocol["episodes"] if protocol else 2
        expected_indices = protocol["indices"] if protocol else [0, 1]
        expected_protocol_sha = protocol["sha256"] if protocol else None
        if (manifest["purpose"] != "development" or manifest["seed"] != expected_seed or
            manifest["episodes"] != expected_episodes or manifest["init_state_indices"] != expected_indices or
            manifest["n_envs"] != 1 or manifest["tasks"] != TASKS or set(tasks) != set(TASKS) or
            manifest["initial_state_protocol"] != "libero10_official_bank_v1" or
            summary["tasks_complete"] != 10):
            raise ValueError(f"{arm}: not a complete declared development run")
        if expected_protocol_sha is not None and manifest.get("protocol_sha256") != expected_protocol_sha:
            raise ValueError(f"{arm}: evaluation used another protocol file")
        signatures, per_task = [], {}
        for index, task in enumerate(TASKS):
            row = tasks[task]
            if row["returncode"] != 0 or len(row["results"]) != expected_episodes:
                raise ValueError(f"{arm}/{task}: incomplete episode results")
            resets = validate_resets(row, expected_seed+1000*index, expected_indices)
            signatures.extend([{k:r[k] for k in ("episode_index", "seed", "init_state_index",
                "init_state_bank_sha256", "restored_state_sha256", "initial_state_sha256")} for r in resets])
            per_task[task] = {"results": row["results"], "successes": sum(row["results"]),
                              "episodes": expected_episodes,
                              "success_rate": sum(row["results"])/expected_episodes}
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
        total_episodes = 10 * expected_episodes
        reports[arm] = {"checkpoint": manifest["checkpoint"], "successes": successes,
                        "episodes": total_episodes, "macro_success_rate": successes/total_episodes,
                        "per_task": per_task, "environment_pairing_verified": True,
                        "wall_seconds_including_server_loads": summary["wall_seconds_including_server_loads"]}
    selected = None
    if reports:
        base_rate = Fraction(reports["bf16"]["successes"], reports["bf16"]["episodes"])
        if protocol and protocol["high_fp4_rule"]:
            # The v3 rule is explicit: only a sufficiently degraded but still
            # usable PTQ arm can be selected as the recovery stress window.
            candidates = protocol["selection"].get("pressure_candidates", expected_order[1:])
            pressure_rule = protocol["selection"]["pressure_rule"]
            gap_threshold = Fraction(str(pressure_rule["min_drop_from_bf16"]))
            floor = Fraction(str(pressure_rule["min_absolute_success"]))
            for arm in candidates:
                if arm not in reports:
                    continue
                rate = Fraction(reports[arm]["successes"], reports[arm]["episodes"])
                if base_rate - rate >= gap_threshold and rate >= floor:
                    selected = arm
            # Candidate order is increasing FP4 coverage.  All candidates must
            # finish before the highest qualifying pressure can be frozen.
            if any(arm not in reports for arm in candidates):
                selected = None
            rule = (f"Highest-FP4 declared pressure candidate with development macro success at least "
                    f"{float(gap_threshold):.2f} below BF16 and at least {float(floor):.2f} absolute; "
                    "evaluate all candidates; no fallback if none qualifies")
        else:
            base = reports["bf16"]["successes"]
            for arm in arms[2:]:
                if Fraction(base-reports[arm]["successes"], reports[arm]["episodes"]) >= Fraction(1,10):
                    selected = arm
                    break
            if selected is None and "calib" in reports:
                selected = "calib"
            rule = "First promoted level at least 0.10 below BF16 development macro; otherwise maximal calib"
    else:
        rule = "No development arms completed"
    result = {"selection_rule": rule,
            "selection_uses_heldout": False, "selected_recipe": selected, "arms": reports,
            "source_sha256": sources,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "paired_scope": "environment initial states; not per-episode diffusion random streams"}
    if protocol:
        result.update({"protocol_file": str(protocol["path"]),
                       "protocol_sha256": protocol["sha256"],
                       "protocol_version": protocol["data"].get("version"),
                       "development_partition": {"seed": protocol["seed"],
                                                   "episodes_per_task": protocol["episodes"],
                                                   "init_state_indices": protocol["indices"]},
                       "pressure_candidates": protocol["selection"].get("pressure_candidates")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol-file", help="Versioned protocol JSON; omitted for the historical ladder")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    report = audit_development(args.root,args.arms,args.protocol_file)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"selected_recipe":report["selected_recipe"],
                      "scores":{k:f"{v['successes']}/{v['episodes']}" for k,v in report["arms"].items()}}))
