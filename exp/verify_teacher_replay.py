"""CPU audit of successful BF16 rollout snapshots before recovery training.

This validates the tensor payload and its join to scored reset identities, not
just the capture manifest. It never constructs a policy or uses CUDA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from run_recovery_eval import TASKS, parse_log, validate_resets


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit_replay(root, protocol_file, teacher, minimum_episodes=2):
    import torch
    root, protocol_file, teacher = (Path(x).resolve() for x in (root, protocol_file, teacher))
    # Accept the evaluation directory or its observations child, and bind both
    # to their original evaluation. Arbitrary piles of .pt files are invalid.
    evaluation = root if (root / "eval_manifest.json").is_file() else root.parent
    observations = evaluation / "observations"
    require(root in (evaluation, observations), "Replay root must be teacher evaluation or observations directory")
    require(type(minimum_episodes) is int and minimum_episodes > 0, "Invalid minimum episode count")
    protocol, manifest = read(protocol_file), read(evaluation / "eval_manifest.json")
    part = protocol["partitions"]["teacher_supervision"]
    protocol_sha = digest(protocol_file)
    for key, value in {"purpose": "teacher_supervision", "checkpoint": str(teacher),
                       "protocol_sha256": protocol_sha, "tasks": TASKS,
                       "seed": part["seed"], "episodes": part["episodes_per_task"],
                       "init_state_indices": part["init_state_indices"]}.items():
        require(manifest.get(key) == value, f"Teacher evaluation mismatch: {key}")
    heldout = set(protocol["partitions"]["heldout"]["init_state_indices"])
    require(not heldout.intersection(part["init_state_indices"]), "Teacher/heldout bank overlap")
    rows = read(evaluation / "task_results.json")
    require(set(rows) == set(TASKS), "Teacher results must contain exactly ten tasks")
    tasks, accepted_paths = {}, set()
    statistics_sha, config_sha = digest(teacher / "statistics.json"), digest(teacher / "config.json")
    teacher_weights = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                       for p in sorted(teacher.glob("*.safetensors"))}
    require(bool(teacher_weights), "Teacher has no weight shards")
    for index, task in enumerate(TASKS):
        row = rows[task]
        require(row.get("returncode") == 0, f"Incomplete teacher rollout: {task}")
        raw = parse_log(evaluation / (task + ".log"))
        for key in ("results", "resets", "log_sha256", "episodes", "successes"):
            require(row.get(key) == raw[key], f"Teacher raw log mismatch: {task}/{key}")
        resets = validate_resets(row, part["seed"] + 1000 * index, part["init_state_indices"])
        folder = observations / task
        capture = read(folder / "capture_manifest.json")
        for key, value in {"source_kind": "teacher_rollout", "checkpoint_role": "teacher",
                           "finalized": True, "finalized_purpose": "teacher_supervision",
                           "student_checkpoint": str(teacher), "task_name": task,
                           "protocol_sha256": protocol_sha,
                           "student_statistics_sha256": statistics_sha,
                           "student_config_sha256": config_sha}.items():
            require(capture.get(key) == value, f"Teacher capture mismatch: {task}/{key}")
        events = folder / "reset_events.jsonl"
        require(capture.get("reset_event_file_sha256") == digest(events), f"Reset events changed: {task}")
        episodes, sources = Counter(), []
        samples = sorted(folder.glob("sample_*.pt"))
        require(len(samples) == capture.get("accepted_samples"), f"Accepted sample accounting mismatch: {task}")
        for path in samples:
            accepted_paths.add(path)
            sample = torch.load(path, map_location="cpu", weights_only=True)
            episode = sample.get("episode_index")
            require(type(episode) is int and 0 <= episode < len(resets), f"Invalid episode: {path}")
            require(row["results"][episode] is True and sample.get("episode_success") is True,
                    f"Unsuccessful or unfinalized teacher sample: {path}")
            for key, value in {"source_kind": "teacher_rollout", "checkpoint_role": "teacher",
                               "task_name": task, "student_checkpoint": str(teacher),
                               "student_statistics_sha256": statistics_sha,
                               "episode_seed": resets[episode]["seed"],
                               "init_state_index": resets[episode]["init_state_index"]}.items():
                require(sample.get(key) == value, f"Teacher sample mismatch: {path}/{key}")
            identity = sample.get("reset_identity", {})
            for key in ("seed", "episode_index", "init_state_index", "initial_state_sha256",
                        "restored_state_sha256", "init_state_bank_sha256"):
                require(identity.get(key) == resets[episode][key], f"Sample reset mismatch: {path}/{key}")
            inputs = sample.get("inputs")
            require(isinstance(inputs, dict) and bool(inputs), f"Missing model inputs: {path}")
            for key, tensor in inputs.items():
                require(torch.is_tensor(tensor), f"Non-tensor input: {path}/{key}")
                if tensor.is_floating_point():
                    require(bool(torch.isfinite(tensor).all()), f"Nonfinite input: {path}/{key}")
            action, mask = inputs.get("action"), inputs.get("action_mask")
            require(torch.is_tensor(action) and torch.is_tensor(mask), f"Missing action/mask: {path}")
            require(action.shape == mask.shape and action.ndim == 3 and action.shape[0] == 1,
                    f"Invalid action/mask shape: {path}")
            require(bool(((mask == 0) | (mask == 1)).all()) and bool(mask.any()), f"Invalid mask: {path}")
            episodes[episode] += 1
            sources.append({"path": str(path.relative_to(observations)), "sha256": digest(path)})
        # A successful episode without captured states contributes no training
        # supervision, so count distinct episodes represented in the payload.
        require(len(episodes) >= minimum_episodes,
                f"Too few successful episodes with samples: {task}: {len(episodes)} < {minimum_episodes}")
        require(sorted(episodes) == capture.get("successful_episode_indices"), f"Capture episode accounting mismatch: {task}")
        tasks[task] = {"samples": len(samples), "contributing_successful_episodes": len(episodes),
                       "samples_per_episode": dict(sorted(episodes.items())), "source_files": sources,
                       "capture_manifest_sha256": digest(folder / "capture_manifest.json"),
                       "raw_log_sha256": raw["log_sha256"]}
    require(set(root.rglob("sample_*.pt")) == accepted_paths, "Unexpected samples outside the ten declared teacher tasks")
    return {"format": "successful_teacher_replay_audit_v1", "status": "verified",
            "evaluation": str(evaluation), "observations": str(observations),
            "protocol_sha256": protocol_sha, "teacher": str(teacher),
            "teacher_weights": teacher_weights, "minimum_contributing_successful_episodes": minimum_episodes,
            "task_count": len(tasks), "sample_count": sum(t["samples"] for t in tasks.values()),
            "tasks": tasks, "implementation_sha256": digest(__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--protocol-file", required=True)
    p.add_argument("--teacher", required=True)
    # The frozen teacher partition has four episodes per task.  Requiring two
    # successful contributing episodes gives every task a useful replay while
    # avoiding a brittle all-tasks-3/4 gate on a stochastic closed-loop run.
    p.add_argument("--minimum-episodes", type=int, default=2)
    args = p.parse_args()
    print(json.dumps(audit_replay(args.root, args.protocol_file, args.teacher, args.minimum_episodes),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
