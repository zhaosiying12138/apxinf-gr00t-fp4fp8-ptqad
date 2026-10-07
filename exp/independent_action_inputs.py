"""Freeze independent full-trajectory diagnostic observations without training views."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_diagnostic_inputs(capture_root, protocol_file):
    """Audit complete raw rollouts, then select prespecified time ranks and seeds.

    No sample files are materialized; failed episodes remain eligible. Repeated
    observation paths represent different declared noise seeds, not new states.
    """
    import torch
    from eval.run_recovery_eval import TASKS, capture_sampling_config, parse_log, validate_resets
    from exp.independent_action_protocol import validate_diagnostic_protocol
    from rl.capture_sampling import _verify_finalized, select_indices
    from rl.checkpoint_identity import checkpoint_files

    capture_root, protocol_file = (Path(x).expanduser().resolve() for x in (capture_root, protocol_file))
    reference = validate_diagnostic_protocol(protocol_file)
    protocol, config = reference["data"], reference["settings"]
    source = capture_root.parent
    require(capture_root == source / "observations", "Expected raw diagnostic observations directory")
    manifest, summary, results = (read(source / name) for name in (
        "eval_manifest.json", "summary.json", "task_results.json"))
    partition = protocol["partitions"]["diagnostics"]
    expected_checkpoint = reference["allowed_teacher"]["checkpoint"]
    reference_sha = hashlib.sha256(json.dumps(reference, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    require(manifest.get("purpose") == "diagnostics" and manifest.get("tasks") == TASKS
            and set(results) == set(TASKS), "Independent diagnostics require all ten diagnostic tasks")
    require(manifest.get("protocol_sha256") == sha(protocol_file)
            and Path(manifest["protocol_file"]).resolve() == protocol_file,
            "Diagnostic capture protocol identity changed")
    require(Path(manifest["checkpoint"]).resolve() == Path(expected_checkpoint).resolve(),
            "Diagnostic capture used a different final BF16 reference")
    require(manifest.get("checkpoint_files") == checkpoint_files(expected_checkpoint)
            and summary.get("checkpoint_files_verified_unchanged") is True,
            "Diagnostic capture lacks unchanged reference checkpoint evidence")
    require(manifest.get("diagnostic_reference") == reference
            and manifest.get("training_eligible") is False
            and summary.get("diagnostic_reference_verified_unchanged") is True
            and summary.get("training_eligible") is False,
            "Diagnostic reference receipt or training eligibility changed")
    require(manifest.get("n_envs") == 1
            and manifest.get("episodes") == partition["episodes_per_task"]
            and manifest.get("init_state_indices") == partition["init_state_indices"]
            and manifest.get("seed") == partition["seed"], "Diagnostic capture partition differs")
    contract = protocol["evaluation_contract"]
    for key in ("max_episode_steps", "task_seed_stride", "episode_seed_stride", "settle_steps",
                "n_action_steps", "server_seed_offset", "initial_state_protocol"):
        require(manifest.get(key) == contract[key], "Diagnostic rollout contract differs: " + key)
    sampling = capture_sampling_config(partition, "diagnostics", manifest["episodes"], manifest["max_episode_steps"])
    require(sampling is not None and manifest.get("capture_sampling") == sampling,
            "Diagnostic full-trajectory sampling differs")
    samples, source_files, temporal, total_successes, observations = [], {}, {}, 0, 0
    stats = set()
    for task_index, task in enumerate(TASKS):
        log = source / (task + ".log")
        raw = parse_log(log)
        result = results[task]
        seed = partition["seed"] + task_index * contract["task_seed_stride"]
        require(result.get("returncode") == 0 and result.get("seed") == seed
                and raw["episodes"] == manifest["episodes"]
                and all(result.get(key) == value for key, value in raw.items()),
                "Incomplete or changed diagnostic rollout: " + task)
        validate_resets(raw, seed, partition["init_state_indices"])
        directory, capture, counts, candidates = _verify_finalized(capture_root / task, task)
        require(capture.get("purpose") == capture.get("finalized_purpose") == "diagnostics"
                and capture.get("source_kind") == "diagnostic_rollout"
                and capture.get("checkpoint_role") == "diagnostic_reference"
                and capture.get("training_eligible") is False
                and capture.get("diagnostic_reference") == reference,
                "Diagnostic capture mislabeled as training data")
        require(capture.get("student_checkpoint") == str(Path(expected_checkpoint).resolve())
                and capture.get("protocol_sha256") == manifest["protocol_sha256"]
                and capture.get("seed") == seed
                and capture.get("init_state_indices") == ",".join(map(str, partition["init_state_indices"])),
                "Diagnostic candidate source identity differs")
        require(capture.get("scored_result") == {"results": raw["results"], "resets": raw["resets"]}
                and result.get("capture") == capture.get("finalization_summary"),
                "Diagnostic candidate outcomes differ from raw rollout")
        require(capture.get("every_server_calls") == sampling["every_server_calls"]
                and capture.get("per_episode_limit") == sampling["safety_candidates_per_episode"]
                and capture.get("per_task_limit") == sampling["safety_candidates_per_task"]
                and capture.get("total_limit") == sampling["safety_candidates_per_task"],
                "Diagnostic runtime query sampling differs")
        groups = defaultdict(list)
        for path, sample in candidates:
            require(sample.get("source_kind") == "diagnostic_rollout"
                    and sample.get("checkpoint_role") == "diagnostic_reference"
                    and sample.get("purpose") == "diagnostics"
                    and sample.get("training_eligible") is False
                    and sample.get("diagnostic_reference_sha256") == reference_sha,
                    "Candidate is not an independent diagnostic observation")
            inputs = sample.get("inputs", {})
            require(inputs and all(torch.is_tensor(value) and (not value.is_floating_point()
                    or bool(torch.isfinite(value).all())) for value in inputs.values()), "Invalid observation tensors")
            stats.add(sample.get("student_statistics_sha256"))
            groups[sample["episode_index"]].append((path, sample))
        task_temporal = []
        for episode, rows in sorted(groups.items()):
            rows.sort(key=lambda item: item[1]["episode_call"])
            ranks = select_indices(len(rows), config["windows_per_episode"], "stratified")
            task_temporal.append({"episode_index": episode, "candidate_count": len(rows),
                "actual_queries": counts["episode_query_counts"][str(episode)],
                "selected_ranks": ranks, "episode_success": raw["results"][episode]})
            for rank in ranks:
                path, sample = rows[rank]
                observation_index = observations
                observations += 1
                for noise_index, noise_seed in enumerate(config["noise_seeds"]):
                    samples.append({"path": str(path), "sha256": sha(path), "task": task,
                        "seed": noise_seed, "noise_index": noise_index,
                        "observation_index": observation_index,
                        "init_state_index": sample["init_state_index"], "episode_index": episode,
                        "server_call": sample["server_call"], "episode_call": sample["episode_call"],
                        "episode_success": sample["episode_success"]})
        temporal[task] = task_temporal
        total_successes += raw["successes"]
        for path in (log, directory / "capture_manifest.json", directory / "capture_counts.json"):
            source_files[str(path)] = sha(path)
    require(summary.get("tasks_complete") == len(TASKS)
            and summary.get("total_episodes") == len(TASKS) * manifest["episodes"]
            and summary.get("total_successes") == total_successes
            and summary.get("purpose") == "diagnostics", "Diagnostic capture summary differs")
    require(len(stats) == 1 and next(iter(stats)) == sha(Path(expected_checkpoint) / "statistics.json"),
            "Diagnostic observation normalization differs")
    for name in ("eval_manifest.json", "task_results.json", "summary.json"):
        source_files[str(source / name)] = sha(source / name)
    require(sha(protocol_file) == reference["diagnostic_protocol_sha256"], "Diagnostic protocol changed during audit")
    return {"format": "gr00t_action_inputs_v2", "purpose": "diagnostics",
        "interpretation": "independent_observation_diagnostic_not_training_or_success_rate",
        "protocol_file": str(protocol_file), "protocol_sha256": reference["diagnostic_protocol_sha256"],
        "original_model_protocol_sha256": reference["original_model_protocol_sha256"],
        "final_reference": reference["source_final"], "capture_root": str(capture_root),
        "statistics_sha256": next(iter(stats)), "selection": "stratified query ranks; no success filtering",
        "windows_per_episode": config["windows_per_episode"], "noise_seeds": config["noise_seeds"],
        "seed": config["noise_seeds"][0], "unique_observations": observations,
        "sample_count": len(samples), "samples": samples, "temporal_coverage": temporal,
        "source_files": source_files, "implementation_sha256": sha(__file__)}
