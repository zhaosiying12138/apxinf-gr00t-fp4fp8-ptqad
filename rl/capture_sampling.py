"""CPU-only finalization and deterministic views of complete-query captures.

Candidate coverage means every declared query interval through the last observed
policy query. It says nothing about semantic phases or environment completion;
the supplied scored rollout results remain the authority for completion.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

CANDIDATE_SCHEMA = "fp4vla_capture_candidates_v1"
PLAN_SCHEMA = "fp4vla_capture_view_plan_v1"
VIEW_SCHEMA = "fp4vla_capture_view_v1"
MODE = "full_trajectory_candidates"
HASH_FIELDS = ("initial_state_sha256", "restored_state_sha256", "init_state_bank_sha256")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    _require(not temporary.exists(), f"Unfinished output exists: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def select_indices(n, k=4, mode="stratified"):
    """Select zero-based ranks; each floor-partition contributes its lower median."""
    _require(type(n) is int and n >= 0, "n must be a nonnegative integer")
    _require(type(k) is int and k > 0, "k must be a positive integer")
    _require(mode in ("head", "stratified"), "Unknown selection mode")
    if mode == "head" or n <= k:
        return list(range(min(n, k)))
    return [(j * n // k + (j + 1) * n // k - 1) // 2 for j in range(k)]


def _load_candidates(directory, task_name, result, purpose):
    """Validate all inputs before finalization writes any success labels."""
    import torch
    directory = Path(directory).resolve()
    manifest = _read(directory / "capture_manifest.json")
    counts = _read(directory / "capture_counts.json")
    _require(purpose in ("teacher_supervision", "collection"), "Unsupported capture purpose")
    expected_source = "teacher_rollout" if purpose == "teacher_supervision" else "student_rollout"
    _require(manifest.get("sampling_mode") == MODE, "Not a full candidate capture")
    _require(type(manifest.get("n_envs")) is int and manifest["n_envs"] == 1,
             "Full candidate capture requires one environment")
    _require(manifest.get("sampling_interval_basis") == "episode_call"
             and manifest.get("candidate_prefix") == "candidate_", "Candidate sampling contract differs")
    _require(isinstance(manifest.get("protocol_sha256"), str)
             and re.fullmatch(r"[0-9a-f]{64}", manifest["protocol_sha256"]),
             "Missing protocol hash; external protocol identity must also be checked by the caller")
    _require(manifest.get("task_name") == task_name, "Manifest task differs")
    _require(manifest.get("source_kind") == expected_source, "Manifest source kind differs")
    _require(manifest.get("purpose") == purpose, "Manifest purpose differs")
    _require(not list(directory.rglob("sample_*.pt")) and not list(directory.rglob("rejected_*.pt")),
             "Raw candidates must not contain training-view files")
    every = manifest.get("every_server_calls")
    safety = manifest.get("per_episode_limit")
    _require(type(every) is int and every > 0, "Invalid query interval")
    _require(type(safety) is int and safety > 0, "Missing per-episode safety bound")
    _require(counts.get("safety_limit_hit") is False, "Capture hit or omitted its safety status")
    values, resets = result.get("results"), result.get("resets")
    _require(isinstance(values, list) and values and isinstance(resets, list)
             and len(values) == len(resets), "Scored results/reset count differs or is empty")
    _require(all(type(x) is bool or (type(x) is int and x in (0, 1)) for x in values),
             "Scored outcomes must be bool or 0/1")
    query_counts = counts.get("episode_query_counts")
    _require(isinstance(query_counts, dict) and set(query_counts) == {str(i) for i in range(len(values))},
             "Missing scored episode queries or extra unscored episode queries")
    _require(all(type(x) is int and x > 0 for x in query_counts.values()),
             "Every scored episode must have observed policy queries")
    _require(type(counts.get("calls")) is int and counts["calls"] == sum(query_counts.values()),
             "Task query total differs from episode totals")
    files = sorted(directory.glob("candidate_*.pt"))
    _require(not list(directory.glob("*.tmp")), "Unfinished capture write exists")
    _require(files and len(list(directory.rglob("candidate_*.pt"))) == len(files),
             "Candidates must be nonempty and directly inside the task directory")
    _require(type(counts.get("saved")) is int and counts["saved"] == len(files)
             and type(counts.get("candidates_saved")) is int and counts["candidates_saved"] == len(files),
             "Candidate file count differs from recorded saved count")
    expected = []
    prefix = 0
    for episode, reset in enumerate(resets):
        _require(isinstance(reset, dict), "Malformed reset identity")
        _require(reset.get("task_name", task_name) == task_name
                 and reset.get("episode_index", episode) == episode, "Scored reset identity differs")
        _require(type(reset.get("seed")) is int and type(reset.get("init_state_index")) is int,
                 "Scored reset lacks seed/index")
        for key in HASH_FIELDS:
            _require(isinstance(reset.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", reset[key]),
                     "Scored reset lacks a SHA-256: " + key)
        queries = query_counts[str(episode)]
        calls = list(range(1, queries + 1, every))
        _require(len(calls) <= safety, "Candidate episode exceeds safety bound")
        expected.extend((episode, query, prefix + query) for query in calls)
        prefix += queries
    _require(len(files) == len(expected), "Incomplete query-interval coverage through last observed query")
    loaded = []
    for index, (path, (episode, episode_call, server_call)) in enumerate(zip(files, expected)):
        _require(path.name == f"candidate_{index:06d}.pt", "Candidate filenames are not contiguous")
        sample = torch.load(path, map_location="cpu", weights_only=True)
        _require(isinstance(sample, dict) and isinstance(sample.get("inputs"), dict)
                 and sample["inputs"], "Candidate has no model inputs")
        reset = resets[episode]
        identity = sample.get("reset_identity")
        _require(isinstance(identity, dict), "Candidate lacks reset identity")
        fields = {"task_name": task_name, "episode_index": episode, "episode_seed": reset["seed"],
                  "init_state_index": reset["init_state_index"], "capture_index": index,
                  "episode_call": episode_call, "server_call": server_call, "source_kind": expected_source,
                  "sampling_mode": MODE}
        for key, value in fields.items():
            _require(type(sample.get(key)) is type(value) and sample[key] == value,
                     f"Candidate {path.name} differs in {key}")
        for key in ("student_checkpoint", "student_statistics_sha256", "checkpoint_role"):
            if key in manifest:
                _require(sample.get(key) == manifest[key], f"Candidate {path.name} differs in {key}")
        for key in ("task_sha256", "instruction_sha256"):
            if key in sample:
                _require(isinstance(sample.get("task_text"), str) and sample[key] ==
                         hashlib.sha256(sample["task_text"].encode()).hexdigest(),
                         f"Candidate {path.name} instruction hash differs")
        identity_fields = {"task_name": task_name, "episode_index": episode,
                           "seed": reset["seed"], "init_state_index": reset["init_state_index"]}
        identity_fields.update({key: reset[key] for key in HASH_FIELDS})
        for key, value in identity_fields.items():
            _require(identity.get(key) == value, f"Candidate {path.name} reset differs in {key}")
        if "settled_state_sha256" in reset:
            _require(identity.get("settled_state_sha256") == reset["settled_state_sha256"],
                     "Candidate settled reset differs")
        success = sample.get("episode_success")
        _require(success is None or (type(success) is bool and success == bool(values[episode])),
                 "Candidate has conflicting success metadata")
        loaded.append((path, sample))
    if "per_episode" in counts or "per_task" in counts:
        per_episode, per_task = defaultdict(int), defaultdict(int)
        for _, sample in loaded:
            task = sample.get("task_sha256")
            _require(isinstance(task, str) and re.fullmatch(r"[0-9a-f]{64}", task),
                     "Runtime task counts require instruction identities")
            per_episode[f"{task}:{sample['episode_index']}"] += 1
            per_task[task] += 1
        if "per_episode" in counts:
            _require(counts["per_episode"] == dict(per_episode), "Per-episode saved counts differ")
        if "per_task" in counts:
            _require(counts["per_task"] == dict(per_task), "Per-task saved counts differ")
    return directory, manifest, counts, loaded


def finalize_candidates(directory, task_name, result, purpose):
    """Finalize raw candidates in place; preserve failures and never create sample files."""
    import torch
    directory, manifest, counts, loaded = _load_candidates(directory, task_name, result, purpose)
    scored = {"results": [bool(x) for x in result["results"]], "resets": result["resets"]}
    if manifest.get("finalized") is True:
        _require(manifest.get("schema") == CANDIDATE_SCHEMA and manifest.get("scored_result") == scored,
                 "Capture was finalized against a different result")
        _verify_finalized(directory, task_name)
        return manifest["finalization_summary"]
    before = [{"filename": path.name, "sha256": _sha(path)} for path, _ in loaded]
    for path, sample in loaded:
        sample.update(episode_success=scored["results"][sample["episode_index"]],
                      episode_result_source="run_recovery_eval.task_results",
                      episode_result_index=sample["episode_index"])
        temporary = path.with_name(path.name + ".tmp")
        _require(not temporary.exists(), f"Unfinished candidate exists: {temporary}")
        torch.save(sample, temporary)
        temporary.replace(path)
    records = [{"filename": path.name, "sha256": _sha(path), "bytes": path.stat().st_size,
                "capture_index": sample["capture_index"], "episode_index": sample["episode_index"],
                "episode_call": sample["episode_call"], "server_call": sample["server_call"],
                "episode_success": sample["episode_success"]} for path, sample in loaded]
    successes = sum(sample["episode_success"] for _, sample in loaded)
    summary = {"total_candidates": len(loaded), "successful_candidates": successes,
               "failed_candidates": len(loaded) - successes,
               "scored_episode_count": len(scored["results"]),
               "successful_episode_count": sum(scored["results"]),
               "successful_episode_indices": [i for i, ok in enumerate(scored["results"]) if ok],
               "training_samples_materialized": 0}
    manifest.update(schema=CANDIDATE_SCHEMA, finalized=True, finalized_purpose=purpose,
                    scored_result=scored, scored_result_sha256=_json_sha(scored),
                    capture_counts_sha256=_sha(directory / "capture_counts.json"),
                    pre_finalization_candidates=before, candidates=records,
                    finalization_summary=summary,
                    coverage_scope="declared query intervals through each scored episode's last observed query; not semantic phases",
                    legacy_audit_compatible=False)
    event_file = manifest.get("reset_event_file")
    if event_file and Path(event_file).is_file():
        manifest["reset_event_file_sha256"] = _sha(event_file)
    _write(directory / "capture_manifest.json", manifest)
    return summary


def _verify_finalized(directory, task_name):
    directory = Path(directory).resolve()
    manifest = _read(directory / "capture_manifest.json")
    _require(manifest.get("schema") == CANDIDATE_SCHEMA and manifest.get("finalized") is True,
             "Raw observations must be fully finalized")
    result = manifest.get("scored_result")
    _require(isinstance(result, dict) and manifest.get("scored_result_sha256") == _json_sha(result),
             "Scored result identity changed")
    _require(manifest.get("capture_counts_sha256") == _sha(directory / "capture_counts.json"),
             "Capture query counts changed after finalization")
    event_hash = manifest.get("reset_event_file_sha256")
    if event_hash is not None:
        event_path = manifest.get("reset_event_file")
        _require(isinstance(event_path, str) and Path(event_path).is_file()
                 and _sha(event_path) == event_hash, "Reset event file changed after finalization")
    _, _, counts, loaded = _load_candidates(directory, task_name, result, manifest["finalized_purpose"])
    records = manifest.get("candidates")
    _require(isinstance(records, list) and len(records) == len(loaded), "Candidate inventory is incomplete")
    for (path, sample), record in zip(loaded, records):
        _require(record == {"filename": path.name, "sha256": _sha(path), "bytes": path.stat().st_size,
                            **{key: sample[key] for key in ("capture_index", "episode_index", "episode_call",
                                                           "server_call", "episode_success")}},
                 "Candidate changed after finalization: " + path.name)
        _require(type(sample["episode_success"]) is bool, "Candidate success was not finalized")
    return directory, manifest, counts, loaded


def plan_capture_view(directory, task_name, *, mode, windows_per_episode=4):
    """Plan without modifying source; selection uses query ranks, never tensor values."""
    select_indices(0, windows_per_episode, mode)
    directory, manifest, counts, loaded = _verify_finalized(directory, task_name)
    grouped = defaultdict(list)
    for path, sample in loaded:
        grouped[sample["episode_index"]].append((path, sample))
    entries, episodes = [], []
    for episode, candidates in sorted(grouped.items()):
        ranks = select_indices(len(candidates), windows_per_episode, mode)
        episodes.append({"episode_index": episode, "candidate_count": len(candidates),
                         "actual_queries": counts["episode_query_counts"][str(episode)],
                         "selected_candidate_ranks": ranks, "selected_count": len(ranks),
                         "short_episode": len(candidates) < windows_per_episode})
        for rank in ranks:
            path, sample = candidates[rank]
            index = len(entries)
            accepted = manifest["finalized_purpose"] == "collection" or sample["episode_success"]
            prefix = "sample" if accepted else "rejected"
            entries.append({"source_filename": path.name, "source_sha256": _sha(path),
                            "source_capture_index": sample["capture_index"],
                            "source_server_call": sample["server_call"],
                            "episode_index": episode, "episode_call": sample["episode_call"],
                            "episode_success": sample["episode_success"],
                            "capture_index": index, "accepted_for_training": bool(accepted),
                            "output_filename": f"{prefix}_{index:06d}.pt"})
    return {"schema": PLAN_SCHEMA, "source_directory": str(directory), "task_name": task_name,
            "source_manifest_sha256": _sha(directory / "capture_manifest.json"),
            "source_counts_sha256": _sha(directory / "capture_counts.json"),
            "purpose": manifest["finalized_purpose"], "mode": mode,
            "windows_per_episode": windows_per_episode,
            "selection_rule": "head: first k; stratified: lower median of contiguous floor partitions; n<k: all",
            "episodes": episodes, "selected": entries,
            "coverage_scope": "query-rank sampling across recorded candidates; no semantic-phase coverage claim"}


def materialize_capture_view(plan, output_directory):
    """Write a new per-task view; all input tensors and source/reset identities are retained."""
    import torch
    _require(isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA, "Unknown view plan")
    fresh = plan_capture_view(plan["source_directory"], plan["task_name"],
                              mode=plan["mode"], windows_per_episode=plan["windows_per_episode"])
    _require(plan == fresh, "View plan or source evidence changed")
    output = Path(output_directory).resolve()
    source = Path(plan["source_directory"])
    _require(output != source and source not in output.parents and output not in source.parents,
             "View destination must be separate from source")
    if output.exists():
        raise FileExistsError(f"Refusing existing view directory: {output}")
    output.mkdir(parents=True)
    records = []
    for entry in plan["selected"]:
        sample = torch.load(source / entry["source_filename"], map_location="cpu", weights_only=True)
        sample.update(source_capture_index=entry["source_capture_index"],
                      source_server_call=entry["source_server_call"],
                      capture_index=entry["capture_index"],
                      source_candidate_filename=entry["source_filename"],
                      source_candidate_sha256=entry["source_sha256"],
                      capture_view_mode=plan["mode"])
        path = output / entry["output_filename"]
        torch.save(sample, path)
        records.append({**entry, "output_sha256": _sha(path), "output_bytes": path.stat().st_size})
    raw_manifest = _read(source / "capture_manifest.json")
    accepted = sum(row["accepted_for_training"] for row in records)
    view = {key: value for key, value in raw_manifest.items() if key not in (
        "candidates", "pre_finalization_candidates", "capture_counts_sha256", "finalization_summary")}
    view.update(schema=VIEW_SCHEMA, sampling_mode="derived_training_view", selection_mode=plan["mode"],
                source_directory=str(source), source_manifest_sha256=plan["source_manifest_sha256"],
                source_counts_sha256=plan["source_counts_sha256"], plan_sha256=_json_sha(plan),
                data_view="derived from one finalized rollout; this view is not a new evaluation run",
                legacy_audit_compatible=False, per_episode_limit=plan["windows_per_episode"],
                total_samples=len(records), accepted_samples=accepted, rejected_samples=len(records) - accepted,
                unsuccessful_samples=sum(not row["episode_success"] for row in records),
                selected_sources=records, episodes=plan["episodes"], coverage_scope=plan["coverage_scope"])
    _write(output / "selection_plan.json", plan)
    _write(output / "capture_manifest.json", view)
    _write(output / "capture_counts.json", {
        "schema": VIEW_SCHEMA, "saved": len(records), "accepted_samples": accepted,
        "rejected_samples": len(records) - accepted, "selection_mode": plan["mode"],
        "source_actual_queries": sum(row["actual_queries"] for row in plan["episodes"]),
        "source_counts_sha256": plan["source_counts_sha256"],
        "per_episode_selected": {str(row["episode_index"]): row["selected_count"] for row in plan["episodes"]},
        "data_view": "selected files only; no inference calls were made"})
    return view
