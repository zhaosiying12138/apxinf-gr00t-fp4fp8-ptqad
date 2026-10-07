"""Serial LIBERO-10 collection/held-out evaluation with explicit seed manifests.

Run with the standard-library Python; model and simulation use their own venvs.
One server/task and one environment guarantee independent per-episode reset
seeds. Do not run concurrently with GPU training, baking or another evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import time

TASKS = [
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]

ENV_SUMMARY_KEYS = (
    "FP4VLA_QUANT", "FP4VLA_W4A4", "FP4VLA_W4A4_ADAPTER",
    "FP4VLA_SATURATE_F16_ACTIVATIONS", "QAD_W4A4", "QAD_MAX_GRAD_NORM",
    "QAD_LORA_SCOPE", "QAD_LORA_R", "QAD_LORA_ALPHA", "QAD_LR",
    "QAD_STEPS", "QAD_OPD_MSE_W", "OPD_EVERY", "TRAIN_SEED",
)


def environment_summary(env):
    """Keep evaluation provenance explicit without copying the full environment."""
    return {"variables": {key: env.get(key) for key in ENV_SUMMARY_KEYS}}


INIT_INDICES = {"development": [0, 1], "collection": [2, 3],
                "heldout": list(range(10, 20)), "smoke": [0]}


def validate_recovery_checkpoint(checkpoint):
    """Reject raw Trainer outputs before a plain loader can discard LoRA.

    Recovery deployment uses the completed merge manifest to load the frozen
    PTQ base and adapter separately. A training checkpoint's recovery manifest
    alone does not arm that path, even if a PTQ recipe was copied beside it.
    """
    checkpoint = Path(checkpoint)
    merge_path = checkpoint / "merge_manifest.json"
    if not merge_path.is_file():
        if (checkpoint / "recovery_manifest.json").is_file():
            raise ValueError(
                f"Raw recovery checkpoint cannot be evaluated: {checkpoint}. "
                "Run rl/lora_merge_bake.py and pass its completed deployment "
                "directory; loading checkpoint-* directly discards LoRA and W4A4.")
        return
    try:
        manifest = json.loads(merge_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid recovery deployment manifest: {merge_path}") from exc
    if (not isinstance(manifest, dict) or manifest.get("status") != "complete" or
            not all(isinstance(manifest.get(key), str) and manifest[key].strip()
                    for key in ("base", "training_checkpoint"))):
        raise ValueError(
            f"Recovery deployment requires a complete merge_manifest.json "
            f"with base/training_checkpoint: {merge_path}")


def protocol_entry(path, purpose):
    """Read a versioned bank partition from a protocol file.

    The original protocol is kept as the default for backwards compatibility.
    New studies pass an explicit file so that a new partition cannot be
    silently evaluated under the old hard-coded indices.  Both the compact
    ``{"partitions": {purpose: {...}}}`` form and the original top-level
    purpose keys are accepted; all other fields remain provenance in the
    manifest and are not interpreted here.
    """
    protocol = json.loads(Path(path).read_text())
    partitions = protocol.get("partitions", protocol)
    entry = partitions.get(purpose)
    if not isinstance(entry, dict) or "init_state_indices" not in entry:
        raise ValueError(f"Protocol {path} has no {purpose}.init_state_indices partition")
    indices = entry["init_state_indices"]
    if (not isinstance(indices, list) or not indices or
            any(type(x) is not int for x in indices) or
            len(set(indices)) != len(indices) or
            any(x < 0 or x >= 50 for x in indices)):
        raise ValueError(f"Protocol {path} has invalid {purpose}.init_state_indices")
    return entry


def protocol_partition(path, purpose):
    """Return the validated initial-state indices for ``purpose``."""
    return protocol_entry(path, purpose)["init_state_indices"]


def validate_resets(result, seed, indices):
    """Fail closed: result-only logs cannot certify paired evaluation."""
    resets = result["resets"]
    # New parser output distinguishes resets belonging to scored episodes from
    # the simulator's terminal auto-resets.  Once those counters are present,
    # require an exact one-to-one scored reset list; silently accepting extra
    # records would make the paired manifest ambiguous.
    if "scored_reset_count" in result:
        if result.get("scored_reset_count") != len(resets):
            raise ValueError("Scored reset count disagrees with parsed reset records")
        raw_count = result.get("raw_reset_count")
        ignored = result.get("ignored_auto_reset_count")
        if (type(raw_count) is not int or type(ignored) is not int or
                raw_count < len(resets) or ignored != raw_count - len(resets)):
            raise ValueError("Raw/ignored reset accounting is inconsistent")
        if len(resets) != len(indices):
            raise ValueError("Scored reset count differs from declared episode count")
    if len(resets) < len(indices):
        raise ValueError("Missing episode initial-state records")
    for episode, bank_index in enumerate(indices):
        record = resets[episode]
        if (record.get("episode_index") != episode or record.get("seed") != seed + episode or
            record.get("init_state_index") != bank_index or record.get("settle_steps") != 10):
            raise ValueError(f"Episode {episode} reset differs from declared bank/seed protocol")
        for key in ("initial_state_sha256", "restored_state_sha256", "init_state_bank_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", record.get(key, "")):
                raise ValueError(f"Episode {episode} missing valid {key}")
    return resets[:len(indices)]


def parse_log(path):
    text = Path(path).read_text(errors="replace")
    records = re.findall(r"results:\s*\('[^']+',\s*\[([^\]]*)\]", text)
    values = [x == "True" for x in re.findall(r"\b(?:True|False)\b", records[-1])] if records else []
    raw_resets = [json.loads(x) for x in re.findall(r"FP4VLA_EPISODE_RESET (\{[^\n]+\})", text)]
    # rollout_seeded emits an additional reset after a terminal episode.  The
    # result list is the authoritative scored-episode count, so retain only
    # the first N reset records and expose the discarded tail explicitly.
    scored_count = len(values)
    resets = raw_resets[:scored_count]
    return {"results": values, "successes": sum(values), "episodes": len(values),
            "success_rate": sum(values) / len(values) if values else None,
            "resets": resets, "raw_reset_count": len(raw_resets),
            "scored_reset_count": len(resets),
            "ignored_auto_reset_count": max(0, len(raw_resets) - len(resets)),
            "log_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def capture_sampling_config(partition, purpose, episodes, max_episode_steps=720):
    """Opt in to full candidates only in an explicit training partition.

    The safety capacity covers even one policy query per environment step,
    plus the initial observation. It is not a quota for dataset selection.
    Absent configuration preserves the frozen prefix-capture contract.
    """
    config = (partition or {}).get("capture_sampling")
    if config is None:
        return None
    if purpose not in ("teacher_supervision", "collection"):
        raise ValueError("capture_sampling is only allowed in training capture partitions")
    allowed = {"mode", "every_server_calls", "safety_candidates_per_episode"}
    if not isinstance(config, dict) or set(config) != allowed:
        raise ValueError(f"capture_sampling must specify exactly {sorted(allowed)}")
    if config["mode"] != "full_trajectory_candidates":
        raise ValueError("Explicit capture_sampling must use full_trajectory_candidates")
    every, safety = config["every_server_calls"], config["safety_candidates_per_episode"]
    if any(type(x) is not int or x <= 0 for x in (every, safety, episodes, max_episode_steps)):
        raise ValueError("Capture interval, safety bound and episode limits must be positive integers")
    minimum = (max_episode_steps + 1 + every - 1) // every
    if safety < minimum:
        raise ValueError(f"Candidate safety capacity must be at least {minimum} per episode")
    return {**config, "safety_candidates_per_task": safety * episodes,
            "n_envs": 1, "max_episode_steps": max_episode_steps,
            "capacity_basis": "at most one query per environment step plus initial observation"}


def validate_capture_partition_separation(path, purpose):
    """Check all declared evaluation partitions; teacher/student training may overlap.

    This verifies this protocol's partition ledger, not the independent-test
    status of bank indices across every historical study.
    """
    protocol = json.loads(Path(path).read_text())
    partitions = protocol.get("partitions", protocol)
    capture = protocol_entry(path, purpose)
    if type(capture.get("seed")) is not int:
        raise ValueError("Full-candidate capture requires an explicit integer partition seed")
    indices = capture["init_state_indices"]
    for name in ("development", "heldout", "smoke"):
        if name not in partitions:
            continue
        evaluation = protocol_entry(path, name)
        if set(indices) & set(evaluation["init_state_indices"]):
            raise ValueError(f"Capture and {name} bank indices overlap")
        if type(evaluation.get("seed")) is not int:
            raise ValueError(f"Partition {name} must declare an integer seed")
        # Include the terminal auto-reset, as in the held-out runner contract.
        train_seeds = {capture["seed"] + 1000 * task + episode
                       for task in range(10) for episode in range(len(indices) + 1)}
        eval_seeds = {evaluation["seed"] + 1000 * task + episode
                      for task in range(10) for episode in range(len(evaluation["init_state_indices"]) + 1)}
        if train_seeds & eval_seeds:
            raise ValueError(f"Capture and {name} reset seeds overlap")


def _capture_checkpoint_files(checkpoint):
    # Keep the shared helper and its import path out of the unchanged prefix
    # evaluation path. This branch runs before any policy server is launched.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rl.checkpoint_identity import checkpoint_files
    return checkpoint_files(checkpoint)


def record_capture_checkpoint(manifest):
    """Bind opt-in candidate capture to current checkpoint/base/adapter bytes."""
    if manifest.get("capture_sampling") is not None:
        manifest["checkpoint_files"] = _capture_checkpoint_files(manifest["checkpoint"])


def verify_capture_checkpoint(manifest, summary):
    """Reject a changed candidate producer before publishing a complete summary."""
    if manifest.get("capture_sampling") is None:
        return
    recorded = manifest.get("checkpoint_files")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Full candidate capture lacks initial checkpoint identity")
    if _capture_checkpoint_files(manifest["checkpoint"]) != recorded:
        raise ValueError("Checkpoint files changed during full candidate capture")
    summary["checkpoint_files_verified_unchanged"] = True


def configure_capture_environment(env, output, task, purpose, manifest, seed, indices):
    """Configure optional rollout capture variables for one task.

    The reset-event path is deliberately exported even when the rollout has
    not created the file yet.  ``serve_recovery`` creates that file while it
    runs; conditioning the environment update on ``event_file.exists()``
    would therefore disable capture for every fresh output directory.
    """
    env.pop("OPD_CAPTURE_DIR", None)
    env.pop("FP4VLA_CAPTURE_SAMPLING_MODE", None)
    if purpose not in ("teacher_supervision", "collection"):
        return None

    capture_dir = Path(output) / "observations" / task
    capture_dir.mkdir(parents=True, exist_ok=True)
    event_file = capture_dir / "reset_events.jsonl"
    if event_file.exists():
        event_file.unlink()
    env.update({
        "OPD_CAPTURE_DIR": str(capture_dir),
        "FP4VLA_CAPTURE_EVENT_FILE": str(event_file),
        "FP4VLA_CAPTURE_TASK_NAME": task,
        "FP4VLA_CAPTURE_PURPOSE": purpose,
        "FP4VLA_CAPTURE_SEED": str(seed),
        "FP4VLA_CAPTURE_INIT_STATE_INDICES": ",".join(map(str, indices)),
        "FP4VLA_CAPTURE_PROTOCOL_SHA256": manifest["protocol_sha256"],
        # The recovery protocol captures a fixed number of snapshots from
        # every scored training episode.  Without these explicit values
        # serve_recovery's small historical defaults would silently keep only
        # the first episode/task and invalidate teacher coverage.
        "OPD_CAPTURE_EVERY": "4",
        "OPD_CAPTURE_PER_TASK": "16",
        "OPD_CAPTURE_LIMIT": "160",
        "OPD_CAPTURE_PER_EPISODE": "4",
        "FP4VLA_CAPTURE_SAMPLING_MODE": "prefix",
    })
    config = manifest.get("capture_sampling")
    if config is not None:
        # Validate even for direct helper callers; do not trust inherited env.
        verified = capture_sampling_config(
            {"capture_sampling": {key: config[key] for key in (
                "mode", "every_server_calls", "safety_candidates_per_episode")}},
            purpose, len(indices), manifest.get("max_episode_steps", 720))
        if config != verified:
            raise ValueError("Capture manifest configuration differs from derived safety capacity")
        env.update({
            "FP4VLA_CAPTURE_SAMPLING_MODE": config["mode"],
            "OPD_CAPTURE_EVERY": str(config["every_server_calls"]),
            "OPD_CAPTURE_PER_EPISODE": str(config["safety_candidates_per_episode"]),
            "OPD_CAPTURE_PER_TASK": str(config["safety_candidates_per_task"]),
            "OPD_CAPTURE_LIMIT": str(config["safety_candidates_per_task"]),
        })
    return capture_dir


def finalize_capture_samples(directory, task_name, result, purpose, sampling_mode="prefix"):
    """Bind captured inputs to scored episodes and retain successful samples.

    The policy server records reset identity while the rollout is still
    running, but only the rollout process knows the terminal success label.
    This post-pass joins the two sources using ``episode_index`` and fails
    closed on any mismatch.  Teacher-supervision captures are then restricted
    to successful episodes: rejected samples are renamed (and preserved) so a
    recursive ``QAD_CAPTURE_DATASET`` cannot accidentally train on failures.
    """
    manifest_file = Path(directory) / "capture_manifest.json"
    recorded_mode = (json.loads(manifest_file.read_text()).get("sampling_mode", "prefix")
                     if manifest_file.is_file() else "prefix")
    if recorded_mode != sampling_mode:
        raise ValueError("Recorded capture mode differs from evaluation configuration")
    if sampling_mode == "full_trajectory_candidates":
        if not __package__:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rl.capture_sampling import finalize_candidates
        return finalize_candidates(directory, task_name, result, purpose)
    if sampling_mode != "prefix":
        raise ValueError(f"Unknown capture mode: {sampling_mode}")
    paths = sorted(Path(directory).rglob("sample_*.pt"))
    if not paths:
        return {"total_samples": 0, "accepted_samples": 0, "rejected_samples": 0,
                "unsuccessful_samples": 0,
                "successful_episode_indices": []}
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Capture finalization requires torch to read sample metadata") from exc
    resets = result.get("resets", [])
    values = result.get("results", [])
    if len(resets) != len(values):
        raise ValueError(f"Capture/result episode counts differ for {task_name}: "
                         f"{len(resets)} resets vs {len(values)} results")
    accepted, rejected = 0, 0
    episodes = set()
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=True)
        episode = sample.get("episode_index")
        if type(episode) is not int or not 0 <= episode < len(values):
            raise ValueError(f"Sample {path} lacks a valid scored episode_index")
        reset = resets[episode]
        checks = {
            "task_name": (sample.get("task_name"), task_name),
            "episode_index": (sample.get("episode_index"), episode),
            "episode_seed": (sample.get("episode_seed"), reset.get("seed")),
            "init_state_index": (sample.get("init_state_index"), reset.get("init_state_index")),
        }
        identity = sample.get("reset_identity")
        if not isinstance(identity, dict):
            raise ValueError(f"Sample {path} lacks reset_identity")
        checks["reset_identity.task_name"] = (identity.get("task_name"), task_name)
        checks["reset_identity.episode_index"] = (identity.get("episode_index"), episode)
        for key in ("initial_state_sha256", "restored_state_sha256", "init_state_bank_sha256"):
            checks[f"reset_identity.{key}"] = (identity.get(key), reset.get(key))
        mismatches = [key for key, (actual, expected) in checks.items() if actual != expected]
        if mismatches:
            raise ValueError(f"Sample {path} reset identity mismatch: {mismatches}")
        success = bool(values[episode])
        sample["episode_success"] = success
        sample["episode_result_source"] = "run_recovery_eval.task_results"
        sample["episode_result_index"] = episode
        tmp = path.with_name(path.name + ".tmp")
        torch.save(sample, tmp)
        tmp.replace(path)
        if success:
            accepted += 1
            episodes.add(episode)
        elif purpose == "teacher_supervision":
            target = path.with_name(path.name.replace("sample_", "rejected_", 1))
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite rejected capture {target}")
            path.rename(target)
            rejected += 1
    manifest_path = Path(directory) / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    unsuccessful = len(paths) - accepted
    manifest.update({
        "finalized": True,
        "finalized_purpose": purpose,
        "episode_success_source": "run_recovery_eval.task_results",
        "total_samples": len(paths),
        "accepted_samples": accepted,
        "rejected_samples": rejected,
        "unsuccessful_samples": unsuccessful,
        "successful_episode_indices": sorted(episodes),
        "successful_episode_count": sum(bool(x) for x in values),
        "scored_episode_count": len(values),
    })
    event_file = manifest.get("reset_event_file")
    if event_file and Path(event_file).is_file():
        manifest["reset_event_file_sha256"] = hashlib.sha256(
            Path(event_file).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"total_samples": len(paths), "accepted_samples": accepted,
            "rejected_samples": rejected, "unsuccessful_samples": unsuccessful,
            "successful_episode_indices": sorted(episodes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--purpose", required=True, choices=("development", "teacher_supervision", "collection", "heldout", "smoke"))
    ap.add_argument("--protocol-file", help="Versioned protocol JSON supplying the bank partition; defaults to exp/recovery_protocol.json")
    ap.add_argument("--collection-manifest", help="Required for heldout: prove disjoint reset seeds")
    ap.add_argument("--episodes", type=int, help="Default 2 development/collection, 10 heldout, 1 smoke")
    ap.add_argument("--init-state-indices", help="Explicit bank indices; formal purposes must match fixed partition")
    ap.add_argument("--port", type=int, default=5595)
    ap.add_argument("--gr00t", default=os.environ.get("GR00T_REPO", str(Path.home() / "codebase/groot-fsdp2/Isaac-GR00T")))
    ap.add_argument("--server-python", default=os.environ.get("PTQAD_PYTHON"))
    ap.add_argument("--rollout-python", default=os.environ.get("LIBERO_PYTHON"))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--task-count", type=int, default=10, help="Only reduce for smoke, never a ten-task score")
    args = ap.parse_args()
    validate_recovery_checkpoint(args.checkpoint)
    project = Path(__file__).resolve().parents[1]
    protocol_path = Path(args.protocol_file).resolve() if args.protocol_file else project / "exp/recovery_protocol.json"
    if not protocol_path.is_file():
        ap.error(f"Protocol file does not exist: {protocol_path}")
    protocol_data = protocol_entry(protocol_path, args.purpose) if args.protocol_file else None
    protocol_indices = protocol_data["init_state_indices"] if protocol_data else INIT_INDICES[args.purpose]
    indices = [int(x) for x in args.init_state_indices.split(",")] if args.init_state_indices else protocol_indices
    if args.protocol_file and indices != protocol_indices:
        ap.error("Explicit bank indices must exactly match the selected versioned protocol partition")
    if protocol_data and protocol_data.get("seed") is not None and args.seed != int(protocol_data["seed"]):
        ap.error(f"Seed {args.seed} does not match the selected protocol partition ({protocol_data['seed']})")
    if (protocol_data and args.episodes is not None and
            protocol_data.get("episodes_per_task") is not None and
            args.episodes != int(protocol_data["episodes_per_task"])):
        ap.error("episodes does not match the selected versioned protocol partition")
    if not args.protocol_file and args.purpose != "smoke" and indices != INIT_INDICES[args.purpose]:
        ap.error("Formal evaluation must use the fixed official-bank partition")
    declared_episodes = protocol_data.get("episodes_per_task") if protocol_data else None
    args.episodes = args.episodes if args.episodes is not None else (declared_episodes or len(indices))
    if args.episodes != len(indices) or len(set(indices)) != len(indices):
        ap.error("episodes must equal the number of unique official bank indices")
    if not 1 <= args.task_count <= 10 or not 1 <= args.episodes < 1000:
        ap.error("task-count must be 1..10 and episodes 1..999")
    if args.task_count != 10 and args.purpose != "smoke":
        ap.error("Development, collection and heldout protocols require all 10 tasks")
    sampling_config = capture_sampling_config(protocol_data, args.purpose, args.episodes)
    if sampling_config is not None:
        validate_capture_partition_separation(protocol_path, args.purpose)
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"New evidence requires a new output directory: {output}")
    if args.purpose == "heldout":
        if not args.collection_manifest:
            ap.error("heldout requires --collection-manifest")
        collection = json.loads(Path(args.collection_manifest).read_text())
        if collection["purpose"] != "collection":
            raise ValueError("Expected a collection manifest")
        if collection.get("initial_state_protocol") != "libero10_official_bank_v1":
            raise ValueError("Collection manifest does not certify official bank initialization")
        if set(collection["init_state_indices"]) & set(indices):
            raise ValueError("Collection and heldout official bank indices overlap")
        for i in range(10):
            train = set(range(collection["seed"] + 1000*i, collection["seed"] + 1000*i + collection["episodes"] + 1))
            test = set(range(args.seed + 1000*i, args.seed + 1000*i + args.episodes + 1))
            if train & test:
                raise ValueError("Collection and heldout reset seeds overlap")
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))  # refuse to attach to another server
    output.mkdir(parents=True)
    gr00t = Path(args.gr00t).resolve()
    python = args.server_python or str(gr00t / ".venv/bin/python")
    rollout_python = args.rollout_python or str(gr00t / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python")
    media = os.environ.get("PTQAD_MEDIA_LIB")
    # PTQAD_MEDIA_LIB points to <ffmpeg environment>/lib, so its CLI is a
    # sibling bin directory. PTQAD_MEDIA_BIN explicitly overrides that default.
    media_bin = os.environ.get("PTQAD_MEDIA_BIN") or (str(Path(media).parent / "bin") if media else None)
    video_path = (media_bin + os.pathsep if media_bin else "") + os.environ.get("PATH", "")
    ffmpeg = shutil.which("ffmpeg", path=video_path)
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg executable missing: set PTQAD_MEDIA_LIB=<environment>/lib or PTQAD_MEDIA_BIN=<environment>/bin")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_is_quantized = any(
        (checkpoint_path / name).is_file()
        for name in ("ptq_recipe.json", "category_ptq_recipe.json", "merge_manifest.json")
    )
    checkpoint_w4a4 = checkpoint_is_quantized
    checkpoint_adapter_w4a4 = checkpoint_is_quantized and (checkpoint_path / "merge_manifest.json").is_file()
    # A merged recovery export carries the training contract beside its
    # weights.  Copy just the two values relevant to numerical provenance into
    # the eval receipt; do not infer them from a stale shell environment.
    recovery_contract = {"max_grad_norm": None, "f16_activation_saturation": False}
    recovery_manifest = checkpoint_path / "recovery_manifest.json"
    if recovery_manifest.is_file():
        try:
            recovery = json.loads(recovery_manifest.read_text())
            recovery_contract = {
                "max_grad_norm": recovery.get("max_grad_norm"),
                "f16_activation_saturation": recovery.get("f16_activation_saturation"),
            }
        except (OSError, ValueError):
            raise ValueError(f"Invalid recovery_manifest.json: {recovery_manifest}")
    manifest = vars(args) | {"checkpoint": str(checkpoint_path),
                              "gr00t": str(gr00t), "n_envs": 1,
                              "task_seed_stride": 1000, "episode_seed_stride": 1,
                              "server_seed_offset": 10000000, "tasks": TASKS[:args.task_count],
                              "n_action_steps": 8, "max_episode_steps": 720}
    manifest.update({"initial_state_protocol": "libero10_official_bank_v1",
                     "init_state_indices": indices, "settle_steps": 10,
                     "ffmpeg_executable": ffmpeg, "media_library_dir": media,
                     "video_root": str(output / "videos"),
                     "paired_scope": "environment initial states; policy noise is seeded per task",
                     "environment_summary": {"variables": {
                         "FP4VLA_QUANT": "0",
                         "FP4VLA_W4A4": "1" if checkpoint_w4a4 else "0",
                         "FP4VLA_W4A4_ADAPTER": "1" if checkpoint_adapter_w4a4 else "0",
                         "FP4VLA_SATURATE_F16_ACTIVATIONS": "1" if checkpoint_w4a4 else "0",
                         "PTQAD_ZMQ_TIMEOUT_MS": os.environ.get("PTQAD_ZMQ_TIMEOUT_MS", "120000"),
                     }},
                     "recovery_contract": recovery_contract})
    manifest["protocol_file"] = str(protocol_path)
    manifest["protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if sampling_config is not None:
        manifest["capture_sampling"] = sampling_config
        record_capture_checkpoint(manifest)
    (output / "eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = {}
    started = time.time()
    for index, task in enumerate(TASKS[:args.task_count]):
        seed = args.seed + 1000 * index
        env = os.environ.copy()
        # Infer the numerical path from the checkpoint itself.  This prevents
        # a stale shell export from turning a BF16 baseline or teacher replay
        # into an activation-quantized run.  Recovery merge checkpoints carry
        # the PTQ recipe/manifest and therefore remain on the W4A4 path.
        w4a4 = checkpoint_w4a4
        adapter_w4a4 = checkpoint_adapter_w4a4
        env.update({"HF_HUB_OFFLINE": "1",
                    # Both PTQ and adapter evaluation use the frozen
                    # checkpoint weights as-is.  W4A4 is an explicit
                    # activation-only path for PTQ; setting FP4VLA_QUANT=1
                    # here would quantize an already baked checkpoint twice.
                    "FP4VLA_QUANT": "0",
                    "FP4VLA_W4A4": "1" if w4a4 else "0",
                    "FP4VLA_W4A4_ADAPTER": "1" if adapter_w4a4 else "0",
                    "FP4VLA_SATURATE_F16_ACTIVATIONS": "1" if w4a4 else "0",
                    "PTQAD_ZMQ_TIMEOUT_MS": os.environ.get("PTQAD_ZMQ_TIMEOUT_MS", "120000"),
                    "PYTHONPATH": str(gr00t),
                    "GR00T_EVAL_SEED": str(seed + 10000000),
                    "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
        media = env.get("PTQAD_MEDIA_LIB")
        if media:
            env["LD_LIBRARY_PATH"] = media + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env["PATH"] = video_path
        configure_capture_environment(
            env, output, task, args.purpose, manifest, seed, indices)
        server_cmd = [python, str(project / "eval/serve_recovery.py"), "--model-path", manifest["checkpoint"],
                      "--embodiment-tag", "LIBERO_PANDA", "--use-sim-policy-wrapper", "--port", str(args.port)]
        with open(output / f"{task}.server.log", "w") as log:
            server = subprocess.Popen(server_cmd, cwd=gr00t, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                ready = False
                for _ in range(180):
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited; see {log.name}")
                    try:
                        with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                            ready = True
                            break
                    except OSError:
                        time.sleep(1)
                if not ready:
                    raise TimeoutError("Server did not become ready")
                command = [rollout_python, str(project / "eval/rollout_seeded.py"),
                           "--env-name", f"libero_sim/{task}", "--n-episodes", str(args.episodes),
                           "--n-envs", "1", "--seed", str(seed), "--max-episode-steps", "720",
                           "--init-state-indices", ",".join(map(str, indices)),
                           "--video-dir", str(output / "videos" / task),
                           "--n-action-steps", "8", "--policy-client-host", "127.0.0.1",
                           "--policy-client-port", str(args.port)]
                with open(output / f"{task}.log", "w") as rollout_log:
                    completed = subprocess.run(command, cwd=gr00t, env=env, stdout=rollout_log,
                                               stderr=subprocess.STDOUT, timeout=args.timeout)
                result = parse_log(output / f"{task}.log")
                result.update({"returncode": completed.returncode, "seed": seed})
                results[task] = result
                (output / "task_results.json").write_text(json.dumps(results, indent=2) + "\n")
                if completed.returncode or result["episodes"] != args.episodes:
                    raise RuntimeError(f"Incomplete task {task}: rc={completed.returncode}, episodes={result['episodes']}")
                validate_resets(result, seed, indices)
                if args.purpose in ("teacher_supervision", "collection"):
                    capture_summary = finalize_capture_samples(
                        output / "observations" / task, task, result, args.purpose,
                        sampling_mode=sampling_config["mode"] if sampling_config else "prefix")
                    result["capture"] = capture_summary
                    # Rewrite task_results after the join so its provenance
                    # records exactly how many usable teacher samples remain.
                    (output / "task_results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"[eval] {task}: {result['successes']}/{result['episodes']}", flush=True)
            finally:
                stop(server)
    total_successes = sum(x["successes"] for x in results.values())
    total_episodes = sum(x["episodes"] for x in results.values())
    summary = {"tasks_complete": len(results), "total_successes": total_successes,
               "total_episodes": total_episodes,
               # Keep both denominators explicit.  The task-macro value is
               # the primary summary rate; micro is useful for raw accounting.
               "macro_success_rate": sum(x["success_rate"] for x in results.values()) / 10 if len(results) == 10 else None,
               "micro_success_rate": total_successes / total_episodes if total_episodes else None,
               "score_definition": "macro_success_rate=unweighted mean of ten task rates; micro_success_rate=total successes/episodes",
               "purpose": args.purpose, "seed": args.seed,
               "wall_seconds_including_server_loads": time.time() - started}
    verify_capture_checkpoint(manifest, summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
