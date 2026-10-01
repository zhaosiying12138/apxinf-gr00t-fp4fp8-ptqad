"""Run an existing GR00T server while capturing actual student-visited inputs.

Example (from GR00T):
 OPD_CAPTURE_DIR=/mnt/c/qad_rollout_inputs python /path/to/rl/capture_onpolicy.py \
   gr00t/eval/run_gr00t_server_fp4vla.py --model-path <QAD-merged> ...

The wrapper makes no additional GPU model calls and never changes returned
actions. Captures are CPU-only, individually re-collated from each environment's
processed observation: flattened Qwen image patches cannot be sliced as Bx...
The student's normalized action chunk defines the clean endpoint for subsequent
teacher/student velocity matching. This is one iteration of on-policy data
collection; a cache does not refresh itself after the policy is updated.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from probe_distill import file_sha256, tensor_tree
from gr00t_runtime import action_mask, libero_action_spec


def policy_input_precision(value):
    """Mirror Gr00tPolicy's post-collation BF16 conversion, retaining ints."""
    if torch.is_tensor(value):
        return value.to(torch.bfloat16) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: policy_input_precision(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(policy_input_precision(v) for v in value)
    return value


def install_capture(directory, student_checkpoint, every=8, per_task=16, limit=160,
                    event_file=None, capture_metadata=None, per_episode=None):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("sample_*.pt")):
        raise FileExistsError(f"Capture directory already contains observations: {directory}")
    if min(every, per_task, limit) < 1:
        raise ValueError("Capture intervals and budgets must be positive")
    if per_episode is not None and (type(per_episode) is not int or per_episode < 1 or not event_file):
        raise ValueError("Per-episode capture quota requires a positive integer and reset events")
    checkpoint = Path(student_checkpoint).resolve()
    action_spec = libero_action_spec(checkpoint)
    capture_metadata = dict(capture_metadata or {})
    event_path = Path(event_file).resolve() if event_file else None
    source_kind = capture_metadata.get("source_kind", "student_rollout")
    metadata = {"student_checkpoint": str(checkpoint),
                "student_config_sha256": file_sha256(checkpoint / "config.json"),
                "student_statistics_sha256": file_sha256(checkpoint / "statistics.json"),
                "every_server_calls": every, "per_task_limit": per_task, "total_limit": limit,
                "per_episode_limit": per_episode,
                "source_kind": source_kind,
                "action_endpoint": f"{source_kind} normalized action_pred",
                "checkpoint_role": "teacher" if source_kind == "teacher_rollout" else "student",
                "task_key": "sha256 of actual processed instruction text",
                "reset_event_file": str(event_path) if event_path else None,
                "reset_identity_fields": ["task_name", "episode_index", "seed",
                                          "init_state_index", "initial_state_sha256",
                                          "restored_state_sha256",
                                          "init_state_bank_sha256"],
                "observation_float_dtype": "bfloat16", "action_mask": action_spec}
    metadata.update(capture_metadata)
    (directory / "capture_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    original_collate = Gr00tN1d7DataCollator.__call__
    original_action = Gr00tN1d7.get_action
    pending = {}
    counts = {}
    episode_counts = {}
    state = {"calls": 0, "saved": 0}
    event_state = {"offset": 0, "current": None}

    def consume_reset_events():
        """Read complete reset records without blocking the inference call."""
        if event_path is None:
            return event_state["current"]
        try:
            data = event_path.read_bytes()
        except FileNotFoundError:
            return event_state["current"]
        offset = event_state["offset"]
        if offset > len(data):
            raise RuntimeError(f"Capture reset event file was truncated: {event_path}")
        chunk = data[offset:]
        # Reset writer appends newline-terminated records.  Ignore one
        # incomplete trailing line and retry it on the next policy call.
        complete, separator, _ = chunk.rpartition(b"\n")
        if not separator:
            return event_state["current"]
        consumed = len(complete) + 1
        for raw in complete.splitlines():
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid reset event in {event_path}") from exc
            if record.get("event", "reset") != "reset":
                continue
            required = ("task_name", "episode_index", "seed", "init_state_index",
                        "initial_state_sha256", "restored_state_sha256",
                        "init_state_bank_sha256")
            if any(key not in record for key in required):
                raise RuntimeError(f"Reset event lacks identity fields: {record}")
            previous = event_state["current"]
            if previous is not None and record["episode_index"] < previous["episode_index"]:
                raise RuntimeError("Reset event episode index moved backwards")
            event_state["current"] = record
        event_state["offset"] = offset + consumed
        return event_state["current"]

    def collate(self, features):
        result = original_collate(self, features)
        pending["features"] = features
        pending["collator"] = self
        return result

    def get_action(self, inputs, *args, **kwargs):
        features = pending.pop("features", None)
        collator = pending.pop("collator", None)
        result = original_action(self, inputs, *args, **kwargs)
        state["calls"] += 1
        if state["saved"] >= limit or (state["calls"] - 1) % every:
            return result
        if features is None or collator is None:
            raise RuntimeError("Capture needs the policy's raw per-environment collator features")
        actions = result["action_pred"]
        if len(features) != actions.shape[0]:
            raise ValueError("Captured observation/action batch mismatch")
        # Rotate environment slot rather than always observing slot zero.
        slot = ((state["calls"] - 1) // every) % len(features)
        feature = features[slot]
        text = feature.get("vlm_content", {}).get("text", "")
        if not text:
            raise ValueError("Cannot identify the rollout task without instruction text")
        task = hashlib.sha256(text.encode()).hexdigest()
        if counts.get(task, 0) >= per_task:
            return result
        reset = consume_reset_events()
        if event_path is not None and reset is None:
            raise RuntimeError(
                "Cannot save an on-policy sample before its reset identity is available; "
                f"expected a reset event in {event_path}"
            )
        if reset is not None and capture_metadata.get("task_name"):
            if reset.get("task_name") != capture_metadata["task_name"]:
                raise RuntimeError(
                    f"Reset task {reset.get('task_name')!r} differs from capture task "
                    f"{capture_metadata['task_name']!r}"
                )
        episode_key = f"{task}:{reset['episode_index']}" if reset is not None else None
        if per_episode is not None and episode_counts.get(episode_key, 0) >= per_episode:
            return result
        # Executed inside the caller's inference_mode: leave it before cloning,
        # so saved inputs can later participate in an autograd-enabled forward.
        with torch.inference_mode(False):
            single = tensor_tree(policy_input_precision(original_collate(collator, [feature])["inputs"]))
            single["action"] = actions[slot:slot + 1].detach().float().cpu().clone()
            single["action_mask"] = action_mask(single["action"], action_spec)
        sample = {"source_kind": metadata["source_kind"], "inputs": single,
                  "student_checkpoint": str(checkpoint),
                  "checkpoint_role": metadata["checkpoint_role"],
                  "student_statistics_sha256": metadata["student_statistics_sha256"],
                  "task_text": text, "task_sha256": task,
                  "instruction_sha256": task,
                  "task_name": reset.get("task_name") if reset else capture_metadata.get("task_name"),
                  "task_name_sha256": (hashlib.sha256(reset["task_name"].encode()).hexdigest()
                                       if reset else None),
                  "episode_index": reset.get("episode_index") if reset else None,
                  "episode_seed": reset.get("seed") if reset else None,
                  "init_state_index": reset.get("init_state_index") if reset else None,
                  "reset_identity": reset,
                  # Filled by run_recovery_eval once the scored episode result
                  # is known.  None is deliberately not treated as success.
                  "episode_success": None,
                  "environment_slot": slot,
                  "server_call": state["calls"], "capture_index": state["saved"]}
        torch.save(sample, directory / f"sample_{state['saved']:06d}.pt")
        state["saved"] += 1
        counts[task] = counts.get(task, 0) + 1
        if episode_key is not None:
            episode_counts[episode_key] = episode_counts.get(episode_key, 0) + 1
        (directory / "capture_counts.json").write_text(json.dumps(
            {"calls": state["calls"], "saved": state["saved"], "per_task": counts,
             "per_episode": episode_counts}, indent=2) + "\n")
        print(f"[onpolicy-capture] saved={state['saved']} task={task[:12]} "
              f"task_count={counts[task]} slot={slot} call={state['calls']}", flush=True)
        return result

    Gr00tN1d7DataCollator.__call__ = collate
    Gr00tN1d7.get_action = get_action


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: capture_onpolicy.py <existing-server.py> [server arguments]")
    script, args = sys.argv[1], sys.argv[2:]
    try:
        checkpoint = args[args.index("--model-path") + 1]
    except (ValueError, IndexError):
        raise SystemExit("Capture wrapper requires explicit --model-path")
    sys.path.insert(0, os.getcwd())
    install_capture(os.environ["OPD_CAPTURE_DIR"], checkpoint,
                    every=int(os.environ.get("OPD_CAPTURE_EVERY", "8")),
                    per_task=int(os.environ.get("OPD_CAPTURE_PER_TASK", "16")),
                    limit=int(os.environ.get("OPD_CAPTURE_LIMIT", "160")))
    sys.argv = [script, *args]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
