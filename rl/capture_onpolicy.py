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


def install_capture(directory, student_checkpoint, every=8, per_task=16, limit=160):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("sample_*.pt")):
        raise FileExistsError(f"Capture directory already contains observations: {directory}")
    if min(every, per_task, limit) < 1:
        raise ValueError("Capture intervals and budgets must be positive")
    checkpoint = Path(student_checkpoint).resolve()
    action_spec = libero_action_spec(checkpoint)
    metadata = {"student_checkpoint": str(checkpoint),
                "student_config_sha256": file_sha256(checkpoint / "config.json"),
                "student_statistics_sha256": file_sha256(checkpoint / "statistics.json"),
                "every_server_calls": every, "per_task_limit": per_task, "total_limit": limit,
                "source_kind": "student_rollout", "action_endpoint": "student normalized action_pred",
                "task_key": "sha256 of actual processed instruction text",
                "observation_float_dtype": "bfloat16", "action_mask": action_spec}
    (directory / "capture_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    original_collate = Gr00tN1d7DataCollator.__call__
    original_action = Gr00tN1d7.get_action
    pending = {}
    counts = {}
    state = {"calls": 0, "saved": 0}

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
        # Executed inside the caller's inference_mode: leave it before cloning,
        # so saved inputs can later participate in an autograd-enabled forward.
        with torch.inference_mode(False):
            single = tensor_tree(policy_input_precision(original_collate(collator, [feature])["inputs"]))
            single["action"] = actions[slot:slot + 1].detach().float().cpu().clone()
            single["action_mask"] = action_mask(single["action"], action_spec)
        sample = {"source_kind": "student_rollout", "inputs": single,
                  "student_checkpoint": str(checkpoint),
                  "student_statistics_sha256": metadata["student_statistics_sha256"],
                  "task_text": text, "task_sha256": task, "environment_slot": slot,
                  "server_call": state["calls"], "capture_index": state["saved"]}
        torch.save(sample, directory / f"sample_{state['saved']:06d}.pt")
        state["saved"] += 1
        counts[task] = counts.get(task, 0) + 1
        (directory / "capture_counts.json").write_text(json.dumps(
            {"calls": state["calls"], "saved": state["saved"], "per_task": counts}, indent=2) + "\n")
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
