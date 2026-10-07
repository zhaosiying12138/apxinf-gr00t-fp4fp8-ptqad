"""Seeded checkpoint server, optionally recording student-visited observations."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

if __package__:
    from .run_recovery_eval import validate_diagnostic_capture, validate_recovery_checkpoint
else:
    from run_recovery_eval import validate_diagnostic_capture, validate_recovery_checkpoint


def main():
    # Validate before importing GR00T or initializing any device. This also
    # protects direct server invocations which bypass run_recovery_eval.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-path")
    checkpoint_args, _ = parser.parse_known_args()
    if checkpoint_args.model_path:
        validate_recovery_checkpoint(checkpoint_args.model_path)
    diagnostic_reference = None
    if os.environ.get("OPD_CAPTURE_DIR") and os.environ.get("FP4VLA_CAPTURE_PURPOSE") == "diagnostics":
        if not checkpoint_args.model_path:
            raise ValueError("Diagnostics capture requires explicit --model-path")
        diagnostic_reference = validate_diagnostic_capture(
            os.environ.get("FP4VLA_CAPTURE_PROTOCOL_FILE"), checkpoint_args.model_path)
        expected = os.environ.get("FP4VLA_DIAGNOSTIC_REFERENCE_JSON")
        if expected is not None and json.loads(expected) != diagnostic_reference:
            raise ValueError("Diagnostic source identity changed before server initialization")
    sys.path.insert(0, os.getcwd())
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "rl"))
    from gr00t.utils.determinism import seed_everything
    seed_everything(int(os.environ["GR00T_EVAL_SEED"]))
    if (os.environ.get("FP4VLA_QUANT", "0") != "0" and
            os.environ.get("FP4VLA_W4A4", "0") != "1" and
            os.environ.get("FP4VLA_W4A4_ADAPTER", "0") != "1"):
        raise ValueError("Recovery evaluation loads baked/merged weights; use the explicit W4A4 adapter path")
    if os.environ.get("OPD_CAPTURE_DIR"):
        from capture_onpolicy import install_capture
        checkpoint = sys.argv[sys.argv.index("--model-path") + 1]
        capture_metadata = {
            key: value for key, value in {
                "task_name": os.environ.get("FP4VLA_CAPTURE_TASK_NAME"),
                "purpose": os.environ.get("FP4VLA_CAPTURE_PURPOSE"),
                "source_kind": {"teacher_supervision": "teacher_rollout", "diagnostics": "diagnostic_rollout"}.get(
                    os.environ.get("FP4VLA_CAPTURE_PURPOSE"), "student_rollout"),
                "seed": (int(os.environ["FP4VLA_CAPTURE_SEED"])
                         if os.environ.get("FP4VLA_CAPTURE_SEED") else None),
                "init_state_indices": (os.environ.get("FP4VLA_CAPTURE_INIT_STATE_INDICES")),
                "protocol_sha256": os.environ.get("FP4VLA_CAPTURE_PROTOCOL_SHA256"),
            }.items() if value is not None
        }
        if diagnostic_reference is not None:
            capture_metadata.update(checkpoint_role="diagnostic_reference", training_eligible=False,
                protocol_file=os.environ["FP4VLA_CAPTURE_PROTOCOL_FILE"],
                diagnostic_reference=diagnostic_reference)
        install_capture(os.environ["OPD_CAPTURE_DIR"], checkpoint,
                        every=int(os.environ.get("OPD_CAPTURE_EVERY", "4")),
                        per_task=int(os.environ.get("OPD_CAPTURE_PER_TASK", "16")),
                        limit=int(os.environ.get("OPD_CAPTURE_LIMIT", "16")),
                        event_file=os.environ.get("FP4VLA_CAPTURE_EVENT_FILE"),
                        capture_metadata=capture_metadata,
                        per_episode=(int(os.environ["OPD_CAPTURE_PER_EPISODE"])
                                     if os.environ.get("OPD_CAPTURE_PER_EPISODE") else None),
                        sampling_mode=os.environ.get("FP4VLA_CAPTURE_SAMPLING_MODE", "prefix"))
    script = Path(__file__).with_name("run_gr00t_server_fp4vla.py")
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
