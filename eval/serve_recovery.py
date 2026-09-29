"""Seeded checkpoint server, optionally recording student-visited observations."""
import os
from pathlib import Path
import runpy
import sys


def main():
    sys.path.insert(0, os.getcwd())
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "rl"))
    from gr00t.utils.determinism import seed_everything
    seed_everything(int(os.environ["GR00T_EVAL_SEED"]))
    if os.environ.get("FP4VLA_QUANT", "0") != "0":
        raise ValueError("Recovery evaluation loads baked/merged weights; no second fake quantization")
    if os.environ.get("OPD_CAPTURE_DIR"):
        from capture_onpolicy import install_capture
        checkpoint = sys.argv[sys.argv.index("--model-path") + 1]
        install_capture(os.environ["OPD_CAPTURE_DIR"], checkpoint,
                        every=int(os.environ.get("OPD_CAPTURE_EVERY", "4")),
                        per_task=int(os.environ.get("OPD_CAPTURE_PER_TASK", "16")),
                        limit=int(os.environ.get("OPD_CAPTURE_LIMIT", "16")))
    script = Path(__file__).with_name("run_gr00t_server_fp4vla.py")
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
