#!/usr/bin/env python3
"""W4A4 QAD/OPD recovery entry point.

This is the public, W4A4-named facade for the audited serial driver in
``run_high_fp4_v3.py``. Keeping one implementation avoids two orchestration
contracts drifting apart. Unless the caller supplies ``--protocol-file``,
the frozen full-coverage W4A4 v12 protocol is selected automatically.

The driver still performs all of the fail-closed checks: a development-only
PTQ selection, a finalized BF16 teacher capture dataset, two QAD learning
rates, continued-QAD, two OPD weights, and the paired five-arm held-out run.
This module only adapts the command name and protocol default; it never starts
training or evaluation at import time.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:  # package import (CPU tests and tooling)
    from .run_high_fp4_v3 import main as _driver_main
except ImportError:  # direct ``python exp/run_w4a4_recovery.py`` invocation
    from run_high_fp4_v3 import main as _driver_main

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "exp" / "recovery_protocol_v12_rtn_w4a4.json"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    has_protocol = any(
        arg == "--protocol-file" or arg.startswith("--protocol-file=")
        for arg in args
    )
    if not has_protocol:
        args.extend(["--protocol-file", str(DEFAULT_PROTOCOL)])
    return _driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
