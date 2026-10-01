#!/usr/bin/env python3
"""Run the v7 mixed-pressure development gate and, when it passes, recovery.

The command creates exactly two fresh development evaluations: BF16 and the
mixed NVFP4/FP8 candidate declared by the protocol.  Existing evaluation
directories are reusable only after ``eval_audit`` succeeds; an incomplete
directory or a stale development log is an error and is never removed.  Each
recovery invocation gets a separate attempt log; its downstream validators
decide which completed stages can be reused and reject partial training.
A nonblocking Linux file lock prevents concurrent invocations for one run.
The resulting
``selection.json`` is development-only and is accepted by
``run_mixed_pressure_recovery.py`` only when the pressure rule is met.

``--validate-only`` checks protocol, checkpoints and launchers without making
directories.  ``--dry-run`` performs the same checks and prints the planned
commands.  Neither mode starts a simulator or training process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from fractions import Fraction
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_high_fp4_v3 as v3
import run_mixed_pressure_recovery as mixed
import recovery_invocation

CANDIDATE = "mixed"
DEFAULT_PROTOCOL = ROOT / "exp/recovery_protocol_v7_mixed_pressure.json"
DEFAULT_BASE = ROOT / "weights/GR00T-N1.7-LIBERO/libero_10"


def _error(message: str) -> v3.OrchestrationError:
    return v3.OrchestrationError(message)


def _resolve_existing(path: str | Path, label: str, *, directory: bool = True) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.exists() or (directory and not value.is_dir()):
        raise _error(f"{label} does not exist: {value}")
    return value


def _media_library(path: str | Path) -> Path:
    value = _resolve_existing(path, "media library")
    if not (value.parent / "bin" / "ffmpeg").is_file():
        raise _error(f"ffmpeg is missing beside media library: {value}")
    return value


def _launchers(args: argparse.Namespace, groot: Path) -> tuple[Path, Path]:
    python = Path(args.python or groot / ".venv/bin/python").expanduser().absolute()
    rollout = Path(args.rollout_python or
                   groot / ".venv-libero/bin/python").expanduser().absolute()
    if not python.is_file():
        raise _error(f"training/evaluation Python launcher is missing: {python}")
    if not rollout.is_file():
        raise _error(f"LIBERO rollout Python launcher is missing: {rollout}")
    return python, rollout


def setup(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    """Resolve all inputs before any output directory is created."""
    base = _resolve_existing(args.base or DEFAULT_BASE, "BF16 base checkpoint")
    groot = _resolve_existing(args.gr00t_repo or
                              Path.home() / "codebase/groot-fsdp2/Isaac-GR00T",
                              "GR00T repository")
    dataset = _resolve_existing(args.dataset or groot / "demo_data/libero_demo", "QAD dataset")
    media = _media_library(args.media_lib or
                           os.environ.get("PTQAD_MEDIA_LIB",
                                          "/home/zhaosiying/miniforge3/envs/media7/lib"))
    python, rollout = _launchers(args, groot)
    candidate = Path(protocol["selection"]["pressure_candidate_checkpoints"][CANDIDATE]).expanduser().resolve()
    mixed._validate_mixed_checkpoint(candidate, protocol)
    protocol_path = Path(protocol["path"]).resolve()
    return {"base": base, "groot": groot, "dataset": dataset, "media": media,
            "python": python, "rollout": rollout, "candidate": candidate,
            "protocol_path": protocol_path}


def _env(media: Path) -> dict[str, str]:
    value = os.environ.copy()
    value["PTQAD_MEDIA_LIB"] = str(media)
    old = value.get("LD_LIBRARY_PATH")
    value["LD_LIBRARY_PATH"] = str(media) + (os.pathsep + old if old else "")
    value.setdefault("HF_HUB_OFFLINE", "1")
    value.setdefault("TRANSFORMERS_OFFLINE", "1")
    return value


def development_command(kind: str, checkpoint: Path, out: Path, protocol: dict[str, Any],
                        cfg: dict[str, Any], port: int) -> list[str]:
    part = protocol["partitions"]["development"]
    return [str(cfg["python"]), str(ROOT / "eval/run_recovery_eval.py"),
            "--checkpoint", str(checkpoint), "--out", str(out),
            "--purpose", "development", "--seed", str(part["seed"]),
            "--episodes", str(part["episodes_per_task"]), "--gr00t", str(cfg["groot"]),
            "--server-python", str(cfg["python"]), "--rollout-python", str(cfg["rollout"]),
            "--port", str(port), "--protocol-file", str(cfg["protocol_path"])]


def recovery_command(args: argparse.Namespace, selection: Path, run: Path,
                     protocol: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    command = [str(cfg["python"]), str(HERE / "run_mixed_pressure_recovery.py"),
               "--run-dir", str(run / "recovery"), "--protocol-file", str(cfg["protocol_path"]),
               "--ptq-selection", str(selection), "--base", str(cfg["base"]),
               "--gr00t-repo", str(cfg["groot"]), "--python", str(cfg["python"]),
               "--rollout-python", str(cfg["rollout"]), "--dataset", str(cfg["dataset"]),
               "--port-base", str(args.port_base + 10), "--cleanup-duplicates", "--until", args.until]
    return command


@contextmanager
def _run_lock(run: Path) -> Iterator[int]:
    """Lock the run without PID files or locks that survive a system restart.

    Keep the lock file: unlinking it would let another process lock a different
    inode under the same path.  Closing the descriptor releases the kernel
    lock.  Subprocesses inherit the descriptor so a surviving recovery process
    still excludes a second wrapper if this wrapper is interrupted.
    """
    with (run / ".mixed-pressure-study.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise _error(f"another mixed-pressure study is active for run: {run}") from exc
        yield stream.fileno()


def _recovery_log(run: Path) -> Path:
    """Choose the next attempt under the run lock, retaining every prior log."""
    pattern = re.compile(r"mixed-recovery-attempt-(\d+)\.log")
    attempts = [int(match.group(1)) for path in (run / "logs").glob("mixed-recovery-attempt-*.log")
                if (match := pattern.fullmatch(path.name))]
    attempts += [int(match.group(1)) for path in (run / "invocations").glob("mixed-recovery-attempt-*")
                 if (match := re.fullmatch(r"mixed-recovery-attempt-(\d+)", path.name))]
    return run / "logs" / f"mixed-recovery-attempt-{max(attempts, default=0) + 1:04d}.log"


def _run_logged(name: str, command: list[str], cwd: Path, log: Path, env: dict[str, str],
                *, lock_fd: int | None = None) -> None:
    if log.exists():
        raise _error(f"refusing an existing stage log for {name}: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command, "cwd": str(cwd)}, ensure_ascii=False) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=str(cwd), env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   pass_fds=() if lock_fd is None else (lock_fd,))
        stream.write(f"RETURN_CODE {completed.returncode}\n")
    if completed.returncode != 0:
        raise _error(f"{name} failed; inspect {log}")


def _audit_or_run(kind: str, checkpoint: Path, out: Path, protocol: dict[str, Any],
                  cfg: dict[str, Any], port: int, log: Path,
                  *, lock_fd: int | None = None) -> dict[str, Any]:
    if out.exists():
        try:
            return v3.eval_audit(out, protocol, "development", checkpoint)
        except Exception as exc:
            raise _error(f"{kind} development directory exists but is incomplete or stale: {out}: {exc}") from exc
    if log.exists():
        raise _error(f"{kind} has a stage log but no complete evaluation; refusing to overwrite: {log}")
    _run_logged(kind, development_command(kind, checkpoint, out, protocol, cfg, port),
                cfg["groot"], log, _env(cfg["media"]), lock_fd=lock_fd)
    return v3.eval_audit(out, protocol, "development", checkpoint)


def choose(audits: dict[str, dict[str, Any]], protocol: dict[str, Any]) -> tuple[str | None, list[str]]:
    base = Fraction(audits["bf16"]["successes"], audits["bf16"]["episodes"])
    mixed_score = Fraction(audits[CANDIDATE]["successes"], audits[CANDIDATE]["episodes"])
    rule = protocol["selection"]["pressure_rule"]
    qualifies = (base - mixed_score >= Fraction(str(rule["min_drop_from_bf16"])) and
                 mixed_score >= Fraction(str(rule["min_absolute_success"])))
    return (CANDIDATE if qualifies else None), ([CANDIDATE] if qualifies else [])


def _selection(run: Path, protocol: dict[str, Any], cfg: dict[str, Any],
               audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected, qualifying = choose(audits, protocol)
    arms = {
        "bf16": {"checkpoint": str(cfg["base"]), **{k: audits["bf16"][k] for k in
                  ("successes", "episodes", "macro_success_rate", "pairing_sha256")},
                  "environment_pairing_verified": True},
        CANDIDATE: {"checkpoint": str(cfg["candidate"]), **{k: audits[CANDIDATE][k] for k in
                  ("successes", "episodes", "macro_success_rate", "pairing_sha256")},
                    "environment_pairing_verified": True},
    }
    source = {f"{arm}/{name}": v3.sha(run / "development" / arm / name)
              for arm in ("bf16", CANDIDATE)
              for name in ("eval_manifest.json", "task_results.json", "summary.json")}
    return {"status": "complete" if selected else "no_qualifying_candidate",
            "protocol_file": str(cfg["protocol_path"]), "protocol_sha256": protocol["sha256"],
            "selection_uses_heldout": False, "selection_rule": protocol["selection"]["rule"],
            "pressure_candidates": [CANDIDATE], "selected_recipe": selected,
            "qualifying_candidates": qualifying,
            "pairing_sha256": audits["bf16"]["pairing_sha256"], "arms": arms,
            "source_sha256": source}


def _write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise _error(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _plan(args: argparse.Namespace, protocol: dict[str, Any], cfg: dict[str, Any], run: Path) -> dict[str, Any]:
    dev = run / "development"
    return {"protocol_sha256": protocol["sha256"], "run_dir": str(run),
            "commands": [development_command("bf16", cfg["base"], dev / "bf16", protocol, cfg, args.port_base),
                          development_command(CANDIDATE, cfg["candidate"], dev / CANDIDATE,
                                              protocol, cfg, args.port_base + 1)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--protocol-file", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--base", default=os.environ.get("PTQAD_BASE"))
    parser.add_argument("--gr00t-repo", default=os.environ.get("GR00T_REPO"))
    parser.add_argument("--python", default=os.environ.get("PTQAD_PYTHON"))
    parser.add_argument("--rollout-python", default=os.environ.get("LIBERO_PYTHON"))
    parser.add_argument("--dataset", default=os.environ.get("QAD_DATASET"))
    parser.add_argument("--media-lib", default=os.environ.get("PTQAD_MEDIA_LIB"))
    parser.add_argument("--port-base", type=int, default=5890)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--until", choices=("qad_dev", "qad_selection", "recovery_dev",
                                              "opd_selection", "all"), default="all")
    args = parser.parse_args(argv)
    if args.validate_only and args.development_only:
        parser.error("--validate-only and --development-only are mutually exclusive")
    try:
        protocol_path = _resolve_existing(args.protocol_file, "protocol", directory=False)
        protocol = mixed.load_protocol(protocol_path)
        cfg = setup(args, protocol)
        run = Path(args.run_dir).expanduser().resolve()
        if args.validate_only or args.dry_run:
            result = _plan(args, protocol, cfg, run)
            result["mode"] = "dry-run" if args.dry_run else "validate-only"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if run.exists() and not run.is_dir():
            raise _error(f"run directory is not a directory: {run}")
        run.mkdir(parents=True, exist_ok=True)
        with _run_lock(run) as lock_fd:
            dev = run / "development"
            audits = {
                "bf16": _audit_or_run("eval-bf16", cfg["base"], dev / "bf16", protocol, cfg,
                                       args.port_base, run / "logs/eval-bf16.log", lock_fd=lock_fd),
                CANDIDATE: _audit_or_run("eval-mixed", cfg["candidate"], dev / CANDIDATE, protocol, cfg,
                                         args.port_base + 1, run / "logs/eval-mixed.log", lock_fd=lock_fd),
            }
            v3.require_pairing(audits)
            # Keep selection.json beside development/bf16 and development/mixed.
            # The shared recovery validator resolves those arms relative to the
            # selection file's parent directory.
            selection_path = run / "development" / "selection.json"
            if selection_path.exists():
                selection = v3.jread(selection_path)
                if selection.get("status") == "no_qualifying_candidate":
                    expected = _selection(run, protocol, cfg, audits)
                    if (selection.get("protocol_sha256") != expected["protocol_sha256"] or
                            selection.get("source_sha256") != expected["source_sha256"] or
                            selection.get("arms") != expected["arms"] or
                            selection.get("selected_recipe") is not None):
                        raise _error("existing no-qualification selection disagrees with audited development evidence")
                else:
                    # A completed selection is itself evidence: re-audit its raw
                    # files before allowing any recovery subprocess to run.
                    selection = mixed.validate_ptq(selection_path, protocol)
            elif (run / "selection.json").is_file():
                # Migrate the original root-level selection emitted by the first
                # implementation without changing its recorded source hashes.
                selection = v3.jread(run / "selection.json")
                _write_new(selection_path, selection)
            else:
                selection = _selection(run, protocol, cfg, audits)
                _write_new(selection_path, selection)
            if not selection.get("selected_recipe"):
                print(json.dumps(selection, ensure_ascii=False, indent=2))
                return 0
            if args.development_only:
                print(json.dumps(selection, ensure_ascii=False, indent=2))
                return 0
            command = recovery_command(args, selection_path, run, protocol, cfg)
            recovery_log = _recovery_log(run)
            invocation = recovery_invocation.begin(ROOT, run / "recovery", recovery_log.stem,
                                                   command, protocol["sha256"], selection_path)
            failure = None
            try:
                _run_logged("mixed-recovery", command, ROOT, recovery_log, _env(cfg["media"]),
                            lock_fd=lock_fd)
            except BaseException as exc:
                failure = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                recovery_invocation.finish(invocation, ROOT, run / "recovery", recovery_log,
                                           error=failure)
            print(json.dumps({"selection": selection, "recovery_run": str(run / "recovery"),
                              "recovery_log": str(recovery_log)},
                             ensure_ascii=False, indent=2))
            return 0
    except (v3.OrchestrationError, FileNotFoundError, ValueError) as exc:
        print(f"[mixed-pressure-study] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
