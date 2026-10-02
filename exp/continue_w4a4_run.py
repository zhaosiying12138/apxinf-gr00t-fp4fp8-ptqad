#!/usr/bin/env python3
"""Wait for one identified W4A4 driver, then continue its frozen run.

This Linux operations wrapper does not resume partial training, modify runtime
metrics, or relax the driver's evidence checks. The only repaired stage is an
empty first-QAD development directory left by the exact ffmpeg preflight error.
Run under a service manager; this process waits and keeps its flock while the
continuation driver runs. Checkpoints may contain only model weights, not Adam
state. An interrupted training stage remains an error for the existing driver.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
STAGE = "dev_qad_lr_5e-05"
FFMPEG_ERROR = ("FileNotFoundError: ffmpeg executable missing: set "
                "PTQAD_MEDIA_LIB=<environment>/lib or PTQAD_MEDIA_BIN=<environment>/bin")


class ContinuationError(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"not a JSON object: {path}")
    return value


def load_manifest(run: Path) -> dict:
    """Read mutable stage state afresh while verifying immutable source hashes."""
    if not run.is_dir():
        raise ContinuationError(f"run-dir must already exist: {run}")
    manifest = read_json(run / "run_manifest.json")
    if not isinstance(manifest.get("stages"), dict):
        raise ContinuationError("run_manifest requires a stages object")
    for path_key, hash_key in (("protocol_file", "protocol_sha256"),
                               ("selection_file", "selection_sha256")):
        path, expected = manifest.get(path_key), manifest.get(hash_key)
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or digest(Path(path)) != expected):
            raise ContinuationError(f"run_manifest {hash_key} does not match its file")
    return manifest


def option(argv: list[str], name: str, default: str | None = None) -> str:
    values = []
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ContinuationError(f"missing value for {name}")
            values.append(argv[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg[len(name) + 1:])
    if len(values) > 1 or (not values and default is None):
        raise ContinuationError(f"expected one {name}")
    return values[0] if values else str(default)


def _process(pid: int, proc_root: Path = Path("/proc")) -> dict | None:
    folder = proc_root / str(pid)
    try:
        before = (folder / "stat").read_text()
        fields = before[before.rfind(")") + 2:].split()
        starttime, state = fields[19], fields[0]
        argv = [x.decode() for x in (folder / "cmdline").read_bytes().split(b"\0") if x]
        cwd = (folder / "cwd").resolve(strict=True)
        after = (folder / "stat").read_text()
    except FileNotFoundError:
        return None
    if before != after:
        # CPU counters normally change. Identity must not change between reads.
        latest = after[after.rfind(")") + 2:].split()
        if latest[19] != starttime:
            raise ContinuationError(f"PID {pid} was reused while reading its identity")
    return {"pid": pid, "starttime": starttime, "state": state,
            "argv": argv, "cwd": str(cwd)}


def bind_process(pid: int, run: Path, proc_root: Path = Path("/proc")) -> dict:
    if pid <= 1 or pid == os.getpid():
        raise ContinuationError("wait-pid must identify another live driver")
    identity = _process(pid, proc_root)
    if identity is None or identity["state"] == "Z":
        raise ContinuationError(f"cannot bind absent/exited wait-pid {pid}")
    argv, cwd = identity["argv"], Path(identity["cwd"])
    if len(argv) < 2 or (cwd / argv[1]).resolve() != ROOT / "exp/run_w4a4_recovery.py":
        raise ContinuationError("wait-pid is not this repository's W4A4 driver")
    if (cwd / option(argv, "--run-dir")).resolve() != run.resolve():
        raise ContinuationError("wait-pid belongs to a different run-dir")
    identity["port_base"] = int(option(argv, "--port-base", "5790"))
    if not 1 <= identity["port_base"] <= 65500:
        raise ContinuationError("invalid original port-base")
    return identity


def wait_for_process(identity: dict, proc_root: Path = Path("/proc")) -> None:
    while True:
        current = _process(identity["pid"], proc_root)
        if current is None:
            return
        if current["starttime"] != identity["starttime"]:
            raise ContinuationError("wait-pid was reused; refusing continuation")
        if current["state"] == "Z":
            return
        if current["argv"] != identity["argv"]:
            raise ContinuationError("bound driver's command line changed")
        time.sleep(10)


def continuation_command(run: Path, manifest: dict, port: int,
                         allow_opd_nonimprovement: bool = False) -> list[str]:
    if manifest.get("w4a4") is not True:
        raise ContinuationError("run_manifest does not declare W4A4")
    fields = {"--protocol-file": "protocol_file", "--ptq-selection": "selection_file",
              "--base": "base", "--gr00t-repo": "gr00t_repo", "--python": "python",
              "--rollout-python": "rollout_python", "--dataset": "dataset"}
    values = {}
    for flag, key in fields.items():
        value = manifest.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ContinuationError(f"manifest requires absolute {key}")
        values[flag] = value
    capture = manifest.get("capture_dataset_identity", {}).get("root")
    if not isinstance(capture, str) or not Path(capture).is_absolute():
        raise ContinuationError("manifest requires capture_dataset_identity.root")
    command = [values["--python"], str(ROOT / "exp/run_w4a4_recovery.py"), "--run-dir", str(run)]
    for flag, value in values.items():
        command.extend([flag, value])
    command.extend(["--capture-dataset", capture, "--port-base", str(port),
                    "--until", "all", "--adopt-complete", "--cleanup-duplicates"])
    if allow_opd_nonimprovement:
        command.append("--allow-opd-nonimprovement")
    return command


def continuation_environment(media: Path, inherited: dict | None = None) -> dict[str, str]:
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(("QAD_", "OPD_")) or key in (
                "GR00T_BASE_CKPT", "TRAIN_SEED", "PROTOCOL_FILE", "PTQAD_PROTOCOL_FILE"):
            del env[key]
    env.update({"PTQAD_MEDIA_LIB": str(media), "PTQAD_MEDIA_BIN": str(media.parent / "bin"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "NO_ALBUMENTATIONS_UPDATE": "1", "PTQAD_LOCAL_HF_METADATA": "1"})
    env["LD_LIBRARY_PATH"] = str(media) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def require_idle_gpu(proc_root: Path = Path("/proc")) -> None:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=20)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if lines:
        raise ContinuationError(f"GPU compute jobs exist or query is indeterminate: {lines}")
    # Some WSL driver versions expose an empty compute-apps view. Also refuse
    # live project training/evaluation processes, including a second driver.
    scripts = {"run_w4a4_recovery.py", "run_high_fp4_v3.py", "lora_qad.py",
               "serve_recovery.py", "rollout_seeded.py", "run_recovery_eval.py"}
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            argv = [x.decode() for x in (folder / "cmdline").read_bytes().split(b"\0") if x]
        except FileNotFoundError:
            continue
        if any(Path(arg).name in scripts for arg in argv):
            raise ContinuationError(f"another recovery process remains: PID {folder.name}")


@contextmanager
def run_lock(run: Path):
    # Keep the inode; unlinking a lock could allow two different locked inodes.
    with (run / ".continue-w4a4.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContinuationError("another W4A4 continuation is active") from exc
        yield stream.fileno()


def recover_ffmpeg_preflight(run: Path, manifest: dict) -> Path | None:
    """Archive only the known preflight failure; never modify recorded results."""
    log, output = run / "logs" / (STAGE + ".log"), run / "artifacts" / STAGE
    marker = run / "stages" / (STAGE + ".json")
    if marker.exists() or STAGE in manifest.get("stages", {}):
        return None  # The frozen driver revalidates completed stages.
    if not log.exists() and not output.exists():
        return None
    if log.is_symlink() or output.is_symlink() or not log.is_file() or not output.is_dir():
        raise ContinuationError("incomplete development stage is not the known empty preflight failure")
    if (output / "eval_manifest.json").exists() or any(output.iterdir()):
        raise ContinuationError("refusing nonempty development directory")
    text = log.read_text(encoding="utf-8")
    lines = text.splitlines()
    # ``run_recovery_eval.py`` writes its UTC/CWD/COMMAND receipt after the
    # traceback, so the known preflight error is not necessarily the final
    # traceback line.  Keep the identity check strict while allowing that
    # deterministic receipt trailer.
    if (lines.count(FFMPEG_ERROR) != 1 or lines.count("Traceback (most recent call last):") != 1
            or not lines or lines[-1] != "RETURN_CODE 1"):
        raise ContinuationError("development log does not prove the exact ffmpeg preflight failure")
    errors = [line for line in lines if re.match(r"^[A-Za-z_][\w.]*(?:Error|Exception):", line)]
    if errors != [FFMPEG_ERROR] or any(line.startswith("COMPLETED_UTC") for line in lines):
        raise ContinuationError("development log contains another failure or completion")
    commands = [line[8:] for line in lines if line.startswith("COMMAND ")]
    if len(commands) != 1:
        raise ContinuationError("development log lacks one recorded command")
    argv = shlex.split(commands[0])
    if (len(argv) < 2 or Path(argv[1]).resolve() != ROOT / "eval/run_recovery_eval.py"
            or Path(option(argv, "--out")).resolve() != output
            or option(argv, "--purpose") != "development"
            or option(argv, "--protocol-file") != manifest["protocol_file"]):
        raise ContinuationError("preflight log command belongs to another evaluation")
    require_idle_gpu()
    source_sha = digest(log)
    archive = run / "operations" / ("ffmpeg-preflight-" + uuid.uuid4().hex)
    archive.mkdir(parents=True, exist_ok=False)
    archived = archive / log.name
    receipt = {"status": "prepared", "operation": "archive_empty_ffmpeg_preflight",
               "source_log": str(log), "archived_log": str(archived), "log_sha256": source_sha,
               "empty_directory": str(output), "protocol_sha256": manifest["protocol_sha256"],
               "recorded_exception": FFMPEG_ERROR, "created_unix": time.time()}
    receipt_path = archive / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    # Refuse if another actor changed the evidence after its initial inspection.
    if digest(log) != source_sha or marker.exists() or any(output.iterdir()):
        raise ContinuationError("preflight evidence changed before archiving")
    log.rename(archived)
    output.rmdir()  # Deliberately never recursive; any new file makes this fail.
    receipt["status"] = "complete"
    pending = archive / "receipt.complete.json"
    pending.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    pending.replace(receipt_path)
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--media-lib", required=True)
    parser.add_argument("--allow-opd-nonimprovement", action="store_true")
    args = parser.parse_args(argv)
    try:
        run = Path(args.run_dir).expanduser().resolve(strict=True)
        media = Path(args.media_lib).expanduser().resolve(strict=True)
        ffmpeg = media.parent / "bin/ffmpeg"
        if not media.is_dir() or not ffmpeg.is_file() or not os.access(ffmpeg, os.X_OK):
            raise ContinuationError("media-lib requires an executable sibling bin/ffmpeg")
        manifest = load_manifest(run)
        bound = bind_process(args.wait_pid, run)
        command = continuation_command(run, manifest, bound["port_base"], args.allow_opd_nonimprovement)
        env = continuation_environment(media)
        subprocess.run([str(ffmpeg), "-version"], env=env, check=True,
                       capture_output=True, text=True, timeout=20)
        with run_lock(run) as lock_fd:
            operation = run / "operations" / ("continuation-" + uuid.uuid4().hex)
            operation.mkdir(parents=True, exist_ok=False)
            (operation / "invocation.json").write_text(json.dumps({
                "bound_process": bound, "command": command, "media_library": str(media),
                "protocol_sha256": manifest["protocol_sha256"],
                "selection_sha256": manifest["selection_sha256"]}, indent=2) + "\n", encoding="utf-8")
            print(f"Waiting for bound PID {args.wait_pid}; operations: {operation}", flush=True)
            wait_for_process(bound)
            latest = load_manifest(run)
            if (command != continuation_command(run, latest, bound["port_base"], args.allow_opd_nonimprovement)
                    or any(latest.get(k) != manifest.get(k) for k in ("protocol_sha256", "selection_sha256"))):
                raise ContinuationError("run manifest input identity changed while waiting")
            require_idle_gpu()
            recover_ffmpeg_preflight(run, latest)
            with (operation / "driver.log").open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
                log.write(f"\nCONTINUATION_RETURN_CODE {result.returncode}\n")
            if result.returncode:
                raise ContinuationError(f"driver refused or failed; inspect {operation / 'driver.log'}")
            return 0
    except (ContinuationError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"[continue-w4a4] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
