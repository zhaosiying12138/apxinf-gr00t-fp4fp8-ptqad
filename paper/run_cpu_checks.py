#!/usr/bin/env python3
"""Record CPU unittest and publication suites in a new --out directory.

Run with the installed recovery environment's Python. GR00T_REPO defaults to
third_party/Isaac-GR00T; set it when the dependency checkout lives elsewhere.
No pretrained checkpoint, recovery training, GPU benchmark or closed-loop
evaluation is launched; small synthetic CPU model fixtures are allowed.
The combined cpu-tests.log contains every suite's complete output; per-suite
logs are convenience copies. Copy the JSON and combined log together when
publishing a reviewed receipt. This runner never replaces published receipts.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def suites():
    paper_tests = sorted((ROOT / "paper/validation").glob("test_*.py"))
    if not paper_tests:
        raise ValueError("No paper CPU validation suites found")
    return [
        ("unittest-discovery", ["-m", "unittest", "discover", "-s", "tests", "-v"],
         sorted((ROOT / "tests").rglob("test_*.py"))),
        *[(path.stem.removeprefix("test_").replace("_", "-"),
           ["-O", path.relative_to(ROOT).as_posix()], [path]) for path in paper_tests],
    ]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snap(paths):
    return {path.relative_to(ROOT).as_posix(): sha(path) if path.is_file() else None
            for path in sorted(paths)}


def source_files():
    tree = ast.parse((ROOT / "paper/capture_runtime.py").read_text())
    runtime = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                   and target.id == "DEFAULT_SOURCE_FILES" for target in node.targets))
    sources = {ROOT / name for name in runtime}
    # Git excludes ignored dependency checkouts and virtual environments.
    # Read only project source bytes, never checkpoints or runtime outputs.
    project_paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
         "baselines", "quant", "rl", "eval", "exp", "setup"], cwd=ROOT).decode().split("\0")
    sources.update(ROOT / name for name in project_paths if name.endswith(".py") and (ROOT / name).is_file())
    sources.update((ROOT / "paper").glob("*.py"))
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="New scratch directory for cpu-tests.json, cpu-tests.log and suite logs")
    args = parser.parse_args(argv)
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = utc()
    tick = time.monotonic()
    groot = Path(os.environ.get("GR00T_REPO") or ROOT / "third_party/Isaac-GR00T").expanduser().resolve()
    recorded_env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "GR00T_REPO": str(groot), "NO_ALBUMENTATIONS_UPDATE": "1",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    env = {**os.environ, **recorded_env}
    selected_suites = suites()
    sources = source_files()
    tests = {path for _, _, paths in selected_suites for path in paths}
    source_before = snap(sources)
    tests_before = snap(tests)
    missing = [name for name, value in {**source_before, **tests_before}.items() if value is None]
    if missing:
        raise FileNotFoundError("Missing recorded source/test files: " + ", ".join(missing))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    version = subprocess.check_output([sys.executable, "--version"], env=env, text=True).strip()
    log_path = output / "cpu-tests.log"
    rows = []
    with log_path.open("wb") as combined:
        combined.write((f"Complete CPU unittest collection and {len(selected_suites) - 1} paper suites\n"
                        "No GPU/pretrained-model/closed-loop evaluation; test fixtures are synthetic.\n"
                        + "ENVIRONMENT " + json.dumps(recorded_env, sort_keys=True) + "\n").encode())
        combined.flush()
        for name, suite_args, paths in selected_suites:
            command = [sys.executable, *suite_args]
            command_text = shlex.join(command)
            suite_log = output / (name + ".log")
            suite_started = utc()
            suite_tick = time.monotonic()
            with suite_log.open("wb") as stream:
                process = subprocess.run(command, cwd=ROOT, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT)
            payload = suite_log.read_bytes()
            text = payload.decode(errors="replace")
            runs = re.findall(r"^Ran (\d+) tests? in ([0-9.]+)s$", text, re.M)
            endings = re.findall(r"^(OK|FAILED)(?: \(([^\n]*)\))?$", text, re.M)
            ending = endings[-1] if endings else ("missing", "")
            counters = {key: int(value) for key, value in re.findall(
                r"(failures|errors|skipped|expected failures|unexpected successes)=(\d+)", ending[1])}
            count = int(runs[-1][0]) if runs else None
            row = {"suite": name, "status": "passed" if process.returncode == 0 and
                   ending[0] == "OK" and count else "failed", "command": command_text,
                   "command_argv": command, "environment": recorded_env, "cwd": str(ROOT),
                   "started_utc": suite_started, "completed_utc": utc(),
                   "returncode": process.returncode, "tests": count,
                   "unittest_elapsed_seconds": float(runs[-1][1]) if runs else None,
                   "process_wall_seconds": time.monotonic() - suite_tick,
                   "test_files": [str(path.relative_to(ROOT)) for path in paths],
                   "log": suite_log.name, "log_sha256": sha(suite_log)}
            for key in ("failures", "errors", "skipped", "expected failures", "unexpected successes"):
                row[key.replace(" ", "_")] = counters.get(key, 0)
            rows.append(row)
            combined.write(("\nSUITE " + name + "\nCOMMAND " + command_text + "\n").encode())
            combined.write(payload)
            combined.write(("\nRECEIPT " + json.dumps(row, sort_keys=True) + "\n").encode())
            combined.flush()
            print(json.dumps({key: row[key] for key in
                              ("suite", "status", "tests", "skipped", "failures", "errors", "returncode")}), flush=True)
    # Re-enumerate as well as re-hash, so added and removed tests/sources fail
    # the receipt rather than silently changing the inventory during a run.
    source_after = snap(source_files())
    tests_after = snap({path for _, _, paths in suites() for path in paths})
    changed = sorted({name for before, after in ((source_before, source_after), (tests_before, tests_after))
                      for name in before.keys() | after.keys() if before.get(name) != after.get(name)})
    command = [sys.executable, str(Path(__file__).resolve()), "--out", str(output)]
    report = {"status": "passed" if all(row["status"] == "passed" for row in rows) and not changed else "failed",
              "scope": "Complete tests/ unittest discovery and all paper/validation/test_*.py CPU suites. "
                       "Synthetic fixtures do not provide model accuracy, GPU parity or publication approval.",
              "command": " ".join(shlex.quote(key + "=" + value) for key, value in recorded_env.items()) +
                         " " + shlex.join(command),
              "command_argv": command,
              "cwd": str(ROOT), "environment": recorded_env, "python_version": version,
              "source_head_before": head, "started_utc": started, "completed_utc": utc(),
              "returncode": 0 if all(row["status"] == "passed" for row in rows) and not changed else 1,
              "suites": rows, "tests": sum(row["tests"] or 0 for row in rows),
              "unittest_elapsed_seconds": sum(row["unittest_elapsed_seconds"] or 0 for row in rows),
              "process_wall_seconds": time.monotonic() - tick,
              "gpu_execution": False, "publication_zip_generated": False,
              "test_file_sha256": tests_before, "source_sha256": source_before,
              "test_file_sha256_after": tests_after, "source_sha256_after": source_after,
              "source_hash_scope": "Git-enumerated current project Python under baselines/quant/rl/eval/exp/setup, "
                                   "top-level paper Python and explicit runtime files. Identity inventory, not coverage.",
              "files_changed_during_check": changed,
              "runner_sha256": sha(__file__), "log": log_path.name,
              "log_path_base": "receipt directory",
              "log_scope": "cpu-tests.log embeds every suite payload and receipt; per-suite logs are convenience copies",
              "log_sha256": sha(log_path)}
    for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes"):
        report[key] = sum(row[key] for row in rows)
    (output / "cpu-tests.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "tests", "skipped", "failures", "errors", "files_changed_during_check")}), flush=True)
    return report["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
