"""CPU-only, append-only launch provenance for recovery wrapper invocations.

These snapshots attest file bytes at launch and completion. They do not infer
which source bytes an earlier, unrecorded Python process had imported.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

SOURCES = ("exp/run_mixed_pressure_study.py", "exp/run_mixed_pressure_recovery.py",
           "exp/run_high_fp4_v3.py", "exp/recovery_invocation.py")
FORMAT = "mixed_pressure_recovery_invocation_v1"


def identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _json(path: Path, value: Any) -> None:
    _write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def _copy(source: Path, target: Path) -> dict[str, Any]:
    data = source.read_bytes()
    _write(target, data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _stages(run: Path, target: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for source in sorted((run / "stages").glob("*.json")):
        record = json.loads(source.read_bytes())
        if record.get("status") != "complete":
            raise ValueError("Incomplete stage marker: " + str(source))
        result[source.stem] = _copy(source, target / source.name)
    return result


def begin(root: Path, run: Path, attempt: str, command: list[str],
          protocol_sha256: str, selection: Path) -> Path:
    """Write once before launch; any existing attempt is an error."""
    # Keep launch records beside the recovery directory. The driver requires
    # a new recovery directory to remain absent/empty until it creates its
    # own run_manifest.json.
    folder = run.parent / "invocations" / attempt
    folder.mkdir(parents=True, exist_ok=False)
    sources = {relative: _copy(root / relative, folder / "source" / relative)
               for relative in SOURCES}
    stages = _stages(run, folder / "stages_before")
    _json(folder / "invocation.json", {
        "format": FORMAT, "attempt": attempt,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "cwd": str(root), "run_dir": str(run),
        "protocol_sha256": protocol_sha256,
        "selection": {"path": str(selection), **identity(selection)},
        "sources": sources, "completed_stages_before": stages,
        "scope": "Source file bytes immediately before this launch. Stages already complete before this invocation are not attributed to this snapshot.",
    })
    return folder


def finish(folder: Path, root: Path, run: Path, log: Path,
           *, error: str | None = None) -> None:
    """Append a separate result; never rewrite the start record or snapshots."""
    manifest = json.loads((folder / "invocation.json").read_bytes())
    text = log.read_text(errors="replace") if log.is_file() else ""
    codes = re.findall(r"^RETURN_CODE (-?\d+)\s*$", text, re.MULTILINE)
    code = int(codes[-1]) if codes else None
    stages = _stages(run, folder / "stages_after")
    _json(folder / "result.json", {
        "format": FORMAT, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "invocation_identity": identity(folder / "invocation.json"),
        "status": "completed" if code == 0 and error is None else "failed",
        "returncode": code, "error": error,
        "log": {"path": str(log), **identity(log)} if log.is_file() else None,
        "sources_at_end": {relative: identity(root / relative) for relative in SOURCES},
        "completed_stages_after": stages,
        "newly_completed_stages": sorted(set(stages) - set(manifest["completed_stages_before"])),
    })


def verify(folder: Path) -> dict[str, Any]:
    """Verify an invocation without importing or executing its producer."""
    invocation_path = folder / "invocation.json"
    invocation = json.loads(invocation_path.read_bytes())
    if invocation.get("format") != FORMAT:
        raise ValueError("unknown invocation format")
    for relative, recorded in invocation.get("sources", {}).items():
        snapshot = folder / "source" / relative
        if identity(snapshot) != {k: recorded[k] for k in ("bytes", "sha256")}:
            raise ValueError("invocation source snapshot hash mismatch: " + relative)
    before = invocation.get("completed_stages_before", {})
    for name, recorded in before.items():
        snapshot = folder / "stages_before" / (name + ".json")
        if identity(snapshot) != {k: recorded[k] for k in ("bytes", "sha256")}:
            raise ValueError("invocation stage snapshot hash mismatch: " + name)
    result_path = folder / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_bytes())
        if result.get("invocation_identity") != identity(invocation_path):
            raise ValueError("invocation result does not bind invocation.json")
        after = result.get("completed_stages_after", {})
        for name, recorded in after.items():
            snapshot = folder / "stages_after" / (name + ".json")
            if identity(snapshot) != {k: recorded[k] for k in ("bytes", "sha256")}:
                raise ValueError("invocation completion snapshot hash mismatch: " + name)
    return invocation
