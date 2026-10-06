"""Read-only screenshot provenance checks shared by the v12 README entry points.

The publication validator owns the existing 15 current / 2 approved retained
capture contract.  This module adds release identity and managed-window checks
for the seven v12-dependent figures, plus the separately published compile shot.
It never runs a capture, edits receipts, or constructs a policy.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import validate_publication as publication
from publication_guard import capture_crop_contract, check_record, file_record, require
from v12_publication_contract import PROTOCOL_NAME, PROTOCOL_SHA256, validate_v12_results

CAPTURE_ORDER = (
    "shot_bake", "shot_collect", "shot_evalserver", "shot_fp8probe",
    "shot_gemm", "shot_gr00t", "shot_nvfp4", "shot_opbench", "shot_opd",
    "shot_opdcache", "shot_packed", "shot_pi05", "shot_probe", "shot_qad",
    "shot_qat", "shot_rollout", "shot_verify",
)
V12_CAPTURES = frozenset({"shot_bake", "shot_collect", "shot_evalserver",
                          "shot_opd", "shot_opdcache", "shot_qad", "shot_rollout"})
COMPILE_PREFIX = "docs/evidence/ubuntu-compile/"
COMPILE_FILES = frozenset({"docs/images/ubuntu-compile.png", *(
    COMPILE_PREFIX + name for name in (
        "run.sh", "capture.log", "capture.json", "capture.crop.json",
        "capture.png.window-binding.json"))})


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"Evidence must be a JSON object: {path}")
    return value


def _regular(root: Path, relative: str) -> Path:
    """Reject aliases and escapes before the common hash checker follows paths."""
    require(isinstance(relative, str) and "\\" not in relative,
            f"Evidence requires a portable relative path: {relative!r}")
    value = PurePosixPath(relative)
    require(not value.is_absolute() and value.as_posix() == relative and
            all(part not in (".", "..") for part in value.parts),
            f"Unsafe evidence path: {relative!r}")
    path = root
    for part in value.parts:
        path = path / part
        require(not path.is_symlink(), f"Evidence must not be a symlink: {path}")
    require(path.is_file(), f"Missing evidence file: {path}")
    require(path.resolve().is_relative_to(root.resolve()), f"Evidence escapes root: {path}")
    return path


def _record(root: Path, identity: dict[str, Any]) -> Path:
    require(isinstance(identity, dict) and type(identity.get("bytes")) is int and
            identity["bytes"] > 0 and isinstance(identity.get("sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]), "Invalid evidence identity")
    path = _regular(root, identity.get("path"))
    check_record(identity, root)
    return path


@contextmanager
def _publication_root(paper: Path):
    # Existing publication functions use one module-level paper directory.
    # Restore it even on rejection so fixture audits cannot leak their root.
    previous = publication.P
    publication.P = paper
    try:
        yield
    finally:
        publication.P = previous


def validate_window_binding(binding: dict[str, Any], dimensions: list[int]) -> None:
    """Prove the captured foreground window was the selected managed terminal."""
    require(binding.get("status") == "captured", "Window binding did not capture a terminal")
    token = binding.get("window_title_token")
    require(isinstance(token, str) and re.fullmatch(r"FP4VLA-[0-9a-f]{32}", token),
            "Window binding has no managed title token")
    require(binding.get("foreground_before_capture_return") is True,
            "Capture terminal was not foreground before capture")
    selected = binding.get("selected")
    require(isinstance(selected, dict), "Window binding lacks selected terminal")
    for key in ("hwnd", "pid"):
        require(type(selected.get(key)) is int and selected[key] > 0,
                f"Invalid selected window {key}")
    for name in ("selected", "capture_window", "capture_foreground"):
        window = binding.get(name)
        require(isinstance(window, dict), f"Window binding lacks {name}")
        require(all(window.get(key) == selected[key] for key in ("hwnd", "pid")),
                f"Window identity changed: {name}")
        require(window.get("title") == token and window.get("owner") == "WindowsTerminal" and
                window.get("windowClass") == "CASCADIA_HOSTING_WINDOW_CLASS",
                f"Capture belongs to another terminal: {name}")
        require(window.get("exists") is True and window.get("visible") is True and
                window.get("minimized") is False and type(window.get("cloaked")) is int and
                window["cloaked"] == 0, f"Capture window is hidden or cloaked: {name}")
    before = binding.get("before")
    require(isinstance(before, list) and all(isinstance(row, dict) for row in before),
            "Window binding lacks the pre-launch inventory")
    require(all(row.get("hwnd") != selected["hwnd"] for row in before),
            "Managed capture reused a pre-existing terminal window")
    window = binding["capture_window"]
    require(all(type(window.get(key)) is int for key in ("left", "top", "right", "bottom")),
            "Capture window bounds are missing")
    require(window["left"] <= 0 and window["top"] <= 0 and
            window["right"] >= dimensions[0] and window["bottom"] >= dimensions[1],
            "Capture window does not cover the published crop")


def _support(row: dict[str, Any], paper: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for identity in row.get("supporting_files", []):
        path = _record(paper, identity)
        require(path.name not in files, "Duplicate capture dependency: " + path.name)
        files[path.name] = path
    return files


def _release_binding(row: dict[str, Any], paper: Path, manifest_sha: str,
                     support: dict[str, Path]) -> Path:
    """Require a hash-bound plan for the exact script and frozen final release."""
    sidecar = _load(paper / row["capture_sidecar"])
    source_name = PurePosixPath(str(sidecar.get("script", "")).replace("\\", "/")).name
    require(bool(source_name), f"Capture sidecar lacks script identity: {row['figure']}")
    plans = []
    for path in support.values():
        if path.suffix != ".json" or path.name.endswith(".window-binding.json"):
            continue
        data = _load(path)
        scripts = data.get("script_files", {})
        if isinstance(scripts, dict) and source_name in scripts:
            plans.append((path, data, scripts[source_name]))
    require(len(plans) == 1, f"Capture needs one registered release plan: {row['figure']}")
    path, plan, script = plans[0]
    require(plan.get("protocol_sha256") == PROTOCOL_SHA256 and
            plan.get("final_manifest_sha256") == manifest_sha,
            f"Capture is not bound to the final v12 release: {row['figure']}")
    entry = paper / row["script"]
    require(script.get("sha256") == row["script_sha256"] and
            script.get("bytes") == entry.stat().st_size,
            f"Capture script differs from release plan: {row['figure']}")
    dependencies = plan.get("script_dependencies", {}).get("all_entrypoints", [])
    require(isinstance(dependencies, list) and len(dependencies) == len(set(dependencies)),
            f"Invalid capture dependencies: {row['figure']}")
    for name in dependencies:
        require(name in support and name in plan["script_files"],
                f"Capture shared script is not archived: {name}")
        source = support[name]
        identity = plan["script_files"][name]
        require(identity.get("sha256") == publication.sha(source) and
                identity.get("bytes") == source.stat().st_size,
                f"Capture shared script differs from release plan: {name}")
    return path


def validate_compile_capture(root: Path, approved_sample: dict[str, Any]) -> dict[str, Any]:
    """Validate the sixth-file compilation proof without invoking the compiler."""
    root = root.resolve()
    manifest_path = _regular(root, COMPILE_PREFIX + "manifest.json")
    manifest = _load(manifest_path)
    require(manifest.get("visually_verified") is True and type(manifest.get("exit_status")) is int and
            manifest["exit_status"] == 0 and manifest.get("dimensions") == [3840, 2280],
            "Compile screenshot lacks successful visual acceptance")
    require(manifest.get("approved_sample_sha256") == approved_sample.get("sha256") and
            approved_sample.get("confirmed_by_user") is True,
            "Compile screenshot uses another approved sample")
    rows = manifest.get("files")
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
            "Compile screenshot requires file identities")
    require(len(rows) == len(COMPILE_FILES) and {row.get("path") for row in rows} == COMPILE_FILES,
            "Compile screenshot must contain exactly image, script, log, sidecar, crop and window binding")
    files = {row["path"]: _record(root, row) for row in rows}
    sidecar = _load(files[COMPILE_PREFIX + "capture.json"])
    crop = _load(files[COMPILE_PREFIX + "capture.crop.json"])
    image = files["docs/images/ubuntu-compile.png"]
    capture_crop_contract(sidecar, crop, image, [3840, 2280])
    publication.check_png(image, [3840, 2280], terminal=True)
    require(manifest.get("captured_at") == sidecar.get("captured_at"),
            "Compile screenshot timestamp differs from accepted capture")
    validate_window_binding(_load(files[COMPILE_PREFIX + "capture.png.window-binding.json"]), [3840, 2280])
    require(manifest.get("targets") == ["fp4_gemm_bench", "fp4_opbench", "fp8_probe"] and
            manifest.get("arch") == "sm_120", "Compile screenshot targets or architecture differ")
    require(isinstance(manifest.get("scope"), str) and "compilation only" in manifest["scope"],
            "Compile screenshot must describe compilation-only evidence")
    log = files[COMPILE_PREFIX + "capture.log"].read_text(encoding="utf-8")
    for target in manifest["targets"]:
        require(re.search(r"nvcc[^\n]+-arch=sm_120[^\n]+ -o " + target + r"(?:\s|$)", log),
                f"Compile log lacks the declared target command: {target}")
    return {"manifest": file_record(manifest_path, root), "files": rows, "verified": True}


def validate_capture_provenance(root: Path) -> dict[str, Any]:
    """Audit all 17 figures and compile supplement; reject old release evidence."""
    root = root.resolve()
    paper = root / "paper"
    final = _load(_regular(root, "paper/evidence/final_results.json"))
    validate_v12_results(final, root=root)
    protocol = _regular(paper, "evidence/protocol/" + PROTOCOL_NAME)
    require(publication.sha(protocol) == PROTOCOL_SHA256, "Archived screenshot protocol is not frozen v12")
    final_path = _regular(paper, "evidence/final_manifest.json")
    manifest = _load(final_path)
    require(manifest.get("format") == "w4a4_recovery_v12_final_manifest" and
            manifest.get("protocol_sha256") == PROTOCOL_SHA256 and
            manifest.get("selection_uses_heldout") is False,
            "Screenshots require the final v12 manifest")
    inputs: set[Path] = set()
    with _publication_root(paper):
        captures = publication.load_capture_manifests(set(CAPTURE_ORDER), inputs)
        rows = captures["screenshots"]
        for row in rows:
            _regular(paper, "figs/" + row["figure"] + ".png")
            if row.get("retained_unaffected") is not True:
                for field in ("raw_log", "script", "capture_sidecar", "crop_manifest"):
                    _regular(paper, row.get(field))
        checked = publication.capture_records(captures, set(CAPTURE_ORDER), inputs)
    by_name = {row["figure"]: row for row in rows}
    for name in CAPTURE_ORDER:
        row = by_name[name]
        if row.get("retained_unaffected") is True:
            continue  # Only the two explicitly approved Git-anchored BF16 captures.
        support = _support(row, paper)
        sidecar = _load(paper / row["capture_sidecar"])
        require(row.get("captured_at") == sidecar.get("captured_at"),
                f"Capture timestamp differs from sidecar: {name}")
        bindings = [path for path in support.values() if path.name.endswith(".window-binding.json")]
        require(len(bindings) <= 1, f"Ambiguous managed window binding: {name}")
        if name in V12_CAPTURES:
            require(len(bindings) == 1, f"v12 capture lacks registered managed window binding: {name}")
            _release_binding(row, paper, publication.sha(final_path), support)
        if bindings:
            image_name = PurePosixPath(str(sidecar.get("image", "")).replace("\\", "/")).name
            require(bindings[0].name == image_name + ".window-binding.json",
                    f"Window binding belongs to another capture image: {name}")
            validate_window_binding(_load(bindings[0]), [3840, 2280])
    compile_capture = validate_compile_capture(root, captures["approved_sample"])
    return {"protocol_sha256": PROTOCOL_SHA256, "screenshots": checked,
            "capture_rows": by_name, "compile": compile_capture,
            "package_inputs": [file_record(path, root) for path in sorted(inputs)]}
