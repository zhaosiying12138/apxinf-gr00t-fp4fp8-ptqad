#!/usr/bin/env python3
"""Export and replay the five-arm action diagnostic without models or .pt files.

``collect`` validates original local artifacts and selected checkpoint bytes.
``verify`` only reads the resulting folder: finite tensor JSON, original run
receipts, input/final manifests, logs and a byte-preserved CPU comparator. It
recomputes the producer's four comparisons, not a second metric implementation.
The archive supports metric replay, not inference from absent camera/model data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")
PROTOCOL_SHA256 = "31addae911f92db65dda3c1c3145e8b798dca46c49a5ae12bbe10d232b5a4521"
REPLAY = {
    "producer": "exp/action_chunk_diagnostics.py",
    "action_contract": "rl/gr00t_runtime.py",
    "producer_import": "rl/probe_distill.py",
}
TENSOR_TAG = "__finite_torch_tensor_v1__"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Expected regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def cpu_only():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    require(not torch.cuda.is_initialized(), "CPU export/replay requires a process without initialized CUDA")
    require(sys.byteorder == "little", "Tensor byte identity currently requires a little-endian host")
    return torch


def tensor_sha(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(cpu_only().uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def encode(value):
    """Round-trip the exact finite tensor dtype, shape and bytes through JSON."""
    torch = cpu_only()
    if torch.is_tensor(value):
        require(value.layout == torch.strided and not value.is_complex(), "Unsupported tensor layout/dtype")
        require(bool(torch.isfinite(value).all()), "Nonfinite tensor cannot be published")
        return {TENSOR_TAG: True, "dtype": str(value.dtype).removeprefix("torch."),
                "shape": list(value.shape), "values": value.detach().cpu().reshape(-1).tolist(),
                "tensor_sha256": tensor_sha(value)}
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value) and TENSOR_TAG not in value,
                "Payload requires string keys without reserved tensor tag")
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode(v) for v in value]
    require(value is None or type(value) in (str, bool, int, float), "Unsupported payload value")
    require(type(value) is not float or math.isfinite(value), "Nonfinite scalar")
    return value


def decode(value):
    torch = cpu_only()
    if isinstance(value, dict) and TENSOR_TAG in value:
        require(set(value) == {TENSOR_TAG, "dtype", "shape", "values", "tensor_sha256"}
                and value[TENSOR_TAG] is True, "Malformed tensor record")
        allowed = {"float16", "bfloat16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool"}
        require(value["dtype"] in allowed, "Unsupported tensor dtype")
        shape = value["shape"]
        require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), "Malformed tensor shape")
        size = math.prod(shape)
        require(size <= 100_000_000 and isinstance(value["values"], list)
                and len(value["values"]) == size, "Tensor element count differs")
        result = torch.tensor(value["values"], dtype=getattr(torch, value["dtype"]), device="cpu").reshape(shape)
        require(bool(torch.isfinite(result).all()), "Nonfinite reconstructed tensor")
        require(tensor_sha(result) == value["tensor_sha256"], "Tensor byte identity differs")
        return result
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


@contextmanager
def comparator(source_root):
    """Import exactly the recorded three-file CPU implementation, then restore imports."""
    cpu_only()
    names = ("gr00t_runtime", "probe_distill")
    saved = {name: sys.modules.pop(name, None) for name in names}
    path_before = sys.path[:]
    bytecode_before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    source_root = Path(source_root)
    try:
        sys.path.insert(0, str(source_root / "rl"))
        spec = importlib.util.spec_from_file_location("_archived_action_diagnostics", source_root / REPLAY["producer"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = path_before
        sys.dont_write_bytecode = bytecode_before
        for name, value in saved.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value


def _local(folder, relative):
    require(isinstance(relative, str) and "\\" not in relative, "Archive path must be relative POSIX")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and path.parts and all(p not in (".", "..") for p in path.parts),
            "Unsafe archive path")
    result = Path(folder).joinpath(*path.parts)
    require(result.resolve().is_relative_to(Path(folder).resolve()) and not result.is_symlink(),
            "Archive path escaped folder")
    return result


def _copy(source, folder, relative):
    original = identity(source)
    target = _local(folder, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "Duplicate archive destination: " + relative)
    shutil.copyfile(source, target)
    require(identity(target) == original == identity(source), "Source changed while copying: " + str(source))
    return {"path": relative, **original}


def _publish_new(stage, out):
    """Atomically publish without replacing even a concurrently created directory."""
    if sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        # AT_FDCWD=-100, RENAME_NOREPLACE=1. Available on the WSL Linux host.
        if rename(-100, os.fsencode(stage), -100, os.fsencode(out), 1):
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number), str(out))
    elif os.name == "nt":
        # Windows rename rejects an existing destination, including directories.
        stage.rename(out)
    else:
        raise ValueError("Atomic no-replace publication currently supports Linux/Windows; verify is portable")


def _completion_log(path, arm, count):
    pattern = re.compile(r"\[action-diagnostic\] " + re.escape(arm) + r" (\d+)/(\d+)")
    observed = []
    for line in Path(path).read_text(encoding="utf-8", errors="strict").splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            observed.append(tuple(int(x) for x in match.groups()))
    require(observed == [(i, count) for i in range(1, count + 1)], f"Incomplete/duplicated action log: {arm}")


def _selected(final, selection, arm):
    if arm == "bf16":
        return selection["arms"]["bf16"]["model_identity"]
    if arm == "ptq":
        return selection["arms"][final["selected_pressure_recipe"]]["model_identity"]
    return final[{"qad": "selected_qad_model_identity", "continued_qad": "selected_continued_model_identity",
                  "qad_opd": "selected_opd_model_identity"}[arm]]


def _model_receipt(receipt, selected):
    """Check archived selected identities without opening their original model paths."""
    require(receipt["checkpoint"] == selected["path"], "Selected checkpoint path differs")
    files = receipt["checkpoint_files"]
    root = PurePosixPath(selected["path"])
    recorded = {PurePosixPath(p).name for p in files
                if PurePosixPath(p).parent == root and p.endswith(".safetensors")}
    require(recorded == {r["name"] for r in selected["shards"]}, "Selected checkpoint shard set differs")
    for item in selected["shards"]:
        require(files.get(str(root / item["name"])) == {k: item[k] for k in ("bytes", "sha256")},
                "Selected checkpoint weight identity differs")
    for name, sha in selected["metadata"].items():
        require(files.get(str(root / name), {}).get("sha256") == sha, "Selected checkpoint metadata differs")


def _validate_payload(payload, receipt, arm, inputs, inputs_sha, final_sha):
    require(receipt.get("format") == "gr00t_action_run_v1" and receipt.get("status") == "complete", "Incomplete action receipt")
    require(receipt.get("arm") == payload.get("arm") == arm, "Action arm identity differs")
    require(payload.get("format") == "gr00t_action_outputs_v1", "Unknown action payload")
    require(receipt.get("inputs_sha256") == payload.get("inputs_sha256") == inputs_sha,
            "Action input identity differs")
    require(receipt.get("final_manifest_sha256") == payload.get("final_manifest_sha256") == final_sha,
            "Action final manifest identity differs")
    require(payload.get("input_manifest") == inputs, "Embedded diagnostic inputs differ")
    require(receipt.get("source_files") == payload.get("source_files"), "Action source identities differ")
    require(receipt.get("sample_count") == len(payload.get("records", [])) == len(inputs["samples"]) > 0,
            "Action sample count differs")
    require(receipt.get("interpretation") == inputs["interpretation"], "Diagnostic interpretation differs")
    statistics = str(PurePosixPath(receipt["checkpoint"]) / "statistics.json")
    require(receipt["checkpoint_files"].get(statistics, {}).get("sha256") == inputs["statistics_sha256"],
            "Diagnostic checkpoint/input normalization differs")


def _final_binding(folder, manifest, external):
    final = read(folder / "final_manifest.json")
    selection = read(folder / "selection.json")
    inputs = read(folder / "inputs.json")
    final_sha = identity(folder / "final_manifest.json")["sha256"]
    require(manifest["final_manifest_sha256"] == final_sha, "Archived final manifest identity differs")
    if external is not None:
        require(identity(external)["sha256"] == final_sha, "Installed final manifest identity differs")
    require(final.get("format") == "w4a4_recovery_v12_final_manifest"
            and final.get("selection_uses_heldout") is False
            and set(final.get("required_arms", [])) == set(ARMS)
            and final.get("selected_pressure_recipe") == "rtn_w4a4_category", "Final v12 five-arm selection is incomplete")
    require(final["protocol_sha256"] == manifest["protocol_sha256"] == inputs["protocol_sha256"]
            == identity(folder / "protocol.json")["sha256"] == PROTOCOL_SHA256, "Frozen v12 protocol differs")
    require(final["selection_sha256"] == identity(folder / "selection.json")["sha256"]
            and selection["protocol_sha256"] == PROTOCOL_SHA256, "Archived selection identity differs")
    require(inputs.get("format") == "gr00t_action_inputs_v1" and inputs.get("purpose") in
            ("teacher_supervision", "collection", "development"), "Undeclared or heldout diagnostic inputs")
    protocol = read(folder / "protocol.json")
    allowed = set(protocol["partitions"][inputs["purpose"]]["init_state_indices"])
    require(not allowed.intersection(protocol["partitions"]["heldout"]["init_state_indices"]), "Diagnostic/heldout partitions overlap")
    require(all(s["init_state_index"] in allowed for s in inputs["samples"]), "Diagnostic sample outside partition")
    expected_interpretation = ("development_distribution_diagnostic_not_heldout" if inputs["purpose"] == "development"
                               else "training_distribution_fit_diagnostic")
    require(inputs["interpretation"] == expected_interpretation, "Input partition interpretation differs")
    require(type(inputs["samples_per_task"]) is int and inputs["samples_per_task"] > 0, "Invalid per-task sample count")
    tasks = {s["task"] for s in inputs["samples"]}
    require(tasks and all(sum(s["task"] == task for s in inputs["samples"]) == inputs["samples_per_task"]
                          for task in tasks), "Frozen task sample counts differ")
    require(len({s["path"] for s in inputs["samples"]}) == len(inputs["samples"]), "Repeated diagnostic sample")
    require([s["seed"] for s in inputs["samples"]] == list(range(inputs["seed"], inputs["seed"] + len(inputs["samples"]))),
            "Frozen diagnostic seed sequence differs")
    require(manifest["inputs_sha256"] == identity(folder / "inputs.json")["sha256"], "Archived input manifest identity differs")
    evidence = read(folder / "final_evidence.json")
    require(evidence.get("status") == "complete" and evidence["source"]["final_manifest"]["sha256"] == final_sha,
            "Completed final evidence binding differs")
    return final, selection, inputs


def verify(folder, final_manifest=None):
    """Verify a moved archive and recompute all four comparisons on CPU only."""
    cpu_only()
    folder = Path(folder).resolve(strict=True)
    manifest = read(folder / "manifest.json")
    require(manifest.get("format") == "gr00t_action_publication_v1" and manifest.get("status") == "complete",
            "Incomplete action publication archive")
    listed = manifest.get("files", {})
    require(isinstance(listed, dict) and listed, "Missing archive file inventory")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(actual == set(listed) | {"manifest.json"}, "Archive file inventory differs")
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Symlinks are not portable evidence")
    require(not any(p.endswith((".pt", ".safetensors")) for p in actual), "Model/raw .pt files are not publication evidence")
    for relative, recorded in listed.items():
        require(identity(_local(folder, relative)) == recorded, "Archive file identity differs: " + relative)
    final, selection, inputs = _final_binding(folder, manifest, final_manifest)
    final_sha, inputs_sha = manifest["final_manifest_sha256"], manifest["inputs_sha256"]
    sources = manifest["sources"]
    require(set(manifest["arms"]) == set(ARMS), "Exactly five diagnostic arms are required")
    payloads = {}
    for arm in ARMS:
        receipt = read(folder / "runs" / arm / "manifest.json")
        payload = decode(read(folder / "runs" / arm / "outputs.json"))
        _validate_payload(payload, receipt, arm, inputs, inputs_sha, final_sha)
        _model_receipt(receipt, _selected(final, selection, arm))
        _completion_log(folder / "runs" / arm / "collect.log", arm, receipt["sample_count"])
        raw = manifest["arms"][arm]["raw_actions"]
        require(type(raw["bytes"]) is int and raw["bytes"] > 0 and raw["sha256"] == receipt["actions_sha256"],
                "Raw .pt provenance differs")
        for origin, sha in payload["source_files"].items():
            require(origin in sources and sources[origin]["sha256"] == sha
                    and identity(_local(folder, sources[origin]["path"]))["sha256"] == sha,
                    "Recorded inference source missing/changed: " + origin)
        payloads[arm] = payload
    replay_root = folder / "sources/recompute"
    for role, relative in REPLAY.items():
        recorded = manifest["recompute_dependencies"][role]
        require(recorded["path"] == "sources/recompute/" + relative
                and identity(_local(folder, recorded["path"])) == {k: recorded[k] for k in ("bytes", "sha256")},
                "CPU replay dependency differs")
    producer_sha = manifest["recompute_dependencies"]["producer"]["sha256"]
    require(inputs["implementation_sha256"] == producer_sha, "Input producer source differs")
    for arm in ARMS:
        require(producer_sha in payloads[arm]["source_files"].values(), "Inference producer not in source receipt")
    with comparator(replay_root) as module:
        comparisons = {arm: module.compare(payloads["bf16"], payloads[arm]) for arm in ARMS[1:]}
    summary = read(folder / "summary.json")
    require(summary == {"format": "gr00t_action_comparisons_v1", "final_manifest_sha256": final_sha,
                        "inputs_sha256": inputs_sha, "comparisons": comparisons},
            "Published action comparisons disagree with tensor replay")
    steps = {record["integration_steps"] for payload in payloads.values() for record in payload["records"]}
    require(len(steps) == 1, "Integration step count differs across diagnostic observations")
    return {"comparisons": comparisons, "manifest": manifest,
            "manifest_sha256": identity(folder / "manifest.json")["sha256"],
            "files": [folder / relative for relative in sorted(actual)],
            "action_spec": payloads["bf16"]["action_spec"], "integration_steps": next(iter(steps)),
            "sample_count": len(inputs["samples"]), "task_count": len({s["task"] for s in inputs["samples"]})}


def collect(source_root, final_manifest, out):
    """Validate live originals, then atomically create one new portable archive."""
    cpu_only()
    source_root, final_path, out = Path(source_root).resolve(), Path(final_manifest).resolve(), Path(out).absolute()
    require(not out.exists() and not out.is_symlink(), "Refusing to overwrite action publication archive")
    require(final_path.name == "final_manifest.json", "Expected original final_manifest.json")
    # This checks all five completed heldout arms, exact frozen protocol, and
    # final comparison identities before any output directory is constructed.
    sys.path.insert(0, str(ROOT))
    from paper.extract_final_evidence import extract
    completed = extract(final_path.parent)
    final, inputs = read(final_path), read(source_root / "inputs.json")
    require(completed["source"]["final_manifest"]["sha256"] == identity(final_path)["sha256"], "Final changed during extraction")
    require(inputs["protocol_sha256"] == final["protocol_sha256"] == PROTOCOL_SHA256, "Diagnostic final protocol differs")
    initial_final = identity(final_path)
    initial_inputs = identity(source_root / "inputs.json")
    payloads, receipts, raw_records, log_paths = {}, {}, {}, {}
    with comparator(ROOT) as module:
        require(inputs["implementation_sha256"] == identity(ROOT / REPLAY["producer"])["sha256"], "Frozen input producer differs")
        require(inputs == module.freeze_inputs(inputs["capture_root"], inputs["protocol_file"],
                                               inputs["samples_per_task"], inputs["seed"]), "Frozen input sources differ")
        for arm in ARMS:
            folder = source_root / arm
            raw_records[arm] = {"source_path": str(folder / "actions.pt"), **identity(folder / "actions.pt")}
            payload = module.load_outputs(folder)
            receipt = read(folder / "manifest.json")
            _validate_payload(payload, receipt, arm, inputs, initial_inputs["sha256"], initial_final["sha256"])
            checkpoint, files = module.bind_final_checkpoint(final_path, arm, inputs)
            require(receipt["checkpoint"] == str(checkpoint) and receipt["checkpoint_files"] == files,
                    "Actual checkpoint bytes differ from diagnostic receipt")
            for origin, sha in payload["source_files"].items():
                require(identity(origin)["sha256"] == sha, "Inference source changed: " + origin)
            logs = [p for p in (folder / "collect.log", source_root / (arm + ".log")) if p.is_file()]
            require(len(logs) == 1, f"Expected exactly one original collect log for {arm}")
            _completion_log(logs[0], arm, receipt["sample_count"])
            payloads[arm], receipts[arm], log_paths[arm] = payload, receipt, logs[0]
        comparisons = {arm: module.compare(payloads["bf16"], payloads[arm]) for arm in ARMS[1:]}
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + out.name + ".incomplete-", dir=out.parent))
    try:
        _copy(final_path, stage, "final_manifest.json")
        _copy(final["selection_file"], stage, "selection.json")
        _copy(final["protocol_file"], stage, "protocol.json")
        _copy(source_root / "inputs.json", stage, "inputs.json")
        write(stage / "final_evidence.json", completed)
        sources = {}
        for origin, sha in payloads["bf16"]["source_files"].items():
            relative = "sources/recorded/" + sha + "/" + Path(origin).name
            if (stage / relative).exists():
                require(identity(stage / relative) == identity(origin), "Colliding source identities differ")
                sources[origin] = {"path": relative, **identity(origin)}
            else:
                sources[origin] = _copy(origin, stage, relative)
        dependencies = {role: _copy(ROOT / relative, stage, "sources/recompute/" + relative)
                        for role, relative in REPLAY.items()}
        _copy(__file__, stage, "sources/recompute/paper/collect_action_diagnostics.py")
        for arm in ARMS:
            _copy(source_root / arm / "manifest.json", stage, "runs/" + arm + "/manifest.json")
            _copy(log_paths[arm], stage, "runs/" + arm + "/collect.log")
            encoded = encode(payloads[arm])
            # A byte-level check inside decode supplements exact metric replay.
            decode(encoded)
            write(stage / "runs" / arm / "outputs.json", encoded)
            require(identity(source_root / arm / "actions.pt") ==
                    {k: raw_records[arm][k] for k in ("bytes", "sha256")}, "Raw actions changed during export")
        write(stage / "summary.json", {"format": "gr00t_action_comparisons_v1",
              "final_manifest_sha256": initial_final["sha256"], "inputs_sha256": initial_inputs["sha256"],
              "comparisons": comparisons})
        manifest = {"format": "gr00t_action_publication_v1", "status": "complete",
                    "final_manifest_sha256": initial_final["sha256"], "inputs_sha256": initial_inputs["sha256"],
                    "protocol_sha256": PROTOCOL_SHA256, "sources": sources, "recompute_dependencies": dependencies,
                    "arms": {arm: {"raw_actions": raw_records[arm]} for arm in ARMS},
                    "scope": "Exact finite-tensor metric replay; camera inputs and model weights are identified by hash only. Raw .pt bytes/hash are recorded, not copied.",
                    "replay_dependency_scope": "CPU helper sources are snapshotted at export; recorded inference source hashes retain their original collection identities.",
                    "files": {p.relative_to(stage).as_posix(): identity(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
        write(stage / "manifest.json", manifest)
        verify(stage, final_path)
        require(identity(final_path) == initial_final and identity(source_root / "inputs.json") == initial_inputs,
                "Final/input manifest changed during export")
        require(not out.exists() and not out.is_symlink(), "Refusing to overwrite action publication archive")
        _publish_new(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify(out, final_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("collect")
    export.add_argument("--source-root", required=True)
    export.add_argument("--final-manifest", required=True)
    export.add_argument("--out", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--folder", required=True)
    check.add_argument("--final-manifest")
    args = parser.parse_args()
    result = (collect(args.source_root, args.final_manifest, args.out) if args.command == "collect"
              else verify(args.folder, args.final_manifest))
    print(json.dumps({"status": "verified", "final_manifest_sha256": result["manifest"]["final_manifest_sha256"],
                      "comparisons": result["comparisons"], "files": len(result["files"])}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
