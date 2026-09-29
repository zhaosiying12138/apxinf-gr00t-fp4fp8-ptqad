"""Prepare a new native pi0.5 packed artifact and model overlay, using CPU only.

Run the .sh entry point to enforce invisible CUDA devices and two CPU threads.
Stages are intentionally separate; existing stage outputs are never overwritten.
The checkpoint remains in its original directory. The overlay copies small
metadata files and links only the large safetensors file and the new packed dir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = ["exp/prepare_native_pi05.sh", "exp/prepare_native_pi05.py",
           "quant/nvfp4_convert_packed.py", "quant/nvfp4_convert.py", "quant/fp4_quant.py"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def pack(args) -> None:
    output = args.out / "packed"
    for path in (output, args.out / "pack.log", args.out / "pack_input_manifest.json"):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing stage output: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.base / "model.safetensors"
    source_hashes = {name: digest(PROJECT / name) for name in SOURCES}
    command = [sys.executable, str(PROJECT / "quant/nvfp4_convert_packed.py"),
               "--ckpt", str(checkpoint), "--out", str(output), "--scope", "all",
               "--lang-depth", "18", "--act-depth", "18", "--vision-depth", "27",
               "--chunk-rows", str(args.chunk_rows)]
    frozen = {"schema_version": 1, "status": "input_frozen",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_checkpoint": str(checkpoint), "source": file_record(checkpoint),
              "source_code_sha256": source_hashes, "command": command,
              "cpu_environment": {name: os.environ[name] for name in
                                  ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True))}
    write_new(args.out / "pack_input_manifest.json", frozen)
    print("Frozen source and input hashes; starting CPU-only pack", flush=True)
    with (args.out / "pack.log").open("x") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=os.environ.copy())
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if proc.wait():
            raise RuntimeError(f"packing failed with exit code {proc.returncode}; partial output retained")
    if source_hashes != {name: digest(PROJECT / name) for name in SOURCES}:
        raise RuntimeError("source code changed during packing; artifact is not accepted")
    if frozen["source"] != file_record(checkpoint):
        raise RuntimeError("checkpoint changed during packing; artifact is not accepted")
    m = json.loads((output / "manifest.json").read_text())
    names = set()
    files = {}
    for tensor in m["tensors"]:
        name = tensor["name"]
        if name in names:
            raise ValueError(f"duplicate tensor: {name}")
        names.add(name)
        rows, k = tensor["shape"]
        if tensor["block"] != 16 or tensor["tscale"] != 1.0 or k % 16:
            raise ValueError(f"invalid native NVFP4 contract: {name}")
        for suffix, n in [(".packed.u8", rows*k//2),
                          (".scale.u8", 512*((k//16+3)//4)*((rows+127)//128))]:
            path = output / (name.replace("/", ".") + suffix)
            if path.stat().st_size != n:
                raise ValueError(f"wrong packed size: {path}")
            files[path.name] = file_record(path)
    root = "paligemma_with_expert."
    expected = {f"{root}{branch}.layers.{i}.{projection}.weight"
                for branch, depth, projections in [
                    ("paligemma.model.language_model", 18, ("qkv", "gate_up")),
                    ("gemma_expert.model", 18, ("qkv", "gate_up")),
                    ("paligemma.model.vision_tower.vision_model.encoder", 27, ("qkv",))]
                for i in range(depth) for projection in projections}
    if names != expected or m["summary"]["tensors"] != 99:
        raise ValueError("expected exact 18/18/27 layer set and 99 projection tensors")
    if m["summary"]["packed_bytes"] != sum(f["bytes"] for f in files.values()):
        raise ValueError("packed byte total disagrees with actual files")
    producer = {**frozen, "status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": frozen["source"]["sha256"],
                "input_manifest_sha256": digest(args.out / "pack_input_manifest.json"),
                "manifest_sha256": digest(output / "manifest.json"), "summary": m["summary"],
                "format": "NVFP4 tscale=1; per-16 E4M3 physical swizzle; language norms folded",
                "files": files,
                "storage_note": "packed files are additional native artifacts; engine retains BF16 weights"}
    write_new(output / "producer_manifest.json", producer)
    print(f"Verified 99 tensors; producer_manifest SHA256 {digest(output / 'producer_manifest.json')}")


def model_overlay(args) -> None:
    output, model = args.out / "packed", args.out / "model"
    if model.exists() or model.is_symlink():
        raise FileExistsError(f"refusing existing model overlay: {model}")
    producer = json.loads((output / "producer_manifest.json").read_text())
    checkpoint = args.base / "model.safetensors"
    if producer["status"] != "complete" or producer["source"] != file_record(checkpoint):
        raise ValueError("base checkpoint does not match the verified packed artifact")
    if producer["manifest_sha256"] != digest(output / "manifest.json"):
        raise ValueError("packed manifest changed")
    for name, record in producer["files"].items():
        if file_record(output / name) != record:
            raise ValueError(f"packed tensor changed: {name}")
    for required in ("config.json", "norm_stats.json"):
        if not (args.base / required).is_file():
            raise FileNotFoundError(args.base / required)
    model.mkdir(parents=True, exist_ok=False)
    copied = {}
    for path in sorted(args.base.iterdir()):
        if path.name in (".cache", "fp4") or not path.is_file():
            continue
        if path.name == "model.safetensors":
            (model / path.name).symlink_to(path)
        else:
            shutil.copy2(path, model / path.name)
            copied[path.name] = file_record(model / path.name)
    (model / "fp4").symlink_to(output, target_is_directory=True)
    record = {"schema_version": 1, "status": "complete", "model_dir": str(model),
              "source_checkpoint": str(checkpoint), "source_sha256": producer["source_sha256"],
              "packed_dir": str(output), "producer_manifest_sha256": digest(output / "producer_manifest.json"),
              "copied_metadata": copied,
              "note": "Original base/fp4 link is untouched; AutoPolicy loads this new directory."}
    write_new(args.out / "overlay_manifest.json", record)
    print(f"New model overlay: {model}")
    print(f"overlay_manifest SHA256 {digest(args.out / 'overlay_manifest.json')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["pack", "model-overlay"])
    ap.add_argument("--base", type=Path, default=PROJECT / "weights/pi05_libero_base")
    ap.add_argument("--out", type=Path, required=True, help="new absolute run root")
    ap.add_argument("--chunk-rows", type=int, default=512)
    args = ap.parse_args()
    if not args.out.is_absolute() or args.chunk_rows < 1:
        ap.error("--out must be absolute and --chunk-rows must be positive")
    args.base, args.out = args.base.resolve(strict=True), args.out.resolve()
    if args.out == args.base or args.base in args.out.parents or args.out in args.base.parents:
        ap.error("output root must be separate from the source model directory")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
            os.environ.get(name) != "2" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        ap.error("use prepare_native_pi05.sh to enforce CPU-only, two-thread preparation")
    (pack if args.stage == "pack" else model_overlay)(args)


if __name__ == "__main__":
    main()
