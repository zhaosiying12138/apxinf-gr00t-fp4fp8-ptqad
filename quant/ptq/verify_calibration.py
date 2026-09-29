"""Read-only CPU acceptance checks for a completed full-model calibration cache.

The cache is memory mapped; numerical checks allocate at most a row chunk of
one Hessian at a time. No model is loaded, CUDA is hidden, and inputs are never
modified. The default metadata file is calib_meta.json beside calib.pt.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

EXPECTED_ARCHITECTURE = {"language_layers": 16, "dit_layers": 32, "vl_layers": 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def checkpoint_shapes(base, records):
    """Read only safetensors JSON headers; independently verify source bytes."""
    shapes = {}
    for relative, record in records.items():
        path = (base / relative).resolve()
        require(path.is_relative_to(base), f"Source shard escapes base: {relative}")
        require(path.stat().st_size == record["bytes"], f"Source shard size differs: {relative}")
        require(file_sha256(path) == record["sha256"], f"Source shard SHA256 differs: {relative}")
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            require(len(length_bytes) == 8, f"Missing safetensors header: {relative}")
            length = struct.unpack("<Q", length_bytes)[0]
            require(0 < length <= min(path.stat().st_size - 8, 64 * 1024 * 1024),
                    f"Invalid safetensors header length: {relative}")
            header = json.loads(stream.read(length))
        for name, item in header.items():
            if name == "__metadata__":
                continue
            require(name not in shapes, f"Duplicate checkpoint key: {name}")
            shapes[name] = item["shape"]
    return shapes


def check_architecture(meta, config, shapes):
    require(meta.get("architecture") == EXPECTED_ARCHITECTURE,
            f"Expected full architecture {EXPECTED_ARCHITECTURE}; got {meta.get('architecture')}")
    config_counts = {"language_layers": config.get("select_layer"),
                     "dit_layers": config.get("diffusion_model_cfg", {}).get("num_layers"),
                     "vl_layers": config.get("vl_self_attention_cfg", {}).get("num_layers")}
    require(config_counts == EXPECTED_ARCHITECTURE, f"Source config architecture differs: {config_counts}")
    patterns = {"language_layers": r"backbone\.model\.model\.language_model\.layers\.(\d+)\.",
                "dit_layers": r"action_head\.model\.transformer_blocks\.(\d+)\.",
                "vl_layers": r"action_head\.vl_self_attention\.transformer_blocks\.(\d+)\."}
    for group, pattern in patterns.items():
        indices = {int(match.group(1)) for key in shapes if (match := re.match(pattern, key))}
        require(indices == set(range(EXPECTED_ARCHITECTURE[group])),
                f"Checkpoint block coverage differs for {group}: {sorted(indices)}")


def check_matrix(name, entry, shape, row_count, calls, chunk_rows, atol, rtol):
    hessian, absolute = entry["H"], entry["abs"]
    require(len(shape) == 2, f"Not a matrix weight: {name}: {shape}")
    width = shape[1]
    require(isinstance(hessian, torch.Tensor) and hessian.device.type == "cpu" and
            hessian.dtype == torch.float32 and tuple(hessian.shape) == (width, width),
            f"H dtype/device/shape differs from weight input dimension: {name}")
    require(isinstance(absolute, torch.Tensor) and absolute.device.type == "cpu" and
            absolute.dtype == torch.float32 and tuple(absolute.shape) == (width,),
            f"Absolute-activation statistic differs from weight input dimension: {name}")
    require(isinstance(entry["n"], int) and entry["n"] == row_count and row_count > 0,
            f"Activation row count differs: {name}")
    require(isinstance(entry["calls"], int) and entry["calls"] == calls and calls > 0,
            f"Forward call count differs: {name}")
    require(bool(torch.isfinite(absolute).all()) and bool((absolute >= 0).all()),
            f"Invalid absolute-activation statistic: {name}")
    max_asymmetry, max_value = 0.0, 0.0
    for start in range(0, width, chunk_rows):
        stop = min(start + chunk_rows, width)
        rows = hessian[start:stop, :]
        require(bool(torch.isfinite(rows).all()), f"Nonfinite Hessian at {name} rows {start}:{stop}")
        transpose_rows = hessian[:, start:stop].T
        difference = (rows - transpose_rows).abs()
        require(bool((difference <= atol + rtol * transpose_rows.abs()).all()),
                f"Asymmetric Hessian at {name} rows {start}:{stop}")
        max_asymmetry = max(max_asymmetry, float(difference.max()))
        max_value = max(max_value, float(rows.abs().max()))
    require(bool((hessian.diagonal() >= 0).all()), f"Negative Gram diagonal: {name}")
    return {"input_width": width, "h_bytes": hessian.numel() * hessian.element_size(),
            "rows": row_count, "calls": calls, "max_abs_asymmetry": max_asymmetry,
            "max_abs_entry": max_value}


def verify(args):
    start = time.monotonic()
    require(not torch.cuda.is_initialized(), "CUDA was initialized before CPU-only verification")
    torch.set_num_threads(args.cpu_threads)
    cache = Path(args.calib).resolve()
    if cache.is_dir():
        cache = cache / "calib.pt"
    metadata_path = Path(args.meta).resolve() if args.meta else cache.with_name("calib_meta.json")
    metadata = json.loads(metadata_path.read_text())
    require(metadata.get("status") == "complete", "Calibration metadata is not complete")
    require(metadata.get("version") == "full-model-cpu-hessian-v2", "Unsupported calibration format")
    require(metadata.get("windows_requested") == args.expected_windows and
            metadata.get("windows_consumed") == args.expected_windows,
            f"Expected exactly {args.expected_windows} requested/consumed windows")
    require(metadata.get("batch", 0) > 0 and metadata.get("forwards") ==
            math.ceil(args.expected_windows / metadata["batch"]), "Window/batch/forward counts disagree")
    require(metadata.get("accumulation_dtype") == "float32" and
            metadata.get("accumulation_device") == "cpu" and
            metadata.get("model_dtype") == "bfloat16" and metadata.get("tf32_matmul") is False,
            "Calibration precision contract differs")
    print(f"[calib-verify] CPU-only read-only; mmap; chunk_rows={args.chunk_rows}", flush=True)
    print(f"[calib-verify] cache={cache}", flush=True)
    cache_hash = file_sha256(cache)
    require(cache_hash == metadata.get("cache_sha256"), "Calibration cache SHA256 differs from metadata")
    print(f"[calib-verify] cache SHA256 verified: {cache_hash}", flush=True)
    base = Path(metadata["base"]).resolve()
    require(file_sha256(base / "config.json") == metadata["base_config_sha256"], "Base config SHA256 differs")
    if metadata.get("base_statistics_sha256") is not None:
        require(file_sha256(base / "statistics.json") == metadata["base_statistics_sha256"],
                "Base statistics SHA256 differs")
    records = metadata.get("base_weight_files", {})
    require(bool(records), "Missing source checkpoint shard records")
    shapes = checkpoint_shapes(base, records)
    check_architecture(metadata, json.loads((base / "config.json").read_text()), shapes)
    print(f"[calib-verify] source shard hashes, config, statistics verified; architecture=16/32/4", flush=True)
    entries = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    require(isinstance(entries, dict) and len(entries) == metadata["n_layers"] and len(entries) > 0,
            "Calibration layer count differs")
    require(set(entries) == set(metadata["rows_per_layer"]) == set(metadata["calls_per_layer"]),
            "Calibration layer keys differ from metadata")
    if metadata.get("recipe_targets") == "calib":
        aliases = set(metadata.get("tied_weight_aliases", {}))
        nonlinear = set(metadata.get("nonlinear_targets", {}))
        eligible = {key.removesuffix(".weight") for key, shape in shapes.items()
                    if key.endswith(".weight") and len(shape) == 2 and shape[1] % 16 == 0
                    and key not in aliases and key.removesuffix(".weight") not in nonlinear}
        require(set(entries) == eligible, "Full-calibration H coverage differs from eligible source matrices")
    checked = {}
    for i, (name, entry) in enumerate(entries.items(), 1):
        require(name + ".weight" in shapes, f"Calibration key missing from source checkpoint: {name}")
        checked[name] = check_matrix(name, entry, shapes[name + ".weight"],
                                     metadata["rows_per_layer"][name], metadata["calls_per_layer"][name],
                                     args.chunk_rows, args.atol, args.rtol)
        if i % 64 == 0 or i == len(entries):
            print(f"[calib-verify] numerical checks {i}/{len(entries)} layers", flush=True)
    total_bytes = sum(item["h_bytes"] for item in checked.values())
    require(math.isclose(total_bytes / 1e9, metadata["H_GB_f32"], rel_tol=1e-12, abs_tol=1e-12),
            "Hessian byte total differs from metadata")
    require(not torch.cuda.is_initialized(), "CPU verification initialized CUDA")
    max_asymmetry = max(item["max_abs_asymmetry"] for item in checked.values())
    report = {"status": "passed", "scope": "Calibration artifact acceptance; not a replay of collection",
              "cache": str(cache), "cache_sha256": cache_hash, "metadata_sha256": file_sha256(metadata_path),
              "source_shards_verified": len(records), "architecture": EXPECTED_ARCHITECTURE,
              "windows": args.expected_windows, "batch": metadata["batch"], "forwards": metadata["forwards"],
              "layers": len(checked), "H_bytes": total_bytes, "finite": True,
              "max_abs_asymmetry": max_asymmetry, "symmetry_atol": args.atol, "symmetry_rtol": args.rtol,
              "cuda_initialized": torch.cuda.is_initialized(), "elapsed_seconds": time.monotonic() - start,
              "layer_checks": checked}
    print(f"[calib-verify] PASS full_model=16/32/4 windows={args.expected_windows} "
          f"layers={len(checked)} H_GB={total_bytes/1e9:.9f}", flush=True)
    print(f"[calib-verify] shapes/counts/finite/diagonal/symmetry=PASS "
          f"max_abs_asymmetry={max_asymmetry:.9g} CUDA_initialized=False "
          f"elapsed={report['elapsed_seconds']:.2f}s", flush=True)
    if args.report:
        report_path = Path(args.report).resolve()
        require(report_path not in (cache, metadata_path) and not report_path.is_relative_to(base),
                "Report must not overwrite calibration or base checkpoint inputs")
        require(not report_path.exists(), f"Refusing to overwrite report: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[calib-verify] report={report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", required=True, help="calib.pt or its containing directory")
    parser.add_argument("--meta", help="Optional metadata path; default adjacent calib_meta.json")
    parser.add_argument("--expected-windows", type=int, default=128)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--report", help="Optional new JSON report (never overwrites inputs)")
    args = parser.parse_args()
    if (min(args.expected_windows, args.chunk_rows, args.cpu_threads) < 1 or
            not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol))):
        parser.error("Counts must be positive and tolerances nonnegative")
    verify(args)


if __name__ == "__main__":
    main()
