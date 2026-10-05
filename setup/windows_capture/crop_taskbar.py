#!/usr/bin/env python3
"""Crop only the bottom taskbar and record a verifiable before/after link.

The input PNG is replaced in place. A new <stem>.crop.json binds the original
captured bytes to the final PNG; an existing crop record is never overwritten.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from PIL import Image
import numpy as np


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: crop_taskbar.py <captured.png>")
    src = Path(sys.argv[1]).resolve(strict=True)
    record_path = src.with_suffix(".crop.json")
    if record_path.exists():
        raise FileExistsError(f"Refusing existing crop record: {record_path}")
    source_bytes = src.read_bytes()
    source_hash = sha256(source_bytes)
    with Image.open(io.BytesIO(source_bytes)) as original:
        original.load()
        width, height = original.size
        # Use RGB only for detection; crop/save preserves the original mode.
        pixels = np.asarray(original.convert("RGB"))
        row_mean = pixels.mean(axis=(1, 2))
        cut = height
        for y in range(height - 1, int(height * .7), -1):
            if row_mean[y] < 100:
                cut = y + 1
                break
        cropped = original.crop((0, 0, width, cut))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=src.parent, suffix=".png", delete=False) as output:
                temporary = Path(output.name)
            cropped.save(temporary, format="PNG")
            expected_output_hash = sha256(temporary.read_bytes())
            if sha256(src.read_bytes()) != source_hash:
                raise RuntimeError("Screenshot changed during taskbar crop")
            os.replace(temporary, src)
            temporary = None
            # Hash the final persisted file, not only an encoded image buffer.
            output_hash = sha256(src.read_bytes())
            if output_hash != expected_output_hash:
                raise RuntimeError("Persisted cropped screenshot hash mismatch")
            with Image.open(src) as persisted:
                output_dimensions = list(persisted.size)
            record = {
                "operation": "taskbar_crop_only",
                "source_sha256": source_hash,
                "output_sha256": output_hash,
                "source_dimensions": [width, height],
                "output_dimensions": output_dimensions,
                "crop_box": [0, 0, width, cut],
            }
            # A failure here is an error, never an accepted image without provenance.
            with record_path.open("x", encoding="utf-8") as output:
                json.dump(record, output, indent=2)
                output.write("\n")
        finally:
            cropped.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"cropped {src}: {width}x{height} -> {width}x{cut}; record={record_path}")


if __name__ == "__main__":
    main()
