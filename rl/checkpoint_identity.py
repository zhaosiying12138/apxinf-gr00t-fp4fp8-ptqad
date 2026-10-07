"""Standard-library identities for checkpoint files actually used by deployment.

Adapter exports reference a frozen base and a training checkpoint. Their files
belong to the identity alongside the export itself; hashing uses bounded reads.
"""
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(checkpoint):
    """Hash all loaded base/adapter weights, including referenced deployment files."""
    checkpoint = Path(checkpoint).resolve()
    roots = {checkpoint}
    merge = checkpoint / "merge_manifest.json"
    if merge.is_file():
        info = read(merge)
        require(info.get("status") == "complete", "Incomplete deployment manifest")
        roots.update(Path(info[k]).resolve() for k in ("base", "training_checkpoint"))
    records = {}
    for root in sorted(roots):
        paths = sorted(root.glob("*.safetensors"))
        require(paths, f"No weight shards: {root}")
        paths += sorted(root.glob("*.json"))
        for path in paths:
            records[str(path)] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return records
