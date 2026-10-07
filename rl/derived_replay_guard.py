"""CPU provenance guard for training from derived teacher-capture views."""
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import sys


def audit_report_sha256(report):
    """Use the same deterministic serialization as the orchestration request."""
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()


@dataclass
class DerivedReplayGuard:
    report: dict
    source_hashes: dict

    @property
    def paths(self):
        return sorted(self.source_hashes)

    def load_sample(self, path):
        """Deserialize only the exact bytes verified against the audit."""
        import torch
        path = Path(path).resolve()
        if path not in self.source_hashes:
            raise ValueError(f"Captured sample is absent from derived audit: {path}")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != self.source_hashes[path]:
            raise ValueError(f"Captured sample changed after derived audit: {path}")
        return torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)


def prepare_derived_replay(root, protocol_file, env=None):
    """Return a strict view guard, or None for an unchanged legacy dataset."""
    root = Path(root).expanduser().resolve()
    protocol_file = Path(protocol_file).expanduser().resolve()
    protocol = json.loads(protocol_file.read_text())
    env = os.environ if env is None else env
    if (root / "views_manifest.json").exists():
        raise ValueError("Select one derived view, not the paired view root")
    mode_root = root.parent if root.name == "observations" else root
    marker = mode_root.parent / "views_manifest.json"
    if not marker.exists():
        if "capture_views" in protocol:
            raise ValueError("Protocol declares capture_views but derived view marker is missing")
        return None
    manifest = json.loads(marker.read_text())
    if manifest.get("schema") != "fp4vla_paired_capture_views_v1":
        raise ValueError("Unknown paired capture view schema")
    if mode_root.name not in ("head", "stratified"):
        raise ValueError("Expected a head or stratified derived capture view")
    teacher, expected_sha = (env.get(key) for key in (
        "QAD_CAPTURE_TEACHER", "QAD_CAPTURE_AUDIT_SHA256"))
    if not teacher or not expected_sha:
        raise ValueError("Derived replay requires QAD_CAPTURE_TEACHER and QAD_CAPTURE_AUDIT_SHA256")
    project = str(Path(__file__).resolve().parents[1])
    if project not in sys.path:
        sys.path.insert(0, project)
    from exp.derive_capture_views import audit_training_view
    report = audit_training_view(root, protocol_file, Path(teacher).expanduser().resolve(),
                                 minimum_episodes=2)
    if report.get("format") != "derived_teacher_replay_audit_v1" or report.get("status") != "verified":
        raise ValueError("Derived replay audit did not verify the expected format")
    if audit_report_sha256(report) != expected_sha:
        raise ValueError("Derived replay audit SHA differs from the orchestration request")
    observations = mode_root / "observations"
    reported = Path(report.get("observations", ""))
    if not reported.is_absolute() or reported.resolve() != observations:
        raise ValueError("Derived replay audit observations differ from the selected view")
    sources = {}
    for task, task_report in report.get("tasks", {}).items():
        for record in task_report.get("source_files", []):
            relative = Path(record["path"])
            path = (observations / relative).resolve()
            digest = record.get("sha256")
            if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                    or relative.parts[0] != task or observations not in path.parents
                    or not path.name.startswith("sample_") or path.suffix != ".pt"):
                raise ValueError("Derived replay audit has an invalid source path")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Derived replay audit has an invalid source SHA")
            if path in sources:
                raise ValueError("Derived replay audit repeats a source sample")
            sources[path] = digest
    actual = {path.resolve() for path in root.rglob("sample_*.pt")}
    if not sources or actual != set(sources) or report.get("sample_count") != len(sources):
        raise ValueError("Derived replay sample inventory differs from the audit")
    return DerivedReplayGuard(report, sources)
