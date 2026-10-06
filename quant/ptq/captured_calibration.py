"""Audited teacher-rollout inputs for the existing PTQ Hessian collectors.

Freezing and validation use the existing successful-teacher replay auditor on
CPU. This module does not implement quantization or Hessian accumulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
from exp.verify_teacher_replay import audit_replay
from probe_distill import file_sha256, tensor_tree

FORMAT = "ptq_successful_teacher_capture_v1"
METADATA_FILES = ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze_capture(root, protocol_file, teacher):
    protocol_file, teacher = Path(protocol_file).resolve(), Path(teacher).resolve()
    audit = audit_replay(root, protocol_file, teacher)
    record = {"format": FORMAT, "purpose": "PTQ calibration from training-only successful teacher states",
            "protocol_file": str(protocol_file), "source_audit": audit,
            "teacher_metadata": {name: file_sha256(teacher / name) for name in METADATA_FILES},
            "order": "task order in teacher audit, then filename; consume every observation exactly once",
            "calibration_forward": "training velocity forward with captured teacher action endpoint",
            "implementation_sha256": file_sha256(__file__)}
    # The audit includes integer episode keys. JSON stores object keys as
    # strings, so normalize both the frozen and recomputed records identically.
    return json.loads(json.dumps(record))


class CapturedCalibration:
    """Do not re-collate flattened VLM image patches or silently repeat windows."""
    def __init__(self, manifest, checkpoint, windows, batch):
        self.path = Path(manifest).resolve()
        self.data = json.loads(self.path.read_text())
        require(self.data.get("format") == FORMAT, "Unknown captured calibration format")
        audit = self.data["source_audit"]
        require(self.data == freeze_capture(audit["evaluation"], self.data["protocol_file"], audit["teacher"]),
                "Captured calibration no longer matches its source audit")
        require(type(batch) is int and batch == 1, "Captured calibration requires batch=1")
        require(type(windows) is int and windows == audit["sample_count"],
                f"Captured calibration requires all {audit['sample_count']} windows exactly once")
        checkpoint = Path(checkpoint).resolve()
        teacher = Path(audit["teacher"])
        for name in METADATA_FILES:
            require(file_sha256(checkpoint / name) == file_sha256(teacher / name),
                    "Captured teacher and calibration model metadata differ: " + name)
        self.paths = [Path(audit["observations"]) / row["path"]
                      for task in audit["tasks"].values() for row in task["source_files"]]
        self.hashes = [row["sha256"] for task in audit["tasks"].values() for row in task["source_files"]]
        require(len(self.paths) == windows and len(set(self.paths)) == windows,
                "Captured calibration has duplicate or missing windows")
        self.record = {"path": str(self.path), "bytes": self.path.stat().st_size,
                       "sha256": file_sha256(self.path), "source_audit": audit,
                       "protocol_file": self.data["protocol_file"], "order": self.data["order"],
                       "calibration_forward": self.data["calibration_forward"],
                       "implementation_sha256": file_sha256(__file__)}

    def validate_root_weights(self, weights):
        """Ordinary collector passes BF16 weights; category passes root_bf16."""
        expected = self.data["source_audit"]["teacher_weights"]
        require(set(weights) == set(expected), "Calibration root shard set differs from teacher")
        for name, record in expected.items():
            require(all(weights[name].get(field) == record[field] for field in ("bytes", "sha256")),
                    "Calibration root weights differ from captured teacher: " + name)

    def __iter__(self):
        import torch
        for path, sha256 in zip(self.paths, self.hashes):
            require(file_sha256(path) == sha256, "Captured input changed after validation: " + str(path))
            raw = torch.load(path, map_location="cpu", weights_only=True)
            # Audit already checked all tensors, masks, reset identities and
            # byte hashes. Each item retains its original single-example axes.
            yield tensor_tree(raw["inputs"])


def load_captured_model(checkpoint, model_config):
    """Exact full-model loader already used by the teacher-labeling workflow."""
    from opd_probe_cache import use_local_hf_metadata
    use_local_hf_metadata()
    from transformers import AutoModel
    model, info = AutoModel.from_pretrained(
        checkpoint, config=model_config, local_files_only=True,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
        output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Captured calibration model loading was not exact: {info}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Verified teacher evaluation or observations directory")
    parser.add_argument("--protocol-file", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite calibration manifest: {out}")
    data = freeze_capture(args.root, args.protocol_file, args.teacher)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"manifest": str(out.resolve()), "samples": data["source_audit"]["sample_count"],
                      "tasks": data["source_audit"]["task_count"], "status": "CPU_verified_no_calibration_run"}))


if __name__ == "__main__":
    main()
