"""Dispatch derived views to their full provenance audit, never the legacy path."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from exp.verify_teacher_replay import audit_replay


class TeacherReplayViewDispatchTests(unittest.TestCase):
    def fixture(self, root, schema="fp4vla_paired_capture_views_v1"):
        paired = root / "paired"
        paired.mkdir()
        (paired / "views_manifest.json").write_text(json.dumps({"schema": schema}))
        for mode in ("head", "stratified"):
            (paired / mode / "observations").mkdir(parents=True)
        module = ModuleType("exp.derive_capture_views")
        module.audit_training_view = Mock(return_value={
            "format": "derived_teacher_replay_audit_v1", "status": "verified",
            "task_count": 10, "sample_count": 148, "source_ancestry": {"verified": True}})
        return paired, module

    def test_each_view_root_and_observations_dispatch_without_legacy_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired, module = self.fixture(root)
            for mode in ("head", "stratified"):
                for child in (Path("."), Path("observations")):
                    with self.subTest(mode=mode, child=str(child)):
                        selected = paired / mode / child
                        module.audit_training_view.reset_mock()
                        with patch.dict(sys.modules, {module.__name__: module}):
                            report = audit_replay(selected, root / "protocol.json", root / "teacher", 3)
                        module.audit_training_view.assert_called_once_with(
                            selected.resolve(), (root / "protocol.json").resolve(),
                            (root / "teacher").resolve(), 3)
                        self.assertIs(report, module.audit_training_view.return_value)

    def test_paired_root_cannot_concatenate_both_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired, module = self.fixture(root)
            with patch.dict(sys.modules, {module.__name__: module}):
                with self.assertRaisesRegex(ValueError, "not the paired view root"):
                    audit_replay(paired, root / "protocol.json", root / "teacher")
            module.audit_training_view.assert_not_called()

    def test_unknown_view_schema_fails_without_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired, module = self.fixture(root, schema="unrecognized")
            with patch.dict(sys.modules, {module.__name__: module}):
                with self.assertRaisesRegex(ValueError, "Unknown paired capture view schema"):
                    audit_replay(paired / "head", root / "protocol.json", root / "teacher")
            module.audit_training_view.assert_not_called()

    def test_raw_candidates_are_rejected_even_alongside_sample_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observation = root / "raw/observations/task"
            observation.mkdir(parents=True)
            (observation / "candidate_000000.pt").write_bytes(b"raw candidate")
            (observation / "sample_000000.pt").write_bytes(b"must not bypass source audit")
            for selected in (root / "raw", root / "raw/observations"):
                with self.subTest(root=str(selected)), self.assertRaisesRegex(ValueError, "Raw candidate"):
                    audit_replay(selected, root / "protocol.json", root / "teacher")

    def test_derived_auditor_rejection_is_never_downgraded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired, module = self.fixture(root)
            module.audit_training_view.side_effect = ValueError("Collection views are not QAD teacher supervision")
            with patch.dict(sys.modules, {module.__name__: module}):
                with self.assertRaisesRegex(ValueError, "Collection views"):
                    audit_replay(paired / "stratified", root / "protocol.json", root / "teacher")


if __name__ == "__main__":
    unittest.main()
