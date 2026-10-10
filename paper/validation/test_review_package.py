"""Regressions for review-only packaging; no models or final gates are run."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package_review import MANIFEST, PENDING, check_receipt_hashes, verify_archive

class ReviewPackageTests(unittest.TestCase):
    def make_archive(self, root, *, altered=False, hidden_pending=False):
        good = b"complete science; screenshots pending"
        data = b"tampered" if altered else good
        entries = {"README.md": data}
        if hidden_pending:
            entries["paper/figs/renamed.png"] = b"old pending screenshot bytes"
        records = {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                   for name, payload in entries.items()}
        if altered:
            records["README.md"] = {"bytes": len(good), "sha256": hashlib.sha256(good).hexdigest()}
        manifest = {"not_final_publication": True, "pending_screenshots": sorted(PENDING),
                    "files": records,
                    "omitted_capture_sha256": [hashlib.sha256(b"old pending screenshot bytes").hexdigest()]}
        archive = root / "test.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name, payload in entries.items():
                z.writestr(name, payload)
            z.writestr(MANIFEST, json.dumps(manifest))
        return archive

    def test_integrity_and_pending_image_alias_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(verify_archive(self.make_archive(root))["not_final_publication"])
            with self.assertRaisesRegex(RuntimeError, "integrity failure"):
                verify_archive(self.make_archive(root, altered=True))
            with self.assertRaisesRegex(RuntimeError, "Pending capture bytes"):
                verify_archive(self.make_archive(root, hidden_pending=True))

    def test_passed_browser_receipt_requires_current_meta_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.json"
            path.write_text("review version 1")
            receipt = {"pass": True, "meta_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            check_receipt_hashes(receipt, {"meta_sha256": path})
            path.write_text("review version 2")
            with self.assertRaisesRegex(RuntimeError, "Stale review receipt"):
                check_receipt_hashes(receipt, {"meta_sha256": path})

if __name__ == "__main__":
    unittest.main()
