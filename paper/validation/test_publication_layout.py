"""Small archive-layout fixtures; no final science validation or GPU execution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_publication as package
from publication_guard import file_record


class PublicationLayout(unittest.TestCase):
    def fixture(self, root):
        paper = root / "paper"
        paper.mkdir()
        contents = {
            "README.md": "[Article](paper/paper.html) [Guide](docs/guide.md) "
                         "[Compile](docs/evidence/ubuntu-compile/manifest.json) "
                         "[Runs](results/engine/) [Recorded run](results/engine/run.json)",
            "paper/README.md": "[Project](../README.md) [Article](paper.html)",
            "paper/paper.html": "<html>offline article</html>",
            "docs/guide.md": "[Diagram](images/diagram.png) [Details](details.md) "
                             "[External](https://example.org/model.safetensors)",
            "docs/details.md": "[Back](guide.md#scope)",
            "docs/images/diagram.png": "diagram fixture",
            "results/engine/run.json": "{}",
            "results/engine/unlinked-private.tmp": "must not be included",
            "docs/images/ubuntu-compile.png": "real-capture fixture identity",
            "docs/evidence/ubuntu-compile/run.sh": "printf compile\n",
            "docs/evidence/ubuntu-compile/capture.log": "compile exit 0\n",
        }
        for name, text in contents.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        compile_files = ["docs/images/ubuntu-compile.png", "docs/evidence/ubuntu-compile/run.sh",
                         "docs/evidence/ubuntu-compile/capture.log"]
        (root / "docs/evidence/ubuntu-compile/manifest.json").write_text(json.dumps({
            "files": [file_record(root / name, root) for name in compile_files]}))
        shots = ("bake", "collect", "evalserver", "fp8probe", "gemm", "gr00t", "nvfp4", "opbench",
                 "opd", "opdcache", "packed", "pi05", "probe", "qad", "qat", "rollout", "verify")
        for name in shots:
            target = paper / "figs" / ("shot_" + name + ".png")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(("distinct screenshot " + name).encode())
        files = [paper / "README.md", paper / "paper.html", *sorted((paper / "figs").glob("*.png"))]
        report = {"passed": True, "package_inputs": [file_record(root / "README.md", root)]}
        return paper, files, report

    def test_archive_preserves_repo_layout_links_bytes_and_all_seventeen_shots(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paper, files, report = self.fixture(root)
            with (mock.patch.object(package, "P", paper),
                  mock.patch.object(package, "publication_files", return_value=files),
                  mock.patch.object(package, "validate", return_value=report) as validate,
                  mock.patch.object(sys, "argv", ["package_publication.py"])):
                package.main()
                validate.assert_called_once_with(write_report=False)
            with zipfile.ZipFile(paper / package.DESTINATION) as archive:
                names = set(archive.namelist())
                for name in ("README.md", "paper/README.md", "paper/paper.html", "docs/guide.md",
                             "docs/details.md", "docs/images/diagram.png", "docs/images/ubuntu-compile.png",
                             "docs/evidence/ubuntu-compile/run.sh", "docs/evidence/ubuntu-compile/capture.log"):
                    self.assertEqual(archive.read(name), (root / name).read_bytes())
                self.assertNotIn("project-inputs/README.md", names)
                self.assertNotIn("results/engine/unlinked-private.tmp", names)
                self.assertEqual(len([n for n in names if n.startswith("paper/figs/shot_")]), 17)
                self.assertEqual(json.loads(archive.read("PROJECT-INPUTS.json"))["files"]["README.md"], "README.md")
                self.assertEqual(json.loads(archive.read("paper/validation/publication-validation.json")), report)
                notice = archive.read("PUBLICATION-PACKAGE.md").decode()
                self.assertIn("[paper/paper.html](paper/paper.html)", notice)
                self.assertIn("不承诺完整离线安装", notice)
                manifest = json.loads(archive.read("PACKAGE-MANIFEST.json"))
                for name, row in manifest.items():
                    data = archive.read(name)
                    self.assertEqual(row, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def test_linked_compile_provenance_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paper, files, report = self.fixture(root)
            (root / "docs/evidence/ubuntu-compile/capture.log").write_text("changed")
            with mock.patch.object(package, "P", paper):
                with self.assertRaisesRegex(RuntimeError, "provenance changed"):
                    package.publication_entries(files, report["package_inputs"])

    def test_local_links_fail_for_missing_files_escape_weights_and_symlinks(self):
        for kind in ("missing", "escape", "weights", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                paper, files, _ = self.fixture(root)
                target = {"missing": "missing.txt", "escape": "../outside.txt",
                          "weights": "model.safetensors", "symlink": "linked.txt"}[kind]
                if kind == "weights":
                    (root / target).write_bytes(b"not permitted")
                if kind == "symlink":
                    (root / target).symlink_to(root / "docs/details.md")
                (root / "README.md").write_text(f"[Dependency]({target})")
                with mock.patch.object(package, "P", paper):
                    with self.assertRaises(RuntimeError):
                        package.publication_entries(files, [])

    def test_directory_link_requires_a_bundled_member_not_recursive_copy(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paper, files, _ = self.fixture(root)
            (root / "README.md").write_text("[Private directory](results/engine/)")
            with mock.patch.object(package, "P", paper):
                with self.assertRaisesRegex(RuntimeError, "Unbundled document directory"):
                    package.publication_entries(files, [])

    def test_check_mode_never_writes_an_archive(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paper, files, report = self.fixture(root)
            with (mock.patch.object(package, "P", paper),
                  mock.patch.object(package, "publication_files", return_value=files),
                  mock.patch.object(package, "validate", return_value=report),
                  mock.patch.object(sys, "argv", ["package_publication.py", "--check"])):
                package.main()
            self.assertFalse((paper / package.DESTINATION).exists())


if __name__ == "__main__":
    unittest.main()
