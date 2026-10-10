"""CPU negative tests for an explicit unperformed GPTQ supplement."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
import gptq_reference_publication as publication
import validate_publication as validation
from v12_publication_contract import PROTOCOL_NAME, PROTOCOL_SHA256


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def fixture(root):
    paper = root / "paper"
    evidence = paper / "evidence"
    evidence.mkdir(parents=True)
    (root / "exp").mkdir()
    shutil.copyfile(ROOT / "exp" / PROTOCOL_NAME, root / "exp" / PROTOCOL_NAME)
    shutil.copyfile(ROOT / "paper" / publication.SCOPE_FILE, paper / publication.SCOPE_FILE)
    final = {"format": "w4a4_recovery_v12_final_manifest",
             "protocol_sha256": PROTOCOL_SHA256, "selection_uses_heldout": False,
             "required_arms": ["bf16", "ptq", "qad", "continued_qad", "qad_opd"]}
    final_path = evidence / "final_manifest.json"
    dump(final_path, final)
    task = {"episodes": 16, "successes": 8, "success_rate": .5}
    arm = {"episodes": 160, "successes": 80, "success_rate": .5, "macro_success_rate": .5,
           "per_task": {f"task_{n}": copy.deepcopy(task) for n in range(10)}}
    results = {"format": "publication_final_results_v1", "status": "complete",
               "selected_recipe": "rtn_w4a4_category",
               "selection": {"selection_uses_heldout": False},
               "source": {
                   "run_dir": "/original/" + publication.RELEASE_RUN,
                   "protocol": {"path": "/original/exp/" + PROTOCOL_NAME, "sha256": PROTOCOL_SHA256},
                   "final_manifest": {"path": "/original/" + publication.RELEASE_RUN + "/final_manifest.json",
                                      "bytes": final_path.stat().st_size,
                                      "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest()}},
               "public_arms": {name: copy.deepcopy(arm) for name in ("bf16", "ptq", "qad", "qad_opd")},
               "control": {"continued_qad": copy.deepcopy(arm)}}
    dump(evidence / "final_results.json", results)
    return paper


class FrozenGptqScope(unittest.TestCase):
    def test_explicit_receipt_is_not_a_measurement(self):
        with tempfile.TemporaryDirectory() as raw:
            paper = fixture(Path(raw))
            receipt = publication.record_not_performed(paper)
            self.assertEqual(receipt["status"], "not_performed")
            report = publication.load_verified(paper)
            for key in ("raw_logs_replayed", "paired_statistics_recomputed",
                        "tensor_contents_reverified_offline"):
                self.assertIs(report[key], False)
            self.assertNotIn("comparison", report)
            self.assertIn("未执行", publication.render(report))
            with self.assertRaisesRegex(ValueError, "existing GPTQ"):
                publication.record_not_performed(paper)

    def test_missing_receipt_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as raw:
            paper = fixture(Path(raw))
            with self.assertRaisesRegex(ValueError, "requires its registered"):
                publication.load_verified(paper)

    def test_missing_scope_cannot_license_an_omission(self):
        with tempfile.TemporaryDirectory() as raw:
            paper = fixture(Path(raw))
            (paper / publication.SCOPE_FILE).unlink()
            with self.assertRaisesRegex(ValueError, "Missing regular"):
                publication.record_not_performed(paper)

    def test_other_protocol_run_or_incomplete_result_rejected(self):
        for field in ("protocol", "run", "incomplete", "counts"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                paper = fixture(Path(raw))
                path = paper / "evidence/final_results.json"
                data = json.loads(path.read_text())
                if field == "protocol":
                    data["source"]["protocol"]["sha256"] = "0" * 64
                elif field == "run":
                    data["source"]["run_dir"] = "/original/results/previous/recovery_v11"
                elif field == "incomplete":
                    data["status"] = "running"
                else:
                    data["public_arms"]["qad"]["episodes"] = 159
                dump(path, data)
                with self.assertRaises(ValueError):
                    publication.record_not_performed(paper)

    def test_changed_scope_final_or_receipt_cannot_pass(self):
        for target in ("scope", "final", "receipt"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                paper = fixture(Path(raw))
                publication.record_not_performed(paper)
                path = {"scope": paper / publication.SCOPE_FILE,
                        "final": paper / "evidence/final_manifest.json",
                        "receipt": paper / "evidence/gptq_reference/not_performed.json"}[target]
                data = json.loads(path.read_text())
                data["unexpected"] = "modified"
                dump(path, data)
                with self.assertRaises(ValueError):
                    publication.load_verified(paper)

    def test_measured_artifacts_cannot_mix_with_nonperformance(self):
        with tempfile.TemporaryDirectory() as raw:
            paper = fixture(Path(raw))
            publication.record_not_performed(paper)
            dump(paper / "evidence/gptq_reference/summary.json", {"successes": 160})
            with self.assertRaisesRegex(ValueError, "cannot coexist"):
                publication.load_verified(paper)

    def test_omission_gate_rejects_leftover_result_claims(self):
        with tempfile.TemporaryDirectory() as raw:
            paper = fixture(Path(raw))
            publication.record_not_performed(paper)
            prose = publication.render(publication.load_verified(paper))
            article = paper / "sections/04-实验.md"
            article.parent.mkdir()
            article.write_text("## 4.6 同覆盖校准 GPTQ 参考\n\n" + prose + "\n\n## 4.7 成本\n")
            readme = paper.parent / "README.md"
            readme.write_text("### 同覆盖的校准 GPTQ 参考\n\n" + prose + "\n\n### 编码预算\n")
            with mock.patch.object(validation, "P", paper):
                validation.check_gptq_reference(set())
                article.write_text(article.read_text().replace(prose, prose + "\n\nGPTQ 160/160。"))
                with self.assertRaisesRegex(RuntimeError, "unsupported results"):
                    validation.check_gptq_reference(set())


if __name__ == "__main__":
    unittest.main()
