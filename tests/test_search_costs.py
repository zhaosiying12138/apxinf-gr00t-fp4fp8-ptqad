"""Search-cost protocol gates run before any completed-stage evidence is copied."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper import collect_search_costs as search

ROOT = Path(__file__).resolve().parents[1]


class SearchCostProtocolTests(unittest.TestCase):
    def test_v12_accepts_frozen_protocol_then_requires_completed_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            protocol = ROOT / 'exp/recovery_protocol_v12_rtn_w4a4.json'
            state = {'protocol_file': str(protocol),
                     'protocol_sha256': search.costs.identity(protocol)['sha256'],
                     'selected_recipe': 'rtn_w4a4_category'}
            (run / 'run_manifest.json').write_text(json.dumps(state))
            (run / 'stages').mkdir()
            output = run / 'costs'
            with self.assertRaises(FileNotFoundError) as error:
                search.collect(run, output)
            self.assertIn('select_qad_lr.json', str(error.exception))
            self.assertFalse(output.exists())

    def test_changed_v12_protocol_is_rejected_even_with_self_consistent_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            protocol = run / 'protocol.json'
            protocol.write_bytes((ROOT / 'exp/recovery_protocol_v12_rtn_w4a4.json').read_bytes() + b'\n')
            sha = search.costs.identity(protocol)['sha256']
            (run / 'run_manifest.json').write_text(json.dumps({
                'protocol_file': str(protocol), 'protocol_sha256': sha,
                'selected_recipe': 'rtn_w4a4_category'}))
            output = run / 'costs'
            with patch.object(search.costs, 'evaluation_helpers', side_effect=AssertionError('must reject before stages')):
                with self.assertRaisesRegex(ValueError, 'frozen v12'):
                    search.collect(run, output)
            self.assertFalse(output.exists())

    def test_public_recomputation_rejects_rehashed_modified_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            protocol = archive / 'protocol/recovery_protocol.json'
            protocol.parent.mkdir()
            protocol.write_bytes((ROOT / 'exp/recovery_protocol_v12_rtn_w4a4.json').read_bytes() + b'\n')
            (archive / 'inputs.json').write_text(json.dumps({
                'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError, 'supported frozen'):
                search.summarize(archive)


class SearchCostAdoptedReceiptTests(unittest.TestCase):
    def make_receipt(self, folder, adopted=True):
        request = {"protocol_sha256": "a" * 64, "environment": {
            "QAD_LR": "5e-05", "QAD_OPD_MSE_W": "0.0", "QAD_STEPS": "2000"}}
        req = folder / "orchestrator_training_request.json"
        req.write_text(json.dumps(request))
        runtime = folder / "runtime_metrics.json"
        runtime.write_text('{"status":"completed","global_steps":2000}')
        return {"adopted": adopted,
                "training_request_identity": {"path": str(req), **search.costs.identity(req)},
                "runtime_identity": {"path": str(runtime), **search.costs.identity(runtime)}}

    def test_adopted_receipt_uses_verified_original_request(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            receipt = self.make_receipt(folder)
            search.audit_training_stage_receipt(receipt, folder, 5e-5, 0., 2000, "a" * 64)
            receipt["adopted"] = False
            with self.assertRaisesRegex(ValueError, "configuration is missing"):
                search.audit_training_stage_receipt(receipt, folder, 5e-5, 0., 2000, "a" * 64)

    def test_modified_request_bytes_or_rehashed_wrong_settings_rejected(self):
        for mutation in ("bytes", "learning_rate", "weight", "steps", "protocol"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                receipt = self.make_receipt(folder)
                path = folder / "orchestrator_training_request.json"
                request = json.loads(path.read_text())
                if mutation == "protocol":
                    request["protocol_sha256"] = "b" * 64
                elif mutation != "bytes":
                    key = {"learning_rate": "QAD_LR", "weight": "QAD_OPD_MSE_W",
                           "steps": "QAD_STEPS"}[mutation]
                    request["environment"][key] = "999"
                path.write_text(json.dumps(request) + "\n")
                if mutation != "bytes":
                    receipt["training_request_identity"].update(search.costs.identity(path))
                with self.assertRaises(ValueError):
                    search.audit_training_stage_receipt(receipt, folder, 5e-5, 0., 2000, "a" * 64)

    def test_redundant_adopted_fields_cannot_conflict_with_request(self):
        for key in ("learning_rate", "opd_weight", "optimizer_steps"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                receipt = self.make_receipt(folder)
                receipt[key] = 999
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    search.audit_training_stage_receipt(receipt, folder, 5e-5, 0., 2000, "a" * 64)

    def test_archived_adopted_configuration_uses_hashed_copies_without_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            receipt = self.make_receipt(folder)
            for key in ("runtime_identity", "training_request_identity"):
                receipt[key]["path"] = "/unavailable/private/source"
            protocol = {"selection": {"qad_learning_rates": [5e-5, .0001],
                                     "opd_weights": [.25, 1.],
                                     "qad_optimizer_steps": 2000,
                                     "continuation_optimizer_steps": 2000}}
            with patch.object(search, "checked_identity", side_effect=AssertionError("private access")):
                self.assertEqual(search.archived_training_configuration(
                    receipt, folder, "qad", protocol, "a" * 64), (5e-5, 0., 2000))
                path = folder / "orchestrator_training_request.json"
                path.write_text(path.read_text() + "\\n")
                with self.assertRaisesRegex(ValueError, "Archived training identity differs"):
                    search.archived_training_configuration(receipt, folder, "qad", protocol, "a" * 64)

    def test_stage_cannot_point_at_another_request_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            receipt = self.make_receipt(folder)
            with self.assertRaisesRegex(ValueError, "outside their declared"):
                search.audit_training_stage_receipt(receipt, folder / "other", 5e-5, 0., 2000, "a" * 64)


if __name__ == '__main__':
    unittest.main()
