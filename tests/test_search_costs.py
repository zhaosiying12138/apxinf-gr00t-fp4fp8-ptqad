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


if __name__ == '__main__':
    unittest.main()
