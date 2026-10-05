"""CPU-only publication binding; the search collector owns cost arithmetic tests."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
import validate_publication as validation
import capture_runtime


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw);paper=root/'paper';folder=paper/'evidence/search_costs'
        folder.mkdir(parents=True)
        protocol=root/'protocol.json';protocol.write_text('{"version":11,"w4a4":true}')
        for name in ('collect_search_costs.py','collect_training_costs.py'):
            (paper/name).write_text('# synthetic fixture\n')
        records=[]
        for name in ('summary.json','candidates/example/runtime_metrics.json'):
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('{"fixture_only":true}')
            records.append({'published_path':name,'bytes':path.stat().st_size,'sha256':validation.sha(path)})
        (folder/'evidence_manifest.json').write_text(json.dumps({'files':records}))
        report={'status':'verified','files':len(records),'protocol_sha256':validation.sha(protocol),
                'completed_search_totals_by_timing_scope':{}}
        calls=[]
        def verify(path):
            calls.append(path)
            return copy.deepcopy(report)
        fake=SimpleNamespace(verify=verify)
        with patch.object(validation,'P',paper),patch.dict(sys.modules,{'collect_search_costs':fake}):
            yield paper,folder,protocol,report,calls,fake


class SearchCostPublicationTests(unittest.TestCase):
    def test_archive_verifier_is_required_and_all_files_enter_package_inputs(self):
        with fixture() as (paper,folder,protocol,report,calls,_):
            inputs=set()
            self.assertEqual(validation.check_search_costs(inputs,protocol),report)
            self.assertEqual(calls,[folder])
            self.assertEqual(inputs,{folder/'summary.json',folder/'candidates/example/runtime_metrics.json',
                folder/'evidence_manifest.json',paper/'collect_search_costs.py',paper/'collect_training_costs.py'})

    def test_verifier_failure_is_not_ignored(self):
        with fixture() as (_,_,protocol,_,_,fake):
            def reject(path):raise ValueError('search arithmetic mismatch')
            fake.verify=reject
            with self.assertRaisesRegex(ValueError,'arithmetic mismatch'):
                validation.check_search_costs(set(),protocol)

    def test_other_protocol_or_unverified_status_is_rejected(self):
        for field,value in [('protocol_sha256','f'*64),('status','incomplete')]:
            with self.subTest(field=field),fixture() as (_,_,protocol,report,_,_):
                report[field]=value
                with self.assertRaisesRegex(RuntimeError,'different protocol or did not verify'):
                    validation.check_search_costs(set(),protocol)

    def test_changed_archived_input_after_verifier_return_is_rejected(self):
        with fixture() as (_,folder,protocol,_,_,fake):
            real=fake.verify
            def mutate(path):
                result=real(path)
                (folder/'summary.json').write_text('{"changed":true}')
                return result
            fake.verify=mutate
            with self.assertRaisesRegex(RuntimeError,'Changed evidence'):
                validation.check_search_costs(set(),protocol)

    def test_search_collector_is_in_runtime_source_allowlist(self):
        self.assertIn('paper/collect_search_costs.py',capture_runtime.DEFAULT_SOURCE_FILES)
        self.assertIn('paper/collect_search_costs.py',capture_runtime.resolve_source_files(P.parent))


if __name__=='__main__':unittest.main(verbosity=2)
