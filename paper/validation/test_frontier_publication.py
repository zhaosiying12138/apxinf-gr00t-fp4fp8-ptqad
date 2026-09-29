"""CPU-only publication tests using temporary, visibly synthetic fixtures."""
import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P));sys.path.insert(0,str(P.parent/'tests'))
import test_ptq_frontier as fixtures
from frontier_evidence import f,validate_frontier,digest
from collect_frontier_evidence import collect
import make_figs
write=fixtures.write


class PublishedFrontier(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FrontierTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        t=self.fixture;t.heldout()
        patch=mock.patch.object(f,'__file__',str(t.root/'eval/compare_ptq_frontier.py'));patch.start();self.addCleanup(patch.stop)
        self.result=f.compare(t.plan,t.round,t.out)
        write(t.out/'frontier_comparison.json',self.result)
        self.paper=t.root/'published-paper';(self.paper/'evidence').mkdir(parents=True)
        shutil.copyfile(t.budget,self.paper/'evidence/recipe_inventory.json')

    def snapshot(self):return collect(self.fixture.out,self.paper)

    def rewrite(self,path,value):
        write(path,value)
        file=self.paper/'evidence/frontier/evidence_manifest.json';mapping=f.load(file)
        for row in mapping['mapping']:
            if self.paper/row['published_path']==path:row.update(bytes=path.stat().st_size,sha256=digest(path))
        write(file,mapping)

    def test_snapshot_preserves_json_and_validates_without_original_weights_or_logs(self):
        report=self.snapshot();self.assertEqual(report['status'],'complete')
        self.assertEqual((self.paper/'evidence/frontier_comparison.json').read_bytes(),(self.fixture.out/'frontier_comparison.json').read_bytes())
        for folder in (self.fixture.root/'weights',self.fixture.round,self.fixture.out,self.fixture.dev):shutil.rmtree(folder)
        verified,files=validate_frontier(self.paper)
        self.assertEqual(verified,self.result);self.assertGreater(len(files),100)
        self.assertFalse(any(path.suffix=='.safetensors' for path in files))

    def test_missing_raw_log_fails(self):
        self.snapshot();(self.paper/'evidence/frontier/references/heldout_fp8'/(f.TASKS[0]+'.log')).unlink()
        with self.assertRaises((RuntimeError,FileNotFoundError)):validate_frontier(self.paper)

    def test_changed_raw_log_fails_even_if_copy_manifest_is_updated(self):
        self.snapshot();path=self.paper/'evidence/frontier/references/heldout_fp8'/(f.TASKS[0]+'.log')
        path.write_text(path.read_text()+'changed after evaluation\n')
        m=self.paper/'evidence/frontier/evidence_manifest.json';data=f.load(m)
        for row in data['mapping']:
            if self.paper/row['published_path']==path:row.update(bytes=path.stat().st_size,sha256=digest(path))
        write(m,data)
        with self.assertRaisesRegex(ValueError,'raw log'):validate_frontier(self.paper)

    def test_net_residual_omission_rejected(self):
        self.snapshot();path=self.paper/'evidence/frontier_comparison.json';data=f.load(path)
        data['points'][-1]['encoding_budget']['physical']['residual_bytes']=0
        self.rewrite(path,data)
        with self.assertRaisesRegex(RuntimeError,'points differ'):validate_frontier(self.paper)

    def test_reference_omission_rejected(self):
        self.snapshot();path=self.paper/'evidence/frontier_comparison.json';data=f.load(path)
        del data['references']['fp8'];self.rewrite(path,data)
        with self.assertRaisesRegex(RuntimeError,'Reference results'):validate_frontier(self.paper)

    def test_pairing_is_rechecked_from_copied_episodes(self):
        self.snapshot();original=f.read_heldout
        def altered(folder,protocol):
            manifest,arm=original(folder,protocol)
            if Path(folder).name=='heldout_fp8':arm['episodes'][1]['restored_state_sha256']='f'*64
            return manifest,arm
        with mock.patch.object(f,'read_heldout',side_effect=altered):
            with self.assertRaisesRegex(ValueError,'pairing'):validate_frontier(self.paper)

    def test_other_main_round_rejected(self):
        self.snapshot();main=f.load(self.paper/'evidence/frontier/main/paired_comparison.json');main['arms']['ptq']['successes']-=1
        with self.assertRaisesRegex(RuntimeError,'another five-arm'):validate_frontier(self.paper,main)

    def test_snapshot_refuses_overwrite(self):
        self.snapshot()
        with self.assertRaisesRegex(RuntimeError,'overwrite'):self.snapshot()

    def test_chart_missing_data_and_malformed_budget_fail_closed(self):
        with mock.patch.object(make_figs,'read',side_effect=FileNotFoundError('incomplete')):
            with self.assertRaises(FileNotFoundError):make_figs.frontier_rows()
        with mock.patch.object(make_figs,'read',return_value=copy.deepcopy(self.result)):
            self.assertEqual(len(make_figs.frontier_rows()[1]),7)
        changed=copy.deepcopy(self.result);changed['points'][-1]['encoding_budget']['physical']['total_bytes']-=8
        with mock.patch.object(make_figs,'read',return_value=changed):
            with self.assertRaises(ValueError):make_figs.frontier_rows()

    def test_chart_writes_only_temporary_fixture_output(self):
        folder=self.fixture.root/'test-render';folder.mkdir()
        with mock.patch.object(make_figs,'read',return_value=copy.deepcopy(self.result)),mock.patch.object(make_figs,'FIGS',folder):
            make_figs.ptq_frontier()
        import xml.etree.ElementTree as ET
        doc=ET.parse(folder/'ptq_frontier.svg')
        self.assertEqual(doc.getroot().attrib['width'],'1120')


if __name__=='__main__':unittest.main()
