"""CPU-only byte-copy tests using complete synthetic heldout fixtures."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'paper'))
sys.path.insert(0,str(ROOT/'tests'))
import collect_pairing_evidence as collect
import test_ptq_frontier as fixtures


class PairingCopyTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FrontierTests(methodName='test_full_comparison_includes_net_residual')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.heldout();self.origin=self.fixture.round;self.out=self.fixture.root/'publication/evidence'
    def test_all_sixteen_original_bytes_and_idempotent_retry(self):
        report=collect.collect(self.origin,self.out)
        self.assertEqual(report['status'],'complete');self.assertEqual(len(report['files']),16)
        for name in collect.NAMES:self.assertEqual((self.out/name).read_bytes(),(self.origin/name).read_bytes())
        again=collect.collect(self.origin,self.out);self.assertEqual(again['files'],report['files'])
    def test_incomplete_source_creates_no_output(self):
        (self.origin/'heldout_qad/summary.json').unlink()
        with self.assertRaises(FileNotFoundError):collect.collect(self.origin,self.out)
        self.assertFalse(self.out.exists())
    def test_forged_comparison_rejected(self):
        path=self.origin/'paired_comparison.json';data=json.loads(path.read_text());data['arms']['qad']['successes']-=1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'does not reproduce'):collect.collect(self.origin,self.out)
        self.assertFalse(self.out.exists())
    def test_conflict_refused_before_any_public_file_installed(self):
        target=self.out/'heldout_qad/summary.json';target.parent.mkdir(parents=True);target.write_text('conflicting original')
        with self.assertRaises(FileExistsError):collect.collect(self.origin,self.out)
        self.assertEqual(target.read_text(),'conflicting original')
        self.assertEqual([p for p in self.out.rglob('*') if p.is_file()],[target])
    def test_source_changes_during_copy_rejected(self):
        real=collect.shutil.copyfile
        def changed(source,target):
            result=real(source,target)
            if Path(source).name=='task_results.json':Path(source).write_text('{}')
            return result
        with mock.patch.object(collect.shutil,'copyfile',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'Source changed'):collect.collect(self.origin,self.out)
        self.assertFalse(self.out.exists())
    def test_symbolic_output_cannot_escape(self):
        self.out.parent.mkdir(parents=True);actual=self.fixture.root/'other';actual.mkdir()
        self.out.symlink_to(actual,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symbolic output'):collect.collect(self.origin,self.out)
        self.assertEqual(list(actual.iterdir()),[])

if __name__=='__main__':unittest.main(verbosity=2)
