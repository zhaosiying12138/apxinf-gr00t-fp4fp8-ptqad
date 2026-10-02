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

    def test_v11_sixteen_episodes_per_task_are_copied_and_missing_tail_rejected(self):
        protocol=self.fixture.root/'exp/recovery_protocol_v11_w4a4_category.json'
        protocol.write_bytes((ROOT/'exp/recovery_protocol_v11_w4a4_category.json').read_bytes())
        contract=json.loads(protocol.read_text());part=contract['partitions']['heldout']
        sha=hashlib.sha256(protocol.read_bytes()).hexdigest()
        for arm in collect.ARMS:
            folder=self.origin/('heldout_'+arm)
            manifest=json.loads((folder/'eval_manifest.json').read_text())
            manifest.update(seed=part['seed'],episodes=16,init_state_indices=part['init_state_indices'],
                            protocol_file=str(protocol),protocol_sha256=sha,
                            task_count=10,task_seed_stride=1000,episode_seed_stride=1)
            fixtures.write(folder/'eval_manifest.json',manifest)
            tasks=json.loads((folder/'task_results.json').read_text())
            for ti,(task,row) in enumerate(tasks.items()):
                row['results']=[True]*16
                row['resets']=[{'episode_index':i,'seed':part['seed']+ti*1000+i,'init_state_index':bank,
                    'settle_steps':10,'initial_state_sha256':f'{ti*100+i:064x}',
                    'restored_state_sha256':f'{ti*100+i+3000:064x}','init_state_bank_sha256':f'{ti+7000:064x}'}
                    for i,bank in enumerate(part['init_state_indices'])]
                self.fixture.write_task_log(folder,task,row)
            fixtures.write(folder/'task_results.json',tasks)
            fixtures.write(folder/'summary.json',{'tasks_complete':10,'wall_seconds_including_server_loads':1.,
                'total_successes':160,'total_episodes':160,'macro_success_rate':1.,'purpose':'heldout','seed':part['seed']})
        collection=self.origin/'collection/eval_manifest.json'
        data=json.loads(collection.read_text());c=contract['partitions']['collection']
        data.update(seed=c['seed'],episodes=c['episodes_per_task'],init_state_indices=c['init_state_indices'],
                    protocol_file=str(protocol),protocol_sha256=sha)
        fixtures.write(collection,data)
        comparison=fixtures.compare_round(self.origin)
        fixtures.write(self.origin/'paired_comparison.json',comparison)
        report=collect.collect(self.origin,self.out,protocol)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(json.loads((self.out/'paired_comparison.json').read_text())['arms']['qad']['count'],160)
        path=self.origin/'heldout_qad/task_results.json';tasks=json.loads(path.read_text())
        tasks[next(iter(tasks))]['results'].pop();fixtures.write(path,tasks)
        with self.assertRaises(ValueError):collect.collect(self.origin,self.fixture.root/'incomplete',protocol)

if __name__=='__main__':unittest.main(verbosity=2)
