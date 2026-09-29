"""CPU-only, synthetic v3 evidence: never publish fixture experiment results."""
import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'paper'))
sys.path.insert(0,str(ROOT/'tests'))
import test_ptq_frontier as fixtures
from frontier_evidence import validate_frontier
from collect_frontier_evidence import collect as collect_frontier
from collect_pairing_evidence import collect as collect_pairing
import validate_publication as publication
from capture_runtime import DEFAULT_SOURCE_FILES
f=fixtures.f
write=fixtures.write


class V3PublicationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FrontierTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        t=self.fixture;self.t=t
        self.protocol=t.root/'exp/recovery_protocol_v3_high_fp4.json'
        shutil.copyfile(ROOT/'exp/recovery_protocol_v3_high_fp4.json',self.protocol)
        self.contract=f.protocol_contract(self.protocol)
        shutil.rmtree(t.dev)
        for name,old in (('head_lang_vision','fp8'),('calib','head_lang')):
            folder=t.root/'weights'/name;shutil.copytree(t.checkpoints[old],folder)
            recipe=f.load(folder/'ptq_recipe.json');recipe['recipe']=name;write(folder/'ptq_recipe.json',recipe)
            t.checkpoints[name]=folder;t.inventory['recipes'][name]=copy.deepcopy(t.inventory['recipes'][old])
        for name,score in (('bf16',50),('head_lang_vision',30),('calib',20)):
            self.run_fixture(t.dev/name,t.checkpoints[name],'development',score)
        write(t.budget,t.inventory)
        write(t.selection,f.audit_development(t.dev,['bf16','head_lang_vision','calib'],self.protocol))

    def run_fixture(self,folder,checkpoint,purpose,score,collection=None):
        t=self.t;t.make_run(folder,checkpoint,purpose,0,collection)
        contract=self.contract[purpose];indices=contract['init_state_indices'];seed=contract['seed']
        manifest=f.load(folder/'eval_manifest.json')
        manifest.update({**{k:v for k,v in contract.items() if k not in ('episodes_per_task','tasks')},
            'tasks':f.TASKS,'task_count':10,'episodes':len(indices),'protocol_file':str(self.protocol),
            'protocol_sha256':f.sha(self.protocol),'task_seed_stride':1000,'episode_seed_stride':1})
        rows={}
        for ti,task in enumerate(f.TASKS):
            row={'returncode':0,'seed':seed+1000*ti,'results':[ti*len(indices)+i<score for i in range(len(indices))],
                 'resets':[{'episode_index':i,'seed':seed+1000*ti+i,'init_state_index':bank,'settle_steps':10,
                     'initial_state_sha256':f'{ti*100+i:064x}','restored_state_sha256':f'{ti*100+i+3000:064x}',
                     'init_state_bank_sha256':f'{ti+7000:064x}'} for i,bank in enumerate(indices)]}
            t.write_task_log(folder,task,row);rows[task]=row
        write(folder/'eval_manifest.json',manifest);write(folder/'task_results.json',rows)
        write(folder/'summary.json',{'tasks_complete':10,'total_episodes':10*len(indices),'total_successes':score,
            'macro_success_rate':sum(r['success_rate'] for r in rows.values())/10,'purpose':purpose,'seed':seed,
            'wall_seconds_including_server_loads':1.})

    def completed_frontier(self):
        t=self.t;plan=f.freeze(t.selection,t.dev,t.budget,self.protocol);write(t.plan,plan)
        t.round=t.root/'round';t.out=t.root/'frontier';collection=t.round/'collection/eval_manifest.json'
        cp=self.contract['collection'];write(collection,{'purpose':'collection','seed':cp['seed'],
            'episodes':cp['episodes_per_task'],'tasks':f.TASKS,'init_state_indices':cp['init_state_indices'],
            'initial_state_protocol':'libero10_official_bank_v1','n_envs':1,'settle_steps':10,
            'protocol_file':str(self.protocol),'protocol_sha256':f.sha(self.protocol)})
        for arm in f.ARMS:
            checkpoint=t.checkpoints['bf16' if arm=='bf16' else 'calib']
            if arm in ('qad','continued_qad','qad_opd'):
                checkpoint=t.root/'weights'/arm;checkpoint.mkdir()
                write(checkpoint/'merge_manifest.json',{'status':'complete','base':str(t.checkpoints['calib']),
                    'lora_pairs':2,'recovery_manifest':{'rank':1,'alpha':2,'scope':'head+lang_all','trainable_parameters':4}})
            self.run_fixture(t.round/('heldout_'+arm),checkpoint,'heldout',95,collection)
        write(t.round/'paired_comparison.json',f.compare_round(t.round))
        for name in plan['references']:
            self.run_fixture(t.out/('heldout_'+name),t.checkpoints[name],'heldout',94,collection)
        write(t.out/'run_manifest.json',{'plan':f.identity(t.plan),'collection_manifest':f.identity(collection),
              'main_comparison':f.identity(t.round/'paired_comparison.json')})
        return plan

    def test_v3_freezes_only_declared_candidates_and_requires_final_reference(self):
        plan=self.completed_frontier();t=self.t
        self.assertEqual(plan['references'],['head_lang_vision'])
        self.assertEqual(set(plan['checkpoints']),{'bf16','head_lang_vision','calib'})
        result=f.compare(t.plan,t.round,t.out)
        self.assertEqual(result['reference_order'],['head_lang_vision'])
        self.assertEqual(result['selected_recipe'],'calib')
        (t.out/'heldout_head_lang_vision/summary.json').unlink()
        with self.assertRaises(FileNotFoundError):f.compare(t.plan,t.round,t.out)

    def test_v3_public_copy_verifies_without_original_protocol_or_runs(self):
        self.completed_frontier();t=self.t
        with mock.patch.object(f,'__file__',str(t.root/'eval/compare_ptq_frontier.py')):
            result=f.compare(t.plan,t.round,t.out);write(t.out/'frontier_comparison.json',result)
            paper=t.root/'public-paper';(paper/'evidence').mkdir(parents=True)
            shutil.copyfile(t.budget,paper/'evidence/recipe_inventory.json')
            collect_frontier(t.out,paper)
        for folder in (t.root/'weights',t.round,t.out,t.dev):shutil.rmtree(folder)
        self.protocol.unlink()
        actual,_=validate_frontier(paper)
        self.assertEqual(actual,result)

    def test_pairing_collector_and_validator_follow_v3_partition(self):
        self.completed_frontier();t=self.t
        report=f.load(t.round/'paired_comparison.json')
        publication.check_pairing(report,self.protocol)
        with self.assertRaisesRegex(RuntimeError,'outside declared partition'):
            publication.check_pairing(report,ROOT/'exp/recovery_protocol.json')
        output=t.root/'paired-public'
        collect_pairing(t.round,output,self.protocol)
        self.assertEqual((output/'paired_comparison.json').read_bytes(),(t.round/'paired_comparison.json').read_bytes())
        for token in (r'\textcolor{#b42318}{xxx\%}',r'\mathrm{xxx}'):
            with self.assertRaisesRegex(RuntimeError,'Incomplete review draft'):publication.check_finished(token)

    def test_runtime_allowlist_carries_v3_protocol_and_orchestrator(self):
        self.assertIn('exp/recovery_protocol_v3_high_fp4.json',DEFAULT_SOURCE_FILES)
        self.assertIn('exp/run_high_fp4_v3.py',DEFAULT_SOURCE_FILES)


if __name__=='__main__':unittest.main()
