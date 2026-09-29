"""CPU-only temporary fixtures: development-only freeze and paired PTQ frontier."""
import copy
import json
from pathlib import Path
import shutil
import sys
import struct
import io
import contextlib
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eval'))
import compare_ptq_frontier as f
from compare_development import audit_development
from compare_recovery import compare_round
from run_recovery_eval import TASKS


def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class FrontierTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name in f.SOURCES:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        self.patch=mock.patch.multiple(f,ROOT=self.root,PROTOCOL=self.root/'exp/ptq_frontier_protocol.json');self.patch.start();self.addCleanup(self.patch.stop)
        self.dev=self.root/'development';self.selection=self.root/'selection.json';self.plan=self.root/'plan.json'
        self.inventory={'recovery_residual':{'dtype':'bfloat16','bytes_per_element':2,'tensor_elements':4,'target_bytes':8,
                        'rank':1,'alpha':2,'scope':'head+lang_all','linear_modules':2},'recipes':{}}
        self.checkpoints={}
        for name,score,cost in (('bf16',20,1000),('fp8',20,600),('head_ffn',19,500),('head_lang',18,450)):
            folder=self.root/'weights'/name;folder.mkdir(parents=True);self.checkpoints[name]=folder
            for filename in f.META:write(folder/filename,{})
            weight=folder/'model.safetensors';weight.write_bytes(('TEMPORARY TEST WEIGHTS '+name).encode())
            write(folder/'model.safetensors.index.json',{'weight_map':{'example.weight':weight.name}})
            memory={'source_tensor_bytes':1000,'target_full_checkpoint_bytes':cost,
                    'known_tied_alias_deduplicated':{'source_tensor_bytes':800,'all_tensor_elements':400,'target_full_bytes':cost-100}}
            self.inventory['recipes'][name]=memory
            if name!='bf16':
                write(folder/'ptq_recipe.json',{'recipe':name,'base':str(self.checkpoints['bf16']),'memory':memory})
                write(folder/'bake_manifest.json',{'status':'complete','output_weight_files':{weight.name:{'bytes':weight.stat().st_size,'sha256':f.sha(weight)}}})
            self.make_run(self.dev/name,folder,'development',score)
        self.inventory['base']=str(self.checkpoints['bf16'])
        self.inventory['recovery_residual']['source_index_sha256']=f.sha(self.checkpoints['bf16']/'model.safetensors.index.json')
        self.inventory['recovery_residual']['source_config_sha256']=f.sha(self.checkpoints['bf16']/'config.json')
        self.budget=self.root/'budget.json';write(self.budget,self.inventory)
        write(self.selection,audit_development(self.dev,['bf16','fp8','head_ffn','head_lang']))

    def make_run(self,folder,checkpoint,purpose,successes=95,collection=None):
        indices=[0,1] if purpose=='development' else list(range(10,20));seed=330000 if purpose=='development' else 220000
        manifest={'purpose':purpose,'seed':seed,'episodes':len(indices),'init_state_indices':indices,
            'n_envs':1,'tasks':TASKS,'initial_state_protocol':'libero10_official_bank_v1','settle_steps':10,
            'protocol_sha256':f.sha(self.root/'exp/recovery_protocol.json'),'n_action_steps':8,'max_episode_steps':720,
            'checkpoint':str(checkpoint),'gr00t':str(self.root/'external'),'server_python':'/cpu/test/python',
            'rollout_python':'/cpu/test/simulation','server_seed_offset':10000000,
            'collection_manifest':str(collection) if collection else None}
        tasks={}
        for i,task in enumerate(TASKS):
            tasks[task]={'returncode':0,'results':[i*len(indices)+j<successes for j in range(len(indices))],
                'resets':[{'episode_index':j,'seed':seed+1000*i+j,'init_state_index':index,'settle_steps':10,
                           'initial_state_sha256':f'{i*100+j:064x}','restored_state_sha256':f'{i*100+j+3000:064x}',
                           'init_state_bank_sha256':f'{i+7000:064x}'} for j,index in enumerate(indices)]}
        folder.mkdir(parents=True,exist_ok=True)
        for task,row in tasks.items():
            self.write_task_log(folder,task,row)
        for filename,data in (('eval_manifest.json',manifest),('task_results.json',tasks),
                ('summary.json',{'tasks_complete':10,'wall_seconds_including_server_loads':1.,'total_successes':successes,'total_episodes':10*len(indices),'macro_success_rate':sum(row['success_rate'] for row in tasks.values())/10,'purpose':purpose,'seed':seed})):
            write(folder/filename,data)

    def write_task_log(self,folder,task,row):
        log=folder/(task+'.log')
        log.write_text(''.join('FP4VLA_EPISODE_RESET '+json.dumps(reset)+'\n' for reset in row['resets'])+"results: ('fixture', "+repr(row['results'])+")\n")
        row.update(f.parse_log(log));row['seed']=row['resets'][0]['seed']

    def freeze(self):
        value=f.freeze(self.selection,self.dev,self.budget);f.write_new(self.plan,value);return value

    def heldout(self):
        plan=self.freeze();self.round=self.root/'round';self.out=self.root/'frontier'
        collection=self.round/'collection/eval_manifest.json'
        write(collection,{'purpose':'collection','seed':110000,'episodes':2,'tasks':TASKS,'init_state_indices':[2,3],
              'protocol_sha256':f.sha(self.root/'exp/recovery_protocol.json'),'initial_state_protocol':'libero10_official_bank_v1','n_envs':1,'settle_steps':10})
        for arm in ('bf16','ptq','qad','continued_qad','qad_opd'):
            checkpoint=self.checkpoints['bf16' if arm=='bf16' else 'head_lang']
            if arm in ('qad','continued_qad','qad_opd'):
                checkpoint=self.root/'weights'/arm;checkpoint.mkdir()
                write(checkpoint/'merge_manifest.json',{'status':'complete','base':str(self.checkpoints['head_lang']),'lora_pairs':2,
                    'recovery_manifest':{'rank':1,'alpha':2,'scope':'head+lang_all','trainable_parameters':4}})
            self.make_run(self.round/('heldout_'+arm),checkpoint,'heldout',96 if arm=='qad_opd' else 95,collection)
        write(self.round/'paired_comparison.json',compare_round(self.round))
        for name in plan['references']:self.make_run(self.out/('heldout_'+name),self.checkpoints[name],'heldout',94,collection)
        write(self.out/'run_manifest.json',{'plan':f.identity(self.plan),'collection_manifest':f.identity(collection),
              'main_comparison':f.identity(self.round/'paired_comparison.json')})
        return plan

    def test_freeze_all_predecessors_and_never_reads_heldout(self):
        with mock.patch.object(f,'checked_main',side_effect=AssertionError('heldout forbidden')):
            plan=self.freeze()
        self.assertEqual(plan['references'],['fp8','head_ffn']);self.assertFalse(plan['reference_selection_uses_heldout'])
        self.assertEqual(f.check_plan(self.plan)['selected_recipe'],'head_lang')
        with self.assertRaises(FileExistsError):f.write_new(self.plan,{})

    def test_rejects_incomplete_or_forged_selection(self):
        for mutation in ('pending','skip','heldout','score'):
            report=audit_development(self.dev,['bf16','fp8','head_ffn','head_lang'])
            if mutation=='pending':report['selected_recipe']=None
            elif mutation=='skip':del report['arms']['head_ffn']
            elif mutation=='heldout':report['selection_uses_heldout']=True
            else:report['arms']['head_lang']['successes']=17
            write(self.selection,report)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):f.freeze(self.selection,self.dev,self.budget)

    def test_rejects_changed_metadata_weights_and_removed_reference(self):
        self.freeze();path=self.checkpoints['head_ffn']/'model.safetensors';original=path.read_bytes();path.write_bytes(b'changed')
        with self.assertRaises(ValueError):f.check_plan(self.plan)
        path.write_bytes(original)
        plan=f.load(self.plan);plan['references']=['fp8'];write(self.plan,plan)
        with self.assertRaises(ValueError):f.check_plan(self.plan)

    def test_full_comparison_includes_net_residual(self):
        self.heldout();result=f.compare(self.plan,self.round,self.out)
        self.assertEqual(set(result['references']),{'fp8','head_ffn'})
        self.assertEqual(len(result['points']),7)
        opd=next(row for row in result['points'] if row['name']=='qad_opd')
        self.assertEqual(opd['encoding_budget']['physical']['total_bytes'],458)
        self.assertEqual(opd['encoding_budget']['known_alias_deduplicated']['total_bytes'],358)
        self.assertEqual(opd['successes'],96);self.assertTrue(result['environment_pairing_verified'])

    def test_rejects_second_episode_hash_mismatch(self):
        self.heldout();path=self.out/'heldout_head_ffn/task_results.json';data=f.load(path)
        data[TASKS[0]]['resets'][1]['restored_state_sha256']='f'*64;self.write_task_log(path.parent,TASKS[0],data[TASKS[0]]);write(path,data)
        with self.assertRaisesRegex(ValueError,'pairing'):f.compare(self.plan,self.round,self.out)

    def test_rejects_incomplete_reference(self):
        self.heldout();path=self.out/'heldout_fp8/task_results.json';data=f.load(path)
        data[TASKS[0]]['results'].pop();write(path,data)
        with self.assertRaisesRegex(ValueError,'Incomplete'):f.compare(self.plan,self.round,self.out)

    def test_rejects_wrong_residual_budget(self):
        self.heldout();path=self.root/'weights/qad_opd/merge_manifest.json';data=f.load(path)
        data['recovery_manifest']['trainable_parameters']=5;write(path,data)
        with self.assertRaisesRegex(ValueError,'residual'):f.compare(self.plan,self.round,self.out)

    def test_missing_dedup_metadata_requires_real_equal_alias_bytes(self):
        folder=self.root/'alias-test';folder.mkdir();payload=bytes(range(32))*2
        header={'embed.weight':{'dtype':'BF16','shape':[1,16],'data_offsets':[0,32]},
                'head.weight':{'dtype':'BF16','shape':[1,16],'data_offsets':[32,64]}}
        raw=json.dumps(header).encode();(folder/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+payload)
        write(folder/'model.safetensors.index.json',{'weight_map':{k:'model.safetensors' for k in header}})
        memory={'source_tensor_bytes':64,'target_full_checkpoint_bytes':40}
        write(folder/'ptq_recipe.json',{'memory':memory,'layers':{name:{'actual_method':'fp8'} for name in ('head','embed')}})
        inventory={'tied_aliases':{'head.weight':'embed.weight'},'recipes':{'fp8':{**memory,
                    'known_tied_alias_deduplicated':{'source_tensor_bytes':32,'all_tensor_elements':16,'target_full_bytes':20}}}}
        budget,proof=f.budget_for(folder,'fp8',inventory)
        self.assertEqual(budget['known_alias_deduplicated'],{'source_bytes':32,'base_bytes':20})
        self.assertEqual(proof['aliases'][0]['removed_target_bytes'],20)
        file=folder/'model.safetensors';file.write_bytes(file.read_bytes()[:-1]+b'X')
        with self.assertRaisesRegex(ValueError,'values differ'):f.budget_for(folder,'fp8',inventory)

    def test_rejects_raw_log_or_summary_disagreement(self):
        self.heldout();path=self.out/'heldout_fp8/summary.json';data=f.load(path);data['total_successes']+=1;write(path,data)
        with self.assertRaisesRegex(ValueError,'Summary'):f.compare(self.plan,self.round,self.out)
        data['total_successes']-=1;write(path,data)
        log=self.out/'heldout_fp8'/(TASKS[0]+'.log');log.write_text(log.read_text()+'unrecorded change')
        with self.assertRaisesRegex(ValueError,'raw log'):f.compare(self.plan,self.round,self.out)

    def test_dry_run_does_not_launch_or_create_output(self):
        self.heldout();destination=self.root/'dry-run-output'
        with contextlib.redirect_stdout(io.StringIO()),mock.patch.object(f.subprocess,'run',side_effect=AssertionError('CPU test must never execute GPU')):
            f.run(self.plan,self.round,destination,5595,dry_run=True)
        self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()
