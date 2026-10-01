"""Synthetic CPU-only v3 orchestrator cost fixtures, never real experiment evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tests import test_training_costs as legacy

cost=legacy.cost;ROOT=legacy.ROOT;write=legacy.write


def read(path):return json.loads(path.read_text())
def full_identity(path):return {'path':str(path),**cost.identity(path)}


def evaluation_fixture(folder,checkpoint,part,purpose,score,sha):
    evaluation=cost.evaluation_helpers(ROOT/'eval/run_recovery_eval.py')
    folder.mkdir(parents=True,exist_ok=True);rows={};signatures=[]
    for ti,task in enumerate(evaluation.TASKS):
        resets=[{'episode_index':i,'seed':part['seed']+1000*ti+i,'init_state_index':bank,'settle_steps':10,
                 'initial_state_sha256':f'{ti*100+i:064x}','restored_state_sha256':f'{ti*100+i+3000:064x}',
                 'init_state_bank_sha256':f'{ti:064x}'} for i,bank in enumerate(part['init_state_indices'])]
        outcomes=[ti*len(resets)+i<score for i in range(len(resets))]
        log=folder/(task+'.log');log.write_text(''.join('FP4VLA_EPISODE_RESET '+json.dumps(z)+'\n' for z in resets)+
            "results: ('fixture', "+repr(outcomes)+")\n")
        rows[task]=evaluation.parse_log(log)|{'returncode':0,'seed':part['seed']+1000*ti}
        signatures.extend({'task':task,**z} for z in resets)
    write(folder/'eval_manifest.json',{'purpose':purpose,'seed':part['seed'],'episodes':part['episodes_per_task'],
        'init_state_indices':part['init_state_indices'],'tasks':evaluation.TASKS,'n_envs':1,'settle_steps':10,
        'n_action_steps':8,'max_episode_steps':720,'protocol_sha256':sha,'checkpoint':str(checkpoint),
        'initial_state_protocol':'libero10_official_bank_v1'})
    write(folder/'task_results.json',rows)
    write(folder/'summary.json',{'purpose':purpose,'seed':part['seed'],'tasks_complete':10,'total_successes':score,
        'total_episodes':10*part['episodes_per_task'],'macro_success_rate':score/(10*part['episodes_per_task']),
        'wall_seconds_including_server_loads':30.})
    return {'evaluation_path':str(folder),'selection_source':'development_only',
        'evaluation_identity':{name:full_identity(folder/name) for name in ('eval_manifest.json','task_results.json','summary.json')},
        'raw_log_identities':{task:full_identity(folder/(task+'.log')) for task in evaluation.TASKS},
        'pairing_sha256':hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest(),
        'successes':score,'episodes':10*part['episodes_per_task'],'macro_success_rate':score/(10*part['episodes_per_task'])}


def v3_run(root):
    # Reuse opaque legacy CPU tensors, but don't collect any old-format output.
    with patch.object(cost,'collect',return_value={}):_,private=legacy.synthetic_package(root)
    protocol_path=ROOT/'exp/recovery_protocol_v3_high_fp4.json';protocol=read(protocol_path);sha=cost.identity(protocol_path)['sha256']
    run=private/'run';art=run/'artifacts';art.mkdir(parents=True);(run/'logs').mkdir();(run/'stages').mkdir()
    moves={private/'train_qad':art/'train_qad_lr_5e-05',private/'qad':art/'merge_qad_lr_5e-05',
        private/'round/continued_qad':art/'train_continued_qad',private/'round/continued_qad_merged':art/'merge_continued_qad',
        private/'round/qad_opd':art/'train_opd_025',private/'round/qad_opd_merged':art/'merge_opd_025',
        private/'round/collection':art/'collection_qad',
        private/'round/teacher_probes.pt':art/'teacher_cache/teacher_probes.pt',
        private/'round/teacher_probes.json':art/'teacher_cache/teacher_probes.json'}
    for old,new in moves.items():new.parent.mkdir(parents=True,exist_ok=True);old.rename(new)
    for path in private.rglob('*.json'):
        data=path.read_text()
        for old,new in sorted(moves.items(),key=lambda item:-len(str(item[0]))):data=data.replace(str(old),str(new))
        path.write_text(data)
    stages={};merged={};cache=art/'teacher_cache/teacher_probes.pt'
    for arm,tag in (('qad','qad_lr_5e-05'),('continued_qad','continued_qad'),('qad_opd','opd_025')):
        train=art/('train_'+tag);model=art/('merge_'+tag);steps=500 if arm=='qad' else 100
        manifest=read(train/'recovery_manifest.json');manifest.update(protocol_sha256=sha,seed=20260929,train_seed=20260929)
        weight=.25 if arm=='qad_opd' else 0.
        manifest['probe_weight']=weight
        write(train/'recovery_manifest.json',manifest);write(train/f'checkpoint-{steps}/recovery_manifest.json',manifest)
        write(train/'orchestrator_training_request.json',{'protocol_sha256':sha,'environment':{
            'QAD_OUT':str(train),'QAD_STEPS':str(steps),'QAD_LR':'5e-05','TRAIN_SEED':'20260929',
            'QAD_OPD_MSE_W':str(weight),'QAD_ACTIVATION_CHECKPOINTING':'1'}})
        export=read(model/'merge_manifest.json');export['recovery_manifest']=manifest
        export['recovery_manifest_sha256']=cost.identity(train/f'checkpoint-{steps}/recovery_manifest.json')['sha256']
        write(model/'merge_manifest.json',export);merged[arm]=model
        marker={'output':str(train),'status':'complete','protocol_sha256':sha};stages['train_'+tag]=marker
        write(run/'stages'/('train_'+tag+'.json'),marker)
        text='\n' if arm!='qad_opd' else ''.join(f'[opd] step={step} probe=0 mse=0.01 weight=0.25 microbatch=1\n'
                                                   for step in (4,24,44,64,84) for _ in range(16))
        (run/'logs'/('train_'+tag+'.log')).write_text(text)
    collection=art/'collection_qad'
    evaluation_fixture(collection,merged['qad'],protocol['partitions']['collection'],'collection',20,sha)
    meta=read(cache.with_suffix('.json'));sources=[];samples=[]
    for task in cost.evaluation_helpers(ROOT/'eval/run_recovery_eval.py').TASKS:
        obs=collection/'observations'/task;write(obs/'capture_counts.json',{'saved':16,'per_task':{task:16},'calls':16})
        for i in range(16):
            path=obs/f'sample_{i:06d}.pt';path.write_bytes(f'opaque {task}/{i}'.encode());sources.append(full_identity(path))
            samples.append({'provenance':{'source_kind':'student_rollout','student_checkpoint':str(merged['qad']),
                'student_statistics_sha256':meta['teacher_statistics_sha256']}})
    meta.update(count=160,requested_count=160,source_observation_files=sources);write(cache.with_suffix('.json'),meta)
    tensor_cache={'version':3,'metadata':meta,'samples':samples}
    selections={}
    for kind,field,values in (('qad','learning_rate',[.00005,.0001]),('opd','opd_weight',[.25,1.])):
        rows={str(value):{field:value,**evaluation_fixture(art/f'dev_{kind}_{i}',merged['qad' if kind=='qad' else 'qad_opd'],
                protocol['partitions']['development'],'development',45-i,sha)} for i,value in enumerate(values)}
        selections[kind]={'protocol_sha256':sha,'selection_uses_heldout':False,'environment_pairing_verified':True,
            'candidates':rows,'selected_'+field:values[0]}
    heldout=art/'heldout_round';heldout.mkdir()
    for arm in ('bf16','qad','continued_qad','qad_opd'):
        write(heldout/('heldout_'+arm)/'eval_manifest.json',{'checkpoint':str(private/'bf16' if arm=='bf16' else merged[arm]),
            'collection_manifest':str(collection/'eval_manifest.json')})
    write(heldout/'paired_comparison.json',{'environment_pairing_verified':True,
        'arms':{arm:{'count':100} for arm in ('bf16','ptq','qad','continued_qad','qad_opd')}})
    def model_id(folder):return {'path':str(folder),'shards':[{'name':p.name,**cost.identity(p)} for p in folder.glob('*.safetensors')]}
    final={'format':'high_fp4_v3_final_manifest','protocol_sha256':sha,'selection_uses_heldout':False,
        'selected_qad_learning_rate':.00005,'selected_opd_weight':.25,'qad_selection':selections['qad'],'opd_selection':selections['opd'],
        'required_arms':['bf16','ptq','qad','continued_qad','qad_opd'],'heldout_round':str(heldout),
        'heldout_comparison':full_identity(heldout/'paired_comparison.json'),
        'selected_qad_checkpoint_identity':model_id(art/'train_qad_lr_5e-05/checkpoint-500'),
        'selected_pressure_checkpoint':str(private/'ptq'),
        'selected_qad_model_identity':model_id(merged['qad']),'selected_continued_model_identity':model_id(merged['continued_qad']),
        'selected_opd_model_identity':model_id(merged['qad_opd'])}
    write(run/'final_manifest.json',final)
    write(run/'run_manifest.json',{'status':'complete','output_layout':'stable_paths_v2','protocol_sha256':sha,
        'selection_uses_heldout':False,'implementation_sha256':cost.identity(ROOT/'exp/run_high_fp4_v3.py')['sha256'],'stages':stages})
    return run,private,protocol_path,tensor_cache


class V3TrainingCosts(unittest.TestCase):
    def test_current_v7_manifest_format_is_auditable(self):
        self.assertIsNotNone(cost.FINAL_FORMAT_RE.fullmatch('mixed_pressure_v7_final_manifest'))
        self.assertIsNotNone(cost.FINAL_FORMAT_RE.fullmatch('high_fp4_category_final_manifest'))
        self.assertIsNone(cost.FINAL_FORMAT_RE.fullmatch('mixed_pressure_v8_final_manifest'))

    def test_v7_nested_evaluation_contract_normalizes_for_cost_audit(self):
        protocol = json.loads((ROOT / 'exp/recovery_protocol_v7_mixed_pressure.json').read_text())
        normalized = cost.normalized_protocol(protocol, {
            'selected_qad_learning_rate': protocol['selection']['qad_learning_rates'][0],
            'selected_opd_weight': protocol['selection']['opd_weights'][0],
        })
        self.assertEqual(normalized['initial_states']['protocol'],
                         protocol['evaluation_contract']['initial_state_protocol'])
        self.assertEqual(normalized['heldout']['episodes_per_task'], 10)

    def test_orchestrator_schema_survives_protocol_label_bump(self):
        protocol = read(ROOT/'exp/recovery_protocol_v3_high_fp4.json')
        protocol['version'] = 4
        protocol['id'] = 'high-fp4-stress-v4'
        normalized = cost.normalized_protocol(protocol, {
            'selected_qad_learning_rate': protocol['selection']['qad_learning_rates'][0],
            'selected_opd_weight': protocol['selection']['opd_weights'][0],
        })
        self.assertEqual(normalized['collection'], protocol['partitions']['collection'])
        self.assertEqual(normalized['selected_learning_rate'], protocol['selection']['qad_learning_rates'][0])
        self.assertEqual(normalized['continuation']['opd_weight'], protocol['selection']['opd_weights'][0])

    def test_actual_orchestrator_layout_and_portable_public_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,private,protocol,cache=v3_run(root);out=root/'costs'
            with patch.dict(sys.modules,{'torch':types.SimpleNamespace(load=lambda *a,**kw:cache)}):
                result=cost.collect(None,None,None,out,protocol,orchestrator_run=run)
            self.assertEqual(result['collection']['episodes'],40)
            self.assertEqual(result['collection']['captured_observations'],160)
            self.assertEqual(result['training']['qad_opd']['scheduled_teacher_backward_passes'],400)
            self.assertEqual(result['selected_recovery_settings']['opd_weight'],.25)
            self.assertIn('Hyperparameter-search',result['cost_scope'])
            source=run/'final_manifest.json'
            self.assertEqual(source.read_bytes(),(out/'orchestration/final_manifest.json').read_bytes())
            shutil.rmtree(private)
            self.assertEqual(cost.verify_published(out),result)
            standalone=root/'standalone.py';shutil.copyfile(ROOT/'paper/collect_training_costs.py',standalone)
            checked=subprocess.run([sys.executable,'-I','-S',str(standalone),'--verify-published',str(out)],
                                   cwd=root,text=True,capture_output=True)
            self.assertEqual(checked.returncode,0,checked.stderr)

    def test_unfinished_orchestrator_and_forged_winner_refuse_output(self):
        for mutation in ('status','winner'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);run,_,protocol,_=v3_run(root);out=root/'costs'
                if mutation=='status':
                    path=run/'run_manifest.json';data=read(path);data['status']='running'
                else:
                    path=run/'final_manifest.json';data=read(path);data['opd_selection']['selected_opd_weight']=1.
                write(path,data)
                with self.assertRaises(ValueError):cost.collect(None,None,None,out,protocol,orchestrator_run=run)
                self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
