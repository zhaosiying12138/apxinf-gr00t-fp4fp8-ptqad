#!/usr/bin/env python3
"""CPU-only integration checks for public cost evidence and evaluated weights.

All numeric records below are temporary fixtures. The collector owns tests of
the metadata-only arithmetic verifier; these test its publication binding.
"""
from contextlib import contextmanager
import copy
import hashlib
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
PROTOCOL=P.parent/'exp/recovery_protocol_v12_rtn_w4a4.json'
QAD_PRODUCER=P.parent/'rl/lora_qad.py'


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw);paper=root/'paper';folder=paper/'evidence/training'
        folder.mkdir(parents=True);(root/'exp').mkdir()
        protocol=root/'exp/recovery_protocol_v12_rtn_w4a4.json'
        protocol.write_bytes(PROTOCOL.read_bytes())
        settings=json.loads(protocol.read_text())['selection']
        (paper/'collect_training_costs.py').write_text('# fixture only\n')
        residual={'linear_modules':2,'tensor_elements':128,
                  'rank':settings['rank'],'alpha':settings['alpha'],'scope':settings['recovery_scope']}
        (paper/'evidence/recipe_inventory.json').write_text(json.dumps({'recovery_residual':residual}))
        identity=lambda code:{'bytes':100,'sha256':code*64}
        runtime={'checkpoints':{arm:{'path':str(root/'not-published-weights'/arm),
            'files':[{'name':'model.safetensors',**identity(code)}]}
            for arm,code in [('bf16','a'),('ptq','b'),('qad','c'),('continued_qad','d'),('qad_opd','e')]}}
        records=[]
        def write(relative,value,original):
            path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
            records.append({'published_path':relative,'original_absolute_path':str(original),
                'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'role':'temporary fixture'})
        for arm in ('qad','continued_qad','qad_opd'):
            # Publication binding starts after the collector has verified
            # training arithmetic. Both continuation fixtures start from the
            # same selected QAD adapter and use the frozen additional budget.
            recovery={'trainable_parameters':residual['tensor_elements'],
                      **{key:residual[key] for key in ('rank','alpha','scope')},
                      'w4a4_enabled':True,'protocol_sha256':validation.sha(protocol),
                      'initial_adapter':None if arm=='qad' else runtime['checkpoints']['qad']['path']}
            env={'QAD_W4A4':'1'}
            if arm=='qad':
                recovery['recovery_source_sha256']=dict(validation.LEGACY_STRICT_QAD_SOURCES)
            else:
                recovery.update(max_grad_norm=.25,f16_activation_saturation=True)
                env.update(QAD_MAX_GRAD_NORM='0.25',FP4VLA_SATURATE_F16_ACTIVATIONS='1')
            write('stages/'+arm+'/recovery_manifest.json',recovery,root/arm/'recovery_manifest.json')
            write('stages/'+arm+'/orchestrator_training_request.json',
                  {'protocol_sha256':validation.sha(protocol),'environment':env,
                   'initial_adapter_identity':None if arm=='qad' else {'path':recovery['initial_adapter']}},
                  root/arm/'orchestrator_training_request.json')
            merge={'output_weights':{'model.safetensors':{k:v for k,v in runtime['checkpoints'][arm]['files'][0].items() if k!='name'}},
                   'base_weights':{'model.safetensors':identity('b')},
                   'lora_pairs':residual['linear_modules'],'recovery_manifest':recovery}
            write('exports/'+arm+'/merge_manifest.json',merge,Path(runtime['checkpoints'][arm]['path'])/'merge_manifest.json')
        write('teacher/teacher_probes.json',{'teacher':runtime['checkpoints']['bf16']['path'],
            'teacher_weights':{'model.safetensors':identity('a')}},root/'private-round/teacher_probes.json')
        write('collection/eval_manifest.json',{'checkpoint':runtime['checkpoints']['qad']['path']},
              root/'private-round/collection/eval_manifest.json')
        (folder/'evidence_manifest.json').write_text(json.dumps({'status':'complete','files':records}))
        costs={'status':'complete','protocol_sha256':validation.sha(protocol),
               'training':{arm:{'status':'completed','optimizer_steps':settings[
                   'qad_optimizer_steps' if arm=='qad' else 'continuation_optimizer_steps']}
                   for arm in ('qad','continued_qad','qad_opd')}}
        (folder/'costs.json').write_text(json.dumps(costs))
        fake=SimpleNamespace(verify_published=lambda path:copy.deepcopy(costs))
        with patch.object(validation,'P',paper),patch.dict(sys.modules,{'collect_training_costs':fake}):
            yield runtime,folder,costs,fake


def rewrite_record(folder, relative, change):
    """Mutate a fixture while retaining a valid archive hash for semantic tests."""
    path=folder/relative;value=json.loads(path.read_text());change(value)
    path.write_text(json.dumps(value))
    mapping=json.loads((folder/'evidence_manifest.json').read_text())
    row=next(row for row in mapping['files'] if row['published_path']==relative)
    row.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (folder/'evidence_manifest.json').write_text(json.dumps(mapping))


class TrainingPublicationTests(unittest.TestCase):
    def test_verified_public_costs_bind_to_all_evaluated_weights_without_private_files(self):
        with fixture() as (runtime,folder,costs,_):
            self.assertFalse(Path(runtime['checkpoints']['qad']['path']).exists())
            inputs=set()
            self.assertEqual(validation.check_training_costs(runtime,inputs),costs)
            self.assertIn(folder/'costs.json',inputs)
            self.assertIn(folder/'teacher/teacher_probes.json',inputs)

    def test_any_recovery_output_mismatch_rejected(self):
        for arm in ('qad','continued_qad','qad_opd'):
            with self.subTest(arm=arm),fixture() as (runtime,_,_,_):
                runtime['checkpoints'][arm]['files'][0]['sha256']='f'*64
                with self.assertRaisesRegex(RuntimeError,'export weight identity'):
                    validation.check_training_costs(runtime,set())

    def test_ptq_source_mismatch_rejected(self):
        with fixture() as (runtime,_,_,_):
            runtime['checkpoints']['ptq']['files'][0]['sha256']='f'*64
            with self.assertRaisesRegex(RuntimeError,'base weight identity'):
                validation.check_training_costs(runtime,set())

    def test_teacher_mismatch_rejected(self):
        with fixture() as (runtime,_,_,_):
            runtime['checkpoints']['bf16']['files'][0]['sha256']='f'*64
            with self.assertRaisesRegex(RuntimeError,'Teacher identity'):
                validation.check_training_costs(runtime,set())

    def test_evaluated_export_path_mismatch_rejected(self):
        with fixture() as (runtime,_,_,_):
            runtime['checkpoints']['qad']['path']+='_different'
            with self.assertRaisesRegex(RuntimeError,'evaluated checkpoint'):
                validation.check_training_costs(runtime,set())

    def test_other_collection_student_rejected_even_with_valid_metadata_hash(self):
        with fixture() as (runtime,folder,_,_):
            path=folder/'collection/eval_manifest.json'
            path.write_text(json.dumps({'checkpoint':runtime['checkpoints']['continued_qad']['path']}))
            mapping=json.loads((folder/'evidence_manifest.json').read_text())
            row=next(row for row in mapping['files'] if row['published_path']=='collection/eval_manifest.json')
            row.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (folder/'evidence_manifest.json').write_text(json.dumps(mapping))
            with self.assertRaisesRegex(RuntimeError,'Collection student identity'):
                validation.check_training_costs(runtime,set())

    def test_changed_archived_input_rejected_even_after_verifier_return(self):
        with fixture() as (runtime,folder,_,_):
            (folder/'teacher/teacher_probes.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Changed evidence'):
                validation.check_training_costs(runtime,set())

    def test_public_verifier_failure_never_ignored(self):
        with fixture() as (runtime,_,_,verifier):
            def reject(path):raise ValueError('cost arithmetic mismatch')
            verifier.verify_published=reject
            with self.assertRaisesRegex(ValueError,'arithmetic mismatch'):
                validation.check_training_costs(runtime,set())

    def test_recovery_budget_mismatch_rejected_even_with_valid_metadata_hash(self):
        for key in ('lora_pairs','trainable_parameters','rank','alpha','scope'):
            with self.subTest(field=key),fixture() as (runtime,folder,_,_):
                relative='exports/qad_opd/merge_manifest.json'
                path=folder/relative;data=json.loads(path.read_text())
                target=data if key=='lora_pairs' else data['recovery_manifest']
                target[key]='head+lang_all' if key=='scope' else target[key]+1
                path.write_text(json.dumps(data))
                mapping=json.loads((folder/'evidence_manifest.json').read_text())
                row=next(row for row in mapping['files'] if row['published_path']==relative)
                row.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                (folder/'evidence_manifest.json').write_text(json.dumps(mapping))
                with self.assertRaisesRegex(RuntimeError,'plotted LoRA budget'):
                    validation.check_training_costs(runtime,set())

    def test_non_w4a4_protocol_rejected_even_with_matching_protocol_hash(self):
        with fixture() as (runtime,folder,costs,_):
            protocol=folder.parents[2]/'exp/recovery_protocol_v12_rtn_w4a4.json'
            data=json.loads(protocol.read_text());data['w4a4']=False
            protocol.write_text(json.dumps(data));costs['protocol_sha256']=validation.sha(protocol)
            with self.assertRaisesRegex(RuntimeError,'frozen W4A4 protocol'):
                validation.check_training_costs(runtime,set())

    def test_initial_qad_default_is_bound_to_archived_producer_without_rewriting_receipts(self):
        with fixture() as (runtime,folder,costs,_):
            source=folder/'stages/qad/source/rl/lora_qad.py'
            source.parent.mkdir(parents=True)
            source.write_bytes(QAD_PRODUCER.read_bytes())
            mapping=json.loads((folder/'evidence_manifest.json').read_text())
            mapping['files'].append({'published_path':source.relative_to(folder).as_posix(),
                'original_absolute_path':str(QAD_PRODUCER), 'bytes':source.stat().st_size,
                'sha256':validation.sha(source),'role':'synthetic fixture producer copy'})
            (folder/'evidence_manifest.json').write_text(json.dumps(mapping))
            receipt={'initial_adapter':None,'max_grad_norm':1.0,'f16_activation_saturation':False,
                'recovery_source_sha256':{'rl/lora_qad.py':validation.sha(source)},
                'environment_summary':{'variables':{'QAD_W4A4':'1','QAD_MAX_GRAD_NORM':None,
                                                   'FP4VLA_SATURATE_F16_ACTIVATIONS':'0'}}}
            rewrite_record(folder,'stages/qad/recovery_manifest.json',lambda data:data.update(receipt))
            rewrite_record(folder,'exports/qad/merge_manifest.json',lambda data:data['recovery_manifest'].update(receipt))
            rewrite_record(folder,'stages/qad/orchestrator_training_request.json',
                lambda data:data['environment'].update(FP4VLA_SATURATE_F16_ACTIVATIONS='0'))
            before={path:path.read_bytes() for path in folder.rglob('*') if path.is_file()}
            self.assertEqual(validation.check_training_costs(runtime,set()),costs)
            self.assertEqual(before,{path:path.read_bytes() for path in before})

    def test_initial_qad_default_rejects_unbound_or_conflicting_provenance(self):
        protocol_sha=validation.sha(PROTOCOL)
        recovery={'protocol_sha256':protocol_sha,'w4a4_enabled':True,'initial_adapter':None,
            'max_grad_norm':1.0,'f16_activation_saturation':False,
            'recovery_source_sha256':{'rl/lora_qad.py':validation.sha(QAD_PRODUCER)},
            'environment_summary':{'variables':{'QAD_W4A4':'1','QAD_MAX_GRAD_NORM':None,
                                               'FP4VLA_SATURATE_F16_ACTIVATIONS':'0'}}}
        request={'protocol_sha256':protocol_sha,'initial_adapter_identity':None,'environment':{'QAD_W4A4':'1',
                     'FP4VLA_SATURATE_F16_ACTIVATIONS':'0'}}
        for mutation in ('unknown_producer','missing_source','changed_source','wrong_default','missing_summary',
                         'unobserved_env','conflicting_env','empty_override','null_override','continuation',
                         'missing_initial_adapter','nonempty_initial_adapter','conflicting_init_alias','only_init_alias',
                         'missing_initial_identity','conflicting_initial_identity','adapter_env','null_adapter_env',
                         'empty_adapter_env'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as raw:
                source=Path(raw)/'lora_qad.py';source.write_bytes(QAD_PRODUCER.read_bytes())
                rec,req=copy.deepcopy(recovery),copy.deepcopy(request)
                arm='qad'
                if mutation=='unknown_producer':rec['recovery_source_sha256']['rl/lora_qad.py']='a'*64
                elif mutation=='missing_source':source=None
                elif mutation=='changed_source':source.write_bytes(source.read_bytes()+b'\n')
                elif mutation=='wrong_default':rec['max_grad_norm']=.25
                elif mutation=='missing_summary':rec.pop('environment_summary')
                elif mutation=='unobserved_env':rec['environment_summary']['variables'].pop('QAD_MAX_GRAD_NORM')
                elif mutation=='conflicting_env':rec['environment_summary']['variables']['QAD_MAX_GRAD_NORM']='1.0'
                elif mutation=='empty_override':req['environment']['QAD_MAX_GRAD_NORM']=''
                elif mutation=='null_override':req['environment']['QAD_MAX_GRAD_NORM']=None
                elif mutation=='continuation':arm='qad_opd'
                elif mutation=='missing_initial_adapter':rec.pop('initial_adapter')
                elif mutation=='nonempty_initial_adapter':rec['initial_adapter']='/continuation/adapter'
                elif mutation=='conflicting_init_alias':rec['init_adapter']='/continuation/adapter'
                elif mutation=='only_init_alias':rec['init_adapter']=rec.pop('initial_adapter')
                elif mutation=='missing_initial_identity':req.pop('initial_adapter_identity')
                elif mutation=='conflicting_initial_identity':req['initial_adapter_identity']={'path':'/continuation/adapter'}
                elif mutation=='adapter_env':req['environment']['QAD_INIT_ADAPTER']='/continuation/adapter'
                elif mutation=='null_adapter_env':req['environment']['QAD_INIT_ADAPTER']=None
                elif mutation=='empty_adapter_env':req['environment']['QAD_INIT_ADAPTER']=''
                with self.assertRaisesRegex(RuntimeError,'max_grad_norm|numerical request'):
                    validation.check_training_numerical_environment(rec,req,arm,protocol_sha,
                                                                    producer_source=source)

    def test_continuation_missing_numerical_request_rejected_without_summary(self):
        for arm in ('continued_qad','qad_opd'):
            with self.subTest(arm=arm),fixture() as (runtime,folder,_,_):
                rewrite_record(folder,'stages/'+arm+'/orchestrator_training_request.json',
                               lambda data:data['environment'].pop('QAD_MAX_GRAD_NORM'))
                with self.assertRaisesRegex(RuntimeError,'numerical request differs'):
                    validation.check_training_costs(runtime,set())

    def test_conflicting_numerical_request_rejected(self):
        for key,value in [('QAD_MAX_GRAD_NORM','1.0'),('FP4VLA_SATURATE_F16_ACTIVATIONS','0'),
                          ('QAD_W4A4','0')]:
            with self.subTest(key=key),fixture() as (runtime,folder,_,_):
                rewrite_record(folder,'stages/qad_opd/orchestrator_training_request.json',
                               lambda data:data['environment'].update({key:value}))
                with self.assertRaisesRegex(RuntimeError,'numerical request'):
                    validation.check_training_costs(runtime,set())

    def test_legacy_qad_request_cannot_enable_unrecorded_numerical_setting(self):
        with fixture() as (runtime,folder,_,_):
            rewrite_record(folder,'stages/qad/orchestrator_training_request.json',
                           lambda data:data['environment'].update(FP4VLA_SATURATE_F16_ACTIVATIONS='1'))
            with self.assertRaisesRegex(RuntimeError,'Legacy QAD request conflicts'):
                validation.check_training_costs(runtime,set())

    def test_numerical_summary_must_agree_when_present(self):
        for saturation in ('1','0'):
            with self.subTest(saturation=saturation),fixture() as (runtime,folder,_,_):
                summary={'variables':{'QAD_W4A4':'1','QAD_MAX_GRAD_NORM':'0.25',
                                     'FP4VLA_SATURATE_F16_ACTIVATIONS':saturation}}
                rewrite_record(folder,'stages/qad_opd/recovery_manifest.json',
                               lambda data:data.update(environment_summary=summary))
                rewrite_record(folder,'exports/qad_opd/merge_manifest.json',
                               lambda data:data['recovery_manifest'].update(environment_summary=summary))
                if saturation=='1':
                    validation.check_training_costs(runtime,set())
                else:
                    with self.assertRaisesRegex(RuntimeError,'environment summary differs'):
                        validation.check_training_costs(runtime,set())


if __name__=='__main__':unittest.main(verbosity=2)
