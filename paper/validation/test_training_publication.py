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
PROTOCOL=P.parent/'exp/recovery_protocol_v11_w4a4_category.json'


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw);paper=root/'paper';folder=paper/'evidence/training'
        folder.mkdir(parents=True);(root/'exp').mkdir()
        protocol=root/'exp/recovery_protocol_v11_w4a4_category.json'
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
                      'w4a4_enabled':True,
                      'init_adapter':None if arm=='qad' else runtime['checkpoints']['qad']['path']}
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
            protocol=folder.parents[2]/'exp/recovery_protocol_v11_w4a4_category.json'
            data=json.loads(protocol.read_text());data['w4a4']=False
            protocol.write_text(json.dumps(data));costs['protocol_sha256']=validation.sha(protocol)
            with self.assertRaisesRegex(RuntimeError,'frozen v11 W4A4 category protocol'):
                validation.check_training_costs(runtime,set())


if __name__=='__main__':unittest.main(verbosity=2)
