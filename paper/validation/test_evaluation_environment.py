#!/usr/bin/env python3
"""CPU-only publication checks of inference switches versus training receipts."""
from contextlib import contextmanager
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
import validate_publication as validation


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw);paper=root/'paper';paper.mkdir()
        protocol=root/'protocol.json';protocol.write_text(json.dumps({'version':12}))
        runtime={'checkpoints':{}};records=[]
        archive=paper/'evidence/training';archive.mkdir(parents=True)
        for arm in ('bf16','ptq','qad','continued_qad','qad_opd'):
            quant=arm!='bf16';recovery=arm in ('qad','continued_qad','qad_opd')
            checkpoint=root/'private-checkpoints-not-present'/arm
            weights={'model.safetensors':{'bytes':123,'sha256':hashlib.sha256(arm.encode()).hexdigest()}}
            runtime['checkpoints'][arm]={'path':str(checkpoint),
                'files':[{'name':name,**row} for name,row in weights.items()]}
            contract={'max_grad_norm':None,'f16_activation_saturation':False}
            if recovery:
                receipt={'protocol_sha256':validation.sha(protocol),'w4a4_enabled':True}
                if arm=='qad':
                    receipt.update(initial_adapter=None,
                        recovery_source_sha256=dict(validation.LEGACY_STRICT_QAD_SOURCES))
                    contract['f16_activation_saturation']=None
                else:
                    receipt.update(max_grad_norm=.25,f16_activation_saturation=True)
                    contract={'max_grad_norm':.25,'f16_activation_saturation':True}
                relative='exports/'+arm+'/merge_manifest.json'
                exported=archive/relative;exported.parent.mkdir(parents=True)
                exported.write_text(json.dumps({'output_weights':weights,'recovery_manifest':receipt}))
                records.append({'published_path':relative,'original_absolute_path':str(checkpoint/'merge_manifest.json'),
                                'bytes':exported.stat().st_size,'sha256':validation.sha(exported)})
            folder=paper/'evidence'/('heldout_'+arm);folder.mkdir(parents=True)
            (folder/'eval_manifest.json').write_text(json.dumps({
                'environment_summary':{'variables':{
                    'FP4VLA_QUANT':'0','FP4VLA_W4A4':'1' if quant else '0',
                    'FP4VLA_W4A4_ADAPTER':'1' if recovery else '0',
                    'FP4VLA_SATURATE_F16_ACTIVATIONS':'1' if quant else '0',
                    'PTQAD_ZMQ_TIMEOUT_MS':'120000'}},
                'recovery_contract':contract}))
        (archive/'evidence_manifest.json').write_text(json.dumps({'status':'complete','files':records}))
        with patch.object(validation,'P',paper):
            yield runtime,protocol,paper


def change(path, mutate):
    data=json.loads(path.read_text());mutate(data);path.write_text(json.dumps(data))


def change_archived_receipt(paper, arm, mutate):
    relative='exports/'+arm+'/merge_manifest.json';folder=paper/'evidence/training'
    path=folder/relative
    change(path,lambda data:mutate(data['recovery_manifest']))
    manifest=folder/'evidence_manifest.json';data=json.loads(manifest.read_text())
    row=next(row for row in data['files'] if row['published_path']==relative)
    row.update(bytes=path.stat().st_size,sha256=validation.sha(path))
    manifest.write_text(json.dumps(data))


class EvaluationEnvironmentTests(unittest.TestCase):
    def test_source_bound_legacy_qad_and_nonrecovery_ptq_are_accepted_without_rewriting(self):
        with fixture() as (runtime,protocol,paper):
            receipt=paper/'evidence/heldout_qad/eval_manifest.json'
            before=receipt.read_bytes();inputs=set()
            validation.check_evaluation_environment(inputs,runtime,protocol)
            self.assertEqual(receipt.read_bytes(),before)
            self.assertEqual(len(inputs),9)
            self.assertTrue(all(not Path(row['path']).exists() for row in runtime['checkpoints'].values()))

    def test_arbitrary_legacy_qad_producer_is_rejected(self):
        with fixture() as (runtime,protocol,paper):
            change_archived_receipt(paper,'qad',lambda d:d['recovery_source_sha256'].update(
                {'quant/native_activation.py':'f'*64}))
            with self.assertRaisesRegex(RuntimeError,'verified strict producer'):
                validation.check_evaluation_environment(set(),runtime,protocol)

    def test_legacy_qad_eval_cannot_claim_a_new_training_setting(self):
        for key,value in [('max_grad_norm',.25),('f16_activation_saturation',True),
                          ('f16_activation_saturation',False)]:
            with self.subTest(key=key,value=value),fixture() as (runtime,protocol,paper):
                receipt=paper/'evidence/heldout_qad/eval_manifest.json'
                change(receipt,lambda d:d['recovery_contract'].update({key:value}))
                with self.assertRaisesRegex(RuntimeError,'differs from checkpoint'):
                    validation.check_evaluation_environment(set(),runtime,protocol)

    def test_continuation_cannot_use_legacy_missing_or_nonfinite_clip(self):
        for arm in ('continued_qad','qad_opd'):
            for value in (None,0,True,float('inf'),float('nan')):
                with self.subTest(arm=arm,value=value),fixture() as (runtime,protocol,paper):
                    change_archived_receipt(paper,arm,lambda d:d.update(max_grad_norm=value))
                    with self.assertRaisesRegex(RuntimeError,'finite max_grad_norm'):
                        validation.check_evaluation_environment(set(),runtime,protocol)

    def test_continuation_saturation_receipt_required_and_not_truthy_integer(self):
        for value in (None,1):
            with self.subTest(value=value),fixture() as (runtime,protocol,paper):
                receipt=paper/'evidence/heldout_qad_opd/eval_manifest.json'
                change(receipt,lambda d:d['recovery_contract'].update(f16_activation_saturation=value))
                with self.assertRaisesRegex(RuntimeError,'invalid numerical types'):
                    validation.check_evaluation_environment(set(),runtime,protocol)

    def test_inference_switch_is_checked_separately_from_ptq_training_contract(self):
        with fixture() as (runtime,protocol,paper):
            receipt=paper/'evidence/heldout_ptq/eval_manifest.json'
            change(receipt,lambda d:d['environment_summary']['variables'].update(
                FP4VLA_SATURATE_F16_ACTIVATIONS='0'))
            with self.assertRaisesRegex(RuntimeError,'Evaluation environment summary differs'):
                validation.check_evaluation_environment(set(),runtime,protocol)

    def test_changed_archived_export_is_rejected_without_using_private_checkpoint(self):
        with fixture() as (runtime,protocol,paper):
            path=paper/'evidence/training/exports/qad/merge_manifest.json'
            change(path,lambda d:d['recovery_manifest'].update(max_grad_norm=.25))
            with self.assertRaisesRegex(RuntimeError,'Changed evidence'):
                validation.check_evaluation_environment(set(),runtime,protocol)

    def test_archive_export_must_match_runtime_weights_and_original_path(self):
        for mutation in ('weights','path'):
            with self.subTest(mutation=mutation),fixture() as (runtime,protocol,paper):
                if mutation=='weights':runtime['checkpoints']['qad']['files'][0]['sha256']='f'*64
                else:runtime['checkpoints']['qad']['path']+='_other'
                with self.assertRaisesRegex(RuntimeError,'differs|differ'):
                    validation.check_evaluation_environment(set(),runtime,protocol)

    def test_missing_or_ambiguous_archived_export_is_rejected(self):
        for duplicate in (False,True):
            with self.subTest(duplicate=duplicate),fixture() as (runtime,protocol,paper):
                path=paper/'evidence/training/evidence_manifest.json'
                data=json.loads(path.read_text())
                if duplicate:data['files'].append(dict(data['files'][0]))
                else:data['files'].pop(0)
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(RuntimeError,'missing or ambiguous'):
                    validation.check_evaluation_environment(set(),runtime,protocol)


if __name__=='__main__':unittest.main(verbosity=2)
