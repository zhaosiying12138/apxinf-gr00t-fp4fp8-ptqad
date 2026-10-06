"""CPU fixtures for v11 counts, capture tails and immutable per-stage source bytes."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tests import test_training_costs as legacy
from tests.test_v3_training_costs import v3_run, read

cost, ROOT, write = legacy.cost, legacy.ROOT, legacy.write
PROTOCOL_PATH = ROOT / 'exp/recovery_protocol_v11_w4a4_category.json'
PROTOCOL = read(PROTOCOL_PATH)


def capture_stage(root):
    folder, metrics, manifest = legacy.stage(root)
    (folder/'checkpoint-500').rename(folder/'checkpoint-2000')
    sha = cost.identity(PROTOCOL_PATH)['sha256']
    capture = {'root':str(root/'capture'), 'sample_count':148, 'files':[]}
    manifest.update(scope='all_ordinary_linear', protocol_sha256=sha,
                    seed=20261003, train_seed=20261003, w4a4_enabled=True,
                    execution_mode='W4A4 numerical QDQ + BF16 LoRA residual',
                    capture_dataset=capture['root'], capture_dataset_samples=148,
                    capture_dataset_sha256=hashlib.sha256(json.dumps(capture,sort_keys=True).encode()).hexdigest())
    for name in ('rl/w4a4_lora.py','quant/native_activation.py'):
        manifest['recovery_source_sha256'][name] = cost.identity(ROOT/name)['sha256']
    metrics.update(global_steps=2000, requested_optimizer_steps=2000)
    write(folder/'runtime_metrics.json',metrics)
    write(folder/'recovery_manifest.json',manifest)
    write(folder/'checkpoint-2000/recovery_manifest.json',manifest)
    write(folder/'checkpoint-2000/trainer_state.json',{'global_step':2000,'max_steps':2000,'epoch':200.0})
    write(folder/'orchestrator_training_request.json',{'protocol_sha256':sha,'capture_dataset_identity':capture,
        'environment':{'QAD_OUT':str(folder),'QAD_STEPS':'2000','QAD_LR':'5e-05',
        'TRAIN_SEED':'20261003','QAD_OPD_MSE_W':'0.0','QAD_ACTIVATION_CHECKPOINTING':'1'}})
    normalized=cost.normalized_protocol(PROTOCOL,{'selected_qad_learning_rate':.00005,'selected_opd_weight':.25})
    return folder,manifest,normalized,sha


class W4A4TrainingCosts(unittest.TestCase):
    def test_v12_cost_gate_binds_format_recipe_and_frozen_protocol_sha(self):
        protocol_path = ROOT / 'exp/recovery_protocol_v12_rtn_w4a4.json'
        protocol = read(protocol_path)
        sha = cost.identity(protocol_path)['sha256']
        final = {'format': 'w4a4_recovery_v12_final_manifest',
                 'protocol_sha256': sha, 'selected_pressure_recipe': 'rtn_w4a4_category'}
        self.assertTrue(cost.final_format_matches(final, protocol, sha))
        self.assertFalse(cost.final_format_matches(final, protocol, '0' * 64))
        self.assertFalse(cost.final_format_matches({**final, 'protocol_sha256': '0' * 64}, protocol, sha))
        self.assertFalse(cost.final_format_matches({**final, 'selected_pressure_recipe': 'all_nvfp4_gptq_category'}, protocol, sha))
        for value in ('w4a4_recovery_v11_final_manifest', 'high_fp4_v3_final_manifest'):
            self.assertFalse(cost.final_format_matches({**final, 'format': value}, protocol, sha))
        self.assertFalse(cost.final_format_matches(final, {**protocol, 'version': 13}, sha))
        self.assertEqual(cost.partition_budget(protocol, 'heldout'), (10, 16))
        settings = cost.normalized_protocol(protocol, {'selected_qad_learning_rate': .00005,
                                                        'selected_opd_weight': .25})
        self.assertEqual(settings['continuation']['probe_every_optimizer_steps'], 1)

    def test_exact_v11_format_and_160_episode_contract(self):
        final={'format':'w4a4_recovery_v11_final_manifest',
               'required_arms':['bf16','ptq','qad','continued_qad','qad_opd']}
        self.assertTrue(cost.final_format_matches(final,PROTOCOL))
        self.assertFalse(cost.final_format_matches(final,{**PROTOCOL,'version':10}))
        self.assertFalse(cost.final_format_matches({**final,'format':'w4a4_recovery_v12_final_manifest'},PROTOCOL))
        self.assertEqual(cost.partition_budget(PROTOCOL,'heldout'),(10,16))
        rows={'environment_pairing_verified':True,'arms':{arm:{'count':160} for arm in final['required_arms']}}
        cost.audit_final_comparison(rows,final,PROTOCOL)
        rows['arms']['qad']['count']=100
        with self.assertRaisesRegex(ValueError,'incomplete'):cost.audit_final_comparison(rows,final,PROTOCOL)
        bad=copy.deepcopy(PROTOCOL);bad['evaluation_contract']['final_episode_count_per_arm']=100
        with self.assertRaisesRegex(ValueError,'contradicts'):cost.partition_budget(bad,'heldout')

    def test_real_v11_orchestrator_path_is_resolved_by_hash(self):
        state={'implementation_sha256':cost.identity(ROOT/'exp/run_high_fp4_v3.py')['sha256']}
        self.assertEqual(cost.orchestrator_source(PROTOCOL,state),'exp/run_high_fp4_v3.py')

    def test_initialization_resolution_ignores_later_driver_amendment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'exp').mkdir()
            initial=root/'exp/run_high_fp4_v3_producer.py';initial.write_text('initial synthetic source')
            resumed=root/'exp/run_high_fp4_v3.py';resumed.write_text('resumed synthetic source')
            state={'implementation_sha256':cost.identity(initial)['sha256'],
                   'orchestrator_source_sha256':cost.identity(resumed)['sha256']}
            before=copy.deepcopy(state)
            self.assertEqual(cost.orchestrator_source(PROTOCOL,state,root),
                             'exp/run_high_fp4_v3_producer.py')
            self.assertEqual(state,before)
            initial.unlink()
            with self.assertRaisesRegex(ValueError,'no matching source bytes'):
                cost.orchestrator_source(PROTOCOL,state,root)
            initial.write_text('wrong initial bytes')
            with self.assertRaisesRegex(ValueError,'no matching source bytes'):
                cost.orchestrator_source(PROTOCOL,state,root)
            del state['implementation_sha256']
            with self.assertRaisesRegex(ValueError,'initialization implementation SHA'):
                cost.orchestrator_source(PROTOCOL,state,root)

    def test_explicit_initialization_path_must_match_its_own_digest(self):
        state={'implementation_path':'exp/run_high_fp4_v3.py',
               'implementation_sha256':cost.identity(ROOT/'exp/run_high_fp4_v3_producer.py')['sha256'],
               'orchestrator_source_sha256':cost.identity(ROOT/'exp/run_high_fp4_v3.py')['sha256']}
        with self.assertRaisesRegex(ValueError,'path differs from recorded SHA'):
            cost.orchestrator_source(PROTOCOL,state)

    def test_amended_orchestrator_keeps_initial_source_in_portable_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,private,protocol,cache=v3_run(root);out=root/'costs'
            path=run/'run_manifest.json';state=read(path)
            initial='exp/run_high_fp4_v3_producer.py'
            state.update(implementation_sha256=cost.identity(ROOT/initial)['sha256'],
                         orchestrator_source_sha256=cost.identity(ROOT/'exp/run_high_fp4_v3.py')['sha256'])
            write(path,state);original=path.read_bytes()
            with patch.dict(sys.modules,{'torch':types.SimpleNamespace(load=lambda *a,**kw:cache)}):
                result=cost.collect(None,None,None,out,protocol,orchestrator_run=run)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual((out/'source'/initial).read_bytes(),(ROOT/initial).read_bytes())
            self.assertEqual(result['recovery_provenance']['initialization_snapshot']['sha256'],
                             state['implementation_sha256'])
            self.assertEqual(result['recovery_provenance']['launch_snapshot_coverage'],'none')
            shutil.rmtree(private)
            self.assertEqual(cost.verify_published(out),result)

    def test_capture_tail_budget_and_epoch_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            folder,manifest,protocol,sha=capture_stage(Path(directory))
            row,_,_=cost.audit_stage(folder,'qad',2000,protocol,sha,cost.Evidence())
            self.assertEqual(row['demonstration_window_draws'],29600)
            self.assertIn('source-derived',row['demonstration_window_draws_scope'])
            write(folder/'checkpoint-2000/trainer_state.json',{'global_step':2000,'max_steps':2000,'epoch':216.22})
            with self.assertRaisesRegex(ValueError,'epoch disagrees'):
                cost.audit_stage(folder,'qad',2000,protocol,sha,cost.Evidence())

    def test_capture_identity_and_w4a4_source_omissions_fail_closed(self):
        for mutation in ('dataset','source'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                folder,manifest,protocol,sha=capture_stage(Path(directory))
                if mutation=='dataset':manifest['capture_dataset_samples']=160
                else:del manifest['recovery_source_sha256']['quant/native_activation.py']
                write(folder/'recovery_manifest.json',manifest)
                write(folder/'checkpoint-2000/recovery_manifest.json',manifest)
                with self.assertRaises(ValueError):cost.audit_stage(folder,'qad',2000,protocol,sha,cost.Evidence())

    def test_probe_passes_include_short_epoch_tails(self):
        manifest={'micro_batch':1,'gradient_accumulation_steps':16,'capture_dataset_samples':148,
                  'probe_every':4,'probe_weight':.25}
        counts=cost.microbatches_per_update(manifest,2000)
        self.assertEqual(counts[:10],[16]*9+[4])
        text=''.join(f'[opd] step={step} probe=0 mse=0.1 weight=0.25 microbatch=1\n'
                     for step in range(4,2001,20) for _ in range(counts[step-1]))
        result=cost.probe_cost_fields(text,'qad_opd',2000,manifest)
        self.assertEqual(result['scheduled_teacher_backward_passes'],6800)
        self.assertEqual(result['logged_teacher_backward_passes'],1600)
        self.assertEqual(cost.probe_cost_fields('','continued_qad',2000,manifest)['scheduled_teacher_backward_passes'],0)

    def test_per_stage_source_snapshots_survive_portable_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,private,protocol,cache=v3_run(root);out=root/'costs'
            train=run/'artifacts/train_qad_lr_5e-05';model=run/'artifacts/merge_qad_lr_5e-05'
            old=(ROOT/'rl/probe_distill.py').read_bytes()+b'\n# Synthetic earlier fixture bytes.\n'
            sha=hashlib.sha256(old).hexdigest()
            snapshot=run/'operations/opd-tail-fix/source_snapshots'/sha/'rl/probe_distill.py'
            snapshot.parent.mkdir(parents=True);snapshot.write_bytes(old)
            manifest=read(train/'recovery_manifest.json');manifest['recovery_source_sha256']['rl/probe_distill.py']=sha
            write(train/'recovery_manifest.json',manifest);write(train/'checkpoint-500/recovery_manifest.json',manifest)
            export=read(model/'merge_manifest.json');export['recovery_manifest']=manifest
            export['recovery_manifest_sha256']=cost.identity(train/'checkpoint-500/recovery_manifest.json')['sha256']
            write(model/'merge_manifest.json',export)
            with patch.dict(sys.modules,{'torch':types.SimpleNamespace(load=lambda *a,**kw:cache)}):
                result=cost.collect(None,None,None,out,protocol,orchestrator_run=run)
            self.assertEqual((out/'stages/qad/source/rl/probe_distill.py').read_bytes(),old)
            self.assertNotEqual((out/'stages/qad_opd/source/rl/probe_distill.py').read_bytes(),old)
            shutil.rmtree(private)
            self.assertEqual(cost.verify_published(out),result)
            (out/'stages/qad/source/rl/probe_distill.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash differs'):cost.verify_published(out)

    def test_content_addressed_snapshot_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);digest='1'*64
            path=root/'snapshots'/digest/'rl/probe_distill.py';path.parent.mkdir(parents=True);path.write_text('wrong')
            with self.assertRaisesRegex(ValueError,'producer source SHA'):
                cost.source_file('rl/probe_distill.py',digest,root/'live',root/'snapshots')


if __name__=='__main__':unittest.main()
