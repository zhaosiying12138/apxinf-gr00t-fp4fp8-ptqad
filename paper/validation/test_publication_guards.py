#!/usr/bin/env python3
"""CPU-only negative tests of publication gates; all evidence fixtures are temporary."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
import publication_guard as guard
import validate_publication as validation
import package_publication as package
import make_figs


def paired_fixture():
    episodes=[{'task':f'task-{t}','episode_index':i,'init_state_index':i+10,
               'initial_state_sha256':f'{t*10+i:064x}','restored_state_sha256':f'{t*10+i+100:064x}',
               'init_state_bank_sha256':f'{t:064x}','success':i%2==0} for t in range(10) for i in range(10)]
    arm={'episodes':episodes,'count':100,'successes':50,
         'per_task':{f'task-{t}':{'episodes':10,'successes':5} for t in range(10)}}
    return {'environment_pairing_verified':True,'arms':{name:copy.deepcopy(arm) for name in ('bf16','ptq','qad','continued_qad','qad_opd')}}


class Gates(unittest.TestCase):
    def test_require_survives_optimized_python(self):
        code='import sys;sys.path.insert(0,sys.argv[1]);from publication_guard import require;require(False,"EXPECTED_REJECTION")'
        result=subprocess.run([sys.executable,'-O','-c',code,str(P)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('EXPECTED_REJECTION',result.stderr)

    def test_evidence_hash_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            folder=Path(raw);file=folder/'sample.json';file.write_text('{}')
            record=guard.file_record(file,folder);guard.check_record(record,folder)
            file.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'Changed evidence'):guard.check_record(record,folder)

    def test_path_escape_and_absolute_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(RuntimeError):guard.resolve_inside(Path(raw),'../outside')
            with self.assertRaises(RuntimeError):guard.resolve_inside(Path(raw),'/etc/passwd')

    def test_all_pending_block_kinds_rejected(self):
        for name in ('DEVELOPMENT RESULTS','HELDOUT RESULTS','RECOVERY COSTS','PTQ FRONTIER RESULTS','BF16 TIMING RESULTS','NATIVE RESULTS','GEMM RESULTS','OPBENCH RESULTS'):
            with self.subTest(name=name),self.assertRaises(RuntimeError):
                validation.check_finished(f'<!-- BEGIN {name} -->\n本轮结果将在验证后填入。\n<!-- END {name} -->')
        validation.check_finished('<!-- BEGIN NATIVE RESULTS -->\n本轮实际中位数为 1 ms。\n<!-- END NATIVE RESULTS -->')

    def test_markdown_includes_complete_prose_and_caption(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)/'one.md';path.write_text('正文甲。\n\n{{fig:test}}\n')
            args=([path],{'title':'题','meta':'说明'},{'test':{'title':'图题','note':'证据'}})
            original=validation.expected_markdown(*args)
            self.assertIn('![图题](images/test.png)',original)
            path.write_text(path.read_text().replace('正文甲','正文乙'))
            self.assertNotEqual(original,validation.expected_markdown(*args))

    def test_svg_remote_resource_is_detected_but_citation_is_allowed(self):
        page=validation.Page()
        page.feed('<a href="https://example.com/paper">citation</a><svg><image href="https://example.com/tracker.png"/></svg>')
        self.assertEqual(page.resources,['https://example.com/tracker.png'])

    def test_correct_pairing(self):validation.check_pairing(paired_fixture())

    def test_pairing_mismatch_rejected(self):
        data=paired_fixture();data['arms']['qad_opd']['episodes'][0]['initial_state_sha256']='f'*64
        with self.assertRaisesRegex(RuntimeError,'Unpaired'):validation.check_pairing(data)

    def test_duplicate_initial_state_rejected_even_when_all_arms_match(self):
        data=paired_fixture()
        for arm in data['arms'].values():arm['episodes'][0]['init_state_index']=11
        with self.assertRaisesRegex(RuntimeError,'repeated or missing'):validation.check_pairing(data)

    def test_incomplete_or_wrong_outcome_rejected(self):
        data=paired_fixture();data['arms']['qad']['count']=99
        with self.assertRaises(RuntimeError):validation.check_pairing(data)
        data=paired_fixture();data['arms']['qad']['episodes'][0]['success']=1
        with self.assertRaisesRegex(RuntimeError,'Boolean'):validation.check_pairing(data)

    def test_black_png_rejected_and_text_image_accepted(self):
        from PIL import Image,ImageDraw
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)/'capture.png';image=Image.new('RGB',(240,120),'black');image.save(path)
            with self.assertRaisesRegex(RuntimeError,'Blank/dark'):validation.check_png(path,[240,120],terminal=True)
            ImageDraw.Draw(image).text((12,30),'REAL TEST TEXT',fill='white');image.save(path)
            validation.check_png(path,[240,120],terminal=True)

    def test_capture_crop_contract_rejects_failed_or_unlinked_image(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)/'image.png';path.write_bytes(b'temporary-contract-test')
            side={'capture_status':'accepted_not_black','exit_status':0,'sha256_before_taskbar_crop':'a'*64,
                  'image_quality':{'accepted_not_black':True,'width':3840,'height':2400}}
            crop={'operation':'taskbar_crop_only','source_sha256':'a'*64,
                  'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'source_dimensions':[3840,2400],'output_dimensions':[3840,2280],'crop_box':[0,0,3840,2280]}
            guard.capture_crop_contract(side,crop,path,[3840,2280])
            for field,value in (('capture_status','failed'),('exit_status',1)):
                bad=copy.deepcopy(side);bad[field]=value
                with self.assertRaises(RuntimeError):guard.capture_crop_contract(bad,crop,path,[3840,2280])
            bad=copy.deepcopy(crop);bad['source_sha256']='b'*64
            with self.assertRaisesRegex(RuntimeError,'not linked'):guard.capture_crop_contract(side,bad,path,[3840,2280])
            bad=copy.deepcopy(crop);bad['crop_box']=[1,0,3840,2280]
            with self.assertRaisesRegex(RuntimeError,'bottom taskbar'):guard.capture_crop_contract(side,bad,path,[3840,2280])

    def test_previous_shot_verify_hash_cannot_bypass_current_capture_proof(self):
        # Exercise the former exact-hash exception while isolating PNG decoding:
        # even those exact recorded hashes must now reach the sidecar requirement.
        image_sha='a179bbf641af1f5594313d5a75669ce82944fbae68878635f413112a4dc32536'
        log_sha='aefb9e76719f0e3d612ad58318a1b36c329b7c7f6f84ba38c97844a715757224'
        with tempfile.TemporaryDirectory() as raw:
            folder=Path(raw);(folder/'run.log').write_text('original log fixture')
            (folder/'run.sh').write_text('true')
            first={'figure':'shot_verify','verified':True,'sha256':image_sha,
                   'dimensions':[3840,2280],'exit_status':0,'captured_at':'2026-09-29T02:34:12Z',
                   'raw_log':'run.log','raw_log_sha256':log_sha,'script':'run.sh','script_sha256':'1'*64}
            rows=[first]+[dict(first,figure=f'fixture_{i}') for i in range(16)]
            capture={'approved_sample':{'confirmed_by_user':True,'dimensions':[3840,2280]},'screenshots':rows}
            def fixture_sha(path):
                return log_sha if path.name=='run.log' else ('1'*64 if path.name=='run.sh' else image_sha)
            with mock.patch.object(validation,'P',folder),mock.patch.object(validation,'check_png'),\
                 mock.patch.object(validation,'sha',side_effect=fixture_sha):
                with self.assertRaisesRegex(RuntimeError,'Capture lacks capture_sidecar: shot_verify'):
                    validation.capture_records(capture,{x['figure'] for x in rows},set())

    def test_raw_latency_median_and_count(self):
        value={'schema_version':2,'all_outputs_finite':True,'n':4,'lat_all_ms':[10,2,8,4],'latency_ms_p50':6}
        make_figs.timing_record(value,'temporary','latency_ms_p50','lat_all_ms')
        value['latency_ms_p50']=8
        with self.assertRaisesRegex(ValueError,'P50'):make_figs.timing_record(value,'temporary','latency_ms_p50','lat_all_ms')
        value['latency_ms_p50']=6;value['n']=3
        with self.assertRaisesRegex(ValueError,'count'):make_figs.timing_record(value,'temporary','latency_ms_p50','lat_all_ms')

    def test_current_result_missing_never_generates_old_chart(self):
        with tempfile.TemporaryDirectory() as raw,mock.patch.object(make_figs,'ROOT',Path(raw)):
            with self.assertRaises(FileNotFoundError):make_figs.e1_chart()
            self.assertEqual(list(Path(raw).iterdir()),[])

    def test_browser_failure_invalidates_old_report_before_loading(self):
        with tempfile.TemporaryDirectory() as raw:
            folder=Path(raw);(folder/'validation').mkdir()
            report=folder/'validation/browser-validation.json';report.write_text('{"pass":true}')
            script=folder/'qa_browser.cjs';script.write_bytes((P/'qa_browser.cjs').read_bytes())
            result=subprocess.run(['node',str(script)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIs(json.loads(report.read_text())['pass'],False)
            self.assertEqual(json.loads(report.read_text())['status'],'failed')

    def test_package_never_trusts_old_pass(self):
        with tempfile.TemporaryDirectory() as raw:
            folder=Path(raw);old=folder/'publication-validation.json';old.write_text('{"passed":true}')
            with mock.patch.object(package,'P',folder),mock.patch.object(package,'publication_files',return_value=[old]),\
                 mock.patch.object(package,'validate',side_effect=RuntimeError('fresh validation rejects')),\
                 mock.patch.object(sys,'argv',['package_publication.py']):
                with self.assertRaisesRegex(RuntimeError,'fresh validation rejects'):package.main()
            self.assertFalse((folder/package.DESTINATION).exists())
            self.assertFalse((folder/(package.DESTINATION+'.tmp')).exists())


if __name__=='__main__':unittest.main(verbosity=2)
