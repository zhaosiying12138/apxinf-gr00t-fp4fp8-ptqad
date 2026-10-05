"""CPU-only tests for portable screenshot dependencies and their hash binding."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
import record_capture
import validate_publication as validation


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw);paper=root/'paper';incoming=root/'incoming'
        (paper/'figs').mkdir(parents=True);incoming.mkdir()
        names=[f'shot_fixture_{i}' for i in range(17)]
        (paper/'figures.json').write_text(json.dumps(dict.fromkeys(names,{})))
        image=incoming/'capture.png';image.write_bytes(b'synthetic image; PNG decoding is independently tested')
        sample=incoming/'approved.png';sample.write_bytes(b'synthetic approved sample')
        log=incoming/'run.log';log.write_text('fixture command completed\n')
        script=incoming/'run.sh';script.write_text('#!/bin/bash\ntrue\n')
        sidecar=incoming/'capture.json'
        sidecar.write_text(json.dumps({'captured_at':'2026-10-06T00:00:00Z',
            'capture_status':'accepted_not_black','exit_status':0,'sha256_before_taskbar_crop':'f'*64,
            'image_quality':{'accepted_not_black':True,'width':3840,'height':2400}}))
        crop=image.with_suffix('.crop.json')
        crop.write_text(json.dumps({'operation':'taskbar_crop_only','source_sha256':'f'*64,
            'source_dimensions':[3840,2400],'output_dimensions':[3840,2280],
            'crop_box':[0,0,3840,2280],'output_sha256':validation.sha(image)}))
        support=[]
        for name in ('common_v11.sh','plan.json',names[0]+'.png.window-binding.json'):
            path=incoming/name;path.write_text('synthetic dependency: '+name);support.append(path)
        def install(sources=support,figure=names[0]):
            args=['record_capture.py','--figure',figure,'--image',str(image),'--log',str(log),
                  '--script',str(script),'--sidecar',str(sidecar),'--approved-sample',str(sample),
                  '--visually-verified']
            for path in sources:args.extend(['--support-file',str(path)])
            with patch.object(record_capture,'P',paper),patch.object(record_capture,'check_png'),patch.object(sys,'argv',args):
                record_capture.main()
            return json.loads((paper/'evidence/captures.json').read_text())
        def publication_fixture():
            captures=install();first=captures['screenshots'][0]
            captures['screenshots']=[]
            for i,name in enumerate(names):
                (paper/'figs'/(name+'.png')).write_bytes(image.read_bytes())
                row=copy.deepcopy(first);row['figure']=name
                if i>=5:row.pop('supporting_files')
                captures['screenshots'].append(row)
            return captures
        yield paper,incoming,names,support,install,publication_fixture


class CaptureSupportFilesTests(unittest.TestCase):
    def test_cli_archives_original_basenames_and_shared_dependencies_are_reusable(self):
        with fixture() as (paper,_,names,support,install,_):
            captures=install()
            identities=captures['screenshots'][0]['supporting_files']
            self.assertEqual(len(identities),3)
            for source,identity in zip(support,identities):
                self.assertEqual(identity,{'path':'evidence/captures/'+source.name,
                    'bytes':source.stat().st_size,'sha256':validation.sha(source)})
                self.assertEqual((paper/identity['path']).read_bytes(),source.read_bytes())
            self.assertEqual(len(install(figure=names[1])['screenshots']),2)

    def test_duplicate_or_primary_proof_basename_rejected_before_install(self):
        for kind in ('duplicate','primary'):
            with self.subTest(kind=kind),fixture() as (paper,incoming,names,support,install,_):
                extra=incoming/'other';extra.mkdir()
                path=extra/(support[0].name if kind=='duplicate' else names[0]+'.sh')
                path.write_text('fixture collision')
                with self.assertRaisesRegex(RuntimeError,'Duplicate|collides'):
                    install(sources=[support[0],path])
                self.assertFalse((paper/'evidence/captures.json').exists())
                self.assertFalse((paper/'figs'/(names[0]+'.png')).exists())

    def test_existing_support_conflict_rejected_before_main_proof_changes(self):
        with fixture() as (paper,_,names,support,install,_):
            evidence=paper/'evidence/captures';evidence.mkdir(parents=True)
            target=evidence/support[0].name;target.write_text('existing different dependency')
            image=paper/'figs'/(names[0]+'.png');image.write_bytes(b'existing image')
            with self.assertRaisesRegex(RuntimeError,'Conflicting existing'):
                install()
            self.assertEqual(image.read_bytes(),b'existing image')
            self.assertEqual(target.read_text(),'existing different dependency')
            self.assertFalse((paper/'evidence/captures.json').exists())

    def test_publication_verifies_five_supported_and_twelve_legacy_rows(self):
        with fixture() as (paper,_,names,support,_,build):
            captures=build();inputs=set()
            with patch.object(validation,'P',paper),patch.object(validation,'check_png'):
                self.assertEqual(len(validation.capture_records(captures,set(names),inputs)),17)
            self.assertTrue({paper/'evidence/captures'/path.name for path in support}<=inputs)

    def test_publication_rejects_tampered_dependency_or_size(self):
        for mutation in ('bytes','sha256','file'):
            with self.subTest(mutation=mutation),fixture() as (paper,_,names,_,_,build):
                captures=build();identity=captures['screenshots'][0]['supporting_files'][0]
                if mutation=='file':(paper/identity['path']).write_text('changed dependency')
                elif mutation=='sha256':identity[mutation]='0'*64
                else:identity[mutation]+=1
                with patch.object(validation,'P',paper),patch.object(validation,'check_png'),\
                     self.assertRaisesRegex(RuntimeError,'Changed evidence'):
                    validation.capture_records(captures,set(names),set())

    def test_publication_rejects_duplicate_primary_or_nonlocal_support(self):
        for mutation in ('duplicate','primary','nonlocal'):
            with self.subTest(mutation=mutation),fixture() as (paper,_,names,_,_,build):
                captures=build();row=captures['screenshots'][0]
                if mutation=='duplicate':row['supporting_files'].append(copy.deepcopy(row['supporting_files'][0]))
                else:row['supporting_files'][0]['path']=row['script'] if mutation=='primary' else 'evidence/../plan.json'
                with patch.object(validation,'P',paper),patch.object(validation,'check_png'),\
                     self.assertRaisesRegex(RuntimeError,'Unsafe, duplicate or conflicting'):
                    validation.capture_records(captures,set(names),set())


if __name__=='__main__':unittest.main(verbosity=2)
