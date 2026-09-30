"""Private CPU tests of plan path/hash controls, not experiment tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('refresh',Path(__file__).with_name('refresh_public_export_plan.py'))
f=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(f)

class ExportPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def write(self,name,text='evidence'):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);return p
    def test_hash_accept_and_tamper_reject(self):
        p=self.write('ok.json');identity=f.checked_file(self.root,'ok.json')
        self.assertEqual(f.checked_file(self.root,'ok.json',identity),identity)
        p.write_text('other');self.assertRaises(ValueError,f.checked_file,self.root,'ok.json',identity)
    def test_traversal_and_absolute_rejected(self):
        for name in ('../outside','/tmp/outside','a/../../outside','a\\b'):
            with self.subTest(name=name):self.assertRaises(ValueError,f.inside,self.root,name)
    def test_symlink_rejected(self):
        p=self.write('real.json');(self.root/'linked.json').symlink_to(p)
        self.assertRaises(ValueError,f.checked_file,self.root,'linked.json')
    def test_weight_payload_rejected(self):
        self.write('tensor.pt');self.assertRaises(ValueError,f.checked_file,self.root,'tensor.pt')
    def test_incomplete_manifest_rejected(self):
        self.write('manifest.json',json.dumps({'status':'incomplete','files':[]}))
        self.assertRaises(ValueError,f.mapped_files,self.root,'manifest.json','training')
    def test_manifest_bytes_and_namespace(self):
        name='paper/evidence/training/stages/qad/run.json';self.write(name,'{}')
        identity=f.checked_file(self.root,name)
        row={'published_path':'stages/qad/run.json','original_absolute_path':'/private/run.json',**identity}
        p=self.write('manifest.json',json.dumps({'status':'complete','files':[row]}))
        self.assertEqual(f.mapped_files(self.root,'manifest.json','training'),[name])
        p.write_text(json.dumps({'status':'complete','files':[row,row]}))
        self.assertRaises(ValueError,f.mapped_files,self.root,'manifest.json','training')
        row['published_path']='../../other.json'
        p.write_text(json.dumps({'status':'complete','files':[row]}))
        self.assertRaises(ValueError,f.mapped_files,self.root,'manifest.json','training')
    def test_arbitrary_weight_mapping_rejected(self):
        seed={'source_export':[{'source':'weights/secret/config.json','target':'results/secret.json','category':'x'}]}
        self.assertRaises(ValueError,f.refresh,seed,self.root,self.root/'target')
    def test_result_allowlist_is_named_and_current(self):
        self.assertEqual(len(f.RESULT_FILES),32)
        self.assertNotIn('results/engine/shot4_pi05_nvfp4.json',f.RESULT_FILES)
        self.assertNotIn('results/baselines/pi05_lerobot_pt_5090.json',f.RESULT_FILES)
        self.assertTrue(all('*' not in name for name in f.RESULT_FILES))

    def test_final_manifest_is_required_and_complete(self):
        run = self.root / 'results' / 'final-run'
        run.mkdir(parents=True)
        comparison = run / 'heldout' / 'paired_comparison.json'
        comparison.parent.mkdir()
        arms = {name: {} for name in f.ARMS}
        comparison.write_text(json.dumps({'environment_pairing_verified': True,
                                          'protocol_consistency_verified': True,
                                          'arms': arms}))
        identity = {'bytes': comparison.stat().st_size, 'sha256': f.sha(comparison)}
        final = {'format': 'high_fp4_v5_final_manifest',
                 'selection_uses_heldout': False,
                 'required_arms': list(f.ARMS),
                 'heldout_comparison': {'path': str(comparison), **identity}}
        final_path = run / 'final_manifest.json'
        final_path.write_text(json.dumps(final))
        (run / 'run_manifest.json').write_text(json.dumps({'status': 'complete'}))
        path, data = f.validate_final_manifest(final_path, self.root)
        self.assertEqual(path, final_path.resolve())
        self.assertEqual(data['format'], final['format'])

    def test_incomplete_final_manifest_rejected(self):
        run = self.root / 'run'; run.mkdir()
        path = run / 'final_manifest.json'
        path.write_text(json.dumps({'format': 'high_fp4_v3_final_manifest',
                                    'selection_uses_heldout': False,
                                    'required_arms': list(f.ARMS)}))
        self.assertRaises(ValueError, f.validate_final_manifest, path, self.root)

if __name__=='__main__':unittest.main(verbosity=2)
