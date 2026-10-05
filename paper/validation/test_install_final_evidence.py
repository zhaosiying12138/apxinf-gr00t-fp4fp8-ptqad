"""File-only final installation: relocation, proofs, tampering and rollback."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import install_final_evidence as installer


class InstallFinalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paper = self.root / 'paper'
        self.old = self.paper / 'evidence'
        self.old.mkdir(parents=True)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        self.archive = self.paper / '_build/archived-evidence'
        self.mapping = []

        def write(relative, value, mapped=True):
            path = self.bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) + '\n')
            if mapped:
                self.mapping.append({'published_path': relative, 'role': 'fixture_source',
                                     'source': {'path': '/private-author/' + relative,
                                                **installer.identity(path)}})
            return path

        final = write('final_manifest.json', {'format': 'w4a4_recovery_v11_final_manifest'})
        pair = write('evidence/paired_comparison.json', {'paired': True})
        protocol = write('protocol/frozen.local.json', {'version': 11})
        write('protocol/analysis_plan_w4a4.json', {'method': 'paired'})
        write('evidence/recipe_inventory.json', {'recipe': 'all_nvfp4_gptq_category'})
        for arm in installer.ARMS:
            write('evidence/heldout_' + arm + '/eval_manifest.json', {'arm': arm})
        write('evidence/training/costs.json', {'status': 'complete'})
        write('evidence/selected_recipe/category_ptq_recipe.json', {'recipe': 'nvfp4'})
        write('evidence/selected_recipe/category_memory.json', {'memory': 1}, False)
        raw = write('evidence/heldout_raw_logs.json', {'status': 'complete'}, False)
        source = {'run_dir': '/private-author/run',
                  'final_manifest': {'path': '/private-author/final_manifest.json', **installer.identity(final)},
                  'heldout_comparison': {'path': '/private-author/paired_comparison.json', **installer.identity(pair)},
                  'protocol': {'path': '/private-author/protocol.json', 'sha256': installer.identity(protocol)['sha256']}}
        write('final_results.json', {'format': 'publication_final_results_v1', 'status': 'complete', 'source': source}, False)
        write('final_costs_summary.json', {'status': 'complete'}, False)
        self.source_manifest = {'version': 1, 'status': 'complete', 'files': self.mapping,
                                'source_final_manifest': source,
                                'heldout_raw_logs_manifest': {'path': 'evidence/heldout_raw_logs.json', **installer.identity(raw)}}
        write('evidence_manifest.json', self.source_manifest, False)

        (self.old / 'README.md').write_text('Capture evidence.\n')
        (self.old / 'retained_captures.json').write_text(json.dumps({'version': 1, 'screenshots': []}))
        captures = self.old / 'captures'
        captures.mkdir()
        row = {'figure': 'shot_qad'}
        for field, suffix in [('raw_log', '.log'), ('script', '.sh'),
                              ('capture_sidecar', '.capture.json'), ('crop_manifest', '.crop.json')]:
            path = captures / ('shot_qad' + suffix)
            path.write_text(field + '\n')
            row[field] = 'evidence/captures/' + path.name
            row[field + '_sha256'] = installer.identity(path)['sha256']
        row['supporting_files'] = []
        for name in ('common_v11.sh', 'plan.json', 'shot_qad.png.window-binding.json'):
            path = captures / name
            path.write_text(name + '\n')
            row['supporting_files'].append({'path': 'evidence/captures/' + name, **installer.identity(path)})
        (self.old / 'captures.json').write_text(json.dumps({'version': 1, 'screenshots': [row]}))
        (self.old / 'obsolete_result.json').write_text('old result must not be copied')

    def run_install(self):
        return installer.install(self.bundle, self.archive, self.paper)

    def test_install_preserves_bytes_normalizes_paths_and_verifies_without_private_sources(self):
        source_bytes = (self.bundle / 'evidence_manifest.json').read_bytes()
        original_pair = (self.bundle / 'evidence/paired_comparison.json').read_bytes()
        report = self.run_install()
        self.assertEqual(report['status'], 'verified')
        self.assertEqual((self.old / 'source_bundle_manifest.json').read_bytes(), source_bytes)
        self.assertEqual((self.old / 'paired_comparison.json').read_bytes(), original_pair)
        self.assertFalse((self.old / 'obsolete_result.json').exists())
        self.assertTrue((self.archive / 'obsolete_result.json').is_file())
        data = installer.read(self.old / 'evidence_manifest.json')
        self.assertEqual(data['heldout_raw_logs_manifest']['path'], 'heldout_raw_logs.json')
        self.assertTrue(all(not row['published_path'].startswith('evidence/') for row in data['files']))
        self.assertTrue(all((self.old / row['published_path']).is_file() for row in data['files']))
        shutil.rmtree(self.bundle)
        self.assertEqual(installer.verify(self.old)['status'], 'verified')

    def test_all_shared_capture_supports_survive_copy(self):
        before = installer.capture_files(self.old)
        self.run_install()
        self.assertEqual(installer.capture_files(self.old), before)
        self.assertIn('captures/common_v11.sh', before)
        self.assertIn('captures/shot_qad.png.window-binding.json', before)

    def test_source_tampering_refuses_before_replacement(self):
        path = self.bundle / 'evidence/paired_comparison.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Evidence identity differs'):
            self.run_install()
        self.assertTrue((self.old / 'obsolete_result.json').is_file())
        self.assertFalse(self.archive.exists())

    def test_capture_support_tampering_refuses_before_replacement(self):
        (self.old / 'captures/plan.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Capture proof identity differs'):
            self.run_install()
        self.assertFalse(self.archive.exists())

    def test_installed_tampering_is_rejected(self):
        self.run_install()
        (self.old / 'paired_comparison.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Evidence identity differs'):
            installer.verify(self.old)

    def test_updated_installed_hash_cannot_override_original_bundle_identity(self):
        self.run_install()
        path = self.old / 'paired_comparison.json'
        path.write_text('{}')
        manifest = installer.read(self.old / 'evidence_manifest.json')
        next(row for row in manifest['files'] if row['published_path'] == path.name)['source'].update(installer.identity(path))
        (self.old / 'evidence_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Installed mapping differs'):
            installer.verify(self.old)

    def test_unsafe_source_mapping_is_rejected(self):
        self.source_manifest['files'][0]['published_path'] = '../escape.json'
        (self.bundle / 'evidence_manifest.json').write_text(json.dumps(self.source_manifest))
        with self.assertRaisesRegex(ValueError, 'Unsafe evidence path'):
            self.run_install()

    def test_subsequent_runtime_evidence_does_not_change_core_manifest(self):
        self.run_install()
        (self.old / 'runtime').mkdir()
        (self.old / 'runtime/manifest.json').write_text('{}')
        self.assertEqual(installer.verify(self.old)['status'], 'verified')

    def test_failed_replace_rolls_back_archived_evidence(self):
        original = Path.rename

        def fail_ready(path, target):
            if path.name == 'evidence' and path.parent.name.startswith('install-evidence-'):
                raise OSError('simulated install failure')
            return original(path, target)

        with patch.object(Path, 'rename', fail_ready):
            with self.assertRaisesRegex(OSError, 'simulated install failure'):
                self.run_install()
        self.assertTrue((self.old / 'obsolete_result.json').is_file())
        self.assertFalse(self.archive.exists())


if __name__ == '__main__':
    unittest.main()
