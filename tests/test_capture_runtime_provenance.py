"""CPU checks for the explicit final-run boundary of runtime provenance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('capture_runtime', ROOT / 'paper/capture_runtime.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class CaptureRuntimeProvenanceTests(unittest.TestCase):
    def test_complete_final_manifest_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'run'; run.mkdir()
            comparison = run / 'heldout' / 'paired_comparison.json'
            comparison.parent.mkdir()
            comparison.write_text(json.dumps({
                'environment_pairing_verified': True,
                'protocol_consistency_verified': True,
                'arms': {name: {} for name in mod.ARMS},
            }))
            final = {
                'format': 'high_fp4_category_final_manifest',
                'selection_uses_heldout': False,
                'required_arms': list(mod.ARMS),
                'heldout_comparison': {
                    'path': str(comparison), 'bytes': comparison.stat().st_size,
                    'sha256': mod.digest(comparison),
                },
            }
            final_path = run / 'final_manifest.json'
            final_path.write_text(json.dumps(final))
            (run / 'run_manifest.json').write_text(json.dumps({'status': 'complete'}))
            path, data = mod.validate_final_manifest(final_path)
            self.assertEqual(path, final_path.resolve())
            self.assertEqual(data['format'], final['format'])

    def test_v3_or_incomplete_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'final_manifest.json'
            path.write_text(json.dumps({
                'format': 'high_fp4_v3_final_manifest',
                'selection_uses_heldout': False,
                'required_arms': list(mod.ARMS),
            }))
            with self.assertRaises(ValueError):
                mod.validate_final_manifest(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
