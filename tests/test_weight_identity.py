import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('weight_identity', ROOT / 'setup/verify_weights.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WeightIdentityTests(unittest.TestCase):
    def test_changed_bytes_rejected_even_when_size_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model'
            original = b'first'
            path.write_bytes(original)
            expected = {'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()}
            self.assertTrue(MODULE.verify_file(path, expected))
            path.write_bytes(b'other')
            with self.assertRaises(ValueError):
                MODULE.verify_file(path, expected, existing_only=True)
            self.assertEqual(path.read_bytes(), b'other')

    def test_missing_is_not_certified_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'absent'
            expected = {'bytes': 1, 'sha256': '0' * 64}
            self.assertFalse(MODULE.verify_file(path, expected, existing_only=True))
            with self.assertRaises(FileNotFoundError):
                MODULE.verify_file(path, expected)


if __name__ == '__main__':
    unittest.main()
