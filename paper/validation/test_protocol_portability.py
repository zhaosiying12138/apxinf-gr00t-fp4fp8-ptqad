"""Publication protocol lookup must not depend on the author's private paths."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_publication as validation


class ProtocolPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'public-project'
        self.paper = self.root / 'paper'
        self.folder = self.paper / 'evidence/protocol'
        self.folder.mkdir(parents=True)
        self.payload = b'{"version":11,"id":"frozen-protocol"}\n'
        self.local = self.folder / 'recovery_protocol.local.json'
        self.local.write_bytes(self.payload)
        self.manifest = {'protocol_file': '/absent-private-machine/run/protocol.json',
                         'protocol_sha256': hashlib.sha256(self.payload).hexdigest()}
        self.patcher = patch.object(validation, 'P', self.paper)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_missing_private_path_uses_byte_identical_archive(self):
        before = dict(self.manifest)
        self.assertEqual(validation.publication_protocol(self.manifest), self.local)
        self.assertEqual(self.manifest, before)
        record = validation.file_record(self.local, self.root)
        self.assertEqual(record['path'], 'paper/evidence/protocol/recovery_protocol.local.json')

    def test_present_private_original_never_replaces_local_archive(self):
        private = Path(self.temp.name) / 'private-protocol.json'
        private.write_bytes(self.payload)
        self.manifest['protocol_file'] = str(private)
        self.assertEqual(validation.publication_protocol(self.manifest), self.local)
        self.local.unlink()
        with self.assertRaisesRegex(RuntimeError, 'no unique SHA-matching local'):
            validation.publication_protocol(self.manifest)

    def test_changed_archive_is_rejected(self):
        self.local.write_bytes(self.payload + b' ')
        with self.assertRaisesRegex(RuntimeError, 'no unique SHA-matching local'):
            validation.publication_protocol(self.manifest)

    def test_ambiguous_local_copies_are_rejected(self):
        (self.folder / 'other.json').write_bytes(self.payload)
        with self.assertRaisesRegex(RuntimeError, 'no unique SHA-matching local'):
            validation.publication_protocol(self.manifest)

    def test_explicit_copy_must_match_sha_and_stay_in_public_tree(self):
        self.assertEqual(validation.publication_protocol(self.manifest, self.local), self.local)
        private = Path(self.temp.name) / 'private-protocol.json'
        private.write_bytes(self.payload)
        with self.assertRaisesRegex(RuntimeError, 'inside the public repository'):
            validation.publication_protocol(self.manifest, private)
        self.local.write_bytes(b'{}')
        with self.assertRaisesRegex(RuntimeError, 'no unique SHA-matching local'):
            validation.publication_protocol(self.manifest, self.local)

    def test_protocol_symlink_is_rejected(self):
        original = self.folder / 'source.txt'
        self.local.rename(original)
        self.local.symlink_to(original)
        with self.assertRaisesRegex(RuntimeError, 'inside the public repository'):
            validation.publication_protocol(self.manifest)


if __name__ == '__main__':
    unittest.main()
