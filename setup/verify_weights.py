#!/usr/bin/env python3
"""Verify local model bytes against the public source lock, without loading models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_file(path, expected, existing_only=False):
    if not path.is_file():
        if existing_only:
            return False
        raise FileNotFoundError(path)
    before = path.stat()
    actual = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed during verification: {path}')
    if before.st_size != expected['bytes'] or actual != expected['sha256']:
        raise ValueError(f'Model identity mismatch: {path}; existing files are not automatically replaced')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'setup/locks/model-sources.json')
    parser.add_argument('--weights-root', type=Path, default=ROOT / 'weights')
    parser.add_argument('--cosmos-dir', type=Path, help='Explicit existing named backbone directory')
    parser.add_argument('--models', nargs='+', choices=('gr00t', 'cosmos', 'pi05'), default=['gr00t', 'cosmos', 'pi05'])
    parser.add_argument('--skip-assets', action='store_true')
    parser.add_argument('--existing-only', action='store_true', help='Pre-download safety check; missing files are reported, not certified')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verified = missing = 0
    for key in args.models:
        model = manifest['models'][key]
        folder = args.weights_root / model['local_dir']
        if key == 'cosmos' and args.cosmos_dir:
            folder = args.cosmos_dir
        elif key == 'cosmos' and not folder.exists():
            folder = args.weights_root / 'nvidia/Cosmos-Reason2-2B'
        for row in model['files']:
            found = verify_file(folder / row['path'], row, args.existing_only)
            verified += int(found)
            missing += int(not found)
        print(f'[model identity] {key}: selector revision={model["revision"]!r}; {model["revision_confidence"]}')
    if 'pi05' in args.models and not args.skip_assets:
        for row in manifest['external_assets']:
            found = verify_file(args.weights_root / row['path'], row, args.existing_only)
            verified += int(found)
            missing += int(not found)
    print(json.dumps({'verified_files': verified, 'missing_files': missing,
                      'scope': 'existing_files_only_not_a_completed_download' if args.existing_only else 'all_selected_model_bytes_verified'}))


if __name__ == '__main__':
    main()
