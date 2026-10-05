#!/usr/bin/env python3
"""Install a materialized final bundle while preserving verified capture proofs.

The experiment's JSON/log bytes are never rewritten. A new installed manifest
uses paths relative to paper/evidence; the original bundle manifest is retained
byte-for-byte. This is a file-integrity step, not a substitute for the scientific
and publication validators. No original private source path is opened by verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

PAPER = Path(__file__).resolve().parent
ARMS = ('bf16', 'ptq', 'qad', 'continued_qad', 'qad_opd')
EVIDENCE_ENTRIES = {'paired_comparison.json', 'heldout_raw_logs.json',
                    'recipe_inventory.json', 'selected_recipe', 'training'} | {
                        'heldout_' + arm for arm in ARMS}
GENERATED = ('final_results.json', 'evidence/heldout_raw_logs.json',
             'evidence/selected_recipe/category_memory.json')
FORMAT = 'installed_final_evidence_v1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'Missing regular evidence file: ' + str(path))
    payload = path.read_bytes()
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def local(root, relative):
    need(isinstance(relative, str) and '\\' not in relative, 'Invalid evidence path')
    name = PurePosixPath(relative)
    need(relative and not name.is_absolute() and '..' not in name.parts and
         name.as_posix() == relative, 'Unsafe evidence path: ' + relative)
    path = root / relative
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'Evidence symlink: ' + relative)
    need(path.resolve().is_relative_to(root.resolve()), 'Evidence escapes root: ' + relative)
    return path


def checked(root, relative, record):
    path = local(root, relative)
    need(identity(path) == {key: record[key] for key in ('bytes', 'sha256')},
         'Evidence identity differs: ' + relative)
    return path


def installed_path(relative):
    name = PurePosixPath(relative)
    if name.parts and name.parts[0] == 'evidence':
        return name.relative_to('evidence').as_posix()
    return relative


def tree_files(root):
    need(root.is_dir() and not root.is_symlink(), 'Missing regular evidence directory: ' + str(root))
    names = set()
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'Evidence tree contains symlink: ' + str(path))
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
    return names


def bundle_files(bundle):
    """Validate the materializer's exact output layout before copying anything."""
    manifest = read(local(bundle, 'evidence_manifest.json'))
    need(manifest.get('version') == 1 and manifest.get('status') == 'complete',
         'Source bundle is incomplete')
    need({p.name for p in (bundle / 'evidence').iterdir()} == EVIDENCE_ENTRIES,
         'Unexpected source bundle evidence entries')
    rows = manifest.get('files')
    need(isinstance(rows, list) and rows, 'Empty source bundle manifest')
    inputs = {}
    for row in rows:
        relative = row['published_path']
        need(relative not in inputs, 'Duplicate source bundle path: ' + relative)
        checked(bundle, relative, row['source'])
        inputs[relative] = row
    raw = manifest['heldout_raw_logs_manifest']
    need(raw['path'] == 'evidence/heldout_raw_logs.json', 'Unexpected raw-log manifest path')
    checked(bundle, raw['path'], raw)
    final = read(bundle / 'final_results.json')
    need(final.get('format') == 'publication_final_results_v1' and final.get('status') == 'complete',
         'Source bundle has no complete final results')
    checked(bundle, 'final_manifest.json', final['source']['final_manifest'])
    checked(bundle, 'evidence/paired_comparison.json', final['source']['heldout_comparison'])
    need(manifest.get('source_final_manifest') == final['source'], 'Source final identity differs')
    need(read(bundle / 'final_costs_summary.json') == read(bundle / 'evidence/training/costs.json'),
         'Bundle cost summary differs from training evidence')
    expected = set(inputs) | set(GENERATED) | {'evidence_manifest.json', 'final_costs_summary.json'}
    need(tree_files(bundle) == expected, 'Unmapped or missing source bundle files')
    for relative in GENERATED:
        need(relative not in inputs, 'Generated file is unexpectedly double-mapped')
        inputs[relative] = {'published_path': relative, 'role': 'materializer_generated_evidence',
                            'source': {'path': str(bundle / relative), **identity(bundle / relative)}}
    return manifest, inputs


def capture_files(folder):
    """Resolve only the retained README, capture manifests and their proof files."""
    inputs = {name: identity(local(folder, name)) for name in
              ('README.md', 'captures.json', 'retained_captures.json')}
    for name in ('captures.json', 'retained_captures.json'):
        data = read(folder / name)
        need(data.get('version') == 1 and isinstance(data.get('screenshots'), list),
             'Invalid capture manifest: ' + name)
        for row in data['screenshots']:
            records = [] if row.get('retained_unaffected') else [
                {'path': row[field], 'sha256': row[field + '_sha256']}
                for field in ('raw_log', 'script', 'capture_sidecar', 'crop_manifest')]
            records.extend(row.get('supporting_files', []))
            for record in records:
                relative = record['path']
                need(PurePosixPath(relative).parts[:2] == ('evidence', 'captures'),
                     'Unexpected capture proof path: ' + relative)
                target = installed_path(relative)
                actual = identity(local(folder, target))
                need(actual['sha256'] == record['sha256'] and
                     ('bytes' not in record or actual['bytes'] == record['bytes']),
                     'Capture proof identity differs: ' + relative)
                if target in inputs:
                    need(inputs[target] == actual, 'Conflicting shared capture proof')
                inputs[target] = actual
    return inputs


def verify(folder):
    """Verify installed core bytes and current capture proofs without private data."""
    folder = Path(folder).resolve(strict=True)
    manifest = read(folder / 'evidence_manifest.json')
    need(manifest.get('format') == FORMAT and manifest.get('status') == 'complete' and
         manifest.get('path_root') == '.', 'Unsupported installed evidence manifest')
    original_path = checked(folder, 'source_bundle_manifest.json', manifest['source_bundle_manifest'])
    original = read(original_path)
    need(original.get('version') == 1 and original.get('status') == 'complete', 'Incomplete source manifest')
    expected = {installed_path(row['published_path']):
                {**row, 'published_path': installed_path(row['published_path'])}
                for row in original['files']}
    need(len(expected) == len(original['files']), 'Installed source paths collide')
    rows = manifest.get('files')
    need(isinstance(rows, list) and rows, 'Empty installed manifest')
    actual = {}
    for row in rows:
        relative = row['published_path']
        need(relative not in actual, 'Duplicate installed evidence path')
        checked(folder, relative, row['source'])
        actual[relative] = row
    extras = {installed_path(name) for name in GENERATED}
    need(set(actual) == set(expected) | extras and
         all(actual[name] == row for name, row in expected.items()),
         'Installed mapping differs from source bundle')
    for name in extras:
        need(actual[name].get('role') == 'materializer_generated_evidence', 'Unexpected generated evidence role')
    raw = manifest['heldout_raw_logs_manifest']
    need(raw == {**original['heldout_raw_logs_manifest'], 'path': 'heldout_raw_logs.json'},
         'Installed raw-log manifest identity differs')
    checked(folder, raw['path'], raw)
    final = read(folder / 'final_results.json')
    checked(folder, 'final_manifest.json', final['source']['final_manifest'])
    checked(folder, 'paired_comparison.json', final['source']['heldout_comparison'])
    need(final['source'] == original['source_final_manifest'], 'Installed final source differs')
    # Subsequent runtime/search/frontier evidence and updated captures are owned
    # by their own manifests. The materialized scientific core stays immutable.
    for name in EVIDENCE_ENTRIES | {'protocol'}:
        path = folder / name
        current = ({name + '/' + child for child in tree_files(path)} if path.is_dir() else {name})
        recorded = {item for item in actual if item == name or item.startswith(name + '/')}
        need(current == recorded, 'Unmapped installed core files: ' + name)
    captures = capture_files(folder)
    return {'status': 'verified', 'core_files': len(actual), 'capture_proof_files': len(captures),
            'source_bundle_manifest_sha256': identity(original_path)['sha256']}


def install(bundle, archive, paper=PAPER):
    bundle = Path(bundle).absolute()
    paper = Path(paper).resolve(strict=True)
    old = paper / 'evidence'
    archive = Path(archive).absolute()
    need(old.is_dir() and not old.is_symlink(), 'Current evidence must be a regular directory')
    need(not archive.exists() and not archive.is_symlink(), 'Archive must be a new path')
    need(not bundle.resolve().is_relative_to(old.resolve()), 'Bundle cannot be inside evidence being replaced')
    need(not archive.resolve().is_relative_to(old.resolve()) and
         not archive.resolve().is_relative_to(bundle.resolve()) and
         not bundle.resolve().is_relative_to(archive.resolve()), 'Archive overlaps source evidence')
    source_manifest, source_files = bundle_files(bundle)
    captures = capture_files(old)
    bundle_snapshot = {name: identity(bundle / name) for name in tree_files(bundle)}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='install-evidence-', dir=paper) as temp:
        ready = Path(temp) / 'evidence'
        ready.mkdir()
        installed = []
        for relative, row in source_files.items():
            name = installed_path(relative)
            target = local(ready, name)
            need(not target.exists(), 'Installed path collision: ' + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / relative, target)
            installed.append({**row, 'published_path': name})
        shutil.copyfile(bundle / 'evidence_manifest.json', ready / 'source_bundle_manifest.json')
        for relative in captures:
            target = local(ready, relative)
            need(not target.exists(), 'Capture/core path collision: ' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old / relative, target)
        manifest = {**source_manifest, 'format': FORMAT, 'path_root': '.',
                    'source_bundle_manifest': {'path': 'source_bundle_manifest.json',
                                               **identity(ready / 'source_bundle_manifest.json')},
                    'heldout_raw_logs_manifest': {**source_manifest['heldout_raw_logs_manifest'],
                                                 'path': 'heldout_raw_logs.json'},
                    'files': sorted(installed, key=lambda row: row['published_path'])}
        (ready / 'evidence_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        report = verify(ready)
        need(capture_files(ready) == captures, 'Capture evidence changed during copy')
        need(capture_files(old) == captures and
             {name: identity(bundle / name) for name in tree_files(bundle)} == bundle_snapshot,
             'Inputs changed during evidence installation')
        need(not archive.exists(), 'Archive appeared during installation')
        old.rename(archive)
        try:
            ready.rename(old)
        except BaseException:
            archive.rename(old)
            raise
    return {**report, 'installed': str(old), 'previous_evidence_archive': str(archive)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--paper', type=Path, default=PAPER)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.bundle or args.archive:
            parser.error('--verify cannot be combined with --bundle or --archive')
        result = verify(args.verify)
    else:
        if not args.bundle or not args.archive:
            parser.error('installation requires --bundle and --archive')
        result = install(args.bundle, args.archive, args.paper)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
