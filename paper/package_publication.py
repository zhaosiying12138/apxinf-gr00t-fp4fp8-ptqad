#!/usr/bin/env python3
"""Build a publication ZIP only after a fresh, fail-closed validation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from publication_guard import file_record, require
from validate_publication import validate

P = Path(__file__).resolve().parent
DESTINATION = 'apxinf-gr00t-fp4fp8-ptqad-publication.zip'
# Keep review/debug outputs on disk without publishing them.  The first five
# receipts are consumed or produced by the final publication gates; the rest
# are explicitly cited by the source and reproduction appendices.
VALIDATION_REPORTS = frozenset({
    'figure-inputs.json',
    'figure-renders.json',
    'html-build.json',
    'browser-validation.json',
    'publication-validation.json',
    'source_refs.json',
    'source_refs_check.log',
    'cpu-tests.json',
    'cpu-tests.log',
    'native-toolchain-observation.json',
    'native-graph-preparation/README.md',
    'native-graph-preparation/build_manifest.json',
    'native-graph-preparation/evidence_manifest.json',
    'native-graph-preparation/native_patch_check.json',
    'native-graph-preparation/native_fp4_cpu_build.log',
    'native-graph-preparation/native_fp4_model_gate_build.log',
    'native-graph-preparation/native_fp4_opbench_build.log',
    'native-graph-preparation/native_fp4_wheel_build.log',
})


def publication_files():
    registry = json.loads((P/'figures.json').read_text())
    expected_png = {name+'.png' for name in registry}
    expected_figs = expected_png | {name+'.svg' for name in registry if not name.startswith('shot_')} | {'render_pngs.sh'}
    require({p.name for p in (P/'zhihu/images').iterdir() if p.is_file()} == expected_png,
            'Unexpected or missing Zhihu image; remove stale exports before packaging')
    require({p.name for p in (P/'figs').iterdir() if p.is_file()} == expected_figs,
            'Unexpected or missing figure asset; no unregistered figures may enter publication')
    files = [P/name for name in ('paper.html','README.md','meta.json','figures.json','requirements-build.txt',
                                'analysis_plan_w4a4.json')]
    files.extend((P/'sections').glob('*.md'))
    for folder in ('zhihu','figs','evidence','assets'):
        files.extend(path for path in (P/folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    validation = P/'validation'
    files.extend(validation/name for name in VALIDATION_REPORTS if (validation/name).is_file())
    # The checks remain reproducible, but arbitrary JSON/log/review files do
    # not become public simply because a check once wrote them here.
    files.extend(path for path in validation.rglob('*')
                 if path.is_file() and path.suffix in {'.py','.cjs'} and '__pycache__' not in path.parts)
    files.extend(path for path in P.glob('*.py') if path.name != 'snapshot_original.py')
    files.extend(P.glob('*.cjs'))
    for path in files:
        require(not path.is_symlink() and path.resolve().is_relative_to(P), f'External package file: {path}')
        require(path.suffix.lower() not in {'.safetensors','.pt','.pth','.npy','.npz','.zip','.tmp'},
                f'Weights, cache, nested archives or temporary files must not enter publication: {path}')
        require(path.stat().st_size < 128*1024*1024, f'Unexpectedly large publication asset: {path}')
    return sorted(set(files))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate and inventory only; never write a ZIP')
    args=parser.parse_args()
    # Snapshot every candidate before validating, so edits during validation fail.
    files=publication_files()
    before={str(path.relative_to(P)):file_record(path,P) for path in files}
    report=validate(write_report=False)  # Never trust a previously saved passed flag.
    require(report.get('passed') is True,'Fresh publication validation did not pass')
    require(files==publication_files() and before=={str(path.relative_to(P)):file_record(path,P) for path in files},
            'Publication changed during validation; rebuild and retry')
    entries={str(path.relative_to(P)):path for path in files}
    links={}
    for record in report['package_inputs']:
        source=P.parent/record['path']
        require(file_record(source,P.parent)==record, f'Validated evidence changed: {source}')
        if not source.resolve().is_relative_to(P):
            target='project-inputs/'+record['path']
            require(target not in entries,'Duplicate bundled project input')
            entries[target]=source
            links[record['path']]=target
        else:
            require(str(source.resolve().relative_to(P)) in entries,
                    f'Validated internal input is absent from the publication allowlist: {source}')
    # Store the current report even if the standalone validation file was absent/stale.
    generated={'validation/publication-validation.json':(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode(),
               'PROJECT-INPUTS.json':(json.dumps({'scope':'Original repository-relative paths mapped to bundled evidence; model weights are represented only by their recorded hashes.',
                                                'files':links},ensure_ascii=False,indent=2)+'\n').encode()}
    manifest={name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in entries.items()}
    manifest.update({name:{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()} for name,payload in generated.items()})
    if args.check:
        print(f'Publication passed fresh validation: {len(manifest)} files; ZIP not written')
        return
    destination=P/DESTINATION
    temporary=destination.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(manifest):
                payload=generated[name] if name in generated else entries[name].read_bytes()
                require(len(payload)==manifest[name]['bytes'] and hashlib.sha256(payload).hexdigest()==manifest[name]['sha256'],
                        f'File changed while packaging: {name}')
                archive.writestr(name,payload)
            archive.writestr('PACKAGE-MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(destination)
    finally:
        if temporary.exists():temporary.unlink()
    print('Publication ZIP:',destination,'files:',len(manifest),'bytes:',destination.stat().st_size)


if __name__=='__main__':main()
