#!/usr/bin/env python3
"""Install one manually inspected execution capture with its capture/crop proof."""
import argparse
import json
from pathlib import Path
import shutil

from publication_guard import capture_crop_contract, require
from validate_publication import check_png, sha

P = Path(__file__).resolve().parent


def prepare_support_files(sources, evidence, figures):
    """Check every shared dependency before installing any capture output."""
    reserved = {name + suffix for name in figures
                for suffix in ('.png', '.log', '.sh', '.capture.json', '.crop.json')}
    prepared = []; seen = set()
    for value in sources:
        source = Path(value); name = source.name
        require(name not in reserved, 'Support basename collides with capture proof: ' + name)
        require(name not in seen, 'Duplicate support basename: ' + name)
        seen.add(name)
        require(source.is_file() and source.stat().st_size > 0, 'Missing or empty support file: ' + str(source))
        digest = sha(source); size = source.stat().st_size; destination = evidence / name
        require(not destination.is_symlink(), 'Support destination must not be a symlink: ' + name)
        if destination.exists():
            require(destination.is_file() and destination.stat().st_size == size and sha(destination) == digest,
                    'Conflicting existing support file: ' + name)
        prepared.append((source, destination, {'path': destination.relative_to(P).as_posix(),
                                              'sha256': digest, 'bytes': size}))
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('figure','image','log','script','sidecar','approved-sample'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--crop-manifest', help='Defaults to IMAGE with .crop.json suffix')
    parser.add_argument('--support-file', action='append', default=[],
                        help='Archive a dependency under its original basename; may be repeated')
    parser.add_argument('--visually-verified', action='store_true',
                        help='Confirms manual reading of terminal content, command identity, theme and lack of overlays')
    args=parser.parse_args()
    require(args.visually_verified, 'Inspect the real capture before passing --visually-verified')
    registry=json.loads((P/'figures.json').read_text())
    require(args.figure.startswith('shot_') and args.figure in registry,'Unknown screenshot figure')
    src=Path(args.image); sample=Path(args.approved_sample)
    manifest_path=P/'evidence/captures.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'version':1,'approved_sample':{'sha256':sha(sample),'confirmed_by_user':True,
                                     'confirmed_on':'2026-09-29','dimensions':[3840,2280]},'screenshots':[]}
    approved=manifest['approved_sample']
    require(approved.get('confirmed_by_user') is True and sha(sample)==approved['sha256'],
            'Supplied sample differs from the user-approved sample')
    sidecar_path=Path(args.sidecar)
    crop_path=Path(args.crop_manifest) if args.crop_manifest else src.with_suffix('.crop.json')
    sidecar=json.loads(sidecar_path.read_text(encoding='utf-8-sig'))
    crop=json.loads(crop_path.read_text(encoding='utf-8-sig'))
    capture_crop_contract(sidecar,crop,src,approved['dimensions'])
    check_png(src,approved['dimensions'],terminal=True)
    for name in ('log','script'):
        require(Path(getattr(args,name)).is_file() and Path(getattr(args,name)).stat().st_size>0,
                'Missing or empty capture '+name)
    evidence=P/'evidence/captures'
    supporting=prepare_support_files(args.support_file,evidence,registry)
    # No destination is touched before all acceptance checks above pass.
    evidence.mkdir(parents=True,exist_ok=True)
    destination=P/'figs'/(args.figure+'.png')
    shutil.copyfile(src,destination)
    for source,suffix in ((Path(args.log),'.log'),(Path(args.script),'.sh'),
                          (sidecar_path,'.capture.json'),(crop_path,'.crop.json')):
        shutil.copyfile(source,evidence/(args.figure+suffix))
    for source,target,identity in supporting:
        if not target.exists():
            shutil.copyfile(source,target)
        require(target.stat().st_size == identity['bytes'] and sha(target) == identity['sha256'],
                'Support file changed during installation: ' + target.name)
    record={'figure':args.figure,'verified':True,'visual_review':{'verified':True,'scope':'Terminal content, command identity, theme and overlays manually inspected'},
            'sha256':sha(destination),'dimensions':approved['dimensions'],'captured_at':sidecar['captured_at'],
            'exit_status':0,'capture_status':sidecar['capture_status'],'image_quality':sidecar['image_quality'],
            'image_processing':'taskbar crop only'}
    for field,suffix in (('raw_log','.log'),('script','.sh'),('capture_sidecar','.capture.json'),('crop_manifest','.crop.json')):
        path=evidence/(args.figure+suffix)
        record[field]=path.relative_to(P).as_posix();record[field+'_sha256']=sha(path)
    if supporting:
        record['supporting_files']=[identity for _,_,identity in supporting]
    manifest['screenshots']=sorted([r for r in manifest['screenshots'] if r['figure']!=args.figure]+[record],key=lambda r:r['figure'])
    temporary=manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(manifest_path)
    print(f'Recorded {args.figure}: image, raw log, script, capture quality and crop proof hashed')


if __name__=='__main__':main()
