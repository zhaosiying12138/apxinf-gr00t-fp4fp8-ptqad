#!/usr/bin/env python3
"""Copy all sixteen complete five-arm JSON files byte-for-byte after recomputing.

This supplies the top-level heldout mirrors required by validate_publication.py.
Frontier raw logs and training cost metadata are collected by their own tools.
No GPU, model payload, simulation or inference is used here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eval'))
from compare_recovery import ARMS, compare_round

NAMES=('paired_comparison.json',)+tuple(
    f'heldout_{arm}/{name}.json' for arm in ARMS
    for name in ('eval_manifest','task_results','summary'))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def destination(root,name):
    path=root/name
    for candidate in (path,*path.parents):
        require(not candidate.is_symlink(),'Symbolic output path rejected: '+str(candidate))
    return path


def audit(round_dir,recorded):
    actual=compare_round(round_dir)
    require(actual==recorded,'Five-arm comparison does not reproduce')
    require(actual.get('environment_pairing_verified') is True,'Pairing not verified')
    for arm,row in actual['arms'].items():
        require(row['count']==100 and len(row['episodes'])==100,'Incomplete heldout arm: '+arm)
        require(len(row['per_task'])==10 and all(v['episodes']==10 for v in row['per_task'].values()),
                'Not ten tasks with ten episodes: '+arm)
        require(all(type(e['success']) is bool and e['init_state_index'] in range(10,20)
                    for e in row['episodes']),'Invalid outcome/bank index: '+arm)
    return actual


def collect(round_dir,out):
    origin=Path(round_dir).resolve(strict=True)
    # Keep the unresolved spelling long enough to reject symlink destinations.
    output=Path(out).absolute();destination(output,'paired_comparison.json')
    require(output.resolve()!=origin,'Source round cannot be its own publication output')
    recorded=json.loads((origin/'paired_comparison.json').read_text())
    audit(origin,recorded)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pairing-copy-',dir=output.parent) as temp:
        stage=Path(temp);identities={}
        for name in NAMES:
            source,target=origin/name,stage/name
            require(source.is_file() and not source.is_symlink(),'Missing regular source: '+name)
            expected=digest(source);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            require(digest(source)==expected and digest(target)==expected,'Source changed during byte copy: '+name)
            identities[name]={'sha256':expected,'bytes':target.stat().st_size}
        audit(stage,recorded)
        # Check every destination before installing any file. Identical existing
        # copies are accepted so retrying does not destroy unrelated evidence.
        for name,identity in identities.items():
            target=destination(output,name)
            if target.exists():
                require(target.is_file(),'Non-file output already exists: '+name)
                if digest(target)!=identity['sha256']:
                    raise FileExistsError('Refusing different published evidence: '+str(target))
        for name in NAMES:
            target=destination(output,name);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                # Same-parent staging makes hard-link installation atomic and
                # refuses a racing existing destination without replacement.
                os.link(stage/name,target)
        audit(output,recorded)
    return {'status':'complete','scope':'Sixteen byte-identical original JSON files; three complete comparisons checked, no raw logs or model tensors copied.',
            'source_round':str(origin),'output':str(output),'files':identities}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',required=True,help='Completed run_onpolicy_round output')
    parser.add_argument('--out',required=True,help='Publication evidence directory; differing existing files are rejected')
    args=parser.parse_args();print(json.dumps(collect(args.round,args.out),indent=2))

if __name__=='__main__':main()
