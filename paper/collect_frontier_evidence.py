#!/usr/bin/env python3
"""Snapshot a completed frozen PTQ frontier without rewriting experiment JSON.

CPU/file IO only. Revalidates original weight identities through the original
comparison, then copies only logs, JSON, source and checkpoint metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from frontier_evidence import f, validate_frontier, digest
from publication_guard import require

P=Path(__file__).resolve().parent


def collect(frontier,paper=P):
    frontier=Path(frontier).resolve();paper=Path(paper).resolve()
    comparison_path=frontier/'frontier_comparison.json';comparison=f.load(comparison_path)
    plan_path=Path(comparison['plan']['path']);plan=f.load(plan_path)
    main_path=Path(comparison['main_comparison']['path']);round_dir=main_path.parent
    require(comparison==f.compare(plan_path,round_dir,frontier),'Completed frontier comparison does not reproduce')
    destination=paper/'evidence/frontier';final_comparison=paper/'evidence/frontier_comparison.json'
    require(not destination.exists() and not final_comparison.exists(),'Refusing to overwrite published frontier evidence')
    jobs={}
    def add(record,target,role):
        original=record['path']
        if original in jobs:
            require(jobs[original][0]==record,'One source has conflicting identities');return
        jobs[original]=(record,target,role)
    def provenance(record,role):
        name=hashlib.sha256(record['path'].encode()).hexdigest()[:16]+'-'+Path(record['path']).name
        add(record,'evidence/frontier/provenance/'+name,role)
    add(f.identity(comparison_path),'evidence/frontier_comparison.json','frontier_comparison')
    add(comparison['plan'],'evidence/frontier/plan.json','frozen_plan')
    add(f.identity(frontier/'run_manifest.json'),'evidence/frontier/run_manifest.json','reference_run')
    add(comparison['main_comparison'],'evidence/frontier/main/paired_comparison.json','five_arm_comparison')
    add(comparison['collection_manifest'],'evidence/frontier/main/collection/eval_manifest.json','collection_protocol')
    for original,record in comparison['source_files'].items():
        path=Path(original)
        if path.is_relative_to(round_dir):target='evidence/frontier/main/'+path.relative_to(round_dir).as_posix()
        else:
            require(path.is_relative_to(frontier),'Unexpected frontier raw evidence location')
            target='evidence/frontier/references/'+path.relative_to(frontier).as_posix()
        add(record,target,'heldout_raw_log' if path.suffix=='.log' else 'heldout_json')
    for name,record in plan['development_sources'].items():add(record,'evidence/frontier/development/'+name,'development_json')
    development=Path(plan['development_root'])
    for name in plan['checkpoints']:
        for task in f.TASKS:
            path=development/name/(task+'.log')
            add(f.identity(path),'evidence/frontier/development/'+name+'/'+path.name,'development_raw_log')
    provenance(plan['selection'],'development_selection');provenance(plan['budget_source'],'frozen_encoding_inventory')
    for record in plan['source_files'].values():provenance(record,'frozen_source')
    for checkpoint in plan['checkpoints'].values():
        for record in checkpoint['metadata'].values():provenance(record,'checkpoint_metadata')
    for record in comparison['recovery_merge_manifests'].values():provenance(record,'recovery_merge_manifest')
    provenance(comparison['implementation'],'comparison_source')
    require(len({target for _,target,_ in jobs.values()})==len(jobs),'Duplicate public evidence destination')
    work=paper/'_build';work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='frontier-evidence-',dir=work) as raw:
        stage=Path(raw);mapping=[]
        for original,(record,target,role) in sorted(jobs.items()):
            f.check_identity(record)
            output=stage/target;output.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,output)
            require(output.stat().st_size==record['bytes'] and digest(output)==record['sha256'],'Evidence changed during copy')
            mapping.append({'original_absolute_path':original,'published_path':target,'bytes':record['bytes'],'sha256':record['sha256'],'role':role})
        inventory=paper/'evidence/recipe_inventory.json'
        shutil.copyfile(inventory,stage/'evidence/recipe_inventory.json')
        manifest={'version':1,'status':'complete','scope':'Byte-identical original JSON/log/source copies; absolute paths are resolved only through this mapping. Weight payloads are omitted; frozen identities and bake metadata are retained.','mapping':mapping}
        (stage/'evidence/frontier/evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        validate_frontier(stage)
        require(digest(inventory)==digest(stage/'evidence/recipe_inventory.json'),'Publication inventory changed during snapshot')
        # Rename only after every copied log, pair, budget and identity passed.
        (stage/'evidence/frontier').rename(destination)
        (stage/'evidence/frontier_comparison.json').rename(final_comparison)
    return {'status':'complete','copied_files':len(jobs),'manifest':str(destination/'evidence_manifest.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontier',required=True,help='Completed output directory of exp/run_ptq_frontier.sh')
    args=parser.parse_args();print(json.dumps(collect(args.frontier),ensure_ascii=False))
