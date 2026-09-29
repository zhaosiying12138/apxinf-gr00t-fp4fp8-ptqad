"""Portable CPU validation of the published PTQ frontier evidence.

Original experiment JSON stays byte-identical. An explicit path map resolves its
absolute identities to copied files; model payloads remain recorded identities.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

from publication_guard import require, resolve_inside

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'eval'))
import compare_ptq_frontier as f


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def load(path):return json.loads(Path(path).read_text())


class PublishedMap:
    def __init__(self,paper):
        self.paper=Path(paper)
        manifest=self.paper/'evidence/frontier/evidence_manifest.json'
        data=load(manifest)
        require(data.get('version')==1 and data.get('status')=='complete','Frontier evidence copy did not complete')
        self.files={};self.paths={manifest}
        destinations=set()
        for row in data['mapping']:
            original=row['original_absolute_path'];target=row['published_path']
            require(Path(original).is_absolute() and original not in self.files,'Duplicate/non-absolute frontier source mapping')
            require(target not in destinations and isinstance(row.get('role'),str) and row['role'],'Duplicate frontier destination or missing role')
            path=resolve_inside(self.paper,target)
            require(type(row['bytes']) is int and row['bytes']==path.stat().st_size and row['sha256']==digest(path),'Changed frontier evidence copy: '+target)
            require(path.suffix not in {'.safetensors','.pt','.pth'},'Weight payload must not enter frontier publication')
            self.files[original]=(row,path);destinations.add(target);self.paths.add(path)

    def resolve(self,identity):
        require(set(identity)=={'path','bytes','sha256'},'Incomplete frontier file identity')
        require(identity['path'] in self.files,'Missing frontier source copy: '+identity['path'])
        row,path=self.files[identity['path']]
        require(all(row[k]==identity[k] for k in ('bytes','sha256')),'Frontier identity/copy mismatch: '+identity['path'])
        return path

    def source(self,path):
        require(str(path) in self.files,'Missing frontier original path: '+str(path))
        return self.files[str(path)][1]


def validate_frontier(paper,paired=None):
    paper=Path(paper);mapping=PublishedMap(paper)
    comparison=load(paper/'evidence/frontier_comparison.json')
    require(comparison.get('version')==1 and comparison.get('status')=='complete' and comparison.get('environment_pairing_verified') is True,'Incomplete frontier comparison')
    plan_path=mapping.resolve(comparison['plan']);plan=load(plan_path)
    require(plan_path==paper/'evidence/frontier/plan.json','Unexpected published frontier plan path')
    require(plan.get('version')==1 and plan.get('status')=='frozen' and plan.get('reference_selection_uses_heldout') is False,'Frontier plan was not development-only frozen')
    require(set(plan['source_files'])==set(f.SOURCES),'Frozen frontier implementation set differs')
    for name,record in plan['source_files'].items():
        copy=mapping.resolve(record);current=ROOT/name
        require(digest(copy)==digest(current),'Frozen frontier implementation changed: '+name)
    require(mapping.resolve(comparison['implementation'])==mapping.resolve(plan['source_files']['eval/compare_ptq_frontier.py']),'Comparison implementation differs from frozen plan')
    protocol=load(mapping.resolve(plan['source_files']['exp/ptq_frontier_protocol.json']))
    require(plan['heldout_protocol']==protocol['heldout'],'Frozen heldout protocol differs')
    protocol_hash=plan['source_files']['exp/recovery_protocol.json']['sha256']
    selected=plan['selected_recipe'];order=f.ORDER[:f.ORDER.index(selected)+1]
    require(selected in f.ORDER[2:] and plan['references']==order[1:-1],'Frontier references omit a completed development predecessor')
    selection=load(mapping.resolve(plan['selection']))
    development=paper/'evidence/frontier/development'
    require(set(plan['development_sources'])==set(selection['source_sha256']),'Frozen development source set differs')
    for name,record in plan['development_sources'].items():
        require(mapping.resolve(record)==development/name,'Development source mapping differs')
    require(f.audit_development(development,order)==selection,'Published development evidence does not reproduce selection')
    require(selection['selected_recipe']==selected and selection['selection_uses_heldout'] is False,'Frozen selection differs')
    for name in order:
        folder=development/name;manifest=load(folder/'eval_manifest.json');tasks=load(folder/'task_results.json')
        require(manifest['protocol_sha256']==protocol_hash,'Development protocol changed')
        for task,row in tasks.items():
            require(all(type(v) is bool for v in row['results']),'Development outcomes must be Boolean')
            require(all(row.get(k)==v for k,v in f.parse_log(folder/(task+'.log')).items()),'Development JSON differs from copied raw log')
    inventory=load(mapping.resolve(plan['budget_source']))
    require(inventory==load(paper/'evidence/recipe_inventory.json'),'Figure budget differs from frozen inventory')
    require(plan['recovery_residual']==inventory['recovery_residual'],'Frozen residual budget changed')
    require(set(plan['checkpoints'])==set(order),'Frozen checkpoint set differs')
    bf16=plan['checkpoints']['bf16'];residual=plan['recovery_residual']
    require(inventory['base']==bf16['path'] and residual['source_index_sha256']==bf16['metadata']['model.safetensors.index.json']['sha256'] and residual['source_config_sha256']==bf16['metadata']['config.json']['sha256'],'Budget source checkpoint metadata differs')
    require(set(plan['encoding_budgets'])==set(order[1:]) and set(plan['budget_derivations'])==set(order[1:]),'Frozen budget candidate set differs')
    for name,checkpoint in plan['checkpoints'].items():
        require(checkpoint['path']==selection['arms'][name]['checkpoint'],'Checkpoint differs from development selection')
        expected=set(f.META) if name=='bf16' else set(f.META)|{'ptq_recipe.json','bake_manifest.json'}
        require(set(checkpoint['metadata'])==expected,'Incomplete frozen checkpoint metadata')
        for filename,record in checkpoint['metadata'].items():
            require(record['path']==str(Path(checkpoint['path'])/filename),'Checkpoint metadata path differs')
            mapping.resolve(record)
        index=load(mapping.resolve(checkpoint['metadata']['model.safetensors.index.json']))
        require(set(checkpoint['weights'])==set(index['weight_map'].values()) and bool(checkpoint['weights']),'Frozen weight index/shards differ')
        for filename,record in checkpoint['weights'].items():
            require(Path(filename).name==filename and record['path']==str(Path(checkpoint['path'])/filename) and type(record['bytes']) is int and record['bytes']>0 and re.fullmatch('[0-9a-f]{64}',record['sha256']),'Invalid recorded weight identity')
        if name=='bf16':continue
        recipe=load(mapping.resolve(checkpoint['metadata']['ptq_recipe.json']))
        bake=load(mapping.resolve(checkpoint['metadata']['bake_manifest.json']))
        require(recipe['recipe']==name and recipe['base']==plan['checkpoints']['bf16']['path'] and bake['status']=='complete','Checkpoint recipe/bake differs')
        require(set(bake['output_weight_files'])==set(checkpoint['weights']),'Bake output identities incomplete')
        for filename,record in checkpoint['weights'].items():
            require(all(bake['output_weight_files'][filename][k]==record[k] for k in ('bytes','sha256')),'Frozen weight identity differs from bake')
        expected_budget=f.encoding_budget(inventory['recipes'][name]);memory=recipe['memory']
        require(plan['encoding_budgets'][name]==expected_budget and expected_budget['physical']=={'source_bytes':memory['source_tensor_bytes'],'base_bytes':memory['target_full_checkpoint_bytes']},'Frozen encoding budget differs')
        proof=plan['budget_derivations'][name]
        if 'known_tied_alias_deduplicated' in memory:
            require(f.encoding_budget(memory)==expected_budget and proof=={'method':'completed_ptq_metadata'},'Completed alias budget differs')
        else:
            require(proof['method']=='derived_from_actual_equal_tied_tensor_bytes_and_frozen_inventory','Missing explicit alias derivation')
            source=memory['source_tensor_bytes'];target=memory['target_full_checkpoint_bytes']
            require({x['alias']:x['canonical'] for x in proof['aliases']}==inventory['tied_aliases'],'Alias proof mapping differs')
            for row in proof['aliases']:
                tensor=row['tensor'];n,k=tensor['shape'];count=n*k
                require(tensor['dtype']=='BF16' and tensor['bytes']==2*count and re.fullmatch('[0-9a-f]{64}',tensor['sha256']),'Invalid recorded equal-alias tensor identity')
                methods=[recipe['layers'][key.removesuffix('.weight')]['actual_method'] for key in (row['alias'],row['canonical'])]
                require(methods[0]==methods[1],'Alias encoding methods differ')
                method=methods[0]
                removed=count+4*n if method=='fp8' else count//2+count//16+4 if method.startswith('nvfp4') else count*2 if method=='bf16' else None
                require(removed is not None and row['removed_source_bytes']==count*2 and row['removed_target_bytes']==removed,'Alias byte deduction differs')
                source-=row['removed_source_bytes'];target-=removed
            require(expected_budget['known_alias_deduplicated']=={'source_bytes':source,'base_bytes':target},'Derived alias budget differs')
    main_path=mapping.resolve(comparison['main_comparison']);main_root=paper/'evidence/frontier/main'
    require(main_path==main_root/'paired_comparison.json','Unexpected main comparison mapping')
    main=load(main_path);require(f.compare_round(main_root)==main,'Copied five-arm frontier evidence does not reproduce')
    if paired is not None:require(main==paired,'Frontier refers to another five-arm experiment')
    collection=mapping.resolve(comparison['collection_manifest']);collection_data=load(collection)
    require(collection==main_root/'collection/eval_manifest.json','Collection mapping differs')
    require(all(collection_data.get(k)==v for k,v in {'purpose':'collection','seed':110000,'episodes':2,'tasks':f.TASKS,'init_state_indices':[2,3],'protocol_sha256':protocol_hash,'initial_state_protocol':'libero10_official_bank_v1','n_envs':1,'settle_steps':10}.items()),'Collection protocol differs')
    run=load(paper/'evidence/frontier/run_manifest.json')
    require(any(path==paper/'evidence/frontier/run_manifest.json' for _,path in mapping.files.values()),'Reference run manifest lacks source mapping')
    require(all(run[k]==comparison[k] for k in ('plan','main_comparison','collection_manifest')),'Reference run was not bound to this frozen plan/main experiment')
    base_manifest=None;references={};expected_sources=set()
    for name in [*f.ARMS,*plan['references']]:
        is_main=name in f.ARMS;folder=(main_root if is_main else paper/'evidence/frontier/references')/('heldout_'+name)
        manifest,arm=f.read_heldout(folder,protocol_hash)
        if name=='bf16':base_manifest=manifest
        f.pair(arm['episodes'],main['arms']['bf16']['episodes'],name)
        require(all(manifest.get(k)==base_manifest.get(k) for k in (*f.FIELDS,'gr00t','server_python','rollout_python','server_seed_offset')),'Frontier arm execution protocol differs')
        require(manifest['collection_manifest']==comparison['collection_manifest']['path'],'Frontier arm used another collection')
        if is_main:require(arm==main['arms'][name],'Main raw logs differ from paired comparison')
        else:references[name]=arm
        recipe_name='bf16' if name=='bf16' else selected if name=='ptq' else name
        if name in ('bf16','ptq',*plan['references']):require(manifest['checkpoint']==plan['checkpoints'][recipe_name]['path'],'Evaluated checkpoint differs from frozen plan')
        for filename in ('eval_manifest.json','task_results.json','summary.json',*(task+'.log' for task in f.TASKS)):
            path=folder/filename
            matches=[original for original,(_,copy) in mapping.files.items() if copy==path]
            require(len(matches)==1,'Missing uniquely mapped frontier raw evidence')
            expected_sources.add(matches[0])
    require(set(comparison['source_files'])==expected_sources,'Frontier source set omits raw evidence')
    for original,record in comparison['source_files'].items():
        require(original==record['path'],'Frontier source key/path identity differs')
        mapping.resolve(record)
    require(references==comparison['references'],'Reference results differ from raw logs')
    budget=plan['encoding_budgets'][selected]
    full={k:{'source_bytes':v['source_bytes'],'base_bytes':v['source_bytes']} for k,v in budget.items()}
    points=[f.point('bf16','bf16',None,main['arms']['bf16'],full,0)]
    points.extend(f.point(name,'pure_ptq_reference',name,references[name],plan['encoding_budgets'][name],0) for name in plan['references'])
    points.append(f.point('ptq','selected_ptq',selected,main['arms']['ptq'],budget,0))
    require(set(comparison['recovery_merge_manifests'])=={'qad','continued_qad','qad_opd'},'Recovery residual evidence set differs')
    residual=plan['recovery_residual']
    require(residual['dtype']=='bfloat16' and residual['bytes_per_element']==2 and residual['target_bytes']==2*residual['tensor_elements'],'Residual payload cost is invalid')
    for name in ('qad','continued_qad','qad_opd'):
        manifest=load(main_root/('heldout_'+name)/'eval_manifest.json');identity=comparison['recovery_merge_manifests'][name]
        require(identity['path']==str(Path(manifest['checkpoint'])/'merge_manifest.json'),'Recovery merge identity path differs')
        merge=load(mapping.resolve(identity));r=merge['recovery_manifest']
        require(merge['status']=='complete' and merge['base']==plan['checkpoints'][selected]['path'] and merge['lora_pairs']==residual['linear_modules'] and r['trainable_parameters']==residual['tensor_elements'] and all(r[k]==residual[k] for k in ('rank','alpha','scope')),'Recovered model does not match frozen residual budget')
        points.append(f.point(name,'recovery',selected,main['arms'][name],budget,residual['target_bytes']))
    require(points==comparison['points'],'Reported frontier points differ from recomputed success/net bytes')
    descriptive={scope:[a['name'] for a in points if not any(b['encoding_budget'][scope]['total_bytes']<=a['encoding_budget'][scope]['total_bytes'] and b['successes']>=a['successes'] and (b['encoding_budget'][scope]['total_bytes']<a['encoding_budget'][scope]['total_bytes'] or b['successes']>a['successes']) for b in points)] for scope in ('physical','known_alias_deduplicated')}
    require(descriptive==comparison['observed_nondominated_points'],'Reported observed frontier differs')
    mapping.paths.add(paper/'evidence/frontier_comparison.json')
    return comparison,mapping.paths
