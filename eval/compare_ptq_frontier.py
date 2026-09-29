"""Freeze development-only PTQ references, then audit an independent heldout frontier.

Freeze accepts no round/heldout argument. All comparisons use the unchanged
five-arm evaluator and a separately frozen reference list. Standard library only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys

from compare_development import ORDER, audit_development
from compare_recovery import ARMS, compare_round
from run_recovery_eval import TASKS, validate_resets, parse_log

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'exp/ptq_frontier_protocol.json'
SOURCES=('eval/compare_ptq_frontier.py','eval/compare_development.py','eval/compare_recovery.py',
         'eval/run_recovery_eval.py','eval/rollout_seeded.py','eval/serve_recovery.py',
         'exp/run_ptq_frontier.sh','exp/ptq_frontier_protocol.json','exp/recovery_protocol.json')
META=('config.json','statistics.json','processor_config.json','embodiment_id.json',
      'model.safetensors.index.json')
FIELDS=('seed','episodes','tasks','n_envs','n_action_steps','max_episode_steps',
        'initial_state_protocol','init_state_indices','settle_steps','protocol_sha256')


def need(ok,message):
    if not ok:raise ValueError(message)


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def identity(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def check_identity(row):
    need(identity(row['path'])==row,'Frozen evidence changed: '+row['path'])


def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(data,stream,indent=2);stream.write('\n')


def checkpoint_identity(folder,recipe=None):
    folder=Path(folder).resolve(strict=True)
    need(not any(c in str(folder) for c in '\t\r\n'),'Checkpoint path has control characters')
    names=list(META)
    if recipe:
        names+=['ptq_recipe.json','bake_manifest.json']
        need(not (folder/'merge_manifest.json').exists(),'Reference must be pure PTQ, not a recovery export')
        need(load(folder/'ptq_recipe.json')['recipe']==recipe,'Checkpoint recipe differs from selected development arm')
        need(load(folder/'bake_manifest.json')['status']=='complete','Incomplete PTQ checkpoint')
    index=load(folder/'model.safetensors.index.json')
    shards=sorted(set(index['weight_map'].values()))
    need(shards and all(Path(name).name==name and name.endswith('.safetensors') for name in shards),'Invalid checkpoint weight index')
    need({p.name for p in folder.glob('*.safetensors')}==set(shards),'Checkpoint shard files differ from index')
    metadata={name:identity(folder/name) for name in names}
    weights={name:identity(folder/name) for name in shards}
    if recipe:
        recorded=load(folder/'bake_manifest.json')['output_weight_files']
        need(set(recorded)==set(weights),'Bake output shard set differs')
        for name,row in weights.items():
            need(all(recorded[name][k]==row[k] for k in ('bytes','sha256')),'PTQ shard differs from completed bake: '+name)
    return {'path':str(folder),'recipe':recipe,'metadata':metadata,'weights':weights}


def checked_selection(selection_path,development_root):
    selection=load(selection_path)
    need(selection.get('selection_uses_heldout') is False,'Selection must explicitly exclude heldout')
    selected=selection.get('selected_recipe')
    need(selected in ORDER[2:],'Development has not selected a promoted recipe')
    arms=list(selection['arms']);expected=ORDER[:ORDER.index(selected)+1]
    need(set(arms)==set(expected),'Selection must contain exactly the completed prefix through the first selected level')
    audited=audit_development(development_root,expected)
    need(selection==audited,'Selection does not reproduce from current audited development records')
    for name in expected:
        manifest=load(Path(development_root)/name/'eval_manifest.json')
        need(manifest['protocol_sha256']==sha(ROOT/'exp/recovery_protocol.json'),'Development used another recovery protocol')
        tasks=load(Path(development_root)/name/'task_results.json')
        need(all(type(v) is bool for row in tasks.values() for v in row['results']),'Development outcomes must be Boolean')
    return selection,expected[1:-1]


def encoding_budget(memory):
    dedup=memory['known_tied_alias_deduplicated']
    result={'physical':{'source_bytes':memory['source_tensor_bytes'],'base_bytes':memory['target_full_checkpoint_bytes']},
            'known_alias_deduplicated':{'source_bytes':dedup.get('source_tensor_bytes',2*dedup['all_tensor_elements']),
                                        'base_bytes':dedup['target_full_bytes']}}
    need(all(type(v) is int and v>0 for row in result.values() for v in row.values()),'Invalid encoding budget')
    return result


def tensor_identity(folder,key):
    folder=Path(folder);index=load(folder/'model.safetensors.index.json');shard=index['weight_map'][key]
    need(Path(shard).name==shard,'Tensor shard must be a flat file')
    with (folder/shard).open('rb') as stream:
        header_size=struct.unpack('<Q',stream.read(8))[0]
        need(0<header_size<64*1024*1024,'Invalid safetensors header size')
        row=json.loads(stream.read(header_size))[key];start,end=row['data_offsets']
        need(row['dtype']=='BF16' and len(row['shape'])==2,'Expected a BF16 two-dimensional tied weight')
        count=row['shape'][0]*row['shape'][1]
        need(0<=start<end and end-start==count*2 and end+8+header_size<=(folder/shard).stat().st_size,'Invalid tensor byte range')
        stream.seek(8+header_size+start);remaining=end-start;digest=hashlib.sha256()
        while remaining:
            block=stream.read(min(8*1024*1024,remaining));need(bool(block),'Truncated tied tensor')
            digest.update(block);remaining-=len(block)
    return {'shape':row['shape'],'dtype':row['dtype'],'bytes':end-start,'sha256':digest.hexdigest()}


def budget_for(folder,name,inventory):
    actual=load(Path(folder)/'ptq_recipe.json');memory=actual['memory'];expected=encoding_budget(inventory['recipes'][name])
    need(expected['physical']=={'source_bytes':memory['source_tensor_bytes'],'base_bytes':memory['target_full_checkpoint_bytes']},'Physical budget differs from actual PTQ metadata')
    if 'known_tied_alias_deduplicated' in memory:
        result=encoding_budget(memory);proof={'method':'completed_ptq_metadata'}
    else:
        # Some completed bakes recorded only physical storage. Derive the other
        # denominator from verified actual tied bytes; never edit those artifacts.
        source=memory['source_tensor_bytes'];target=memory['target_full_checkpoint_bytes'];aliases=[]
        need(bool(inventory.get('tied_aliases')),'No explicit alias mapping for budget derivation')
        for alias,canonical in inventory['tied_aliases'].items():
            left=tensor_identity(folder,alias);right=tensor_identity(folder,canonical)
            need(left==right,'Tied tensor values differ; refusing alias deduction')
            methods=[actual['layers'][key.removesuffix('.weight')]['actual_method'] for key in (alias,canonical)]
            need(methods[0]==methods[1],'Tied quantization methods differ')
            n,k=left['shape'];count=n*k
            if methods[0]=='fp8':removed=count+4*n
            elif methods[0].startswith('nvfp4'):
                need(k%16==0,'NVFP4 alias width must divide by 16');removed=count//2+count//16+4
            elif methods[0]=='bf16':removed=2*count
            else:raise ValueError('Unknown alias encoding method')
            source-=left['bytes'];target-=removed
            aliases.append({'alias':alias,'canonical':canonical,'tensor':left,'removed_source_bytes':left['bytes'],'removed_target_bytes':removed})
        result={'physical':expected['physical'],'known_alias_deduplicated':{'source_bytes':source,'base_bytes':target}}
        proof={'method':'derived_from_actual_equal_tied_tensor_bytes_and_frozen_inventory','aliases':aliases}
    need(result==expected,'Derived alias budget differs from frozen inventory')
    return result,proof


def freeze(selection_path,development_root,budget_path):
    selection_path=Path(selection_path).resolve();development_root=Path(development_root).resolve()
    selection,names=checked_selection(selection_path,development_root)
    inventory=load(budget_path);residual=inventory['recovery_residual']
    need(residual['dtype']=='bfloat16' and residual['bytes_per_element']==2 and
         residual['target_bytes']==2*residual['tensor_elements'],'Residual budget must include BF16 payload')
    checkpoints={}
    for name in ['bf16',*names,selection['selected_recipe']]:
        checkpoints[name]=checkpoint_identity(selection['arms'][name]['checkpoint'],None if name=='bf16' else name)
    bf16=checkpoints['bf16']
    need(Path(inventory['base']).resolve()==Path(bf16['path']) and inventory['recovery_residual']['source_index_sha256']==bf16['metadata']['model.safetensors.index.json']['sha256'] and inventory['recovery_residual']['source_config_sha256']==bf16['metadata']['config.json']['sha256'],'Budget source checkpoint metadata differs')
    budgets={};derivations={}
    for name in [*names,selection['selected_recipe']]:
        actual=load(Path(checkpoints[name]['path'])/'ptq_recipe.json')
        need(Path(actual['base']).resolve()==Path(checkpoints['bf16']['path']),'PTQ source differs from BF16 development checkpoint')
        budgets[name],derivations[name]=budget_for(checkpoints[name]['path'],name,inventory)
    return {'version':1,'status':'frozen','created_utc':datetime.now(timezone.utc).isoformat(),
            'reference_selection_uses_heldout':False,'selected_recipe':selection['selected_recipe'],'references':names,
            'development_root':str(development_root),'selection':identity(selection_path),
            'development_sources':{name:identity(development_root/name) for name in selection['source_sha256']},
            'source_files':{name:identity(ROOT/name) for name in SOURCES},'budget_source':identity(budget_path),
            'checkpoints':checkpoints,'budget_derivations':derivations,'encoding_budgets':budgets,'recovery_residual':residual,
            'heldout_protocol':load(PROTOCOL)['heldout'],
            'note':'Checkpoint bytes are hashed at freeze and rechecked before evaluation/comparison. Freeze reads no heldout data.'}


def check_plan(path):
    plan=load(path)
    need(plan.get('version')==1 and plan.get('status')=='frozen' and plan.get('reference_selection_uses_heldout') is False,'Expected a frozen development-only plan')
    for record in [plan['selection'],plan['budget_source'],*plan['development_sources'].values(),*plan['source_files'].values()]:check_identity(record)
    need(set(plan['source_files'])==set(SOURCES),'Plan source identity set differs')
    need(plan['heldout_protocol']==load(PROTOCOL)['heldout'],'Plan heldout protocol differs')
    selection,names=checked_selection(plan['selection']['path'],plan['development_root'])
    need(plan['references']==names and plan['selected_recipe']==selection['selected_recipe'],'Frozen reference list was changed')
    need(set(plan['checkpoints'])=={'bf16',*names,plan['selected_recipe']},'Plan checkpoint set differs')
    inventory=load(plan['budget_source']['path'])
    need(plan['recovery_residual']==inventory['recovery_residual'],'Frozen residual budget differs')
    for name,checkpoint in plan['checkpoints'].items():
        need(checkpoint['path']==str(Path(selection['arms'][name]['checkpoint']).resolve()),'Plan checkpoint differs from development selection')
        need(checkpoint_identity(checkpoint['path'],None if name=='bf16' else name)==checkpoint,
             'Checkpoint metadata or weight bytes differ from frozen plan: '+name)
        if name!='bf16':
            current,proof=budget_for(checkpoint['path'],name,inventory)
            need(plan['encoding_budgets'][name]==current and plan['budget_derivations'][name]==proof,'Frozen encoding budget differs')
    return plan


def read_heldout(folder,protocol_hash):
    folder=Path(folder);manifest=load(folder/'eval_manifest.json');results=load(folder/'task_results.json');summary=load(folder/'summary.json')
    expected=load(PROTOCOL)['heldout']
    values={'purpose':'heldout','seed':expected['seed'],'episodes':10,'tasks':TASKS,
            'init_state_indices':expected['init_state_indices'],'protocol_sha256':protocol_hash,
            **{key:expected[key] for key in ('n_envs','n_action_steps','max_episode_steps','settle_steps','initial_state_protocol')}}
    need(all(manifest.get(k)==v for k,v in values.items()),'Heldout protocol differs: '+str(folder))
    need(set(results)==set(TASKS) and summary['tasks_complete']==10,'Heldout task set is incomplete')
    episodes=[];per_task={}
    for index,task in enumerate(TASKS):
        result=results[task]
        need(result.get('returncode')==0 and len(result['results'])==10 and all(type(v) is bool for v in result['results']), 'Incomplete/non-Boolean heldout result: '+task)
        observed=parse_log(folder/(task+'.log'))
        need(all(result.get(key)==value for key,value in observed.items()),'Task JSON differs from raw log: '+task)
        need(result.get('seed')==220000+1000*index,'Task seed differs')
        resets=validate_resets(result,220000+1000*index,list(range(10,20)))
        for outcome,reset in zip(result['results'],resets):
            episodes.append({'task':task,**{k:reset[k] for k in ('episode_index','init_state_index','initial_state_sha256','restored_state_sha256','init_state_bank_sha256')},'success':outcome})
        per_task[task]={'episodes':10,'successes':sum(result['results']),'success_rate':sum(result['results'])/10}
    successes=sum(row['success'] for row in episodes)
    need(summary['total_successes']==successes and summary['total_episodes']==100 and summary['purpose']=='heldout' and summary['seed']==220000 and summary['macro_success_rate']==sum(row['success_rate'] for row in per_task.values())/10,'Summary differs from verified episodes')
    return manifest,{'count':100,'successes':successes,'macro_success_rate':sum(row['success_rate'] for row in per_task.values())/10,'episodes':episodes,'per_task':per_task}


def pair(actual,reference,label):
    need(len(actual)==len(reference)==100,'Expected 100 paired episodes')
    for a,b in zip(actual,reference):
        need({k:v for k,v in a.items() if k!='success'}=={k:v for k,v in b.items() if k!='success'},'Initial-state pairing differs: '+label)


def checked_main(plan,round_dir):
    round_dir=Path(round_dir).resolve()
    main=compare_round(round_dir)
    need(main==load(round_dir/'paired_comparison.json'),'Main five-arm comparison does not reproduce')
    protocol_hash=plan['source_files']['exp/recovery_protocol.json']['sha256']
    for arm in ARMS:
        manifest,actual=read_heldout(round_dir/('heldout_'+arm),protocol_hash)
        need(actual==main['arms'][arm],'Main arm does not satisfy full 100-episode contract')
        pair(actual['episodes'],main['arms']['bf16']['episodes'],arm)
        if arm in ('bf16','ptq'):
            expected=plan['checkpoints']['bf16' if arm=='bf16' else plan['selected_recipe']]['path']
            need(Path(manifest['checkpoint']).resolve()==Path(expected),'Main arm checkpoint differs from frozen development selection')
    collection=round_dir/'collection/eval_manifest.json';record=load(collection)
    need(record['protocol_sha256']==protocol_hash and record['initial_state_protocol']=='libero10_official_bank_v1' and record['n_envs']==1 and record['settle_steps']==10,'Main collection provenance differs')
    need(record['purpose']=='collection' and record['seed']==110000 and record['episodes']==2 and record['tasks']==TASKS and record['init_state_indices']==[2,3], 'Main collection protocol differs')
    for arm in ARMS:
        m=load(round_dir/('heldout_'+arm)/'eval_manifest.json')
        need(Path(m['collection_manifest']).resolve()==collection,'Main arm used a different collection manifest')
    return main,collection


def residual_identity(checkpoint,plan):
    path=Path(checkpoint)/'merge_manifest.json';merge=load(path);r=merge['recovery_manifest'];budget=plan['recovery_residual']
    need(merge['status']=='complete' and Path(merge['base']).resolve()==Path(plan['checkpoints'][plan['selected_recipe']]['path']),'Recovery merge used another base')
    need(r['rank']==budget['rank'] and r['alpha']==budget['alpha'] and r['scope']==budget['scope'] and
         r['trainable_parameters']==budget['tensor_elements'] and merge['lora_pairs']==budget['linear_modules'],'Actual recovery residual differs from frozen encoding budget')
    return identity(path)


def point(name,role,recipe,arm,budget,residual):
    costs={key:{'base_bytes':value['base_bytes'],'residual_bytes':residual,
                'total_bytes':value['base_bytes']+residual,'source_bytes':value['source_bytes'],
                'compression_x':value['source_bytes']/(value['base_bytes']+residual)} for key,value in budget.items()}
    return {'name':name,'role':role,'recipe':recipe,'count':arm['count'],'successes':arm['successes'],
            'macro_success_rate':arm['macro_success_rate'],'per_task':arm['per_task'],'encoding_budget':costs}


def compare(plan_path,round_dir,output_dir):
    plan=check_plan(plan_path);main,collection=checked_main(plan,round_dir)
    round_dir=Path(round_dir).resolve();output_dir=Path(output_dir).resolve()
    run=load(output_dir/'run_manifest.json')
    need(run['plan']==identity(plan_path) and run['collection_manifest']==identity(collection) and
         run['main_comparison']==identity(round_dir/'paired_comparison.json'),'Frontier run identities differ')
    base_manifest=load(round_dir/'heldout_bf16/eval_manifest.json')
    need({p.name for p in output_dir.glob('heldout_*')}=={'heldout_'+name for name in plan['references']},'Frontier output set differs from frozen references')
    references={};source_files={};merges={}
    for name in plan['references']:
        folder=output_dir/('heldout_'+name)
        manifest,arm=read_heldout(folder,plan['source_files']['exp/recovery_protocol.json']['sha256'])
        need(all(manifest.get(key)==base_manifest.get(key) for key in (*FIELDS,'gr00t','server_python','rollout_python','server_seed_offset')),'Reference differs from main evaluation protocol')
        need(Path(manifest['checkpoint']).resolve()==Path(plan['checkpoints'][name]['path']),'Reference checkpoint differs from frozen plan')
        need(Path(manifest['collection_manifest']).resolve()==collection,'Reference used another collection manifest')
        pair(arm['episodes'],main['arms']['bf16']['episodes'],name);references[name]=arm
        for filename in ('eval_manifest.json','task_results.json','summary.json',*(task+'.log' for task in TASKS)):source_files[str(folder/filename)]=identity(folder/filename)
    selected=plan['selected_recipe'];budget=plan['encoding_budgets'][selected]
    full={key:{'source_bytes':value['source_bytes'],'base_bytes':value['source_bytes']} for key,value in budget.items()}
    points=[point('bf16','bf16',None,main['arms']['bf16'],full,0)]
    points.extend(point(name,'pure_ptq_reference',name,references[name],plan['encoding_budgets'][name],0) for name in plan['references'])
    points.append(point('ptq','selected_ptq',selected,main['arms']['ptq'],budget,0))
    for arm in ('qad','continued_qad','qad_opd'):
        m=load(round_dir/('heldout_'+arm)/'eval_manifest.json');merges[arm]=residual_identity(m['checkpoint'],plan)
        points.append(point(arm,'recovery',selected,main['arms'][arm],budget,plan['recovery_residual']['target_bytes']))
    descriptive={}
    for scope in ('physical','known_alias_deduplicated'):
        descriptive[scope]=[a['name'] for a in points if not any(
            b['encoding_budget'][scope]['total_bytes']<=a['encoding_budget'][scope]['total_bytes'] and b['successes']>=a['successes'] and
            (b['encoding_budget'][scope]['total_bytes']<a['encoding_budget'][scope]['total_bytes'] or b['successes']>a['successes']) for b in points)]
    for arm in ARMS:
        for filename in ('eval_manifest.json','task_results.json','summary.json',*(task+'.log' for task in TASKS)):
            path=round_dir/('heldout_'+arm)/filename;source_files[str(path)]=identity(path)
    return {'version':1,'status':'complete','environment_pairing_verified':True,'plan':identity(plan_path),
            'main_comparison':identity(round_dir/'paired_comparison.json'),'collection_manifest':identity(collection),
            'references':references,'points':points,'observed_nondominated_points':descriptive,
            'source_files':source_files,'recovery_merge_manifests':merges,'implementation':identity(__file__),
            'interpretation':'Descriptive success-versus-encoding-budget comparison of the frozen tested set. No formal noninferiority, global PTQ optimum, packed deployment or heldout-driven tuning claim.'}


def run(plan_path,round_dir,output_dir,port,dry_run=False):
    plan=check_plan(plan_path);_,collection=checked_main(plan,round_dir)
    round_dir=Path(round_dir).resolve();output_dir=Path(output_dir).resolve()
    need(not output_dir.exists(),'Use a new frontier output directory')
    reference=load(round_dir/'heldout_bf16/eval_manifest.json');commands=[]
    for name in plan['references']:
        command=[sys.executable,str(ROOT/'eval/run_recovery_eval.py'),'--checkpoint',plan['checkpoints'][name]['path'],
                 '--out',str(output_dir/('heldout_'+name)),'--purpose','heldout','--seed','220000','--episodes','10',
                 '--port',str(port),'--collection-manifest',str(collection),'--gr00t',reference['gr00t']]
        for field in ('server_python','rollout_python'):
            if reference.get(field):command+=['--'+field.replace('_','-'),reference[field]]
        commands.append({'recipe':name,'command':command})
    if dry_run:
        print(json.dumps({'dry_run':True,'commands':commands},indent=2));return
    output_dir.mkdir(parents=True)
    write_new(output_dir/'run_manifest.json',{'plan':identity(plan_path),'collection_manifest':identity(collection),
              'main_comparison':identity(round_dir/'paired_comparison.json'),'commands':commands,
              'started_utc':datetime.now(timezone.utc).isoformat()})
    for job in commands:
        print('[frontier] evaluating '+job['recipe'],flush=True)
        with (output_dir/(job['recipe']+'.eval.log')).open('x') as stream:
            subprocess.run(job['command'],stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    result=compare(plan_path,round_dir,output_dir)
    write_new(output_dir/'frontier_comparison.json',result)
    print(json.dumps({'references':plan['references'],'observed_nondominated_points':result['observed_nondominated_points']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('freeze');p.add_argument('--selection',required=True);p.add_argument('--development-root',required=True)
    p.add_argument('--budget',default=str(ROOT/'paper/evidence/recipe_inventory.json'));p.add_argument('--out',required=True)
    p=sub.add_parser('check-plan');p.add_argument('--plan',required=True)
    for mode in ('run','compare'):
        p=sub.add_parser(mode);p.add_argument('--plan',required=True);p.add_argument('--round',required=True);p.add_argument('--out',required=True)
        if mode=='run':p.add_argument('--port',type=int,default=5595);p.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.mode=='freeze':
        need(not Path(args.out).exists(),'Refusing existing frozen plan')
        result=freeze(args.selection,args.development_root,args.budget);write_new(args.out,result)
        print(json.dumps({'frozen_plan':str(Path(args.out).resolve()),'selected_recipe':result['selected_recipe'],'references':result['references']},indent=2))
    elif args.mode=='check-plan':
        result=check_plan(args.plan);print(json.dumps({'status':'verified','references':result['references']}))
    elif args.mode=='run':run(args.plan,args.round,args.out,args.port,args.dry_run)
    else:write_new(Path(args.out)/'frontier_comparison.json',compare(args.plan,args.round,args.out))


if __name__=='__main__':main()
