"""Refresh an exact export candidate plan; never copy/delete public files.

Run after each final evidence collector/build, then audit_final_curated_export.py.
The manually reviewed seed is explicit. New paths come only from the named
manifests, result files and report names below; no recursive results/weights scan.
A candidate snapshot is never release approval, even if all inputs exist.
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('bf16', 'ptq', 'qad', 'continued_qad', 'qad_opd')
MAX_BYTES = 32 * 1024 * 1024
RESULT_FILES = [
    *[f'results/{folder}/{name}_ptqad_20260929.{ext}'
      for folder, names in (
          ('baselines', ('pi05_pt_bf16', 'gr00t_pt_bf16')),
          ('engine', ('pi05_bf16', 'pi05_nvfp4', 'gr00t_bf16')))
      for name in names for ext in ('json', 'log')],
    *['results/engine/' + name for name in (
        'fp4_opbench_ptqad_20260929.csv', 'fp4_opbench_ptqad_20260929.log',
        'fp4_opbench_verify_ptqad_20260929.log')],
    *[f'results/spike/{name}_ptqad_20260929.{ext}' for name, ext in (
        ('verify','log'), ('fp8_probe','log'), ('device_before','log'),
        ('device_after','log'), ('env','log'), ('gemm','csv'))],
    *['results/native_graph_20260929/' + name for name in (
        'start_utc.txt','end_utc.txt','exit_code.txt','device_before.txt',
        'device_after.txt','input_sha256.txt','wheel_install.log',
        'installed_extension.json','fp4_contract_rowmajor_and_tensor_scale.log',
        'fp4_graph_replay_bf16_and_distinct_scales.log',
        'fp4_activation_padding_zero_after_capture.log','pi05_require_graph.log')],
    'results/ptqad_20261001/mixed_pressure/recovery/run_manifest.json',
]
FINAL_REPORTS = [f'paper/validation/{name}.json' for name in (
    'figure-inputs','figure-renders','html-build','browser-validation',
    'publication-validation','publication-validation-v7')]
CURRENT_SOURCE_FILES = (
    'exp/recovery_protocol_v7_mixed_pressure.json',
    'exp/run_mixed_pressure_study.py',
    'exp/run_mixed_pressure_recovery.py',
    'exp/run_high_fp4_v3.py',
    'paper/build_final_frontier.py',
    'paper/build_html.py',
    'paper/export_zhihu.py',
    'paper/extract_final_evidence.py',
    'paper/materialize_final_evidence.py',
    'paper/evidence/selected_recipe/category_memory.json',
    'paper/evidence/selected_recipe/category_ptq_recipe.json',
    'paper/evidence/selected_recipe/category_bake_manifest.json',
)
RETIRED_SEED_PREFIXES = (
    'results/ptqad_20260929/development/',
    'results/ptqad_20260929/calibration',
    'results/ptqad_20260929/bake_',
    'weights/ptqad_20260929/',
)
# No model payload or weights/ -> results/ metadata mapping is approved in v7.
WEIGHT_MAPPINGS = {}
FORBIDDEN_SUFFIXES = {'.pt','.pth','.safetensors','.so','.whl','.pyc','.bin','.mp4'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, name):
    rel = PurePosixPath(name)
    require(isinstance(name, str) and name and not rel.is_absolute()
            and '..' not in rel.parts and '\\' not in name,
            'Non-relative or escaping path: '+str(name))
    path = root / rel
    for parent in (path, *path.parents):
        if parent == root:
            break
        require(not parent.is_symlink(), 'Symlink rejected: '+str(parent))
    require(path.resolve().is_relative_to(root.resolve()), 'Escaping path: '+name)
    return path


def checked_file(root, name, expected=None):
    path = inside(root, name)
    require(path.is_file(), 'Missing regular file: '+name)
    before = path.stat()
    require(before.st_size <= MAX_BYTES, 'Not a lightweight artifact: '+name)
    require(path.suffix not in FORBIDDEN_SUFFIXES, 'Binary/weight payload rejected: '+name)
    identity = {'bytes':before.st_size, 'sha256':sha(path)}
    after = path.stat()
    require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns),
            'Source changed during read: '+name)
    if expected is not None:
        require(type(expected.get('bytes')) is int
                and identity == {k:expected.get(k) for k in ('bytes','sha256')},
                'Manifest bytes/hash mismatch: '+name)
    return identity


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def read_json(path):
    return json.loads(path.read_text())


def validate_final_manifest(path, root):
    """Accept only a complete final run as the source of an export plan."""
    path = Path(path).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    require(path.name == 'final_manifest.json' and not path.is_symlink(),
            'Final export requires a regular final_manifest.json')
    require(path.is_relative_to(root), 'Final manifest must be inside the source repository')
    data = read_json(path)
    require(re.fullmatch(r'(?:high_fp4_[a-z0-9_]+|mixed_pressure_v7)_final_manifest', str(data.get('format', ''))),
            'Unsupported or incomplete final manifest format')
    require(data.get('selection_uses_heldout') is False,
            'Final selection must be independent of heldout results')
    require(set(data.get('required_arms', ())) == set(ARMS),
            'Final manifest does not declare exactly five arms')
    if data.get('protocol_file'):
        protocol = Path(data['protocol_file']).resolve(strict=True)
        require(protocol.is_file() and data.get('protocol_sha256') == sha(protocol),
                'Final protocol identity differs')
    comparison_info = data.get('heldout_comparison')
    require(isinstance(comparison_info, dict), 'Final manifest lacks heldout comparison identity')
    comparison = Path(comparison_info.get('path', '')).resolve(strict=True)
    require(comparison.is_file() and not comparison.is_symlink(),
            'Final heldout comparison is missing')
    require(comparison_info.get('bytes') == comparison.stat().st_size and
            comparison_info.get('sha256') == sha(comparison),
            'Final heldout comparison identity differs')
    paired = read_json(comparison)
    require(paired.get('environment_pairing_verified') is True and
            paired.get('protocol_consistency_verified') is True,
            'Final heldout comparison has not passed pairing/protocol checks')
    if str(data.get('format', '')).startswith('mixed_pressure_v7_'):
        require(paired.get('source_accounting_verified') is True,
                'v7 heldout comparison lacks source accounting verification')
    require(set(paired.get('arms', ())) == set(ARMS),
            'Final heldout comparison does not contain exactly five arms')
    state = path.parent / 'run_manifest.json'
    if state.is_file():
        require(read_json(state).get('status') == 'complete',
                'Final run_manifest is not complete')
    return path, data


def mapped_files(root, manifest_name, kind):
    """Return only byte-verified manifest members in a single allowed namespace."""
    manifest = read_json(inside(root, manifest_name))
    require(manifest.get('status') == 'complete', 'Incomplete '+kind+' manifest')
    names = []; seen = set()
    field = 'files' if kind == 'training' else 'mapping'
    require(isinstance(manifest.get(field),list) and manifest[field], 'Empty '+kind+' manifest')
    for row in manifest[field]:
        relative = row['published_path']
        path = ('paper/evidence/training/'+relative if kind == 'training' else 'paper/'+relative)
        require(path.startswith('paper/evidence/training/' if kind == 'training'
                                else 'paper/evidence/frontier/')
                or (kind == 'frontier' and path == 'paper/evidence/frontier_comparison.json'),
                'Unexpected '+kind+' destination: '+path)
        require(path not in seen, 'Duplicate manifest destination: '+path)
        source = row.get('source')
        if isinstance(source, dict):
            require(Path(source.get('path', '')).is_absolute(), 'Original identity is not absolute')
            expected = source
        else:
            require(Path(row.get('original_absolute_path', '')).is_absolute(), 'Original identity is not absolute')
            expected = row
        checked_file(root,path,expected);seen.add(path);names.append(path)
    return names


def refresh(seed, root, target, final_manifest=None):
    plan = copy.deepcopy(seed)
    final_path = final_data = None
    if final_manifest is not None:
        final_path, final_data = validate_final_manifest(final_manifest, root)
    rows = {}
    pending = []
    snapshots = {}
    def add(name, category, destination=None, expected=None):
        destination = destination or name
        inside(root,name);inside(target,destination)
        require(not name.startswith(('baselines/Isaac-GR00T/','paper/audit/',
                    'paper/evidence/reevaluation','paper/_build/')), 'Excluded source: '+name)
        require(not any('venv' in x or x=='.git' for x in PurePosixPath(name).parts),
                'Environment/Git path rejected: '+name)
        if name.startswith('weights/'):
            require(WEIGHT_MAPPINGS.get(name)==destination, 'Unapproved weight metadata mapping')
        if destination in rows:
            require(rows[destination]['source']==name, 'Conflicting source for '+destination)
        row=rows.setdefault(destination,{'source':name,'target':destination,
            'category':category,'status':'candidate_only_pending_final_gate','condition':None})
        path=inside(root,name)
        row['exists_at_planning']=path.is_file()
        if path.is_file():
            identity=checked_file(root,name,expected)
            row.update(bytes=identity['bytes'],sha256_snapshot=identity['sha256'])
            snapshots[name]=identity
        else:
            row.update(bytes=None,sha256_snapshot=None)
            pending.append({'path':name,'reason':'Named candidate does not exist; no hash fabricated'})
        return row

    for original in seed['source_export']:
        if any(original['source'].startswith(prefix) for prefix in RETIRED_SEED_PREFIXES):
            continue
        require(original['target'] not in rows, 'Duplicate manually reviewed seed target')
        rows[original['target']]=copy.deepcopy(original)
        add(original['source'],original['category'],original['target'])

    for name in CURRENT_SOURCE_FILES:
        add(name, 'v7_reproducibility_source')

    if final_path is not None:
        final_rel = final_path.relative_to(Path(root).resolve()).as_posix()
        add(final_rel, 'completed_final_manifest', 'paper/evidence/final_manifest.json')

    training='paper/evidence/training/evidence_manifest.json'
    if (root/training).is_file():
        require(read_json(root/'paper/evidence/training/costs.json').get('status')=='complete',
                'Training costs incomplete')
        for name in [training,'paper/evidence/training/costs.json',*mapped_files(root,training,'training')]:
            add(name,'completed_training_manifest_bound_evidence')
    else:pending.append({'path':training,'reason':'Await completed training collector'})

    frontier='paper/evidence/frontier/evidence_manifest.json'
    if (root/frontier).is_file():
        for name in [frontier,*mapped_files(root,frontier,'frontier')]:
            add(name,'completed_frontier_manifest_bound_evidence')
    else:pending.append({'path':frontier,'reason':'Await complete frontier collector'})

    for name in ('paper/evidence/final_results.json', 'paper/evidence/frontier_comparison.json'):
        if (root/name).is_file():
            add(name, 'v7_final_frontier_evidence')
        else:
            pending.append({'path':name, 'reason':'Await v7 final evidence/frontier builder'})

    paired=['paper/evidence/paired_comparison.json'] + [
        f'paper/evidence/heldout_{arm}/{name}.json' for arm in ARMS
        for name in ('eval_manifest','task_results','summary')]
    if all((root/name).is_file() for name in paired):
        for name in paired:add(name,'formal_five_arm_original_json')
    else:pending.append({'paths':[n for n in paired if not(root/n).is_file()],
                         'reason':'Await all16 formal pairing JSON; never fill missing arm with zero'})

    runtime='paper/evidence/runtime/manifest.json'
    if (root/runtime).is_file():
        data=read_json(root/runtime)
        require(set(data.get('checkpoints',{}))==set(ARMS),'Runtime must contain exact five arms')
        for name,identity in data['source_files'].items():
            require(name in rows and rows[name]['source']==name,
                    'Runtime references source outside public whitelist: '+name)
            checked_file(root,name,identity)
        add(runtime,'five_arm_runtime_identity')
    else:pending.append({'path':runtime,'reason':'Await exact five-arm runtime manifest'})

    captures=read_json(root/'paper/evidence/captures.json')
    figures=read_json(root/'paper/figures.json')
    new_captures={name for name in figures if name.startswith('shot_')}-{'shot_gr00t','shot_pi05'}
    accepted=set()
    for row in captures['screenshots']:
        if not row.get('verified'):continue
        require(row['figure'] in new_captures, 'Unexpected new capture key')
        require(row.get('capture_status')=='accepted_not_black' and row.get('exit_status')==0,
                'New capture lacks accepted quality/exit chain')
        image='paper/figs/'+row['figure']+'.png'
        require(sha(inside(root,image))==row['sha256'],'Capture image SHA mismatch')
        for field in ('raw_log','script','capture_sidecar','crop_manifest'):
            name='paper/'+row[field]
            require(name.startswith('paper/evidence/captures/'),'Capture source escaped evidence namespace')
            identity=checked_file(root,name)
            require(identity['sha256']==row[field+'_sha256'],'Capture input SHA mismatch')
            add(name,'new_capture_manifest_bound_evidence')
        accepted.add(row['figure'])
    if accepted!=new_captures:pending.append({'figures':sorted(new_captures-accepted),
                                            'reason':'Await modern capture/crop proof and visual acceptance'})

    for name in RESULT_FILES:
        if (root/name).is_file():add(name,'explicit_completed_current_run_small_artifact')
        else:pending.append({'path':name,'reason':'Exact current result name absent'})
    for name in FINAL_REPORTS:
        if (root/name).is_file():add(name,'named_final_build_report_requires_gate_validation')
        else:pending.append({'path':name,'reason':'Await final artifact build/QA report'})

    # A passed report is merely another file here: only the publication gate certifies it.
    source_head=git(root,'rev-parse','HEAD');target_head=git(target,'rev-parse','HEAD')
    target_status=git(target,'status','--porcelain')
    tracked=git(target,'ls-files','-z').split('\0');tracked=[x for x in tracked if x]
    reasons={x['path']:x['reason'] for x in seed['target_existing_proposed_remove']}
    removed=[]
    for name in tracked:
        if name not in rows:
            identity=checked_file(target,name)
            removed.append({'path':name,'reason':reasons.get(name,'Outside current reviewed whitelist'),
                            'bytes':identity['bytes'],'sha256_snapshot':identity['sha256']})
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    plan.update(generated_at_utc=now,source_head=source_head,target_head=target_head,
                source_status_at_planning=git(root,'status','--porcelain'),
                target_status_at_planning=target_status,
                source_root=str(root),target_root=str(target),release_ready=False,
                source_export=[rows[k] for k in sorted(rows)],
                target_existing_proposed_remove=sorted(removed,key=lambda r:r['path']),
                refresh_pending=pending,
                refresh_scope='Read-only candidate/hash/head refresh; no copying, deletion, GPU, model tensor loading or publication certification.',
                refresh_script='paper/_build/refresh_public_export_plan.py',
                refresh_script_sha256=sha(Path(__file__)),
                final_manifest=({'source': str(final_path), 'sha256': sha(final_path),
                                 'bytes': final_path.stat().st_size,
                                 'format': final_data['format']} if final_path else None),
                last_refresh_note='Exact manifest members and explicitly enumerated current run results only. Final publication gate and replacement of stale rendered artifacts still required.')
    source_tracked=[x for x in git(root,'ls-files','-z').split('\0') if x]
    included={row['source'] for row in rows.values()}
    old_exclusions={x['path']:x for x in seed['source_tracked_proposed_exclude']}
    plan['source_tracked_proposed_exclude']=[old_exclusions.get(name,{'path':name,
        'reason':'Outside manually reviewed seed and explicit final-evidence expansion'})
        for name in sorted(set(source_tracked)-included)]
    for template in plan['required_future_evidence']:
        name=template['path_or_template']
        if '{' not in name and inside(root,name).is_file():
            template.update(exists_at_planning=True,sha256=sha(root/name),
                            status='exists_but_not_certified_by_candidate_plan')
    plan['counts'].update(source_export_entries=len(rows),target_tracked=len(tracked),
        target_proposed_remove=len(removed),source_tracked_proposed_exclude=len(plan['source_tracked_proposed_exclude']))
    plan['refresh_added_paths']=sorted(set(rows)-{x['target'] for x in seed['source_export']})
    plan['refresh_seed_entries']=len(seed['source_export'])
    plan['publication_capture_state']={'registered_screenshots':17,'retained_approved':2,
        'modern_accepted':len(accepted),'awaiting_capture':sorted(new_captures-accepted)}
    # Reject concurrent changes to any bytes observed during this refresh.
    for name,identity in snapshots.items():checked_file(root,name,identity)
    require(git(root,'rev-parse','HEAD')==source_head and git(target,'rev-parse','HEAD')==target_head,
            'Git HEAD changed during snapshot; rerun')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=Path,default=ROOT/'paper/_build/public_export_seed.json')
    parser.add_argument('--out',type=Path,default=ROOT/'paper/_build/public_export_plan.json')
    boundary = parser.add_mutually_exclusive_group(required=True)
    boundary.add_argument('--final-manifest', type=Path,
                          help='Completed final_manifest.json; incomplete runs are rejected')
    boundary.add_argument('--run-dir', type=Path,
                          help='Completed run directory containing final_manifest.json')
    args=parser.parse_args()
    seed=read_json(args.seed)
    root=Path(seed['source_root']).resolve();target=Path(seed['target_root']).resolve()
    out=args.out.resolve()
    require(out.is_relative_to(root/'paper/_build'),'Write plans only inside paper/_build')
    require(args.seed.resolve()!=out,'Never overwrite the reviewed seed')
    baseline_hash=sha(args.seed)
    final_manifest = args.final_manifest or (args.run_dir / 'final_manifest.json')
    plan=refresh(seed,root,target,final_manifest)
    require(sha(args.seed)==baseline_hash,'Seed changed during refresh')
    plan['refresh_seed']={'path':str(args.seed.resolve()),'sha256':baseline_hash}
    temporary=out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');temporary.replace(out)
    md=['# 公开仓导出候选（只读刷新）','',
        f"时间 `{plan['generated_at_utc']}`；源 HEAD `{plan['source_head']}`；目标 HEAD `{plan['target_head']}`。",'',
        f"人工审定基底 {plan['refresh_seed_entries']} 项；当前精确候选 {len(plan['source_export'])} 项。目标 {plan['counts']['target_tracked']} 个跟踪文件，建议移出 {len(plan['target_existing_proposed_remove'])} 项。",'',
        '**仅刷新计划和逐文件 SHA，不复制、删除、提交或修改目标仓；release_ready 始终为 false。**','',
        '扩展限定为 training/frontier 的完整 manifest、五臂16份原始JSON、runtime、已接受截图的四份侧车、指定最终检查报告及明确命名的当前结果。权重只允许六个固定 recipe/bake JSON 映射；不扫描完整 results 或 weights。','',
        '最终发布仍需重建全部生成图稿、完成截图并运行当前 publication gate。保留旧生成文件的候选位置不代表其旧字节获准发布。','',
        '```bash','python3 paper/_build/refresh_public_export_plan.py',
        'python3 paper/_build/audit_final_curated_export.py','```','',
        '如需新增其他源码，先人工审查后编辑 public_export_seed.json；脚本不会把未审核源码自动纳入。','',
        '## 待完成或需最终核验的项目','']
    md += ['- '+json.dumps(row,ensure_ascii=False) for row in plan['refresh_pending']]
    out.with_suffix('.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'source_head':plan['source_head'],'candidates':len(plan['source_export']),
        'added':len(plan['refresh_added_paths']),'pending_groups':len(plan['refresh_pending']),
        'release_ready':False,'target_status':plan['target_status_at_planning']},ensure_ascii=False))

if __name__=='__main__':main()
