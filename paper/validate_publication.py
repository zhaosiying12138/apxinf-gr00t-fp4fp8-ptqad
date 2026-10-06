#!/usr/bin/env python3
"""Fail-closed publication validation. Checks remain active under python -O."""
import base64
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import struct
import sys
import subprocess

from publication_guard import capture_crop_contract, check_record, file_record, html_build_inputs, require, resolve_inside

P = Path(__file__).resolve().parent
CURRENT_INPUTS = {
    'paper/evidence/paired_comparison.json', 'paper/evidence/recipe_inventory.json',
    'paper/evidence/selected_recipe/category_memory.json',
    'paper/evidence/selected_recipe/category_ptq_recipe.json',
    'paper/evidence/selected_recipe/category_bake_manifest.json',
    'paper/evidence/frontier_comparison.json',
    'paper/evidence/selected_recipe/category_bake_manifest.json',
    'results/baselines/pi05_pt_bf16_ptqad_20260929.json',
    'results/baselines/gr00t_pt_bf16_ptqad_20260929.json',
    'results/engine/pi05_bf16_ptqad_20260929.json',
    'results/engine/gr00t_bf16_ptqad_20260929.json',
    'results/engine/pi05_nvfp4_ptqad_20260929.json',
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())


def publication_protocol(manifest, protocol_file=None):
    """Bind recorded protocol bytes to the local publication, never a private path."""
    digest = manifest.get('protocol_sha256')
    require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest),
            'Evaluation manifest lacks a valid protocol SHA256')
    root = P.parent.resolve()
    candidates = ([Path(protocol_file)] if protocol_file is not None else
                  sorted((P / 'evidence/protocol').glob('*.json')))
    matched = []
    for path in candidates:
        require(not path.is_symlink() and path.resolve().is_relative_to(root),
                'Publication protocol must be a regular file inside the public repository')
        if path.is_file() and sha(path) == digest:
            matched.append(path.resolve())
    require(len(matched) == 1,
            'Publication protocol has no unique SHA-matching local archive copy')
    return matched[0]


# Only these audited initial-QAD producers predate the numerical receipt fields.
# The public training collector verifies their archived bytes against these
# manifest digests. Do not turn a missing field in an arbitrary checkpoint into
# an inferred strict training contract.
LEGACY_STRICT_QAD_SOURCES = {
    'rl/lora_qad.py': 'aa755bd3809b44abad97dd373417851addbd7ffdbbf7bbc69eb30c9cada2be2c',
    'quant/native_activation.py': '8606b36e9ef2d4a4e194bcf2ab7ae6fc70d800cfc5cbb13f8552d1d4f842c8d3',
}
# Initial-QAD receipts from this audited producer record the effective Trainer
# max_grad_norm as 1.0 with no request override. The producer bytes alone do not
# determine the upstream default; require the actual receipt and observed env.
DEFAULT_QAD_MAX_GRAD_NORM_PRODUCERS = {
    '6259e854207a30db9ead07088f36de2524c784dc7ac7cfc58850ed44faa69d57': 1.0,
}


def recovery_numerical_contract(recovery, arm, protocol_sha):
    """Read original receipt values; narrowly recognize the strict QAD schema."""
    require(recovery.get('protocol_sha256') == protocol_sha and recovery.get('w4a4_enabled') is True,
            f'Recovery numerical contract has a different W4A4 protocol: {arm}')
    legacy = (arm == 'qad' and 'max_grad_norm' not in recovery and
              'f16_activation_saturation' not in recovery)
    if legacy:
        sources = recovery.get('recovery_source_sha256') or {}
        require(recovery.get('initial_adapter') is None and
                all(sources.get(key) == digest for key, digest in LEGACY_STRICT_QAD_SOURCES.items()),
                'Legacy QAD numerical contract lacks verified strict producer identity')
    else:
        norm = recovery.get('max_grad_norm')
        require(type(norm) in (int, float) and math.isfinite(norm) and norm > 0,
                f'Recovery export lacks finite max_grad_norm: {arm}')
        require(type(recovery.get('f16_activation_saturation')) is bool,
                f'Recovery export lacks f16 activation saturation provenance: {arm}')
    return {key: recovery.get(key) for key in ('max_grad_norm', 'f16_activation_saturation')}, legacy


def check_training_numerical_environment(recovery, request, arm, protocol_sha, *, producer_source=None):
    """Cross-check archived launch request and receipt without filling old fields."""
    contract, legacy = recovery_numerical_contract(recovery, arm, protocol_sha)
    env = request.get('environment')
    require(request.get('protocol_sha256') == protocol_sha and isinstance(env, dict) and
            env.get('QAD_W4A4') == '1', f'Training numerical request lacks W4A4 provenance: {arm}')
    keys = ('QAD_W4A4', 'QAD_MAX_GRAD_NORM', 'FP4VLA_SATURATE_F16_ACTIVATIONS')
    if legacy:
        require(all(key not in env for key in keys[1:]),
                'Legacy QAD request conflicts with strict producer numerical schema')
    else:
        raw_norm = env.get('QAD_MAX_GRAD_NORM')
        if 'QAD_MAX_GRAD_NORM' not in env and arm == 'qad':
            require('initial_adapter' in recovery and recovery['initial_adapter'] is None and
                    recovery.get('init_adapter') is None and
                    'initial_adapter_identity' in request and request['initial_adapter_identity'] is None and
                    'QAD_INIT_ADAPTER' not in env,
                    f'Training default max_grad_norm requires explicit initial-QAD identity: {arm}')
            producer_sha = recovery.get('recovery_source_sha256', {}).get('rl/lora_qad.py')
            expected = DEFAULT_QAD_MAX_GRAD_NORM_PRODUCERS.get(producer_sha)
            source = Path(producer_source) if producer_source is not None else None
            variables = (recovery.get('environment_summary') or {}).get('variables')
            require(expected is not None and source is not None and source.is_file() and
                    not source.is_symlink() and sha(source) == producer_sha and
                    isinstance(variables, dict) and 'QAD_MAX_GRAD_NORM' in variables and
                    variables['QAD_MAX_GRAD_NORM'] is None and contract['max_grad_norm'] == expected,
                    f'Training default max_grad_norm lacks verified producer/receipt provenance: {arm}')
            norm = expected
        else:
            try:
                norm = float(raw_norm) if isinstance(raw_norm, str) else None
            except ValueError:
                norm = None
        require(norm is not None and math.isfinite(norm) and norm == contract['max_grad_norm'] and
                env.get('FP4VLA_SATURATE_F16_ACTIVATIONS') ==
                ('1' if contract['f16_activation_saturation'] else '0'),
                f'Training numerical request differs from recovery receipt: {arm}')
    summary = recovery.get('environment_summary')
    if summary is not None:
        variables = summary.get('variables') if isinstance(summary, dict) else None
        require(isinstance(variables, dict) and all(variables.get(key) == env.get(key) for key in keys),
                f'Training numerical environment summary differs from request: {arm}')
    # A missing summary remains missing. The hash-bound request and explicit
    # continuation fields above provide the older schema's numerical evidence.


class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.anchors=[]; self.images=[]; self.resources=[]; self.figures=[]
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        if tag=='figure': self.figures.append(attrs.get('data-figure'))
        if tag=='a' and attrs.get('href','').startswith('#'): self.anchors.append(attrs['href'][1:])
        if tag=='img' and 'src' in attrs: self.images.append(attrs['src'])
        for key in ('src','href','srcset','poster','xlink:href'):
            value=attrs.get(key)
            if value and not (tag=='a' and key=='href') and not value.startswith(('data:','#')):
                self.resources.append(value)


def check_finished(source):
    require('textcolor{#b42318}' not in source and 'mathrm{xxx}' not in source,
            'Incomplete review draft: red xxx placeholders must be replaced by verified results')
    # Check visible result placeholders independently of their Markdown/HTML
    # styling; missing measurements discussed as limitations remain valid.
    def visible(text):
        return html.unescape(re.sub(r'<[^>]*>','',text))
    prose=visible(source)
    numerical_placeholder=r'(?<![A-Za-z0-9_])(?:x{2,}\s*(?:[/\uFF0F]\s*(?:\d+|x{2,})|[%\uFF05])|\d+\s*[/\uFF0F]\s*x{2,})(?![A-Za-z0-9_])'
    span_placeholders=any(re.search(r'(?<![A-Za-z0-9_])x{2,}(?![A-Za-z0-9_])',visible(body),re.I)
                          for body in re.findall(r'<span\b[^>]*>([\s\S]*?)</span\s*>',source,re.I))
    require(not re.search(numerical_placeholder,prose,re.I) and not span_placeholders,
            'Incomplete review draft: numerical xxx/XX placeholders must be replaced by verified results')
    require('待回填' not in prose,'Incomplete review draft: result marked 待回填')
    phrases=('正式开发结果将在','五臂最终结果将在','本轮执行结果将在',
             '本轮显式精度配置的 PyTorch 与 APXInf 测量结果将在','训练、教师标注、额外学生探针与评测成本分别从实际运行日志统计',
             '本轮矩阵乘时延、吞吐和格式可用性将在','本轮逐形状结果将在')
    for text in phrases: require(text not in source, 'Unfinished experiment section: '+text)
    for label,body in re.findall(r'<!-- BEGIN ([A-Z0-9 ]+) -->([\s\S]*?)<!-- END \1 -->',source):
        require(not re.search(r'将在|待填|待测|待运行|尚未完成|结果占位|运行中',body),
                'Unfinished result block: '+label)
    require('FPVMATHTOKEN' not in source and 'FPVCODETOKEN' not in source,'Unexpected build tokens in sources')


def expected_markdown(sections,meta,figures):
    # export_zhihu.py deliberately omits the optional meta subtitle to avoid
    # repeating the title before the first section heading.
    text='# '+meta['title']+'\n\n'
    text+=''.join(path.read_text().rstrip()+'\n\n' for path in sections)
    def figure(match):
        name=match[1]; info=figures[name]
        return f'![{info["title"]}](images/{name}.png)\n\n*{info["title"]}。{info["note"]}*'
    return re.sub(r'\{\{fig:([\w.-]+)\}\}',figure,text).rstrip()+'\n'


def check_png(path,dimensions,terminal=False):
    payload=path.read_bytes()
    require(payload[:8]==b'\x89PNG\r\n\x1a\n',f'Invalid PNG: {path}')
    require(list(struct.unpack('>II',payload[16:24]))==list(dimensions),f'PNG dimensions differ: {path}')
    if terminal:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width,height=image.size
            body=image.crop((0,int(height*.06),width,int(height*.95))).convert('L')
            histogram=body.histogram()
            require(sum(histogram[180:])/sum(histogram)>0.0002,
                    f'Blank/dark terminal body, not an acceptable execution capture: {path}')


def load_capture_manifests(required,package_inputs):
    """Merge current and explicitly retained provenance without creating a new file."""
    current_path=resolve_inside(P,'evidence/captures.json')
    retained_path=resolve_inside(P,'evidence/retained_captures.json')
    current=load(current_path);retained=load(retained_path)
    require(current.get('version')==1 and retained.get('version')==1,'Unsupported capture manifest version')
    current_rows=current.get('screenshots');retained_rows=retained.get('screenshots')
    require(isinstance(current_rows,list) and isinstance(retained_rows,list), 'Capture manifests require screenshot lists')
    allowed_retained={'shot_gr00t','shot_pi05'}
    require(len(required)==17 and allowed_retained<=required,'Unexpected registered screenshot set')
    rows=current_rows+retained_rows
    require(all(isinstance(row,dict) and isinstance(row.get('figure'),str) for row in rows),'Invalid capture record')
    names=[row['figure'] for row in rows]
    require(len(names)==len(set(names)),'Duplicate screenshot across current/retained manifests')
    require({row['figure'] for row in current_rows}==required-allowed_retained and len(current_rows)==15,
            'Exactly 15 current screenshots are required')
    require({row['figure'] for row in retained_rows}==allowed_retained and len(retained_rows)==2,
            'Exactly the two approved BF16 screenshots may be retained')
    for row in current_rows:
        require(row.get('retained_unaffected',False) is False and row.get('retain_for_publication',False) is False,
                'Current capture cannot claim unaffected retention: '+row['figure'])
    for row in retained_rows:
        require(row.get('retained_unaffected') is True and row.get('retain_for_publication') is True and
                row.get('verified') is True,'Retained capture lacks explicit publication approval: '+row['figure'])
    package_inputs.update((current_path,retained_path))
    return {**current,'screenshots':sorted(rows,key=lambda row:row['figure'])}


def capture_records(captures,required,package_inputs):
    approved=captures['approved_sample']
    require(approved.get('confirmed_by_user') is True,'Capture sample was not user-confirmed')
    require(approved.get('dimensions')==[3840,2280],'Unexpected approved capture dimensions')
    rows=captures['screenshots']; names=[row['figure'] for row in rows]
    require(len(names)==len(set(names)) and set(names)==required and len(names)==17,
            'Capture manifest must contain each of the 17 required screenshots exactly once')
    reserved_support_names = {name + suffix for name in names
                              for suffix in ('.png','.log','.sh','.capture.json','.crop.json')}
    reserved_support_names.update(Path(row[field]).name for row in rows
                                  for field in ('raw_log','script','capture_sidecar','crop_manifest')
                                  if isinstance(row.get(field),str))
    checked=[]
    for row in rows:
        name=row['figure'];path=P/'figs'/(name+'.png');git_anchor=None;acceptance='current_capture_and_crop'
        require(row.get('verified') is True,f'Capture not visually verified: {name}')
        require(sha(path)==row['sha256'],f'Capture image changed: {name}')
        check_png(path,[3840,2280],terminal=True)
        require(row.get('dimensions')==[3840,2280],f'Capture manifest dimensions differ: {name}')
        if row.get('retained_unaffected') is True:
            require(name in {'shot_gr00t','shot_pi05'},f'Capture is not approved for unaffected retention: {name}')
            acceptance='retained_unaffected'
            anchors=[row[key] for key in ('original_git_identity','public_git_identity') if key in row]
            require(anchors,'Retained capture has no Git identity')
            git_anchor=None
            for identity in anchors:
                require(identity.get('byte_identical') is True and identity['sha256']==row['sha256'],
                        f'Retained capture identity differs: {name}')
                require(identity['path']==f'paper/figs/{name}.png' and re.fullmatch(r'[0-9a-f]{40}',identity['commit']),
                        'Invalid retained Git identity')
                original=subprocess.run(['git','show',identity['commit']+':'+identity['path']],cwd=P.parent,capture_output=True)
                if original.returncode==0 and hashlib.sha256(original.stdout).hexdigest()==row['sha256']:
                    git_anchor={'commit':identity['commit'],'path':identity['path']};break
            require(git_anchor is not None,f'No reachable byte-identical Git anchor for retained capture: {name}')
            result=check_record(row['raw_result'],P.parent);package_inputs.add(result)
            actual=load(result)
            require(actual.get('variant')=='bf16' and actual.get('n')==10,'Retained capture raw run differs')
            for key,value in row['visible_result_matches'].items():
                require(actual[key]==value,f'Retained image/result mapping differs: {name}/{key}')
        else:
            require(type(row.get('exit_status')) is int and row['exit_status']==0,f'Capture command did not pass: {name}')
            captured=datetime.fromisoformat(row['captured_at'].replace('Z','+00:00'))
            require(captured.tzinfo is not None and captured>=datetime(2026,9,29,tzinfo=timezone.utc),
                    f'Capture is not a current run: {name}')
            for field in ('raw_log','script'):
                require(isinstance(row.get(field),str) and row[field],f'Capture lacks {field}: {name}')
                evidence=resolve_inside(P,row[field])
                require(evidence.stat().st_size>0 and sha(evidence)==row[field+'_sha256'],f'Capture {field} changed: {name}')
                package_inputs.add(evidence)
            proof = {}
            for field in ('capture_sidecar','crop_manifest'):
                require(isinstance(row.get(field),str) and row[field], f'Capture lacks {field}: {name}')
                evidence = resolve_inside(P,row[field])
                require(sha(evidence) == row[field+'_sha256'], f'Capture proof changed: {name}/{field}')
                package_inputs.add(evidence); proof[field] = json.loads(evidence.read_text(encoding='utf-8-sig'))
            capture_crop_contract(proof['capture_sidecar'],proof['crop_manifest'],path,[3840,2280])
            require(row.get('capture_status') == 'accepted_not_black' and row.get('image_quality') == proof['capture_sidecar']['image_quality'],
                    f'Capture quality metadata differs: {name}')
            require(row.get('visual_review',{}).get('verified') is True, f'Capture lacks explicit manual inspection: {name}')
        if 'supporting_files' in row:
            support=row['supporting_files'];seen=set()
            require(isinstance(support,list),f'Invalid capture supporting_files: {name}')
            for identity in support:
                require(isinstance(identity,dict) and isinstance(identity.get('path'),str) and
                        type(identity.get('bytes')) is int and identity['bytes']>0 and
                        isinstance(identity.get('sha256'),str),f'Invalid capture support record: {name}')
                relative=Path(identity['path']);basename=relative.name
                require(relative.as_posix()=='evidence/captures/'+basename and
                        basename not in reserved_support_names and basename not in seen,
                        f'Unsafe, duplicate or conflicting capture support path: {name}/{identity["path"]}')
                seen.add(basename)
                evidence=check_record(identity,P)
                require(evidence.parent==(P/'evidence/captures').resolve(),
                        f'Capture support escapes capture directory: {name}')
                package_inputs.add(evidence)
        checked.append({'file':path.name,'sha256':row['sha256'],'verified':True,
                        'retained_unaffected':row.get('retained_unaffected') is True,
                        'acceptance':acceptance,'verified_git_anchor':git_anchor})
    return checked


def check_pairing(data,protocol_file=None):
    sys.path.insert(0,str(P.parent/'eval'))
    from compare_ptq_frontier import protocol_contract
    expected=protocol_contract(protocol_file or P.parent/'exp/recovery_protocol_v12_rtn_w4a4.json')['heldout']
    indices=expected['init_state_indices']
    task_count=expected['tasks']
    episodes_per_task=expected['episodes_per_task']
    expected_total=task_count * episodes_per_task
    require(data.get('environment_pairing_verified') is True,'Environment pairing has not passed')
    order={'bf16','ptq','qad','continued_qad','qad_opd'}
    require(set(data['arms'])==order,'Exactly five completed heldout arms are required')
    baseline=data['arms']['bf16']['episodes']
    require(len(baseline)==expected_total,
            f'Expected {expected_total} heldout episodes per arm')
    for name,arm in data['arms'].items():
        episodes=arm['episodes'];require(arm['count']==expected_total and len(episodes)==expected_total,
                                         f'Incomplete heldout arm: {name}')
        require(len(arm['per_task'])==task_count and
                all(x['episodes']==episodes_per_task for x in arm['per_task'].values()),
                f'Expected {episodes_per_task} episodes per task')
        for actual,baseline_episode in zip(episodes,baseline):
            require(type(actual['success']) is bool,'Episode outcome must be Boolean')
            require({k:v for k,v in actual.items() if k!='success'}=={k:v for k,v in baseline_episode.items() if k!='success'},
                    f'Unpaired environment initial state: {name}')
            require(actual['init_state_index'] in indices,'Heldout initial-state index outside declared partition')
            for field in ('initial_state_sha256','restored_state_sha256','init_state_bank_sha256'):
                require(re.fullmatch(r'[0-9a-f]{64}',actual[field]) is not None,'Invalid initial-state hash')
        require(sum(x['success'] for x in episodes)==arm['successes'],'Episode outcome count differs')
        for task,record in arm['per_task'].items():
            task_rows=[x for x in episodes if x['task']==task]
            require({x['init_state_index'] for x in task_rows}==set(indices) and len(task_rows)==episodes_per_task,
                    'Heldout initial states repeated or missing')
            require(sum(x['success'] for x in task_rows)==record['successes'],'Task success count differs')


def check_training_costs(runtime, package_inputs,protocol_file=None):
    """Recompute public costs, then bind them to the evaluated weight identities."""
    from collect_training_costs import verify_published
    folder=P/'evidence/training'
    costs=verify_published(folder)
    # Bind training receipts to the same versioned W4A4 protocol as the
    # held-out evaluation.  The selected pressure recipe may be GPTQ or RTN;
    # the manifest, protocol SHA and runtime contract are the authority.
    protocol=load(protocol_file or P.parent/'exp/recovery_protocol_v12_rtn_w4a4.json')
    if protocol.get('partitions') and protocol.get('selection'):
        require(isinstance(protocol.get('version'), int) and protocol.get('version') >= 12 and
                protocol.get('w4a4') is True,
                'Publication training evidence must use a frozen W4A4 protocol')
    require(costs.get('status')=='complete' and set(costs['training'])=={'qad','continued_qad','qad_opd'},
            'Exactly three completed formal training stages are required')
    require(costs['protocol_sha256']==sha(protocol_file or P.parent/'exp/recovery_protocol_v12_rtn_w4a4.json'),
            'Training costs used a different frozen protocol')
    manifest=load(folder/'evidence_manifest.json')
    records={row['published_path']:row for row in manifest['files']}
    for row in manifest['files']:
        package_inputs.add(check_record({'path':row['published_path'],'bytes':row['bytes'],'sha256':row['sha256']},folder))
    package_inputs.update((folder/'costs.json',folder/'evidence_manifest.json',P/'collect_training_costs.py'))

    def weights(arm):
        rows=runtime['checkpoints'][arm]['files']
        result={row['name']:{'bytes':row['bytes'],'sha256':row['sha256']}
                for row in rows if row['name'].endswith('.safetensors')}
        require(result and len(result)==sum(row['name'].endswith('.safetensors') for row in rows),
                f'Incomplete runtime weight identity: {arm}')
        return result

    for arm in ('qad','continued_qad','qad_opd'):
        relative='exports/'+arm+'/merge_manifest.json'
        merge=load(resolve_inside(folder,relative))
        archived_source=Path(records[relative]['original_absolute_path']).parent
        require(archived_source.resolve()==Path(runtime['checkpoints'][arm]['path']).resolve(),
                f'Training export does not match evaluated checkpoint: {arm}')
        require(merge['output_weights']==weights(arm),f'Training export weight identity differs from heldout: {arm}')
        require(merge['base_weights']==weights('ptq'),f'Training base weight identity differs from evaluated PTQ: {arm}')
        # Bind the plotted LoRA storage budget to the actual selected export,
        # preserving the check formerly provided by the PTQ-ladder validator.
        residual=load(P/'evidence/recipe_inventory.json')['recovery_residual']
        recovery=merge['recovery_manifest']
        require(merge['lora_pairs']==residual['linear_modules'] and
                recovery['trainable_parameters']==residual['tensor_elements'] and
                all(recovery[key]==residual[key] for key in ('rank','alpha','scope')),
                f'Recovery export differs from the plotted LoRA budget: {arm}')
        stage_relative = 'stages/' + arm + '/recovery_manifest.json'
        request_relative = 'stages/' + arm + '/orchestrator_training_request.json'
        require(stage_relative in records and request_relative in records,
                f'Recovery numerical source records are missing: {arm}')
        require(load(resolve_inside(folder, stage_relative)) == recovery,
                f'Recovery export differs from archived training receipt: {arm}')
        request = load(resolve_inside(folder, request_relative))
        producer_relative = 'stages/' + arm + '/source/rl/lora_qad.py'
        producer_source = resolve_inside(folder, producer_relative) if producer_relative in records else None
        check_training_numerical_environment(recovery, request, arm, costs['protocol_sha256'],
                                             producer_source=producer_source)
    collection=load(folder/'collection/eval_manifest.json')
    require(Path(collection['checkpoint']).resolve()==Path(runtime['checkpoints']['qad']['path']).resolve(),
            'Collection student identity differs from evaluated QAD')
    teacher=load(folder/'teacher/teacher_probes.json')
    require(Path(teacher['teacher']).resolve()==Path(runtime['checkpoints']['bf16']['path']).resolve() and
            teacher['teacher_weights']==weights('bf16'),'Teacher identity differs from evaluated BF16')
    return costs


def check_search_costs(package_inputs, protocol_file, folder=None):
    """Require the independently verified search archive and bundle every input."""
    from collect_search_costs import verify
    folder = Path(folder) if folder is not None else P / 'evidence/search_costs'
    report = verify(folder)
    require(report.get('status') == 'verified' and report.get('protocol_sha256') == sha(protocol_file),
            'Search cost archive used a different protocol or did not verify')
    manifest_path = folder / 'evidence_manifest.json'
    manifest = load(manifest_path)
    require(type(report.get('files')) is int and report['files'] == len(manifest['files']),
            'Search cost archive file count differs from verified report')
    for row in manifest['files']:
        package_inputs.add(check_record({'path': row['published_path'],
                                         'bytes': row['bytes'], 'sha256': row['sha256']}, folder))
    if int(load(protocol_file).get('version', 0)) >= 12:
        from search_cost_binding import verify_search_cost_binding
        package_inputs.update(verify_search_cost_binding(
            folder, P / 'evidence/training', P / 'evidence/final_manifest.json'))
    package_inputs.update((manifest_path, P / 'collect_search_costs.py', P / 'collect_training_costs.py'))
    return report


def archived_recovery_receipt(package_inputs, runtime, arm):
    """Resolve the evaluated export's receipt entirely from hash-bound public files."""
    folder = P / 'evidence/training'
    archive_path = folder / 'evidence_manifest.json'
    archive = load(archive_path)
    require(archive.get('status') == 'complete' and isinstance(archive.get('files'), list),
            'Recovery archive identity map is missing or incomplete')
    relative = 'exports/' + arm + '/merge_manifest.json'
    matches = [row for row in archive['files'] if row.get('published_path') == relative]
    require(len(matches) == 1, f'Recovery archive export is missing or ambiguous: {arm}')
    row = matches[0]
    source = row.get('original_absolute_path')
    require(isinstance(source, str) and Path(source).is_absolute() and
            Path(source).parent.resolve() == Path(runtime['checkpoints'][arm]['path']).resolve(),
            f'Archived recovery export differs from evaluated checkpoint: {arm}')
    path = check_record({'path': relative, 'bytes': row['bytes'], 'sha256': row['sha256']}, folder)
    merge = load(path)
    weights = {item['name']: {key: item[key] for key in ('bytes', 'sha256')}
               for item in runtime['checkpoints'][arm]['files'] if item['name'].endswith('.safetensors')}
    require(weights and merge.get('output_weights') == weights,
            f'Archived recovery export weights differ from evaluated checkpoint: {arm}')
    receipt = merge.get('recovery_manifest')
    require(isinstance(receipt, dict), f'Archived recovery export lacks its training receipt: {arm}')
    package_inputs.update((archive_path, path))
    return receipt


def check_evaluation_environment(package_inputs, runtime, protocol_file):
    """Require explicit W4A4 and recovery-contract receipts in every eval arm."""
    protocol = load(protocol_file)
    if int(protocol.get('version', 0)) < 12:
        return
    for arm in ('bf16', 'ptq', 'qad', 'continued_qad', 'qad_opd'):
        path = P / 'evidence' / ('heldout_' + arm) / 'eval_manifest.json'
        manifest = load(path)
        variables = (manifest.get('environment_summary') or {}).get('variables')
        require(isinstance(variables, dict), f'Evaluation manifest lacks environment_summary: {arm}')
        quant = arm != 'bf16'
        expected = {'FP4VLA_QUANT': '0', 'FP4VLA_W4A4': '1' if quant else '0',
                    'FP4VLA_W4A4_ADAPTER': '1' if arm in ('qad', 'continued_qad', 'qad_opd') else '0',
                    'FP4VLA_SATURATE_F16_ACTIVATIONS': '1' if quant else '0',
                    'PTQAD_ZMQ_TIMEOUT_MS': '120000'}
        for key, value in expected.items():
            require(variables.get(key) == value,
                    f'Evaluation environment summary differs from W4A4 contract: {arm}/{key}')
        contract = manifest.get('recovery_contract')
        require(isinstance(contract, dict) and 'max_grad_norm' in contract,
                f'Evaluation manifest lacks recovery contract: {arm}')
        require('f16_activation_saturation' in contract,
                f'Evaluation manifest lacks recovery saturation contract: {arm}')
        if arm in ('bf16', 'ptq'):
            require(contract['max_grad_norm'] is None and contract['f16_activation_saturation'] is False,
                    f'Non-recovery arm carries a recovery training contract: {arm}')
        else:
            recorded = archived_recovery_receipt(package_inputs, runtime, arm)
            expected_contract, legacy = recovery_numerical_contract(recorded, arm, sha(protocol_file))
            if not legacy:
                require(type(contract['max_grad_norm']) in (int, float) and
                        type(contract['f16_activation_saturation']) is bool,
                        f'Evaluation recovery contract has invalid numerical types: {arm}')
            require(contract == expected_contract,
                    f'Evaluation recovery contract differs from checkpoint: {arm}')
        package_inputs.add(path)


def check_heldout_raw_logs(package_inputs, protocol_file):
    """Verify every published rollout/server byte and replay the existing parser."""
    from materialize_final_evidence import audit_heldout_raw_logs
    from run_recovery_eval import TASKS
    folder = P / 'evidence'
    manifest_path = folder / 'heldout_raw_logs.json'
    manifest = load(manifest_path)
    require(manifest.get('version') == 1 and manifest.get('status') == 'complete' and
            manifest.get('protocol_sha256') == sha(protocol_file),
            'Raw heldout log manifest has a different protocol or incomplete status')
    parser = manifest.get('parser')
    require(isinstance(parser, dict) and parser.get('path') == 'eval/run_recovery_eval.py' and
            manifest.get('parser_functions') == ['parse_log', 'validate_resets'],
            'Raw heldout evidence does not identify the evaluation parser')
    package_inputs.add(check_record(parser, P.parent))
    expected = {f'heldout_{arm}/{task}{suffix}'
                for arm in ('bf16', 'ptq', 'qad', 'continued_qad', 'qad_opd')
                for task in TASKS for suffix in ('.log', '.server.log')}
    records = manifest.get('files')
    require(isinstance(records, dict) and set(records) == expected,
            'Raw heldout evidence must contain exactly fifty rollout/server log pairs')
    for relative, record in records.items():
        require(isinstance(record, dict) and set(record) == {'bytes', 'sha256'},
                'Raw heldout log identity is incomplete: ' + relative)
        package_inputs.add(check_record({'path': relative, **record}, folder))
    require(audit_heldout_raw_logs(folder) == records,
            'Published raw heldout logs do not reproduce the task receipts')
    protocol = load(protocol_file)
    if int(protocol.get('version', 0)) >= 12:
        from activation_evidence import validate_activation_log
        for arm in ('bf16', 'ptq', 'qad', 'continued_qad', 'qad_opd'):
            for task in TASKS:
                validate_activation_log(folder / ('heldout_' + arm) / (task + '.server.log'),
                                        protocol, quantized=arm != 'bf16')
        package_inputs.add(P / 'activation_evidence.py')
    package_inputs.add(manifest_path)


def check_v12_frontier(package_inputs, paired):
    """Rebuild the selected W4A4 frontier from verified result inputs.

    The old ``frontier_evidence`` contract froze a PTQ ladder and reference
    runs. The current release deliberately has one selected W4A4 checkpoint,
    so accepting a stale ladder here would make the published chart disagree
    with the paper.
    ``build_final_frontier`` is deterministic and fail-closed.  Origin paths
    remain in copied evidence as provenance; a relocated checkout binds those
    identities to its local byte-identical evidence before reconstruction.
    """
    from build_final_frontier import build
    final_path = P / 'evidence/final_results.json'
    comparison_path = P / 'evidence/frontier_comparison.json'
    inventory_path = P / 'evidence/recipe_inventory.json'
    require(final_path.is_file(), 'final_results.json is missing')
    require(comparison_path.is_file(), 'frontier_comparison.json is missing')
    final = load(final_path)
    source = final.get('source', {}).get('heldout_comparison', {})
    local_pair = P / 'evidence/paired_comparison.json'
    require(local_pair.is_file(), 'local heldout paired comparison is missing')
    require(type(source.get('bytes')) is int and source['bytes'] == local_pair.stat().st_size and
            source.get('sha256') == sha(local_pair),
            'local heldout paired comparison identity differs from final_results')
    require(load(local_pair) == paired, 'v12 frontier source differs from the paper heldout comparison')
    check_uncertainty(final, paired, package_inputs)
    import tempfile
    with tempfile.TemporaryDirectory(prefix='validate-v12-frontier-') as tmp:
        expected_path = Path(tmp) / 'frontier_comparison.json'
        expected = build(final_path, local_pair, inventory_path, expected_path)
    recorded = load(comparison_path)
    # Only the location field of each explicit source identity may relocate.
    # Every data/method field and source digest/size must still match; do not
    # recursively drop arbitrary path keys from numerical or method records.
    recorded_sources = recorded.get('source')
    expected_sources = expected['source']
    require(isinstance(recorded_sources, dict) and set(recorded_sources) == set(expected_sources),
            'frontier source identity set differs')
    for name, current in expected_sources.items():
        previous = recorded_sources[name]
        require(isinstance(previous, dict) and set(previous) == {'path','bytes','sha256'} and
                isinstance(previous['path'], str) and Path(previous['path']).is_absolute() and
                type(previous['bytes']) is int and
                (previous['bytes'], previous['sha256']) == (current['bytes'], current['sha256']),
                'frontier source identity differs: ' + name)
    require({key:value for key,value in recorded.items() if key != 'source'} ==
            {key:value for key,value in expected.items() if key != 'source'},
            'frontier_comparison.json differs from deterministic reconstruction')
    for path in (final_path, comparison_path, inventory_path, local_pair,
                 P / 'evidence/selected_recipe/category_memory.json',
                 P / 'evidence/selected_recipe/category_ptq_recipe.json',
                 P / 'evidence/selected_recipe/category_bake_manifest.json'):
        if path.is_file():
            if path.resolve().is_relative_to(P.parent.resolve()):
                package_inputs.add(path.resolve())
    return recorded


def check_uncertainty(final, paired, package_inputs):
    """Recompute every specified contrast; never accept edited CI/p fields."""
    import paired_uncertainty
    plan = P / 'analysis_plan_w4a4.json'
    require(plan.is_file() and paired_uncertainty.PLAN.resolve() == plan.resolve(),
            'Paired analysis plan is missing or comes from another publication tree')
    require(final.get('uncertainty') == paired_uncertainty.analyze(paired),
            'Published uncertainty differs from paired outcomes and the fixed analysis plan')
    package_inputs.update((plan, P / 'paired_uncertainty.py'))


def check_action_diagnostics(package_inputs):
    from action_diagnostics_publication import load_verified, render
    diagnostic = load_verified(P)
    require(render(diagnostic) in (P / 'sections/04-实验.md').read_text(),
            'Article action diagnostic table is absent or stale')
    require(render(diagnostic, detailed=True) in (P.parent / 'README.md').read_text(),
            'README action diagnostic tables are absent or stale')
    package_inputs.update(diagnostic['files'])
    package_inputs.update((P / 'collect_action_diagnostics.py',
                           P / 'action_diagnostics_publication.py',
                           P / 'requirements-evidence.txt', P.parent / 'README.md',
                           P / 'readme_v12_supplement_commands.md',
                           P.parent / 'docs/ACTION_CHUNK_DIAGNOSTICS.md'))
    return diagnostic


def check_gptq_reference(package_inputs):
    from gptq_reference_publication import load_verified, render
    supplement = load_verified(P)
    require(render(supplement) in (P / 'sections/04-实验.md').read_text(),
            'Article GPTQ reference tables are absent or stale')
    require(render(supplement, detailed=True) in (P.parent / 'README.md').read_text(),
            'README GPTQ reference tables are absent or stale')
    package_inputs.update(supplement['files'])
    package_inputs.update((P / 'collect_gptq_reference.py', P / 'gptq_reference_publication.py',
                           P.parent / 'docs/CAPTURED_PTQ_CALIBRATION.md'))
    return supplement


def validate(write_report=True,protocol_file=None):
    sections=sorted((P/'sections').glob('*.md'))
    require({path.name for path in sections}=={
        '01-摘要与引言.md','02-背景与相关工作.md','03-方法.md','04-实验.md','05-讨论与结论.md',
        '14-附录A-核心源码走读与APXInf框架解析.md','15-附录B-复现与证据索引.md',
        '15-附录C-执行基准与完整测量.md','16-参考文献.md'},
        'Expected all nine complete article source files')
    source='\n\n'.join(path.read_text() for path in sections);check_finished(source)
    metadata=load(P/'meta.json');registry=load(P/'figures.json');required=set(registry)
    figures=re.findall(r'\{\{fig:([\w.-]+)\}\}',source)
    require(set(figures)==required and len(figures)==len(required),'Figure registration and placements differ')
    shots={name for name in required if name.startswith('shot_')};diagrams=required-shots
    md=(P/'zhihu/article.md').read_text();doc=(P/'paper.html').read_text()
    require(md==expected_markdown(sections,metadata,registry),'Zhihu Markdown is stale or differs from the complete source')
    require(not re.search(r'\{\{fig:|FPV(?:MATH|CODE)TOKEN|\[缺图',doc+md),'Unresolved figure/math/code token')
    package_inputs=set()
    from install_final_evidence import verify as verify_installed_evidence
    verify_installed_evidence(P / 'evidence')
    installed_manifest = load(P / 'evidence/evidence_manifest.json')
    package_inputs.update((P / 'evidence/evidence_manifest.json',
                           P / 'evidence/source_bundle_manifest.json', P / 'install_final_evidence.py'))
    package_inputs.update(resolve_inside(P / 'evidence', row['published_path'])
                          for row in installed_manifest['files'])
    provenance=load(P/'validation/figure-inputs.json')
    require(provenance.get('version')==1 and provenance.get('status')=='passed','Figure generation did not pass')
    require(provenance['generator']['path']=='paper/make_figs.py','Unexpected figure generator')
    inputs=provenance['inputs'];svg_rows=provenance['svgs']
    require(len(inputs)==len(CURRENT_INPUTS) and {row['path'] for row in inputs}==CURRENT_INPUTS,'Current figure input set is incomplete or outdated')
    expected_svg={f'paper/figs/{name}.svg' for name in diagrams}
    require(len(svg_rows)==len(expected_svg) and {row['path'] for row in svg_rows}==expected_svg,'Diagram provenance set differs')
    for row in [provenance['generator'],*inputs,*svg_rows]:package_inputs.add(check_record(row,P.parent))
    # Bind benchmark output to the actual current driver, not only its filename.
    for result_name,script_name in (
        ('pi05_pt_bf16_ptqad_20260929','bench_pi05_lerobot.py'),
        ('gr00t_pt_bf16_ptqad_20260929','bench_gr00t_pt.py')):
        record=load(P.parent/'results/baselines'/(result_name+'.json'))
        script=P.parent/'baselines'/script_name
        require(record['script_sha256']==sha(script),f'Baseline source differs from its recorded run: {script_name}')
        package_inputs.add(script)
        if script_name=='bench_pi05_lerobot.py':
            helper=P.parent/'baselines/bench_gr00t_pt.py'
            require(record['shared_helper_sha256']==sha(helper),'Pi05 shared baseline helper differs')
    for result_name in ('pi05_bf16','gr00t_bf16','pi05_nvfp4'):
        record=load(P.parent/'results/engine'/(result_name+'_ptqad_20260929.json'))
        script=P.parent/'exp/bench_engine.py'
        identity=record['runtime_identity']['benchmark_script']
        require(identity['sha256']==sha(script) and identity['bytes']==script.stat().st_size,'Engine benchmark source differs from its recorded run')
        package_inputs.add(script)
    from make_figs import paired_rows, budget_rows
    sys.path.insert(0,str(P.parent/'eval'))
    from compare_ptq_frontier import recorded_protocol
    baseline_manifest=load(P/'evidence/heldout_bf16/eval_manifest.json')
    protocol_file=recorded_protocol(baseline_manifest,P.parent,
                                    publication_protocol(baseline_manifest,protocol_file))
    protocol_paths={baseline_manifest['protocol_file']:protocol_file} if baseline_manifest.get('protocol_file') else None
    package_inputs.add(protocol_file)
    paired,_=paired_rows();check_pairing(paired,protocol_file);budget_rows()
    source_names={f'heldout_{arm}/{name}.json' for arm in ('bf16','ptq','qad','continued_qad','qad_opd')
                  for name in ('eval_manifest','task_results','summary')}
    require(set(paired['source_files'])==source_names, 'Paired source evidence set differs')
    for name,row in paired['source_files'].items():
        package_inputs.add(check_record({'path':'paper/evidence/'+name,**row},P.parent))
    require(paired['implementation_sha256']==sha(P.parent/'eval/compare_recovery.py'), 'Paired comparison used different implementation bytes')
    sys.path.insert(0,str(P.parent/'eval'))
    from compare_recovery import compare_round
    require(compare_round(P/'evidence',protocol_paths)==paired, 'Copied heldout evidence does not reproduce the reported comparison')
    check_heldout_raw_logs(package_inputs, protocol_file)
    action_diagnostics = check_action_diagnostics(package_inputs)
    gptq_reference = check_gptq_reference(package_inputs)
    # v12 publishes one selected W4A4 checkpoint; rebuild that frontier from
    # its heldout-only source instead of accepting the retired PTQ ladder
    # validator and its reference runs.
    check_v12_frontier(package_inputs, paired)
    renders=load(P/'validation/figure-renders.json')
    require(renders.get('status')=='passed' and renders.get('scale')==2.0,'Figure rasterization did not pass')
    require(len(renders['figures'])==len(expected_svg) and {x['svg'] for x in renders['figures']}==expected_svg,'Raster provenance set differs')
    for row in renders['figures']:
        require(row['png']==str(Path(row['svg']).with_suffix('.png')),'Raster target is not the matching figure')
        check_record({'path':row['svg'],'sha256':row['svg_sha256']},P.parent)
        check_record({'path':row['png'],'sha256':row['png_sha256']},P.parent)
    build=load(P/'validation/html-build.json')
    require(build.get('version')==1 and build.get('status')=='passed','HTML build provenance missing')
    require(build['inputs']==html_build_inputs(P),'HTML is stale after source, figure, asset, or builder edits')
    require(build['output']==file_record(P/'paper.html',P),'HTML differs from the completed build')
    checks=capture_records(load_capture_manifests(shots,package_inputs),shots,package_inputs)
    for name in figures:
        png=P/'figs'/(name+'.png');copy=P/'zhihu/images'/png.name
        require(png.read_bytes()==copy.read_bytes(),f'Zhihu image differs: {name}')
        require(f'(images/{name}.png)' in md,f'Zhihu figure missing: {name}')
    page=Page();page.feed(doc)
    require(len(page.figures)==len(required) and set(page.figures)==required,'HTML figure identities differ')
    require(len(page.ids)==len(set(page.ids)),'Duplicate HTML anchors')
    require(set(page.anchors)<=set(page.ids),'Broken internal anchor')
    require(not page.resources and not re.search(r'url\(\s*[\"\']?(?:https?:)?//|@import',doc,re.I),
            'Offline HTML has external runtime resources')
    require(all(value.startswith('data:image/png;base64,') for value in page.images),'HTML image is not an embedded PNG')
    embedded={hashlib.sha256(base64.b64decode(value.split(',',1)[1],validate=True)).hexdigest() for value in page.images}
    require({row['sha256'] for row in checks}<=embedded,'A verified screenshot is absent from the HTML')
    for name in diagrams:require((P/'figs'/(name+'.svg')).read_text() in doc,f'HTML contains a stale SVG: {name}')
    code=lambda text:re.findall(r'```[^\n]*\n([\s\S]*?)```',text)
    html_code=[html.unescape(value) for value in re.findall(r'<pre><code[^>]*>([\s\S]*?)</code></pre>',doc)]
    require([x.rstrip('\n') for x in code(source)]==[x.rstrip('\n') for x in html_code],'HTML changed code fences')
    browser=load(P/'validation/browser-validation.json')
    require(browser.get('pass') is True,'Browser QA has not passed')
    require(browser['html_sha256']==sha(P/'paper.html') and browser['qa_script_sha256']==sha(P/'qa_browser.cjs') and
            browser['figures_manifest_sha256']==sha(P/'figures.json'),'Browser QA is stale')
    require(browser['svg_sha256']=={name:sha(P/'figs'/(name+'.svg')) for name in diagrams},'Browser SVG QA is stale')
    require(browser['png_sha256']=={name:sha(P/'figs'/(name+'.png')) for name in required},'Browser image QA is stale')
    require(set(browser['desktop']['figureNames'])==required and browser['desktop']['figures']==len(required),'Browser did not inspect every figure')
    require(browser['desktop']['mathCount']==build['math_expressions'],'Browser did not render every equation')
    runtime=load(P/'evidence/runtime/manifest.json')
    require(set(runtime['checkpoints'])=={'bf16','ptq','qad','continued_qad','qad_opd'},'Runtime provenance must identify five final checkpoints')
    check_evaluation_environment(package_inputs, runtime, protocol_file)
    from capture_runtime import DEFAULT_SOURCE_FILES
    require(set(DEFAULT_SOURCE_FILES)<=set(runtime['source_files']), 'Runtime source provenance omits maintained implementation files')
    for arm,checkpoint in runtime['checkpoints'].items():
        evaluated=load(P/'evidence'/('heldout_'+arm)/'eval_manifest.json')
        require(Path(checkpoint['path']).resolve()==Path(evaluated['checkpoint']).resolve(),f'Runtime checkpoint does not match evaluated {arm}')
        require(evaluated['protocol_sha256']==sha(protocol_file),f'Evaluated protocol differs from current publication: {arm}')
        files=checkpoint['files']
        require(files and any(row['name'].endswith('.safetensors') for row in files),f'Checkpoint has no recorded weights: {arm}')
        require(len({row['name'] for row in files})==len(files),f'Duplicate checkpoint file identity: {arm}')
        for row in files:
            require(Path(row['name']).name==row['name'] and type(row['bytes']) is int and row['bytes']>0 and
                    re.fullmatch(r'[0-9a-f]{64}',row['sha256']),f'Invalid checkpoint file identity: {arm}')
    for path,row in runtime['source_files'].items():package_inputs.add(check_record({'path':path,**row},P.parent))
    training_costs=check_training_costs(runtime,package_inputs,protocol_file)
    search_costs=check_search_costs(package_inputs,protocol_file)
    report={'passed':True,'source_sections':len(sections),'figures':len(figures),'required_figure_references':len(required),
            'action_diagnostics_recomputed': True,
            'action_diagnostic_comparisons': list(action_diagnostics['comparisons']),
            'gptq_reference_raw_logs_replayed': gptq_reference['raw_logs_replayed'],
            'gptq_reference_statistics_recomputed': gptq_reference['paired_statistics_recomputed'],
            'gptq_tensor_contents_reverified_offline': gptq_reference['tensor_contents_reverified_offline'],
            'verified_screenshots':len(checks),'code_blocks_preserved':len(code(source)),'offline_html':True,
            'table_of_contents_links':len(page.anchors),'screenshots':checks,'training_cost_evidence_verified':True,
            'teacher_backward_accounting':{
                key:training_costs['training']['qad_opd'][key]
                for key in ('logged_teacher_backward_passes','scheduled_teacher_backward_passes','teacher_schedule_note')},
            'raw_heldout_logs_verified': True, 'raw_heldout_log_files': 100,
            'search_cost_evidence_verified': True, 'search_cost_evidence_files': search_costs['files'],
            'outputs':{str(path.relative_to(P)):{'bytes':path.stat().st_size,'sha256':sha(path)} for path in (P/'paper.html',P/'zhihu/article.md')},
            'package_inputs':[file_record(path,P.parent) for path in sorted(package_inputs)]}
    if write_report:
        (P/'validation').mkdir(exist_ok=True)
        temporary=P/'validation/publication-validation.json.tmp'
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(P/'validation/publication-validation.json')
    return report


def main():
    try:
        report=validate()
    except Exception as error:
        # A failed rerun must not leave an apparently current passed report.
        (P/'validation').mkdir(exist_ok=True)
        (P/'validation/publication-validation.json').write_text(json.dumps(
            {'passed':False,'failure':str(error),'validator_sha256':sha(Path(__file__))},ensure_ascii=False,indent=2)+'\n')
        raise
    print(json.dumps({key:value for key,value in report.items() if key not in ('screenshots','outputs','package_inputs')},ensure_ascii=False))


if __name__=='__main__':main()
