#!/usr/bin/env python3
"""Fail-closed publication validation. Checks remain active under python -O."""
import base64
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import sys
import subprocess

from publication_guard import capture_crop_contract, check_record, file_record, html_build_inputs, require, resolve_inside

P = Path(__file__).resolve().parent
CURRENT_INPUTS = {
    'paper/evidence/paired_comparison.json', 'paper/evidence/recipe_inventory.json',
    'paper/evidence/frontier_comparison.json',
    'results/baselines/pi05_pt_bf16_ptqad_20260929.json',
    'results/baselines/gr00t_pt_bf16_ptqad_20260929.json',
    'results/engine/pi05_bf16_ptqad_20260929.json',
    'results/engine/gr00t_bf16_ptqad_20260929.json',
    'results/engine/pi05_nvfp4_ptqad_20260929.json',
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())


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
    phrases=('正式开发结果将在','五臂最终结果将在','本轮执行结果将在',
             '本轮显式精度配置的 PyTorch 与 APXInf 测量结果将在','训练、教师标注、额外学生探针与评测成本分别从实际运行日志统计',
             '本轮矩阵乘时延、吞吐和格式可用性将在','本轮逐形状结果将在')
    for text in phrases: require(text not in source, 'Unfinished experiment section: '+text)
    for label,body in re.findall(r'<!-- BEGIN ([A-Z0-9 ]+) -->([\s\S]*?)<!-- END \1 -->',source):
        require(not re.search(r'将在|待填|待测|待运行|尚未完成|结果占位|运行中',body),
                'Unfinished result block: '+label)
    require('FPVMATHTOKEN' not in source and 'FPVCODETOKEN' not in source,'Unexpected build tokens in sources')


def expected_markdown(sections,meta,figures):
    text='# '+meta['title']+'\n\n'+meta['meta']+'\n\n'
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
        checked.append({'file':path.name,'sha256':row['sha256'],'verified':True,
                        'retained_unaffected':row.get('retained_unaffected') is True,
                        'acceptance':acceptance,'verified_git_anchor':git_anchor})
    return checked


def check_pairing(data,protocol_file=None):
    sys.path.insert(0,str(P.parent/'eval'))
    from compare_ptq_frontier import protocol_contract
    expected=protocol_contract(protocol_file or P.parent/'exp/recovery_protocol.json')['heldout']
    indices=expected['init_state_indices']
    require(data.get('environment_pairing_verified') is True,'Environment pairing has not passed')
    order={'bf16','ptq','qad','continued_qad','qad_opd'}
    require(set(data['arms'])==order,'Exactly five completed heldout arms are required')
    baseline=data['arms']['bf16']['episodes']
    require(len(baseline)==100,'Expected 100 heldout episodes per arm')
    for name,arm in data['arms'].items():
        episodes=arm['episodes'];require(arm['count']==100 and len(episodes)==100,f'Incomplete heldout arm: {name}')
        require(len(arm['per_task'])==10 and all(x['episodes']==10 for x in arm['per_task'].values()),'Expected ten episodes per task')
        for actual,expected in zip(episodes,baseline):
            require(type(actual['success']) is bool,'Episode outcome must be Boolean')
            require({k:v for k,v in actual.items() if k!='success'}=={k:v for k,v in expected.items() if k!='success'},
                    f'Unpaired environment initial state: {name}')
            require(actual['init_state_index'] in indices,'Heldout initial-state index outside declared partition')
            for field in ('initial_state_sha256','restored_state_sha256','init_state_bank_sha256'):
                require(re.fullmatch(r'[0-9a-f]{64}',actual[field]) is not None,'Invalid initial-state hash')
        require(sum(x['success'] for x in episodes)==arm['successes'],'Episode outcome count differs')
        for task,record in arm['per_task'].items():
            task_rows=[x for x in episodes if x['task']==task]
            require({x['init_state_index'] for x in task_rows}==set(indices) and len(task_rows)==10,
                    'Heldout initial states repeated or missing')
            require(sum(x['success'] for x in task_rows)==record['successes'],'Task success count differs')


def check_training_costs(runtime, package_inputs,protocol_file=None):
    """Recompute public costs, then bind them to the evaluated weight identities."""
    from collect_training_costs import verify_published
    folder=P/'evidence/training'
    costs=verify_published(folder)
    require(costs.get('status')=='complete' and set(costs['training'])=={'qad','continued_qad','qad_opd'},
            'Exactly three completed formal training stages are required')
    require(costs['protocol_sha256']==sha(protocol_file or P.parent/'exp/recovery_protocol.json'),
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
    collection=load(folder/'collection/eval_manifest.json')
    require(Path(collection['checkpoint']).resolve()==Path(runtime['checkpoints']['qad']['path']).resolve(),
            'Collection student identity differs from evaluated QAD')
    teacher=load(folder/'teacher/teacher_probes.json')
    require(Path(teacher['teacher']).resolve()==Path(runtime['checkpoints']['bf16']['path']).resolve() and
            teacher['teacher_weights']==weights('bf16'),'Teacher identity differs from evaluated BF16')
    return costs


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
    protocol_file=recorded_protocol(baseline_manifest,P.parent,protocol_file)
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
    from frontier_evidence import validate_frontier
    _,frontier_files=validate_frontier(P,paired)
    package_inputs.update(frontier_files)
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
    report={'passed':True,'source_sections':len(sections),'figures':len(figures),'required_figure_references':len(required),
            'verified_screenshots':len(checks),'code_blocks_preserved':len(code(source)),'offline_html':True,
            'table_of_contents_links':len(page.anchors),'screenshots':checks,'training_cost_evidence_verified':True,
            'teacher_backward_accounting':{
                key:training_costs['training']['qad_opd'][key]
                for key in ('logged_teacher_backward_passes','scheduled_teacher_backward_passes','teacher_schedule_note')},
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
