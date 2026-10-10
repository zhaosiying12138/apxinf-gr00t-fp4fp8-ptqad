#!/usr/bin/env python3
"""Package complete v12 science as an explicitly nonfinal offline review ZIP.

No final screenshot gate is modified or bypassed. Model execution is forbidden
here; only archived evidence and existing document outputs are checked.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import unquote, urlsplit
import zipfile
from publication_guard import check_record, file_record, html_build_inputs, require

P = Path(__file__).resolve().parent
ROOT = P.parent
DESTINATION = 'apxinf-gr00t-fp4fp8-ptqad-review.zip'
MANIFEST = 'REVIEW-MANIFEST.json'
PENDING = frozenset({'shot_bake', 'shot_collect', 'shot_qad', 'shot_rollout',
                     'shot_opdcache', 'shot_opd', 'shot_evalserver'})
EXCLUDED = frozenset({'paper/evidence/captures.json', 'paper/evidence/retained_captures.json',
    'paper/evidence/README.md', 'paper/validation/review-build.json',
    'paper/validation/publication-validation.json'})


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def allowed(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and '..' not in path.parts and '\\' not in name
        and name not in EXCLUDED and not name.startswith('paper/evidence/captures/')
        and not any(path.name == shot + '.png' for shot in PENDING)
        and path.suffix.lower() not in {'.zip', '.tmp', '.pt', '.pth', '.safetensors', '.npy', '.npz'}
        and '__pycache__' not in path.parts)


def check_receipt_hashes(receipt, paths):
    for field, path in paths.items():
        require(receipt.get(field) == digest(path.read_bytes()), 'Stale review receipt: ' + field)


def expected_markdown(meta, figures, source):
    text = '# ' + meta['title'] + '\n\n' + source
    def replace(match):
        name = match[1]
        row = figures[name]
        if name in PENDING:
            return f'**{row["title"]}**\n\n待补本轮真实运行截图。{row["note"]}'
        return f'![{row["title"]}](images/{name}.png)\n\n*{row["title"]}。{row["note"]}*'
    return re.sub(r'\{\{fig:([\w.-]+)\}\}', replace, text).rstrip() + '\n'


def verify_science(inputs):
    from install_final_evidence import verify
    from v12_publication_contract import validate_v12_results
    from validate_publication import (check_pairing, check_heldout_raw_logs, check_v12_frontier,
        check_evaluation_environment, check_training_costs, check_search_costs,
        check_action_diagnostics, check_gptq_reference)
    installed = verify(P / 'evidence')
    manifest = load(P / 'evidence/evidence_manifest.json')
    inputs.update((P / 'evidence/evidence_manifest.json', P / 'evidence/source_bundle_manifest.json'))
    for row in manifest['files']:
        inputs.add(check_record({'path': row['published_path'],
            **{k: row['source'][k] for k in ('bytes', 'sha256')}}, P / 'evidence'))
    protocol = ROOT / 'exp/recovery_protocol_v12_rtn_w4a4.json'
    inputs.add(protocol)
    validate_v12_results(load(P / 'evidence/final_results.json'), root=ROOT)
    paired = load(P / 'evidence/paired_comparison.json')
    check_pairing(paired, protocol)
    sys.path.insert(0, str(ROOT / 'eval'))
    from compare_recovery import compare_round
    original = load(P / 'evidence/heldout_bf16/eval_manifest.json')['protocol_file']
    require(compare_round(P / 'evidence', {original: protocol}) == paired,
            'Task receipts do not reproduce paired comparison')
    require(paired['implementation_sha256'] == digest((ROOT / 'eval/compare_recovery.py').read_bytes()),
            'Paired comparison source changed')
    check_heldout_raw_logs(inputs, protocol)
    check_v12_frontier(inputs, paired)
    runtime = load(P / 'evidence/runtime/manifest.json')
    from capture_runtime import DEFAULT_SOURCE_FILES
    require(set(DEFAULT_SOURCE_FILES) <= set(runtime['source_files']), 'Incomplete runtime source identities')
    require(set(runtime['checkpoints']) == {'bf16', 'ptq', 'qad', 'continued_qad', 'qad_opd'},
            'Runtime must bind five checkpoints')
    for name, row in runtime['source_files'].items():
        inputs.add(check_record({'path': name, **row}, ROOT))
    for arm, checkpoint in runtime['checkpoints'].items():
        evaluated = load(P / 'evidence' / ('heldout_' + arm) / 'eval_manifest.json')
        require(Path(checkpoint['path']).resolve() == Path(evaluated['checkpoint']).resolve(),
                'Runtime checkpoint differs: ' + arm)
        require(evaluated['protocol_sha256'] == digest(protocol.read_bytes()), 'Protocol differs: ' + arm)
    check_evaluation_environment(inputs, runtime, protocol)
    check_training_costs(runtime, inputs, protocol)
    check_search_costs(inputs, protocol)
    action = check_action_diagnostics(inputs)
    gptq = check_gptq_reference(inputs)
    return {'scientific_results_complete': True, 'scientific_archive_files': installed['core_files'],
        'heldout_raw_logs_replayed': True, 'paired_statistics_recomputed': True,
        'action_diagnostics_recomputed': True, 'action_comparisons': list(action['comparisons']),
        'gptq_reference_status': gptq['status'], 'training_and_search_costs_verified': True,
        'protocol_sha256': digest(protocol.read_bytes()),
        'final_manifest_sha256': digest((P / 'evidence/final_manifest.json').read_bytes())}


def verify_outputs(inputs):
    from validate_publication import Page, check_finished, CURRENT_INPUTS
    meta, figures = load(P / 'meta.json'), load(P / 'figures.json')
    require('审阅稿' in meta['status'], 'Review archive requires explicit nonfinal status')
    require({name for name, row in figures.items() if row.get('refresh_pending') is True} == PENDING,
            'Review screenshot scope changed; choose the appropriate publication entry point')
    shots = {name for name in figures if name.startswith('shot_')}
    diagrams = set(figures) - shots
    require(len(shots) == 17 and len(diagrams) == 7, 'Unexpected registered figures')
    source = ''.join(path.read_text().rstrip() + '\n\n' for path in sorted((P / 'sections').glob('*.md')))
    check_finished(source)
    names = re.findall(r'\{\{fig:([\w.-]+)\}\}', source)
    require(set(names) == set(figures) and len(names) == len(figures), 'Figure placements differ')
    require((P / 'zhihu/article.md').read_text() == expected_markdown(meta, figures, source),
            'Zhihu Markdown is stale')
    require({path.name for path in (P / 'zhihu/images').iterdir() if path.is_file()} ==
            {name + '.png' for name in figures if name not in PENDING}, 'Stale or missing exported image')
    for name in set(figures) - PENDING:
        require((P / 'figs' / (name + '.png')).read_bytes() ==
                (P / 'zhihu/images' / (name + '.png')).read_bytes(), 'Exported PNG differs: ' + name)
    provenance = load(P / 'validation/figure-inputs.json')
    require(provenance['status'] == 'passed' and
            {row['path'] for row in provenance['inputs']} == CURRENT_INPUTS, 'Stale figure inputs')
    expected_svg = {'paper/figs/' + name + '.svg' for name in diagrams}
    require({row['path'] for row in provenance['svgs']} == expected_svg, 'Incomplete diagram receipts')
    for row in [provenance['generator'], *provenance['inputs'], *provenance['svgs']]:
        inputs.add(check_record(row, ROOT))
    renders = load(P / 'validation/figure-renders.json')
    require(renders['status'] == 'passed' and len(renders['figures']) == len(diagrams) and
            {row['svg'] for row in renders['figures']} == expected_svg, 'Incomplete PNG receipts')
    for row in renders['figures']:
        require(row['png'] == str(Path(row['svg']).with_suffix('.png')), 'Raster path differs')
        inputs.add(check_record({'path': row['svg'], 'sha256': row['svg_sha256']}, ROOT))
        inputs.add(check_record({'path': row['png'], 'sha256': row['png_sha256']}, ROOT))
    build = load(P / 'validation/html-build.json')
    require(build['status'] == 'passed' and build['inputs'] == html_build_inputs(P), 'Stale HTML build inputs')
    require(build['output'] == file_record(P / 'paper.html', P), 'Stale HTML output')
    browser = load(P / 'validation/browser-validation.json')
    require(browser.get('pass') is True and not browser['errors'], 'Browser QA failed')
    check_receipt_hashes(browser, {'meta_sha256': P / 'meta.json', 'html_sha256': P / 'paper.html',
        'qa_script_sha256': P / 'qa_browser.cjs', 'figures_manifest_sha256': P / 'figures.json'})
    for field, suffix, selected in [('svg_sha256', '.svg', diagrams), ('png_sha256', '.png', set(figures))]:
        require(browser[field] == {name: digest((P / 'figs' / (name + suffix)).read_bytes()) for name in selected},
                'Stale browser image receipt: ' + field)
    for receipt in (build, browser):
        require(receipt['screenshot_slots'] == 17 and receipt['screenshots'] == 10 and
                set(receipt['screenshot_refresh_pending']) == PENDING, 'Incorrect screenshot status')
    desktop = browser['desktop']
    require(set(desktop['pendingNames']) == PENDING and set(desktop['figureNames']) == set(figures)
            and desktop['mathCount'] == build['math_expressions'] and desktop['mathErrors'] == 0,
            'Browser scope differs')
    require({row['figure'] for row in desktop['decoded']} == shots - PENDING and
            all((row['width'], row['height']) == (3840, 2280) for row in desktop['decoded']),
            'Displayed screenshots not decoded at expected dimensions')
    doc = (P / 'paper.html').read_text()
    page = Page(); page.feed(doc)
    require(len(page.figures) == len(figures) and set(page.figures) == set(figures), 'HTML figure set differs')
    require(len(page.ids) == len(set(page.ids)) and set(page.anchors) <= set(page.ids), 'Broken HTML anchors')
    require(not page.resources and not re.search(r'url\(\s*["\']?(?:https?:)?//|@import', doc, re.I),
            'HTML needs external runtime resources')
    require(all(value.startswith('data:image/png;base64,') for value in page.images), 'HTML has external images')
    embedded = {digest(base64.b64decode(value.split(',', 1)[1], validate=True)) for value in page.images}
    displayed = {digest((P / 'figs' / (name + '.png')).read_bytes()) for name in shots - PENDING}
    omitted = {digest((P / 'figs' / (name + '.png')).read_bytes()) for name in PENDING}
    require(embedded == displayed and not embedded & omitted, 'HTML includes missing or pending screenshots')
    for name in diagrams:
        require((P / 'figs' / (name + '.svg')).read_text() in doc, 'Stale embedded SVG: ' + name)
    return {'offline_html': True, 'current_html_markdown_browser_receipts': True,
        'screenshot_slots': 17, 'displayed_screenshots': 10, 'pending_screenshots': sorted(PENDING),
        'omitted_html_build_inputs': [row for row in build['inputs'] if Path(row['path']).stem in PENDING]}


def add_markdown_dependencies(paths):
    inspected = set()
    while True:
        queued = sorted(path for path in (ROOT / 'README.md', P / 'README.md') if path not in inspected)
        if not queued:
            return
        for path in queued:
            inspected.add(path)
            for link in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', path.read_text()):
                parsed = urlsplit(link.strip('<>'))
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = (path.parent / unquote(parsed.path)).resolve()
                require(target.is_relative_to(ROOT), 'Document link escapes repository: ' + link)
                require(target.exists(), 'Broken document link: ' + str(path) + ' -> ' + link)
                require(allowed(target.relative_to(ROOT).as_posix()), 'Excluded review document link: ' + link)
                if target.is_dir():
                    require(any(p.is_relative_to(target) for p in paths), 'Unbundled directory link: ' + link)
                else:
                    paths.add(target)


def collect_files(inputs):
    paths = set(inputs)
    paths.update(ROOT / name for name in ('README.md',))
    paths.update(P / name for name in ('README.md', 'paper.html', 'meta.json', 'figures.json',
        'requirements-build.txt', 'requirements-evidence.txt', 'publication_scope_v12.json',
        'analysis_plan_w4a4.json', 'zhihu/article.md'))
    paths.update(P.glob('*.py')); paths.update(P.glob('*.cjs')); paths.update(P.glob('readme_v12_*.md'))
    for directory in ('sections', 'templates/v12', 'assets', 'zhihu/images'):
        paths.update(p for p in (P / directory).rglob('*') if p.is_file())
    for name in load(P / 'figures.json'):
        if name not in PENDING:
            paths.add(P / 'figs' / (name + '.png'))
            if not name.startswith('shot_'):
                paths.add(P / 'figs' / (name + '.svg'))
    for name in ('figure-inputs.json', 'figure-renders.json', 'html-build.json', 'browser-validation.json'):
        paths.add(P / 'validation' / name)
    paths.update(p for p in (P / 'validation').rglob('*') if p.suffix in {'.py', '.cjs'})
    paths = {path.resolve() for path in paths if allowed(path.relative_to(ROOT).as_posix())}
    add_markdown_dependencies(paths)
    for path in paths:
        require(path.is_file() and path.stat().st_size < 128 * 1024 * 1024,
                'Unsafe or unexpectedly large review input: ' + str(path))
    return {path.relative_to(ROOT).as_posix(): path for path in sorted(paths)}


def verify_archive(path):
    """Verify every archived member; this is not final-publication certification."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate archive members')
        require(all(allowed(name) for name in names), 'Excluded or unsafe archive member')
        manifest = json.loads(archive.read(MANIFEST))
        require(manifest.get('not_final_publication') is True and
                set(manifest.get('pending_screenshots', [])) == PENDING, 'Incorrect review scope')
        require(set(names) == set(manifest['files']) | {MANIFEST}, 'Archive member set differs')
        for name, record in manifest['files'].items():
            payload = archive.read(name)
            require(record == {'bytes': len(payload), 'sha256': digest(payload)}, 'Archive integrity failure: ' + name)
        require(not set(manifest['omitted_capture_sha256']) &
                {row['sha256'] for row in manifest['files'].values()}, 'Pending capture bytes leaked into archive')
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate existing outputs without writing an archive')
    parser.add_argument('--verify', type=Path, help='Verify every member of an existing review ZIP only')
    args = parser.parse_args()
    if args.verify:
        manifest = verify_archive(args.verify)
        print(json.dumps({'status': 'verified_nonfinal_review', 'files': len(manifest['files'])}))
        return
    # Snapshot candidate bytes before validation. Additional checked source
    # dependencies are included below; reject any concurrent edit to candidates.
    inputs = {P / 'evidence' / row['published_path']
              for row in load(P / 'evidence/evidence_manifest.json')['files']}
    inputs.update(ROOT / name for name in load(P / 'evidence/runtime/manifest.json')['source_files'])
    initial_entries = collect_files(inputs)
    initial = {name: file_record(path, ROOT) for name, path in initial_entries.items()}
    science = verify_science(inputs)
    outputs = verify_outputs(inputs)
    entries = collect_files(inputs)
    before = {name: file_record(path, ROOT) for name, path in entries.items()}
    require(initial == {name: file_record(path, ROOT) for name, path in initial_entries.items()},
            'Review candidate changed during validation')
    generated = {'REVIEW-PACKAGE.md': ("""# v12 科学结果审阅包（非最终发布包）

打开 paper/paper.html 可离线阅读全文；知乎稿为 paper/zhihu/article.md。
根目录 README 保留安装、编译、执行、指标与来源，其本地文档链接随包保留。

五臂各 160 回合及动作诊断已完成并冻结。本包重新核验逐回合证据、配对统计、
动作诊断、训练/搜索成本和当前 HTML/Markdown/浏览器收据，不运行模型。

17 个截图位置中展示 10 张未受影响的图片；7 张本轮执行核验截图仍待补拍。
七张旧图片和全部原终端截图来源附件均不在本包。显示图片的字节哈希与浏览器
解码已经核验，完整截图来源的最终验收仍待完成。最终发布检查没有被豁免。

REVIEW-MANIFEST.json 登记除清单自身以外每个成员的字节数与 SHA-256；
清单自身的哈希由 ZIP 外的 paper/validation/review-package.json 登记，避免循环哈希。
在仓库运行 python paper/package_review.py --verify <本 ZIP 路径> 可检查成员完整性。
模型权重、原始训练数据不在包内，身份由原始证据中的哈希记录。

HTML 构建收据保留原始源树输入身份，七张待替换图已明确列为归档省略项。
本包支持离线阅读和证据审阅，不声称省略截图后可重跑完整构建或最终发布门禁。
""").encode()}
    manifest = {'format': 'v12_nonfinal_review_v1', 'not_final_publication': True,
        **science, **outputs, 'terminal_capture_provenance_included': False,
        'omitted_capture_sha256': sorted(digest((P / 'figs' / (name + '.png')).read_bytes()) for name in PENDING),
        'files': {name: {k: row[k] for k in ('bytes', 'sha256')} for name, row in before.items()}}
    manifest['files'].update({name: {'bytes': len(payload), 'sha256': digest(payload)} for name, payload in generated.items()})
    if args.check:
        print(json.dumps({'status': 'validated_nonfinal_review', 'files': len(manifest['files']), **outputs}))
        return
    destination = P / DESTINATION
    temporary = destination.with_suffix('.zip.tmp')
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(manifest['files']):
                payload = generated[name] if name in generated else entries[name].read_bytes()
                require(manifest['files'][name] == {'bytes': len(payload), 'sha256': digest(payload)},
                        'Review input changed while packaging: ' + name)
                archive.writestr(name, payload)
            archive.writestr(MANIFEST, manifest_bytes)
        verify_archive(temporary)
        require(before == {name: file_record(path, ROOT) for name, path in entries.items()},
                'Review inputs changed while packaging')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = {'status': 'verified_nonfinal_review', 'not_final_publication': True,
        'archive': file_record(destination, ROOT), 'files': len(manifest['files']),
        'manifest_sha256': digest(manifest_bytes), **science, **outputs}
    (P / 'validation/review-package.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
