#!/usr/bin/env python3
"""Build a publication ZIP only after a fresh, fail-closed validation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import zipfile

from publication_guard import file_record, require
from validate_publication import validate

P = Path(__file__).resolve().parent
DESTINATION = 'apxinf-gr00t-fp4fp8-ptqad-publication.zip'
DOCUMENT_TEMPLATES = (
    'readme_v12_installation.md', 'readme_v12_legacy_engine.md',
    'readme_v12_workflow.md', 'readme_v12_supplement_commands.md',
    'readme_v12_capture_commands.md', 'readme_v12_engine_commands.md',
)
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
    files = [P/name for name in ('paper.html','README.md','meta.json','figures.json','requirements-build.txt','requirements-evidence.txt',
                                'analysis_plan_w4a4.json','publication_scope_v12.json')]
    files.extend((P/'sections').glob('*.md'))
    files.extend(P/name for name in DOCUMENT_TEMPLATES)
    files.extend((P/'templates/v12').glob('*.md'))
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


def checked_asset(path):
    """Only regular, lightweight files inside the repository enter the ZIP."""
    root = P.parent.resolve()
    path = Path(path).absolute()
    require(path.resolve().is_relative_to(root), f'External package dependency: {path}')
    require(not any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)),
            f'Symlink package dependency: {path}')
    require(path.is_file(), f'Missing package dependency: {path}')
    require(path.suffix.lower() not in {'.safetensors', '.pt', '.pth', '.npy', '.npz', '.zip', '.tmp'}
            and '__pycache__' not in path.parts,
            f'Weights, cache, nested archives or temporary files must not enter publication: {path}')
    require(path.stat().st_size < 128 * 1024 * 1024, f'Unexpectedly large publication asset: {path}')
    return path.resolve()


def add_markdown_dependencies(paths):
    """Keep README-linked documents and assets at their original repo paths.

    Linked Markdown is inspected recursively. Source includes sometimes use
    repository-root paths; those are resolved only when the document-relative
    target does not exist. Original document bytes are never rewritten.
    """
    root = P.parent.resolve()
    queue = [root / 'README.md', P / 'README.md']
    inspected, directories = set(), set()
    while queue:
        document = checked_asset(queue.pop())
        paths.add(document)
        if document in inspected:
            continue
        inspected.add(document)
        source = re.sub(r'^```[^\n]*\n[\s\S]*?^```[^\n]*$', '',
                        document.read_text(encoding='utf-8'), flags=re.M)
        for link in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', source):
            parsed = urlsplit(link.strip('<>'))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            relative = unquote(parsed.path)
            target = document.parent / relative
            require(target.resolve().is_relative_to(root), 'Document link escapes repository: ' + link)
            if not target.exists() and (root / relative).exists():
                target = root / relative
            require(target.exists(), f'Broken document link: {document} -> {link}')
            require(not any(part.is_symlink() for part in (target, *target.parents)
                            if part.absolute().is_relative_to(root)),
                    f'Symlink document dependency: {target}')
            target = target.resolve()
            if target.is_dir():
                directories.add(target)
                continue
            paths.add(checked_asset(target))
            if target.suffix == '.md':
                queue.append(target)
    # The supplementary compile screenshot cites a manifest rather than every
    # source file separately. Include exactly its hash-verified provenance.
    compile_manifest = root / 'docs/evidence/ubuntu-compile/manifest.json'
    if compile_manifest in paths:
        for record in json.loads(compile_manifest.read_text(encoding='utf-8'))['files']:
            target = checked_asset(root / record['path'])
            require(file_record(target, root) == record,
                    f'Compile screenshot provenance changed: {target}')
            paths.add(target)
    for directory in directories:
        require(any(path.is_relative_to(directory) for path in paths),
                f'Unbundled document directory: {directory}')


def publication_entries(files, records):
    """Preserve repository-relative paths, including both README entrypoints."""
    root = P.parent.resolve()
    paths = set(files)
    links = {}
    for record in records:
        source = checked_asset(root / record['path'])
        require(file_record(source, root) == record, f'Validated evidence changed: {source}')
        if source.resolve().is_relative_to(P):
            require(source in paths,
                    f'Validated internal input is absent from the publication allowlist: {source}')
        else:
            paths.add(source)
            links[record['path']] = record['path']
    add_markdown_dependencies(paths)
    entries = {checked_asset(path).relative_to(root).as_posix(): path for path in sorted(paths)}
    return entries, links


def package_readme():
    return """# 论文发布包阅读说明

本包保留仓库相对目录，不需要移动文件即可阅读：

- [README.md](README.md)：项目介绍、完整指标、安装与执行命令。
- [paper/paper.html](paper/paper.html)：包含公式、图表和 17 张 Ubuntu 截图的离线文章。
- [paper/zhihu/article.md](paper/zhihu/article.md)：知乎 Markdown 稿及其 `images/` 图片。
- [paper/README.md](paper/README.md)：论文与证据入口。
- [paper/evidence/](paper/evidence/)：原始回合记录、统计、动作诊断与截图来源。

`PROJECT-INPUTS.json` 将原仓库路径映射到本包相同路径，`PACKAGE-MANIFEST.json` 记录包内文件的字节数与 SHA-256。正文代码与部分源码说明中的 `paper/...`、`docs/...` 路径以本包根目录为基准；文章内跳转不改变命令的工作目录。

这是便于阅读和核查的轻量证据包，包含 README 引用的本地文档、图像和编译截图来源；模型权重、观测张量与教师缓存保留记录的身份哈希，不携带原始大文件。安装依赖、下载模型与数据仍按 README 访问相应上游，受上游许可与网络条件约束。本包不承诺完整离线安装或完整离线重跑训练与评测；仅凭哈希也不能恢复原始权重。
""".encode('utf-8')


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
    entries, links = publication_entries(files, report['package_inputs'])
    # Store the current report at its repository-relative path, even if the
    # standalone receipt was absent or stale. Other source bytes stay intact.
    generated={'paper/validation/publication-validation.json':(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode(),
               'PROJECT-INPUTS.json':(json.dumps({'scope':'Repository-relative paths preserved in this lightweight publication archive; model weights are represented only by their recorded hashes.',
                                                'files':links},ensure_ascii=False,indent=2)+'\n').encode(),
               'PUBLICATION-PACKAGE.md':package_readme()}
    manifest={name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in entries.items()}
    manifest.update({name:{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()} for name,payload in generated.items()})
    require((entries, links) == publication_entries(files, report['package_inputs']),
            'Document dependencies changed while preparing publication')
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
