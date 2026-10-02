#!/usr/bin/env python3
"""Build the offline single-file paper. Requires markdown-it-py and Node.js."""
from __future__ import annotations
import base64
import html
import json
import pathlib
import re
import struct
import subprocess
from markdown_it import MarkdownIt
from publication_guard import file_record, html_build_inputs, require

PAPER = pathlib.Path(__file__).resolve().parent
CSS = r'''
:root{--ink:#172c3c;--muted:#627580;--line:#dce5e9;--blue:#17667a;--paper:#fff;--wash:#f3f7f8}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:#f4f6f7;color:var(--ink);font:17px/1.94 "Noto Serif CJK SC","Source Han Serif SC","Microsoft YaHei",serif}
main{max-width:1020px;margin:32px auto;background:var(--paper);padding:64px 76px 96px;box-shadow:0 4px 24px #172c3c08}
header{border-bottom:2px solid var(--blue);padding-bottom:30px;margin-bottom:36px}
.eyebrow{font:600 12px/1.5 system-ui,sans-serif;letter-spacing:.2em;color:var(--blue)}
h1{font-size:34px;line-height:1.5;letter-spacing:.015em;margin:16px 0 20px;font-weight:700}
.meta{font:14px/1.9 system-ui,"Microsoft YaHei",sans-serif;color:var(--muted)}
h2,h3,h4,h5{font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.55;scroll-margin-top:24px}
h2{font-size:25px;margin:60px 0 22px;border-top:1px solid var(--line);padding-top:28px}
h3{font-size:20px;margin:38px 0 18px}h4{font-size:17px;margin-top:28px}
p{margin:14px 0;overflow-wrap:anywhere}strong{font-weight:700;color:#123e50}
a{color:#12667e;text-underline-offset:3px;text-decoration-thickness:1px}a:hover{color:#a45627}
nav{background:var(--wash);padding:22px 28px;border-radius:8px;font:14px/1.9 system-ui,"Microsoft YaHei",sans-serif}
nav ol{columns:2;column-gap:32px;margin:12px 0 0;padding-left:0;list-style:none}nav li{break-inside:avoid;padding:3px 0}
nav details{border-top:1px solid var(--line);margin-top:14px;padding-top:12px}nav summary{cursor:pointer;color:var(--blue)}
nav .sub{font-size:13px;color:var(--muted)}
figure{margin:32px 0}figure svg,figure img{display:block;width:100%;height:auto;max-width:100%;border:1px solid var(--line);border-radius:4px}
figcaption{font:13px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:var(--muted);margin-top:10px}
.shot img{cursor:zoom-in}.table-wrap{overflow-x:auto;margin:24px 0;border:1px solid var(--line);border-radius:5px}
.pending-shot{border:1px dashed #b42318;border-radius:5px;background:#fff8f6;padding:28px;font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif}.pending-shot strong{color:#b42318}.pending-shot p{margin:8px 0 0}
table{border-collapse:collapse;width:100%;font:13.5px/1.8 system-ui,"Microsoft YaHei",sans-serif}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;min-width:65px}
th{background:#eaf2f5;color:#183c4d;font-weight:650}tbody tr:nth-child(even){background:#f8fafb}
td code{white-space:normal}code{font:13px/1.75 "Noto Sans Mono CJK SC",Consolas,monospace;background:#eef3f5;padding:2px 4px;border-radius:3px}
pre{background:#142734;color:#e0eaf0;padding:20px 24px;border-radius:6px;overflow-x:auto;line-height:1.75;font-size:13px}
pre code{background:none;padding:0;color:inherit;white-space:pre}
blockquote{margin:24px 0;padding:10px 22px;border-left:3px solid #3a8092;background:#f0f6f8;font-size:15px}
li{margin:8px 0}ul,ol{padding-left:1.6em}.katex{font-size:1.03em!important}.katex-display{overflow-x:auto;overflow-y:hidden;padding:6px 0}
.math-block{margin:24px 0}.backtop{position:fixed;right:18px;bottom:18px;font:13px system-ui;background:white;border:1px solid var(--line);border-radius:20px;padding:10px 16px}
footer{border-top:1px solid var(--line);margin-top:50px;padding-top:18px;font:13px/1.8 system-ui;color:var(--muted)}
dialog{width:98vw;max-width:98vw;height:96vh;border:0;padding:32px 12px 12px;background:#111;color:white}dialog::backdrop{background:#000b}
dialog img{width:100%;height:100%;object-fit:contain}dialog button{position:absolute;right:12px;top:6px;cursor:pointer}
@media(max-width:760px){body{font-size:16px}main{margin:0;padding:30px 20px 60px;box-shadow:none}h1{font-size:27px}h2{font-size:23px}nav ol{columns:1}nav{padding:18px}table{font-size:12.5px}pre{padding:16px}.backtop{display:none}}
@media print{body{background:white;font-size:11pt}main{margin:0;max-width:none;padding:0;box-shadow:none}nav,.backtop,dialog{display:none}h2,h3,h4{break-after:avoid}figure,pre,tr{break-inside:avoid}a{color:inherit}pre{white-space:pre-wrap;color:#172c3c;background:#f1f4f6}.table-wrap{overflow:visible}table{font-size:9pt}}
'''

def data_uri(path):
    mime = 'font/woff2' if path.suffix == '.woff2' else 'image/png'
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()

def math_and_code(md):
    """Protect code first so $, backslashes and Markdown inside TeX stay literal."""
    codes, maths = [], []
    def keep_code(m):
        codes.append(m.group()); return f'FPVCODETOKEN{len(codes)-1}END'
    md = re.sub(r'```[^\n]*\n[\s\S]*?```|`[^`\n]+`', keep_code, md)
    def keep_math(m):
        display = m.group().startswith('$$')
        edge = 2 if display else 1
        maths.append({'tex': m.group()[edge:-edge].strip(), 'display': display})
        token = f'FPVMATHTOKEN{len(maths)-1}END'
        return '\n\n' + token + '\n\n' if display else token
    md = re.sub(r'\$\$[\s\S]*?\$\$|(?<!\\)\$(?!\$)[^$\n]+?(?<!\\)\$', keep_math, md)
    md = re.sub(r'FPVCODETOKEN(\d+)END', lambda m: codes[int(m[1])], md)
    rendered = subprocess.run(['node', str(PAPER/'render_math.cjs')], input=json.dumps(maths),
                              text=True, encoding='utf-8', capture_output=True, check=True)
    return md, json.loads(rendered.stdout), maths

def main():
    meta = json.loads((PAPER/'meta.json').read_text(encoding='utf-8'))
    figures = json.loads((PAPER/'figures.json').read_text(encoding='utf-8'))
    pending = {name for name, info in figures.items() if info.get('refresh_pending') is True}
    require(all(name.startswith('shot_') for name in pending), 'Only execution screenshots may await refresh')
    require(not pending or '审阅稿' in meta.get('status', ''),
            'Pending screenshots forbid a final publication; explicitly mark meta.status as a review draft')
    inputs = html_build_inputs(PAPER)
    md = '\n\n'.join(p.read_text(encoding='utf-8') for p in sorted((PAPER/'sections').glob('*.md')))
    # Control markers are editorial metadata, never reader-visible prose.
    md = re.sub(r'<!-- (?:BEGIN|END) [A-Z0-9 ]+ -->', '', md)
    md, maths, math_sources = math_and_code(md)
    parser = MarkdownIt('commonmark', {'html': False, 'typographer': False}).enable('table')
    doc = parser.render(md)
    for i, result in enumerate(maths):
        token = f'FPVMATHTOKEN{i}END'
        if math_sources[i]['display']:
            doc = doc.replace('<p>'+token+'</p>', '<div class="math-block">'+result+'</div>')
        else:
            doc = doc.replace(token, result)
    def fig(m):
        name = m[1]
        info = figures[name]
        if name in pending:
            return (f'<figure class="shot refresh-pending" id="fig-{name}" data-figure="{name}" data-refresh-pending="true">'
                    '<div class="pending-shot" role="note"><strong>待补本轮真实运行截图</strong>'
                    '<p>保留此复现环节，完成实拍与来源核验后补入。</p></div>'
                    f'<figcaption>{html.escape(info["title"])} · {html.escape(info["note"])}</figcaption></figure>')
        path = PAPER/'figs'/(name+'.svg')
        if path.exists():
            content = path.read_text(encoding='utf-8')
        else:
            path = path.with_suffix('.png')
            if not path.exists(): raise FileNotFoundError(path)
            width,height=struct.unpack('>II',path.read_bytes()[16:24])
            content = f'<img width="{width}" height="{height}" src="{data_uri(path)}" alt="{html.escape(info["title"])}">'
        kind = 'shot' if name.startswith('shot_') else 'diagram'
        return f'<figure class="{kind}" id="fig-{name}" data-figure="{name}">{content}<figcaption>{html.escape(info["title"])} · {html.escape(info["note"])}</figcaption></figure>'
    doc = re.sub(r'<p>\{\{fig:([\w.-]+)\}\}</p>', fig, doc)
    if '{{fig:' in doc: raise ValueError('Unresolved figure token')
    toc, toc_main, counter = [], [], [0]
    def heading(m):
        level, title = int(m[1]), m[2]
        counter[0] += 1
        anchor = f'section-{counter[0]}'
        if level <= 2:
            cls = ' class="sub"' if level == 2 else ''
            toc.append(f'<li{cls}><a href="#{anchor}">{title}</a></li>')
            if level == 1: toc_main.append(f'<li><a href="#{anchor}">{title}</a></li>')
        return f'<h{level+1} id="{anchor}">{title}</h{level+1}>'
    doc = re.sub(r'<h([1-4])>(.*?)</h\1>', heading, doc)
    doc = doc.replace('<table>', '<div class="table-wrap" tabindex="0"><table>').replace('</table>', '</table></div>')
    kcss = (PAPER/'assets/katex/katex.min.css').read_text(encoding='utf-8')
    kcss = re.sub(r'src:[^;}]+', lambda m: 'src:url(' + data_uri(PAPER/'assets/katex'/re.search(r'url\(([^)]+\.woff2)\)', m[0])[1]) + ') format("woff2")', kcss)
    out = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{html.escape(meta['meta'])}"><title>{html.escape(meta['title'])}</title>
<style>{kcss}\n{CSS}</style></head><body><main id="top">
<header><div class="eyebrow">FP4-VLA · RESEARCH REPORT · 2026</div><h1>{html.escape(meta['title'])}</h1><div class="meta">{html.escape(meta['meta'])}<br>修订日期：{meta['date']} · 正文、公式、图表与原始截图均可离线阅读</div></header>
<nav aria-label="文章目录"><strong>阅读导航</strong><ol>{''.join(toc_main)}</ol><details><summary>展开完整章节目录</summary><ol>{''.join(toc)}</ol></details></nav>
<article>{doc}</article><footer>由同一 Markdown 源生成 HTML 与知乎发布稿。完整源稿、运行证据、图像哈希及校验报告随发布包保存。数学排版使用 KaTeX 0.16.22（MIT）；详细来源见发布包。</footer>
</main><a class="backtop" href="#top">返回目录 ↑</a><dialog id="viewer"><button aria-label="关闭大图">关闭 ×</button><img alt="原始截图放大"></dialog>
<script>const v=document.getElementById('viewer'); document.querySelectorAll('.shot img').forEach(i=>i.addEventListener('click',()=>{{v.querySelector('img').src=i.src;v.showModal()}}));v.querySelector('button').onclick=()=>v.close();</script>
</body></html>'''
    (PAPER/'paper.html').write_text(out, encoding='utf-8')
    (PAPER/'_build').mkdir(exist_ok=True)
    (PAPER/'_build/build-report.json').write_text(json.dumps({'math_expressions':len(maths),'figures':len(re.findall('data-figure=',doc)), 'headings': counter[0], 'bytes':len(out.encode())}, indent=2)+'\n')
    require(inputs == html_build_inputs(PAPER), 'HTML inputs changed during rendering')
    (PAPER/'validation').mkdir(exist_ok=True)
    manifest = {'version': 1, 'status': 'passed', 'inputs': inputs,
                'output': file_record(PAPER/'paper.html', PAPER),
                'math_expressions': len(maths), 'figures': len(figures),
                'screenshot_slots': sum(name.startswith('shot_') for name in figures),
                'screenshots': sum(name.startswith('shot_') and name not in pending for name in figures),
                'screenshot_refresh_pending': sorted(pending)}
    temporary = PAPER/'validation/html-build.json.tmp'
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(PAPER/'validation/html-build.json')
    print('Built offline paper:', len(maths), 'math expressions;', len(out.encode()), 'bytes')

if __name__ == '__main__':
    main()
