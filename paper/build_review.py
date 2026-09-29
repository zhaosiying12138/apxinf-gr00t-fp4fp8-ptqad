#!/usr/bin/env python3
"""Render an explicitly incomplete reading draft; never certify publication results.

Run with requirements-build.txt. Existing execution screenshots are copied verbatim.
The final publication validator intentionally rejects the red xxx placeholders.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import zipfile

import cairosvg
import make_figs as figures

P = Path(__file__).resolve().parent
RED = '#b42318'


def pending_charts():
    chart = figures.SVG(1100, 510, '闭环恢复：五臂对照',
                        '结构审阅稿 · 红色 xxx% 等待独立评测；不预画收益高度。')
    rows = [('BF16', '原始模型参考'), ('PTQ', '固定混合精度配方'),
            ('QAD', '量化基座＋演示恢复'), ('继续 QAD', '从同一 QAD 检查点继续演示训练'),
            ('QAD + OPD', '从同一 QAD 检查点加入教师监督')]
    for index, (name, detail) in enumerate(rows):
        y = 102 + index * 65
        chart.rect(30, y, 1040, 52, '#f4f7f8')
        chart.text(48, y + 32, name, 18, weight=600)
        chart.text(250, y + 32, detail, 16)
        chart.text(985, y + 32, 'xxx%', 22, RED, 600, 'end')
    chart.text(32, 475, '每臂 10 任务 × 10 回合；最终补齐逐任务计数与 OPD 的两项增量比较。', 16)
    chart.save('ladder')
    chart = figures.SVG(1100, 420, '压缩与成功率：计入恢复参数后的比较',
                        '结构审阅稿 · 最终展示开发阶段选定配方及必要的纯 PTQ 参考。')
    chart.text(45, 132, '比较项', 18, weight=600)
    chart.text(470, 132, '完整目标编码预算', 18, weight=600)
    chart.text(850, 132, '闭环成功率', 18, weight=600)
    for index, name in enumerate(('纯 PTQ 参考', '高 FP4 配方 + QAD', '高 FP4 配方 + QAD + OPD')):
        y = 198 + index * 62
        chart.text(45, y, name, 18)
        chart.text(520, y, 'xxx GB', 21, RED, 600)
        chart.text(880, y, 'xxx%', 21, RED, 600)
    chart.text(45, 387, '预算包含格式缩放、未量化张量和 BF16 低秩参数；图中尚无实测成功率。', 15)
    chart.save('ptq_frontier')


def main():
    source = '\n\n'.join(p.read_text() for p in sorted((P/'sections').glob('*.md')))
    if 'textcolor{#b42318}' not in source or '审阅稿' not in (P/'meta.json').read_text():
        raise RuntimeError('This builder is only for explicitly marked incomplete review drafts')
    names = re.findall(r'\{\{fig:([\w.-]+)\}\}', source)
    registry = json.loads((P/'figures.json').read_text())
    if len(names) != len(set(names)) or set(names) != set(registry):
        raise RuntimeError('Review must preserve each registered figure exactly once')
    if sum(n.startswith('shot_') for n in names) != 17:
        raise RuntimeError('Review must preserve all 17 screenshots')

    registry['ladder']['note'] = '五臂结果位置已固定；红色 xxx% 为待测值，回填完整独立评测后再比较 QAD 与 OPD。'
    registry['ptq_frontier']['note'] = '比较完整编码预算与同协议成功率，恢复臂包含 BF16 低秩残差。审阅稿保留图位，不预设结果曲线。'
    registry['recovery_protocol']['note'] = '开发、学生采集与最终评测使用互不重叠的官方初态。两个续训分支从同一 QAD 检查点出发；教师额外数据和计算单列。'
    registry['shot_qad']['note'] = registry['shot_qad']['note'].replace('不代表正式 500 步恢复结果', '不代表正式恢复结果')
    registry['shot_bake']['note'] = ('读取新采集的 2 窗口 H，以 required 模式写出 head_lang：80 个 NVFP4-GPTQ、253 个 NVFP4-RTN、139 个 FP8 张量。'
        '这是 BF16 容器中的量化格点值短测，独立于正式 128 窗口工件。图中 2.6860× 为可量化物理张量子集的目标预算；'
        '完整模型采用另一分母，数值见正文预算表。两者均非稠密文件实测压缩比。')
    number = 0
    for name in names:
        if not name.startswith('shot_'):
            number += 1
            registry[name]['title'] = re.sub(r'^图 \d+\s*', f'图 {number}　', registry[name]['title'])
    (P/'figures.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')

    # Only the two unfinished result charts use placeholders. These other charts
    # retain their real data or exact mathematical diagram, with existing checks.
    figures.e1_chart()
    figures.budget_ladder()
    figures.recovery_protocol()
    pending_charts()
    for name in names:
        if not name.startswith('shot_'):
            cairosvg.svg2png(url=str(P/'figs'/f'{name}.svg'),
                            write_to=str(P/'figs'/f'{name}.png'), scale=2)
    subprocess.run([sys.executable, str(P/'build_html.py')], check=True)
    subprocess.run([sys.executable, str(P/'export_zhihu.py')], check=True)

    files = [P/'paper.html',P/'zhihu/article.md',P/'meta.json',P/'figures.json']
    files += sorted((P/'sections').glob('*.md'))
    files += [P/'zhihu/images'/f'{name}.png' for name in names]
    manifest = {'status':'incomplete_review_draft', 'not_final_publication':True,
                'screenshots':17, 'figures':len(names), 'files':[
                    {'path':str(p.relative_to(P)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                     'bytes':p.stat().st_size} for p in files]}
    note = ('# 中文论文结构审阅包\n\n打开 paper.html 离线阅读；知乎稿位于 zhihu/article.md。\n\n'
            '红色 xxx 为待完成实验，不是测量值。本文稿用于审阅结构、方法与表达，尚未通过最终结果发布验收。\n'
            '保留 17 张真实运行截图、完整方法图和执行基准表；截图文件未被此构建器修改。\n')
    output = P/'apxinf-gr00t-fp4fp8-ptqad-review.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as bundle:
        for p in files: bundle.write(p,str(p.relative_to(P)))
        bundle.writestr('README.md',note)
        bundle.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (P/'validation/review-build.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Review draft:',output)


if __name__ == '__main__':
    main()
