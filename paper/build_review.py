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
    # Reuse the same audited, single-recipe budget reader as the final figures.
    # Success rates remain unknown; byte counts need not be placeholders.
    _, budgets, residual = figures.budget_rows()
    if len(budgets) != 1:
        raise RuntimeError('The review must use one frozen v11 W4A4 category recipe')
    budget = budgets[0]
    chart = figures.SVG(1100, 510, '闭环恢复：五臂对照',
                        '结构审阅稿 · 红色 xxx% 等待独立评测；不预画收益高度。')
    rows = [('BF16', '原始模型参考'), ('PTQ', '冻结 W4A4 配方'),
            ('QAD', '量化基座＋演示恢复'), ('继续 QAD', '从同一 QAD 检查点继续演示训练'),
            ('QAD + OPD', '从同一 QAD 检查点加入教师监督')]
    for index, (name, detail) in enumerate(rows):
        y = 102 + index * 65
        chart.rect(30, y, 1040, 52, '#f4f7f8')
        chart.text(48, y + 32, name, 18, weight=600)
        chart.text(250, y + 32, detail, 16)
        chart.text(985, y + 32, 'xxx%', 22, RED, 600, 'end')
    chart.text(32, 475, '每臂 10 任务 × 16 回合（160 回合）；最终补齐逐任务计数与 OPD 的两项增量比较。', 16)
    chart.save('ladder')
    chart = figures.SVG(1100, 580, '同一 W4A4 基座：编码预算与闭环恢复',
                        '成功率待最终评测；预算由实际权重形状推导，按已知共享别名去重。')
    chart.text(45, 132, '比较项', 18, weight=600)
    chart.text(470, 132, '完整目标编码预算', 18, weight=600)
    chart.text(850, 132, '闭环成功率', 18, weight=600)
    rows = [('BF16', budget['sources'][1]),
            ('W4A4 PTQ', budget['targets'][1]),
            ('W4A4 PTQ + QAD', budget['targets'][1] + residual),
            ('W4A4 PTQ + 继续 QAD', budget['targets'][1] + residual),
            ('W4A4 PTQ + QAD + OPD', budget['targets'][1] + residual)]
    for index, (name, target) in enumerate(rows):
        y = 198 + index * 62
        chart.text(45, y, name, 18)
        chart.text(520, y, f'{target / 1e9:.4f} GB', 21, weight=600)
        chart.text(880, y, 'xxx%', 21, RED, 600)
    chart.text(45, 506, '量化基座含 payload、格式缩放及未量化权重；恢复臂另计完整 BF16 低秩参数。', 15)
    chart.text(45, 543, '该预算不是稠密 checkpoint 文件体积、实际显存或延迟；不预设恢复收益。', 15)
    chart.save('ptq_frontier')


def main():
    source = '\n\n'.join(p.read_text() for p in sorted((P/'sections').glob('*.md')))
    if '#b42318' not in source or '审阅稿' not in (P/'meta.json').read_text():
        raise RuntimeError('This builder is only for explicitly marked incomplete review drafts')
    names = re.findall(r'\{\{fig:([\w.-]+)\}\}', source)
    registry = json.loads((P/'figures.json').read_text())
    # Every registered figure has one canonical position in the reading draft.
    # Duplicate markers silently duplicated images in the HTML and ZIP and
    # shifted figure numbering, so reject them at build time.
    if len(names) != len(registry) or len(set(names)) != len(names) or set(names) != set(registry):
        raise RuntimeError('Review must include every registered figure exactly once')
    if sum(n.startswith('shot_') for n in set(names)) != 17:
        raise RuntimeError('Review must preserve all 17 screenshots')
    screenshot_hashes = {name: hashlib.sha256((P/'figs'/f'{name}.png').read_bytes()).hexdigest()
                         for name in names if name.startswith('shot_')}

    registry['ladder']['title'] = '图 5　同一 W4A4 基座上的五臂闭环对照'
    registry['ladder']['note'] = 'BF16、W4A4 PTQ、QAD、继续 QAD 与 QAD+OPD 使用同一 v11 最终评测协议；红色占位符为待测值。'
    registry['budget_ladder']['title'] = '图 4　W4A4 基座的编码预算与低秩旁路成本'
    registry['budget_ladder']['note'] = '唯一冻结的 v11 W4A4 category 配方按物理张量和已知共享别名去重两种口径核算，计入未量化权重、格式缩放及完整 BF16 低秩旁路。目标编码预算不等于稠密文件或实测显存。'
    registry['ptq_frontier']['title'] = '图 7　全 NVFP4 W4A4 基座的预算与闭环成功率'
    registry['ptq_frontier']['note'] = '编码预算从唯一冻结的 v11 W4A4 category 配方账本生成，按已知共享别名去重；恢复臂另计完整 BF16 低秩残差。成功率仍待最终评测。'
    registry['recovery_protocol']['note'] = '开发、学生采集与最终评测使用互不重叠的官方初态。两个续训分支从同一 QAD 检查点出发；教师额外数据和计算单列。'
    # The image is retained to satisfy the complete screenshot inventory; its
    # Smoke output is not promoted as a v11 quantitative measurement.
    registry['shot_bake']['title'] = '运行截图 01　W4A4 量化写盘 smoke'
    registry['shot_bake']['note'] = ('保留原始终端画面用于复现流程审阅；该短测只检查 H 统计、格式编码和写盘入口，'
                                     '不承载 v11 的张量计数、压缩比或闭环结论；正式发布时绑定冻结 W4A4 配方与运行清单。')
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
    figures.swizzle_layout()
    figures.gptq_block()
    figures.recovery_protocol()
    pending_charts()
    for name in names:
        if not name.startswith('shot_'):
            cairosvg.svg2png(url=str(P/'figs'/f'{name}.svg'),
                            write_to=str(P/'figs'/f'{name}.png'), scale=2)
    subprocess.run([sys.executable, str(P/'build_html.py')], check=True)
    subprocess.run([sys.executable, str(P/'export_zhihu.py')], check=True)
    if screenshot_hashes != {name: hashlib.sha256((P/'figs'/f'{name}.png').read_bytes()).hexdigest()
                             for name in screenshot_hashes}:
        raise RuntimeError('Screenshot bytes changed during the review build')

    files = [P/'paper.html',P/'zhihu/article.md',P/'meta.json',P/'figures.json',
             P/'analysis_plan_w4a4.json', P/'paired_uncertainty.py']
    files += sorted((P/'sections').glob('*.md'))
    files += [P/'zhihu/images'/f'{name}.png' for name in names]
    # Include each file once in the ZIP.  The exact-once figure check above
    # keeps the document and archive inventories aligned.
    files = list(dict.fromkeys(files))
    manifest = {'status':'incomplete_review_draft', 'not_final_publication':True,
                'experiment_protocol':'v11_w4a4_category', 'selected_recipe':'pending_v11_selection',
                'screenshots':17, 'screenshot_sha256':screenshot_hashes,
                'screenshot_refresh_pending':['shot_bake', 'shot_qad', 'shot_opdcache',
                                              'shot_opd', 'shot_evalserver', 'shot_rollout'],
                'figures':len(names), 'input_evidence':list(figures._DATA_INPUTS.values()), 'files':[
                    {'path':str(p.relative_to(P)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                     'bytes':p.stat().st_size} for p in files]}
    note = ('# 中文论文结构审阅包\n\n打开 paper.html 离线阅读；知乎稿位于 zhihu/article.md。\n\n'
            '红色 xxx 为待完成实验，不是测量值。本文稿用于审阅结构、方法与表达，尚未通过最终结果发布验收。\n'
            '保留 17 张真实运行截图、完整方法图和执行基准表；截图文件未被此构建器修改。\n'
            '受本轮 W4A4 实验影响的执行截图将在最终评测结束后重新实拍；当前截图不作为 v11 正式结果。\n')
    output = P/'apxinf-gr00t-fp4fp8-ptqad-review.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as bundle:
        for p in files: bundle.write(p,str(p.relative_to(P)))
        bundle.writestr('README.md',note)
        bundle.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (P/'validation/review-build.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Review draft:',output)


if __name__ == '__main__':
    main()
