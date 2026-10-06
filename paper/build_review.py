#!/usr/bin/env python3
"""Render an explicitly incomplete reading draft; never certify publication results.

Run with requirements-build.txt. Current screenshots are copied verbatim;
pending refreshes retain their slots without publishing previous image bytes.
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
from v12_publication_contract import PROTOCOL_NAME, PROTOCOL_SHA256

P = Path(__file__).resolve().parent
RED = '#b42318'
PROTOCOL_ID = 'w4a4-recovery-v12-rtn'
EXPECTED_PENDING = {'shot_bake', 'shot_collect', 'shot_qad', 'shot_rollout',
                    'shot_opdcache', 'shot_opd', 'shot_evalserver'}
INDEPENDENT_ENGINE_INPUTS = {
    'results/engine/pi05_nvfp4_ptqad_20260929.json',
    'results/engine/pi05_bf16_ptqad_20260929.json',
    'results/engine/gr00t_bf16_ptqad_20260929.json',
    'results/baselines/pi05_pt_bf16_ptqad_20260929.json',
    'results/baselines/gr00t_pt_bf16_ptqad_20260929.json',
}


def review_protocol(meta):
    status = meta.get('status', '')
    if not (re.search(r'\bv12\b', status, re.I) and
            re.search(r'\bRTN\b', status, re.I) and '审阅稿' in status):
        raise RuntimeError('Review meta.status must explicitly identify v12 RTN 审阅稿')
    path = P.parent/'exp'/PROTOCOL_NAME
    payload = path.read_bytes()
    protocol = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if (digest != PROTOCOL_SHA256 or protocol.get('id') != PROTOCOL_ID or
            protocol.get('version') != 12 or protocol.get('w4a4') is not True or
            protocol.get('quantization_scope', {}).get('recipe') != 'rtn_all'):
        raise RuntimeError('Review must bind the frozen v12 RTN W4A4 protocol')
    return path, {'id': PROTOCOL_ID, 'path': 'protocols/'+PROTOCOL_NAME,
                  'source_path': 'exp/'+PROTOCOL_NAME,
                  'sha256': digest, 'bytes': len(payload)}


def pending_screenshots(registry):
    pending = {name for name, info in registry.items() if info.get('refresh_pending') is True}
    if pending != EXPECTED_PENDING:
        raise RuntimeError('Review requires exactly the seven v12 screenshot refresh slots')
    return pending


def pending_charts():
    # Deliberately read no quantization budgets, recipes or evaluation evidence.
    # Equal-height text rows communicate missing values without predicting gains.
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
    chart = figures.SVG(1100, 580, 'W4A4 编码预算：待冻结产物核算',
                        '结构审阅稿 · 红色 xxx 为待核验数值；不读取其他配方的预算。')
    rows = [('BF16 原始权重', 'xxx GB'), ('NVFP4 payload 与格式缩放', 'xxx GB'),
            ('未量化权重及补零开销', 'xxx GB'), ('完整 BF16 低秩旁路', 'xxx GB'),
            ('含低秩旁路的完整压缩比', 'xxx×')]
    for index, (name, value) in enumerate(rows):
        y = 110 + index * 72
        chart.rect(30, y, 1040, 58, '#f4f7f8')
        chart.text(48, y + 37, name, 19, weight=600)
        chart.text(1000, y + 37, value, 22, RED, 600, 'end')
    chart.text(32, 514, '最终同时报告物理张量与已知共享别名去重口径；恢复臂完整计入 BF16 旁路。', 16)
    chart.text(32, 548, '编码预算不是稠密 checkpoint 文件体积、实际显存或延迟。', 16)
    chart.save('budget_ladder')
    chart = figures.SVG(1100, 580, '同一 W4A4 基座：编码预算与闭环恢复',
                        '结构审阅稿 · 预算与成功率均待冻结产物核验；不预设恢复收益。')
    chart.text(45, 132, '比较项', 18, weight=600)
    chart.text(470, 132, '完整目标编码预算', 18, weight=600)
    chart.text(850, 132, '闭环成功率', 18, weight=600)
    rows = ['BF16', 'W4A4 PTQ', 'W4A4 PTQ + QAD',
            'W4A4 PTQ + 继续 QAD', 'W4A4 PTQ + QAD + OPD']
    for index, name in enumerate(rows):
        y = 198 + index * 62
        chart.text(45, y, name, 18)
        chart.text(520, y, 'xxx GB', 21, RED, 600)
        chart.text(880, y, 'xxx%', 21, RED, 600)
    chart.text(45, 506, '量化基座含 payload、格式缩放及未量化权重；恢复臂另计完整 BF16 低秩参数。', 15)
    chart.text(45, 543, '该预算不是稠密 checkpoint 文件体积、实际显存或延迟；不预设恢复收益。', 15)
    chart.save('ptq_frontier')


def main():
    source = '\n\n'.join(p.read_text(encoding='utf-8') for p in sorted((P/'sections').glob('*.md')))
    protocol_path, protocol_record = review_protocol(json.loads((P/'meta.json').read_text(encoding='utf-8')))
    if '#b42318' not in source or 'xxx' not in source:
        raise RuntimeError('This builder is only for explicitly marked incomplete review drafts')
    names = re.findall(r'\{\{fig:([\w.-]+)\}\}', source)
    registry = json.loads((P/'figures.json').read_text(encoding='utf-8'))
    # Every registered figure has one canonical position in the reading draft.
    # Duplicate markers silently duplicated images in the HTML and ZIP and
    # shifted figure numbering, so reject them at build time.
    if len(names) != len(registry) or len(set(names)) != len(names) or set(names) != set(registry):
        raise RuntimeError('Review must include every registered figure exactly once')
    if sum(n.startswith('shot_') for n in set(names)) != 17:
        raise RuntimeError('Review must preserve all 17 screenshot slots')
    pending = pending_screenshots(registry)
    screenshot_hashes = {name: hashlib.sha256((P/'figs'/f'{name}.png').read_bytes()).hexdigest()
                         for name in names if name.startswith('shot_')}

    registry['ladder']['title'] = '图 5　同一 W4A4 基座上的五臂闭环对照'
    registry['ladder']['note'] = 'BF16、RTN W4A4 PTQ、QAD、继续 QAD 与 QAD+OPD 使用同一 v12 最终评测协议；红色占位符为待测值。'
    registry['budget_ladder']['title'] = '图 4　W4A4 基座的编码预算与低秩旁路成本'
    registry['budget_ladder']['note'] = 'v12 RTN W4A4 的实际编码预算待冻结产物核算；最终计入未量化权重、格式缩放及完整 BF16 低秩旁路。红色 xxx 不代表测量值。'
    registry['ptq_frontier']['title'] = '图 7　全 NVFP4 W4A4 基座的预算与闭环成功率'
    registry['ptq_frontier']['note'] = 'v12 RTN W4A4 五臂的预算与闭环成功率均待测量和核验；恢复臂须计入完整 BF16 低秩残差。图中不预设收益大小或排名。'
    registry['recovery_protocol']['note'] = '开发、学生采集与最终评测使用互不重叠的官方初态。两个续训分支从同一 QAD 检查点出发；教师额外数据和计算单列。'
    # Keep screenshot captions in the shared registry. In particular, do not
    # overwrite its pending-refresh labels with claims about the new run.
    number = 0
    for name in names:
        if not name.startswith('shot_'):
            number += 1
            registry[name]['title'] = re.sub(r'^图 \d+\s*', f'图 {number}　', registry[name]['title'])
    (P/'figures.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')

    # Only independent engine timings read data. All recovery charts are pending;
    # the other three illustrations are mathematical/method diagrams.
    figures._DATA_INPUTS.clear()
    figures._GENERATED_SVGS.clear()
    figures.e1_chart()
    figures.swizzle_layout()
    figures.gptq_block()
    figures.recovery_protocol()
    pending_charts()
    if set(figures._DATA_INPUTS) != INDEPENDENT_ENGINE_INPUTS:
        raise RuntimeError('Review charts may consume only the five independent engine timing inputs')
    for name in names:
        if not name.startswith('shot_'):
            cairosvg.svg2png(url=str(P/'figs'/f'{name}.svg'),
                            write_to=str(P/'figs'/f'{name}.png'), scale=2)
    subprocess.run([sys.executable, str(P/'build_html.py')], check=True)
    subprocess.run([sys.executable, str(P/'export_zhihu.py')], check=True)
    if screenshot_hashes != {name: hashlib.sha256((P/'figs'/f'{name}.png').read_bytes()).hexdigest()
                             for name in screenshot_hashes}:
        raise RuntimeError('Screenshot bytes changed during the review build')

    files = [P/'paper.html',P/'zhihu/article.md',P/'meta.json',P/'figures.json']
    files += sorted((P/'sections').glob('*.md'))
    files += [P/'zhihu/images'/f'{name}.png' for name in names if name not in pending]
    # Include each file once in the ZIP.  The exact-once figure check above
    # keeps the document and archive inventories aligned.
    files = [(p, p.relative_to(P).as_posix()) for p in dict.fromkeys(files)]
    files += [(protocol_path, protocol_record['path'])]
    files += [(P.parent/path, path) for path in sorted(INDEPENDENT_ENGINE_INPUTS)]
    manifest = {'status':'incomplete_review_draft', 'not_final_publication':True,
                'experiment_protocol':protocol_record, 'recipe_candidate':'rtn_w4a4_category',
                'closed_loop_results':'pending', 'quantization_budgets':'pending',
                'screenshot_slots':17, 'screenshots':17-len(pending),
                'screenshot_sha256':{name:value for name,value in screenshot_hashes.items() if name not in pending},
                'screenshot_refresh_pending':sorted(pending),
                'internal_retained_screenshot_sha256':{name:screenshot_hashes[name] for name in sorted(pending)},
                'figures':len(names), 'input_evidence':list(figures._DATA_INPUTS.values()), 'files':[
                    {'path':archive_path, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                     'bytes':p.stat().st_size} for p,archive_path in files]}
    note = ('# 中文论文结构审阅包\n\n打开 paper.html 离线阅读；知乎稿位于 zhihu/article.md。\n\n'
            '红色 xxx 为待完成实验，不是测量值。本文稿用于审阅结构、方法与表达，尚未通过最终结果发布验收。\n'
            'v12 RTN W4A4 冻结协议位于 protocols/；独立引擎计时记录位于 results/，不代表 GR00T 恢复模型的部署性能。\n'
            f'保留全部 17 个执行截图环节：{17-len(pending)} 张真实截图与 {len(pending)} 个待补图位，另含 7 张方法及数据图。\n'
            '待替换原图仅保留在工程内部，不进入本文或 ZIP；其哈希记在清单中供替换核对。\n'
            '待补环节完成本轮实拍和来源核验后才能通过正式发布验收。\n')
    output = P/'apxinf-gr00t-fp4fp8-ptqad-review.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as bundle:
        for p,archive_path in files: bundle.write(p,archive_path)
        bundle.writestr('README.md',note)
        bundle.writestr('review-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (P/'validation/review-build.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    print('Review draft:',output)


if __name__ == '__main__':
    main()
