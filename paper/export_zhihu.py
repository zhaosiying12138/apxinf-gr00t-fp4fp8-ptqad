#!/usr/bin/env python3
"""Export Markdown without rewriting code fences, math, lists or hard wraps."""
import json
import pathlib
import re
import shutil

P = pathlib.Path(__file__).resolve().parent

def main():
    out = P/'zhihu'
    (out/'images').mkdir(parents=True, exist_ok=True)
    info = json.loads((P/'meta.json').read_text(encoding='utf-8'))
    figs = json.loads((P/'figures.json').read_text(encoding='utf-8'))
    pending = {name for name, figure in figs.items() if figure.get('refresh_pending') is True}
    refreshable = {'shot_bake', 'shot_collect', 'shot_qad', 'shot_opdcache', 'shot_opd', 'shot_evalserver', 'shot_rollout'}
    if not pending <= refreshable:
        raise ValueError('Pending screenshot refresh set includes an unaudited slot')
    if pending and '审阅稿' not in info.get('status', ''):
        raise ValueError('Pending screenshots forbid a final publication; explicitly mark meta.status as a review draft')
    # Remove only stale generated copies. Original figs/ and evidence/ assets
    # remain byte-identical and available for the replacement audit.
    for name in pending:
        (out/'images'/(name+'.png')).unlink(missing_ok=True)
    # The title already states the scope; repeating the meta description as a
    # second subtitle made the Zhihu draft read like a duplicated heading.
    md = '# '+info['title']+'\n\n'
    for src in sorted((P/'sections').glob('*.md')):
        md += src.read_text(encoding='utf-8').rstrip()+'\n\n'
    def replace(m):
        name = m[1]
        if name in pending:
            return f'**{figs[name]["title"]}**\n\n待补本轮真实运行截图。{figs[name]["note"]}'
        src = P/'figs'/(name+'.png')
        if not src.exists(): raise FileNotFoundError(src)
        shutil.copyfile(src, out/'images'/src.name)
        return f'![{figs[name]["title"]}](images/{name}.png)\n\n*{figs[name]["title"]}。{figs[name]["note"]}*'
    md = re.sub(r'\{\{fig:([\w.-]+)\}\}', replace, md)
    (out/'article.md').write_text(md.rstrip()+'\n', encoding='utf-8')
    print('Exported Zhihu Markdown, preserving source code and math verbatim.')

if __name__ == '__main__':
    main()
