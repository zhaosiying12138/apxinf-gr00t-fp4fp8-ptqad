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
    md = '# '+info['title']+'\n\n'+info['meta']+'\n\n'
    for src in sorted((P/'sections').glob('*.md')):
        md += src.read_text(encoding='utf-8').rstrip()+'\n\n'
    def replace(m):
        name = m[1]
        src = P/'figs'/(name+'.png')
        if not src.exists(): raise FileNotFoundError(src)
        shutil.copyfile(src, out/'images'/src.name)
        return f'![{figs[name]["title"]}](images/{name}.png)\n\n*{figs[name]["title"]}。{figs[name]["note"]}*'
    md = re.sub(r'\{\{fig:([\w.-]+)\}\}', replace, md)
    (out/'article.md').write_text(md.rstrip()+'\n', encoding='utf-8')
    print('Exported Zhihu Markdown, preserving source code and math verbatim.')

if __name__ == '__main__':
    main()
