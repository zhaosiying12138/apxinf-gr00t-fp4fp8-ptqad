#!/usr/bin/env python3
"""Build the single-file Chinese paper (paper/paper.html) from sections + figures.

- Sections: paper/sections/*.md (ordered by filename)
- Figures:  paper/figs/*.svg inlined into the HTML
- Data:     results/**/*.csv rendered into tables by scripts in paper/tables.py (later)
- Animations: inline <svg> + vanilla JS (swizzle demo) — no external deps, fully offline
Run:  uv run --with numpy,matplotlib python paper/build_html.py
"""
from __future__ import annotations
import html, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

CSS = """
:root{--ink:#1a1a2e;--accent:#0f6db3;--soft:#5b6b7d;--bg:#ffffff;--band:#f4f7fa}
*{box-sizing:border-box}
body{font-family:"Noto Sans SC","Source Han Sans SC","Microsoft YaHei",system-ui,sans-serif;
     color:var(--ink);background:var(--bg);margin:0;line-height:1.75;font-size:16px}
.page{max-width:860px;margin:0 auto;padding:48px 28px 96px}
header{border-bottom:3px solid var(--accent);padding-bottom:20px;margin-bottom:32px}
h1{font-size:28px;line-height:1.35;margin:0 0 8px}
.meta{color:var(--soft);font-size:14px}
h2{font-size:21px;margin-top:44px;border-left:4px solid var(--accent);padding-left:10px}
h3{font-size:17px;margin-top:28px}
table{border-collapse:collapse;margin:18px 0;font-size:14px;width:100%}
th,td{border:1px solid #d5dde5;padding:6px 10px;text-align:center}
th{background:var(--band)}
tr:nth-child(even) td{background:#fafcfe}
figure{margin:24px 0;text-align:center}
figure svg{max-width:100%;height:auto}
figcaption{font-size:13px;color:var(--soft);margin-top:6px}
code,pre{font-family:"JetBrains Mono",Consolas,monospace;font-size:13.5px}
pre{background:#0f172a;color:#dbe7f3;padding:14px 16px;border-radius:8px;overflow-x:auto}
code{background:#eef2f6;padding:1px 5px;border-radius:4px}
pre code{background:none;padding:0}
blockquote{border-left:3px solid var(--soft);margin:16px 0;padding:2px 16px;color:#3c4a58;background:var(--band)}
.kv{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
.kv span{background:var(--band);border-radius:6px;padding:2px 10px;font-size:13px}
"""

SWIZZLE_DEMO_JS = r"""
// animated NVFP4 scale swizzle: logical (row, block) grid -> physical cuBLASLt buffer
// formula decoded in spike: PS*(r/128) + 512*(b/4) + 16*(r%32) + 4*((r/32)%4) + (b%4)
(function(){
const R=8, B=8;                          // demo grid: 8 rows x 8 blocks (formula generalizes)
const PS = 512*Math.ceil(B/4);
const off=(r,b)=> PS*Math.floor(r/128) + 512*Math.floor(b/4) + 16*(r%32) + 4*(Math.floor(r/32)%4) + (b%4);
const svg=document.getElementById('swz'); if(!svg) return;
const cell=18, pad=26;
svg.setAttribute('viewBox',`0 0 ${2*B*cell+3*pad+40} ${R*cell+pad+46}`);
function mk(x,y,w,h,fill,txt,fs=10,stroke='#8fa3b8'){
  const g=document.createElementNS('http://www.w3.org/2000/svg','g');
  const r=document.createElementNS('http://www.w3.org/2000/svg','rect');
  r.setAttribute('x',x);r.setAttribute('y',y);r.setAttribute('width',w);r.setAttribute('height',h);
  r.setAttribute('rx',3);r.setAttribute('fill',fill);r.setAttribute('stroke',stroke);
  const t=document.createElementNS('http://www.w3.org/2000/svg','text');
  t.setAttribute('x',x+w/2);t.setAttribute('y',y+h/2+3.5);t.setAttribute('text-anchor','middle');
  t.setAttribute('font-size',fs);t.setAttribute('font-family','monospace');t.textContent=txt;
  g.appendChild(r);g.appendChild(t);return g;
}
const hue=(r,b)=>`hsl(${(r*47+b*23)%360} 65% 82%)`;
let t=0, phases=[0,1,2];                 // 0: logical  1: flying  2: physical
const logical=[]; for(let r=0;r<R;r++)for(let b=0;b<B;b++)logical.push({r,b});
// physical slots occupied (dedup offsets onto a compact axis)
const offs=[...new Set(logical.map(c=>off(c.r,c.b)))].sort((a,b)=>a-b);
const slotX=o=>pad+ (offs.indexOf(o)/offs.length)*B*cell;
const LX=b=>pad+b*cell, PY=r=>pad+30+r*cell;
let anim=null;
function draw(p){                        // p in [0,1]: 0 logical -> 1 physical
  svg.innerHTML='';
  svg.appendChild(mk(pad-8,pad+16,B*cell+16,R*cell+8,'none','逻辑布局 (row, block)',11,'none')).lastChild.setAttribute('x',pad+B*cell/2);
  svg.appendChild(mk(pad+B*cell+pad,pad+16,B*cell+16,R*cell+8,'none','物理缓冲 (swizzle)',11,'none')).lastChild.setAttribute('x',pad+B*cell+pad+B*cell/2);
  for(const c of logical){
    const x0=LX(c.b), y=PY(c.r), x1=pad+B*cell+2*pad+slotX(off(c.r,c.b)), y1=PY(c.r);
    const x=x0+(x1-x0)*p, y=y0(y0=>y0);  // keep row line
    svg.appendChild(mk(x,PY(c.r)+ (0)*p,cell-2,cell-2,hue(c.r,c.b),'',10));
  }
}
function step(){
  t+=0.008; const p=(1-Math.cos(Math.min(t,Math.PI)))/2;
  draw(p); if(t<Math.PI) anim=requestAnimationFrame(step); else {t=0; setTimeout(()=>{anim=requestAnimationFrame(step)},1200);}
}
draw(0); setTimeout(()=>anim=requestAnimationFrame(step),800);
})();
"""

def md_to_html(md: str) -> str:
    """Tiny markdown subset: headers, paragraphs, bold, code spans, code blocks, tables, lists."""
    out, lines, i = [], md.splitlines(), 0
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"): j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i+1:j])) + "</code></pre>")
            i = j + 1; continue
        if ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"): j += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in lines[i:j] if not re.match(r"^\|[\s:|-]+\|$", r)]
            head, body = rows[0], rows[1:]
            t = "<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
            for r in body: t += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</table>"); i = j; continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            out.append(f"<h{len(m.group(1))+1}>{inline(m.group(2))}</h{len(m.group(1))+1}>"); i += 1; continue
        if re.match(r"^[-*]\s+", ln):
            j = i; items = []
            while j < len(lines) and re.match(r"^[-*]\s+", lines[j]):
                items.append(f"<li>{inline(re.sub(r'^[-*]\s+','',lines[j]))}</li>"); j += 1
            out.append("<ul>" + "".join(items) + "</ul>"); i = j; continue
        if not ln.strip(): i += 1; continue
        out.append(f"<p>{inline(ln)}</p>"); i += 1
    return "\n".join(out)

def main():
    sections = sorted((PAPER / "sections").glob("*.md"))
    body = "\n".join(md_to_html(p.read_text(encoding="utf-8")) for p in sections)
    # inline figures: replace {{fig:NAME}} with the svg file content
    def fig(m):
        f = PAPER / "figs" / (m.group(1) + ".svg")
        return f.read_text(encoding="utf-8") if f.exists() else f"<figure><figcaption>[缺图 {m.group(1)}]</figcaption></figure>"
    body = re.sub(r"\{\{fig:([\w.-]+)\}\}", fig, body)
    title = (PAPER / "meta.json").exists() and json.loads((PAPER / "meta.json").read_text()) or {}
    html_doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title.get('title','FP4-VLA'))}</title><style>{CSS}</style></head>
<body><div class="page">
<header><h1>{html.escape(title.get('title','FP4-VLA'))}</h1>
<div class="meta">{html.escape(title.get('meta',''))}</div></header>
{body}
<hr><p class="meta">本页由 paper/build_html.py 生成 · 数据与代码见仓库 · {title.get('date','')}</p>
</div><script>{SWIZZLE_DEMO_JS}</script></body></html>"""
    out = PAPER / "paper.html"
    out.write_text(html_doc, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    sys.exit(main())
