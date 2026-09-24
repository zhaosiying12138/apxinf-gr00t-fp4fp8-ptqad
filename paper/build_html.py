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
const svg=document.getElementById('swz'); if(!svg) return;
const R=8, B=8;                                // demo grid 8x8 (formula generalizes)
const PS=512*Math.ceil(B/4);
const off=(r,b)=> PS*Math.floor(r/128) + 512*Math.floor(b/4) + 16*(r%32) + 4*(Math.floor(r/32)%4) + (b%4);
const cell=19, gap=3, pad=30, midGap=70;
const cells=[]; for(let r=0;r<R;r++)for(let b=0;b<B;b++)cells.push({r,b});
const offs=[...new Set(cells.map(c=>off(c.r,c.b)))].sort((a,b)=>a-b);   // compact physical axis
const W = pad + B*(cell+gap) + midGap + offs.length*(cell+gap) + pad;
const H = pad + R*(cell+gap) + 34;
svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
svg.setAttribute('width', W);
const NS='http://www.w3.org/2000/svg';
const hue=(r,b)=>`hsl(${(r*47+b*23)%360} 62% 80%)`;
function el(tag,attrs,txt){const e=document.createElementNS(NS,tag);
  for(const k in attrs)e.setAttribute(k,attrs[k]); if(txt!=null)e.textContent=txt; return e;}
function label(x,y,txt,anchor){svg.appendChild(el('text',{x,y,'font-size':11,fill:'#5b6b7d',
  'text-anchor':anchor||'start','font-family':'system-ui'},txt));}
// frame titles
label(pad, 20, '逻辑布局 scale(row, block)');
label(pad + B*(cell+gap) + midGap, 20, '物理缓冲（swizzle 后字节偏移紧凑排列）');
// logical position of cell (r,b): column b, row r
const LX=b=>pad + b*(cell+gap), LY=r=>pad+12 + r*(cell+gap);
// physical position: sorted offset index -> column; row = index of this cell among same-offset? offsets unique per (r,b) in this demo
const colOf={}; offs.forEach((o,i)=>colOf[o]=i);
const PX=c=>pad + B*(cell+gap) + midGap + colOf[off(c.r,c.b)]*(cell+gap);
const PY=c=>LY(c.r);
function draw(p){
  [...svg.querySelectorAll('.anim')].forEach(e=>e.remove());
  for(const c of cells){
    const x0=LX(c.b), y0=LY(c.r), x1=PX(c), y1=PY(c);
    const x=x0+(x1-x0)*p, y=y0+(y1-y0)*p;
    svg.appendChild(el('rect',{x:x,y:y,width:cell,height:cell,rx:3,
      fill:hue(c.r,c.b),stroke:'#8fa3b8','class':'anim'}));
  }
}
let t=0, raf=null;
function step(){
  t+=0.012;
  const cycle = t % (Math.PI+1.2);
  const p = cycle<Math.PI ? (1-Math.cos(cycle))/2 : 1;
  draw(p);
  raf=requestAnimationFrame(step);
}
draw(0); setTimeout(()=>{raf=requestAnimationFrame(step)}, 700);
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
