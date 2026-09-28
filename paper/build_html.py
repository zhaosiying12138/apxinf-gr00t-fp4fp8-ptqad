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
     color:var(--ink);background:var(--bg);margin:0;line-height:1.7;font-size:16px}
.page{max-width:880px;margin:0 auto;padding:48px 28px 96px}
p{margin:9px 0;text-align:justify}
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

SWIZZLE_DEMO_JS = ""

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
        if ln.startswith(">"):
            j = i; buf = []
            while j < len(lines) and lines[j].startswith(">"):
                buf.append(lines[j].lstrip("> ").strip()); j += 1
            out.append(f"<blockquote><p>{inline(''.join(buf))}</p></blockquote>"); i = j; continue
        if ln.startswith("{{fig:"):
            out.append(inline(ln)); i += 1; continue
        if not ln.strip(): i += 1; continue
        # merge consecutive plain lines into ONE dense paragraph (standard MD)
        j = i; buf = []
        while j < len(lines) and lines[j].strip() \
                and not lines[j].startswith(("```", "|", "#", ">")) \
                and not lines[j].startswith("{{fig:") \
                and not re.match(r"^[-*]\s+", lines[j]):
            buf.append(lines[j].strip()); j += 1
        out.append(f"<p>{inline(''.join(buf))}</p>"); i = j; continue
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
</div></body></html>"""
    out = PAPER / "paper.html"
    out.write_text(html_doc, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    sys.exit(main())
