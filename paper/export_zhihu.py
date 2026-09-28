#!/usr/bin/env python3
"""Export the Zhihu package: paper/zhihu/article.md + images/.

Zhihu accepts markdown with image references; animations must be GIF.
- sources sections from paper/sections (same markdown, fig refs -> images/xxx.png)
- converts {{fig:NAME}} to ![](images/NAME.png) (PNG/GIF rendered from matplotlib/SVG by figs scripts)
- copies any figs/*.png|gif into zhihu/images/
Run: python paper/export_zhihu.py
"""
from __future__ import annotations
import pathlib, re, shutil

PAPER = pathlib.Path(__file__).resolve().parent
Z = PAPER / "zhihu"


def unwrap_paragraphs(t: str) -> str:
    """Join hard-wrapped plain-text lines into single markdown paragraphs so
    Zhihu (single newline = line break) renders dense flowing text."""
    out, lines, i = [], t.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(("```", "|", "#", ">", "!")) or re.match(r"^[-*]\s+", ln) or not ln.strip():
            out.append(ln); i += 1; continue
        j = i; buf = []
        while j < len(lines) and lines[j].strip() \
                and not lines[j].startswith(("```", "|", "#", ">", "!")) \
                and not re.match(r"^[-*]\s+", lines[j]):
            buf.append(lines[j].strip()); j += 1
        out.append("".join(buf)); i = j
    return "\n".join(out)


def main():
    Z.mkdir(exist_ok=True); (Z / "images").mkdir(exist_ok=True)
    parts = []
    for p in sorted((PAPER / "sections").glob("*.md")):
        t = p.read_text(encoding="utf-8")
        t = re.sub(r"\{\{fig:([\w.-]+)\}\}", r"![](images/\1.png)", t)
        parts.append(unwrap_paragraphs(t))
    (Z / "article.md").write_text("\n\n".join(parts), encoding="utf-8")
    n = 0
    for f in (PAPER / "figs").glob("*"):
        if f.suffix.lower() in (".png", ".gif", ".jpg"):
            shutil.copy(f, Z / "images" / f.name); n += 1
    print(f"zhihu package: {Z/'article.md'} (+{n} images)")

if __name__ == "__main__":
    main()
