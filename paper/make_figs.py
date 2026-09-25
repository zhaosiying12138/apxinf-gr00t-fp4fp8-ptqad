#!/usr/bin/env python3
"""Generate paper figures (SVG) from results CSVs/JSONs. No matplotlib needed —
pure-stdlib SVG writer keeps the paper build dependency-free.

Outputs to paper/figs/: E1 latency bars, operator speedup heatmap, recovery
schematic placeholder (data fills in when GPU windows land).
Run: python3 paper/make_figs.py
"""
from __future__ import annotations
import csv, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def svg_header(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">'
            f'<rect width="{w}" height="{h}" fill="white"/>'
            f'<text x="20" y="28" font-size="15" font-weight="600" fill="#1a1a2e">{title}</text>')

def hbar_chart(rows, title, unit, out, w=760, cw=340):
    """rows: [(label, value, color)] horizontal bars, two-column layout if long."""
    n = len(rows)
    row_h = 26; top = 48; left = 210; bar_max = w - left - 150
    h = top + n * row_h + 16
    s = [svg_header(w, h, title)]
    vmax = max(v for _, v, _ in rows) or 1
    for i, (label, v, color) in enumerate(rows):
        y = top + i * row_h
        bw = max(2, v / vmax * bar_max)
        s.append(f'<text x="{left-8}" y="{y+14}" font-size="12" text-anchor="end" fill="#3c4a58">{label}</text>')
        s.append(f'<rect x="{left}" y="{y+2}" width="{bw:.0f}" height="{row_h-8}" rx="3" fill="{color}"/>')
        s.append(f'<text x="{left+bw+6:.0f}" y="{y+14}" font-size="12" fill="#1a1a2e">{v:.1f}{unit}</text>')
    s.append("</svg>")
    (FIGS / out).write_text("".join(s), encoding="utf-8")
    print("wrote", out)

def e1_chart():
    """End-to-end latency comparison (E1 main table)."""
    rows = [
        ("π0.5 · PyTorch", 361.7, "#c9d6e3"),
        ("π0.5 · ApxInf BF16", 49.6, "#2f7fd1"),
        ("GR00T N1.7 · PyTorch", 106.9, "#c9d6e3"),
        ("GR00T N1.7 · ApxInf BF16", 33.7, "#2f7fd1"),
        ("π0.5 · NVFP4 v1", 1048.8, "#e8b7b7"),  # unfused routing, no graph (fusion headroom: 1.3-5.3x)
    ]
    hbar_chart(rows, "端到端延迟（batch-1, P50, ms）", " ms", "e1_latency.svg")

def opbench_heatmap():
    """Operator speedups on real pi05 shapes (fp4 vs bf16 pipeline)."""
    csv_path = ROOT / "results/engine/fp4_opbench_realshapes.csv"
    if not csv_path.exists():
        print("skip heatmap (csv missing)"); return
    data = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("ok") == "1":
                tag = r["tag"].split(",")[0]
                data.setdefault(tag, {})[int(r["M"])] = float(r["gbps"]) and float(r["speedup"] if "speedup" in r else 0)
    # simpler: reparse speedup = bf16_us/fp4_us
    data.clear()
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("ok") == "1":
                tag = r["tag"].split(",")[0]
                try:
                    su = float(r["bf16_us_p50"]) / float(r["fp4_us_p50"])
                except (KeyError, ZeroDivisionError):
                    continue
                data.setdefault(tag, {})[int(r["M"])] = su
    tags = list(data)
    ms = sorted({m for d in data.values() for m in d})
    cw, ch, top, left = 86, 30, 52, 130
    w = left + len(ms) * cw + 20
    h = top + len(tags) * ch + 20
    s = [svg_header(w, h, "NVFP4 全管线加速 ×（真实层形状, vs BF16 GEMM）")]
    for j, m in enumerate(ms):
        s.append(f'<text x="{left+j*cw+cw/2}" y="{top-10}" font-size="11" text-anchor="middle">M={m}</text>')
    for i, t in enumerate(tags):
        s.append(f'<text x="{left-6}" y="{top+i*ch+19}" font-size="11" text-anchor="end">{t}</text>')
        for j, m in enumerate(ms):
            su = data[t].get(m)
            if su is None:
                fill, txt = "#eef2f6", "–"
            else:
                alpha = min(1.0, (su - 1) / 4.4 + 0.15)
                fill, txt = f"rgba(47,127,209,{alpha:.2f})", f"{su:.1f}×"
            s.append(f'<rect x="{left+j*cw+2}" y="{top+i*ch+2}" width="{cw-6}" height="{ch-6}" rx="3" fill="{fill}"/>')
            s.append(f'<text x="{left+j*cw+cw/2-1}" y="{top+i*ch+19}" font-size="11" text-anchor="middle" fill="{"#fff" if su and su>2.6 else "#1a1a2e"}">{txt}</text>')
    s.append("</svg>")
    (FIGS / "opbench_heatmap.svg").write_text("".join(s), encoding="utf-8")
    print("wrote opbench_heatmap.svg")

if __name__ == "__main__":
    e1_chart()
    opbench_heatmap()
    print("figures done ->", FIGS)
