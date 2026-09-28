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



def ladder_chart():
    """09-28 final closed-loop recovery ladder (LIBERO-10)."""
    rows = [
        ("BF16 上界", 96.7, "#8aa6c1"),
        ("PTQ rtn 全NVFP4 (3.56x)", 0.0, "#d9534f"),
        ("PTQ calib GPTQ (3.56x)", 0.0, "#d9534f"),
        ("PTQ aggr (2.88x)", 0.0, "#d9534f"),
        ("PTQ fp8 基座 (2.44x)", 43.4, "#e8a33d"),
        ("PTQ mixed 部署级 (2.02x)", 95.1, "#5cb85c"),
        ("QAD-LoRA (2.44x+低秩)", 99.0, "#2f7fd1"),
        ("OPD-LoRA (2.44x+低秩)", 99.0, "#2f7fd1"),
    ]
    hbar_chart(rows, "闭环恢复阶樯：lIBERO-10 成功率（%，10×10 全量）", "%", "ladder.svg")



def swizzle_layout():
    """Static NVFP4 scale-layout principle diagram: logical (row, block) grid
    -> physical swizzled buffer with the offset formula and worked examples.
    Demo grid 8 rows x 8 blocks (real layout: 128-row panels, 32-row clusters)."""
    import colorsys
    W, H = 940, 500
    s = [svg_header(W, H, "NVFP4 \u5757\u7f29\u653e\u7684\u7269\u7406\u5e03\u5c40\u539f\u7406\uff08\u9759\u6001\u793a\u610f\uff0c\u6f14\u793a\u7f51\u683c 8\u884c\u00d78 \u5757\uff09")]
    def hue(r, b):
        h = ((r * 47 + b * 23) % 360) / 360.0
        rr, gg, bb = colorsys.hls_to_rgb(h, 0.78, 0.55)
        return f"#{int(rr*255):02x}{int(gg*255):02x}{int(bb*255):02x}"
    cell, gap = 22, 3
    lx, ly = 150, 96
    s.append(f'<text x="{lx + 4*cell}" y="72" font-size="13" font-weight="600" text-anchor="middle" fill="#1a1a2e">\u903b\u8f91\u89c6\u56fe scale[row, block]</text>')
    for r in range(8):
        for b in range(8):
            x, y = lx + b * (cell + gap), ly + r * (cell + gap)
            s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" fill="{hue(r, b)}" stroke="#8fa3b8"/>')
    s.append(f'<text x="{lx - 14}" y="{ly + 4*cell}" font-size="12" text-anchor="middle" fill="#3c4a58" transform="rotate(-90 {lx-14} {ly+4*cell})">\u884c r</text>')
    s.append(f'<text x="{lx + 4*cell}" y="{ly - 12}" font-size="12" text-anchor="middle" fill="#3c4a58">\u5757 b\uff08\u6bcf\u5757\u7ba1 16 \u4e2a\u6743\u91cd\u5143\u7d20\uff09</text>')
    s.append(f'<text x="{lx + 9*cell + 10}" y="{ly + 4*cell}" font-size="11" fill="#5b6b7d">\u6bcf\u683c = 1 \u5b57\u8282 E4M3</text>')
    s.append(f'<text x="{lx + 9*cell + 10}" y="{ly + 4*cell + 18}" font-size="11" fill="#5b6b7d">\uff08\u4e00\u4e2a 16 \u5143\u7d20\u5757\u7684\u5e45\u5ea6\uff09</text>')
    s.append('<text x="470" y="120" font-size="12.5" fill="#0f6db3" font-family="Consolas,monospace">off(r,b) = PS*(r/128)</text>')
    s.append('<text x="470" y="140" font-size="12.5" fill="#0f6db3" font-family="Consolas,monospace">  + 512*(b/4)</text>')
    s.append('<text x="470" y="160" font-size="12.5" fill="#0f6db3" font-family="Consolas,monospace">  + 16*(r%32)</text>')
    s.append('<text x="470" y="180" font-size="12.5" fill="#0f6db3" font-family="Consolas,monospace">  + 4*((r/32)%4) + (b%4)</text>')
    s.append('<text x="470" y="204" font-size="11.5" fill="#3c4a58">PS = 512*ceil(KB/4)\uff08\u4e00\u4e2a 128 \u884c panel \u7684\u5b57\u8282\u6570\uff09</text>')
    s.append('<text x="470" y="232" font-size="11.5" font-weight="600" fill="#1a1a2e">\u4e3a\u4ec0\u4e48\u8981\u91cd\u6392\uff1f</text>')
    for i, line in enumerate([
        "\u2022 \u4e00\u884c\u7684 4 \u4e2a\u5757 scale \u5728\u7269\u7406\u4e0a\u8fde\u7eed\uff084 \u5b57\u8282\uff09",
        "\u2022 32 \u884c\u7c07 \u00d7 4 \u5757\u7ec4 = 512B \u8fde\u7eed\u533a\u95f4",
        "\u2022 kernel \u6309\u6b64\u987a\u5e8f\u6d88\u8d39\uff1awarp \u5408\u5e76\u8bbf\u95ee\uff0c",
        "  \u6bcf lane \u4e00\u6b21 16B \u5411\u91cf\u8bfb\u8fd8\u4e00\u884c 4 \u5757",
    ]):
        s.append(f'<text x="470" y="{252 + i*18}" font-size="11.5" fill="#3c4a58">{line}</text>')
    px, py = 742, 96
    s.append(f'<text x="{px + 2*cell}" y="72" font-size="13" font-weight="600" text-anchor="middle" fill="#1a1a2e">\u7269\u7406\u7f13\u51b2\uff08\u4e24\u4e2a 512B slab\uff09</text>')
    for slab in range(2):
        ox = px + slab * (4 * cell + 52)
        s.append(f'<text x="{ox + 2*cell}" y="{py - 12}" font-size="11" text-anchor="middle" fill="#0f6db3">slab {slab}\uff08b/4={slab}\uff09\u57fa\u5740 {512*slab}</text>')
        s.append(f'<rect x="{ox - 4}" y="{py - 4}" width="{4*cell + 3*gap + 8}" height="{8*cell + 7*gap + 8}" fill="none" stroke="#0f6db3" stroke-dasharray="4 3"/>')
        for r in range(8):
            for q in range(4):
                b = slab * 4 + q
                off = 512 * (b // 4) + 16 * (r % 32) + 4 * ((r // 32) % 4) + (b % 4)
                x, y = ox + q * (cell + gap), py + r * (cell + gap)
                s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" fill="{hue(r, b)}" stroke="#8fa3b8"/>')
    s.append(f'<text x="{px}" y="{py + 8*cell + 22}" font-size="11" fill="#5b6b7d">\u540c\u8272 = \u540c\u4e00\u4e2a\u903b\u8f91\u5355\u5143\uff1b\u4e00\u5217 4 \u683c = \u4e00\u884c\u7684\u8fde\u7eed 4 \u5b57\u8282</text>')
    s.append('<text x="150" y="356" font-size="12" font-weight="600" fill="#1a1a2e">\u5b9e\u4f8b\u8ba1\u7b97\uff08\u6f14\u793a\u7f51\u683c\uff0cr&lt;32 \u6545\u4ea4\u9519\u9879\u4e3a 0\uff09\uff1a</text>')
    for i, (r, b) in enumerate([(0, 0), (0, 5), (3, 2), (7, 7)]):
        off = 512 * (b // 4) + 16 * (r % 32) + 4 * ((r // 32) % 4) + (b % 4)
        s.append(f'<text x="150" y="{378 + i*20}" font-size="12" fill="#3c4a58" font-family="Consolas,monospace">(r={r}, b={b}) -> \u5b57\u8282\u504f\u79fb {off}</text>')
    s.append('<text x="470" y="400" font-size="11" fill="#5b6b7d">\u771f\u5b9e\u5c3a\u5bf8\uff1a128 \u884c/panel\uff1b\u884c\u5185\u6309 32 \u884c\u7c07\u56db\u7ec4\u4ea4\u9514\uff1b</text>')
    s.append('<text x="470" y="418" font-size="11" fill="#5b6b7d">\u672c\u56fe\u4e3a 8\u00d78 \u7f29\u5c0f\u6f14\u793a\u3002\u5e03\u5c40\u7531\u5355\u5b57\u8282\u63a2\u9488\u5b9e\u9a8c\u6d4b\u5b9a\uff08\u00a73.1\uff09\u3002</text>')
    s.append("</svg>")
    (FIGS / "swizzle_layout.svg").write_text("".join(s), encoding="utf-8")
    print("wrote swizzle_layout.svg")

if __name__ == "__main__":
    e1_chart()
    opbench_heatmap()
    ladder_chart()
    swizzle_layout()
    print("figures done ->", FIGS)
