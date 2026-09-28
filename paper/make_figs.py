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
            f'viewBox="0 0 {w} {h}" font-family="Noto Sans CJK SC,system-ui,sans-serif">'
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
    """Static NVFP4 scale-layout principle diagram with strict geometry budget:
    col1 logical grid 200px | col2 formula text | col3 two physical slabs."""
    import colorsys
    cell, gap = 22, 3
    gw = 8 * (cell + gap)          # 200 left grid width
    sw = 4 * (cell + gap)          # 100 one slab width
    W = 900
    lx, ly = 64, 100               # left grid origin
    mx = 330                       # middle text column x
    sx = 610                       # right slabs x
    gap2 = 40
    H = 500
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Noto Sans CJK SC,system-ui,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="20" y="30" font-size="15" font-weight="600" fill="#1a1a2e">'
         f'NVFP4 \u5757\u7f29\u653e\u7684\u7269\u7406\u5e03\u5c40\u539f\u7406\uff08\u9759\u6001\u793a\u610f\uff0c8 \u884c \u00d7 8 \u5757\uff09</text>']

    def hue(r, b):
        h = ((r * 47 + b * 23) % 360) / 360.0
        rr, gg, bb = colorsys.hls_to_rgb(h, 0.78, 0.55)
        return f"#{int(rr*255):02x}{int(gg*255):02x}{int(bb*255):02x}"

    def off(r, b):
        return 512 * (b // 4) + 16 * (r % 32) + 4 * ((r // 32) % 4) + (b % 4)

    # column headers
    s.append(f'<text x="{lx + gw/2:.0f}" y="66" font-size="13" font-weight="600" text-anchor="middle" fill="#1a1a2e">\u903b\u8f91\u89c6\u56fe scale[row, block]</text>')
    s.append(f'<text x="{mx}" y="66" font-size="13" font-weight="600" fill="#1a1a2e">\u504f\u79fb\u516c\u5f0f</text>')
    s.append(f'<text x="{sx + sw + gap2/2:.0f}" y="66" font-size="13" font-weight="600" text-anchor="middle" fill="#1a1a2e">\u7269\u7406\u7f13\u51b2\uff08\u4e24\u4e2a 512B slab\uff09</text>')

    # left logical grid
    for r in range(8):
        for b in range(8):
            x, y = lx + b * (cell + gap), ly + r * (cell + gap)
            s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" fill="{hue(r,b)}" stroke="#8fa3b8"/>')
    for r in (0, 7):
        s.append(f'<text x="{lx-8}" y="{ly + r*(cell+gap) + 15}" font-size="10" text-anchor="end" fill="#5b6b7d">{r}</text>')
    for b in (0, 7):
        s.append(f'<text x="{lx + b*(cell+gap) + 11}" y="{ly-8}" font-size="10" text-anchor="middle" fill="#5b6b7d">{b}</text>')
    s.append(f'<text x="{lx-40}" y="{ly + gw/2:.0f}" font-size="11" fill="#3c4a58" text-anchor="middle" transform="rotate(-90 {lx-40} {ly + gw/2:.0f})">\u884c r</text>')
    s.append(f'<text x="{lx + gw/2:.0f}" y="{ly - 24}" font-size="10.5" text-anchor="middle" fill="#3c4a58">\u5757 b\uff08\u6bcf\u5757\u7ba1 16 \u4e2a\u6743\u91cd\u5143\u7d20\uff09</text>')
    s.append(f'<text x="{lx}" y="{ly + gw + 26}" font-size="11" fill="#5b6b7d">\u6bcf\u683c = 1 \u5b57\u8282 E4M3 scale</text>')

    # middle: formula + why
    for i, line in enumerate([
        "off(r,b) = PS*(r/128)",
        "  + 512*(b/4)",
        "  + 16*(r%32)",
        "  + 4*((r/32)%4) + (b%4)",
        "",
        "PS = 512*ceil(KB/4)",
        "\uff08\u4e00\u4e2a 128 \u884c panel \u7684\u5b57\u8282\u6570\uff09",
    ]):
        mono = i < 7
        s.append(f'<text x="{mx}" y="{ly + 8 + i*22}" font-size="12.5" fill="#0f6db3" '
                 f'font-family="Noto Sans Mono CJK SC,Consolas,monospace">{line}</text>'
                 if line else f'<text x="{mx}" y="{ly + 8 + i*22}"> </text>')
    wy = ly + 8 + 7*22 + 18
    s.append(f'<text x="{mx}" y="{wy}" font-size="11.5" font-weight="600" fill="#1a1a2e">\u4e3a\u4ec0\u4e48\u8981\u91cd\u6392\uff1f</text>')
    for i, line in enumerate([
        "\u2022 \u4e00\u884c\u7684 4 \u4e2a\u5757 scale \u5728\u7269\u7406\u4e0a\u8fde\u7eed",
        "  \uff084 \u5b57\u8282\uff09\uff1b\u6bcf\u7ebf\u7a0b\u4e00\u6b21 16B \u5411\u91cf\u8bfb\u53d6",
        "\u2022 32 \u884c\u7c07 \u00d7 4 \u5757\u7ec4 = 512B \u8fde\u7eed\u533a\u95f4\uff0c",
        "  \u4e00\u4e2a warp \u7684 32 \u7ebf\u7a0b\u6070\u597d\u5408\u5e76\u6d88\u8d39",
        "\u2022 \u82e5\u6309\u884c\u4e3b\u5e8f\u5b58\u653e\uff0c\u540c\u4e00\u884c\u76f8\u90bb 4 \u5757",
        "  \u7684 scale \u4f1a\u76f8\u8ddd KB \u5b57\u8282\uff0c\u8bfb\u53d6\u8de8\u7f13\u5b58\u884c",
    ]):
        s.append(f'<text x="{mx}" y="{wy + 20 + i*17}" font-size="11.5" fill="#3c4a58">{line}</text>')

    # right: two slabs
    py = ly
    for slab in range(2):
        ox = sx + slab * (sw + gap2)
        s.append(f'<rect x="{ox-7}" y="{py-7}" width="{sw+14}" height="{8*(cell+gap)+14}" fill="none" stroke="#0f6db3" stroke-dasharray="4 3"/>')
        s.append(f'<text x="{ox + sw/2:.0f}" y="{py-10}" font-size="10.5" text-anchor="middle" fill="#0f6db3">slab {slab} \u00b7 \u57fa\u5740 {512*slab}</text>')
        for r in range(8):
            for q in range(4):
                b = slab * 4 + q
                x, y = ox + q * (cell + gap), py + r * (cell + gap)
                s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" fill="{hue(r,b)}" stroke="#8fa3b8"/>')
    s.append(f'<text x="{sx-30}" y="{py + 8*(cell+gap) + 26}" font-size="10.5" fill="#5b6b7d">同色 = 同一逻辑单元；一列 4 格 = 该行连续 4 字节</text>')

    # worked examples bottom-left
    ex = ly + gw + 60
    s.append(f'<text x="{lx}" y="{ex}" font-size="11.5" font-weight="600" fill="#1a1a2e">\u5b9e\u4f8b\uff08\u6f14\u793a\u7f51\u683c r&lt;32\uff0c\u4ea4\u9519\u9879\u4e3a 0\uff09</text>')
    for i, (r, b) in enumerate([(0, 0), (0, 5), (3, 2), (7, 7)]):
        s.append(f'<text x="{lx}" y="{ex + 20 + i*19}" font-size="12" fill="#3c4a58" font-family="Noto Sans Mono CJK SC,Consolas,monospace">(r={r}, b={b}) -> \u5b57\u8282 {off(r, b)}</text>')
    s.append(f'<text x="{mx}" y="{H-16}" font-size="10.5" fill="#5b6b7d">\u771f\u5b9e\u5c3a\u5bf8\uff1a128 \u884c/panel\uff0c\u884c\u5185\u6309 32 \u884c\u7c07\u56db\u7ec4\u4ea4\u9519\uff1b\u5e03\u5c40\u7531\u5355\u5b57\u8282\u63a2\u9488\u5b9e\u9a8c\u6d4b\u5b9a\uff08\u00a73.1\uff09\u3002</text>')
    s.append("</svg>")
    (FIGS / "swizzle_layout.svg").write_text("".join(s), encoding="utf-8")
    print("wrote swizzle_layout.svg")

def gptq_block():
    """GPTQ NVFP4 block-adaptation principle diagram (static).
    Top: weight-matrix strip (quantized | current 16-col block | pending) with
    error feed-forward arc; middle: the three formulas; bottom: clip search
    chips +示意 MSE U-curve bars."""
    W, H = 940, 560
    F = "Noto Sans CJK SC,system-ui,sans-serif"
    M = "Noto Sans Mono CJK SC,Consolas,monospace"
    cell, gap = 14, 2
    pitch = cell + gap
    rows, cols = 5, 36
    lx, ly = 70, 150
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{F}">',
         f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">'
         f'<path d="M0,0 L8,4 L0,8 z" fill="#c05621"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="20" y="30" font-size="15" font-weight="600" fill="#1a1a2e">'
         f'GPTQ 的 NVFP4 块适配：块界重定标 + 逐列误差前馈（静态示意）</text>']

    def x(c): return lx + c * pitch

    # matrix cells: cols 0-15 quantized, 16-31 current block, 32-35 pending
    for r in range(rows):
        for c in range(cols):
            fill = "#c7d0da" if c <= 15 else ("#d8e7f7" if c <= 31 else "#ffffff")
            s.append(f'<rect x="{x(c)}" y="{ly + r*pitch}" width="{cell}" height="{cell}" rx="2" fill="{fill}" stroke="#8fa3b8"/>')
    # current column i = col 20
    ci = 20
    for r in range(rows):
        s.append(f'<rect x="{x(ci)}" y="{ly + r*pitch}" width="{cell}" height="{cell}" rx="2" fill="#2f7fd1" stroke="#173a5e"/>')
    # current block outer frame (cols 16..31)
    s.append(f'<rect x="{lx + 16*pitch - 4}" y="{ly - 4}" width="{16*pitch + 8}" height="{rows*pitch + 8}" fill="none" stroke="#0f6db3" rx="5"/>')
    # zone labels
    s.append(f'<text x="{lx + 128}" y="126" font-size="11" text-anchor="middle" fill="#3c4a58">已量化 · 块 0…m−1（E2M1 格点已定格）</text>')
    s.append(f'<text x="{lx + 384}" y="126" font-size="11" text-anchor="middle" fill="#0f6db3" font-weight="600">当前块 m（16 列）：入口重算逐行 s_r</text>')
    s.append(f'<text x="{lx + 576 + 10}" y="126" font-size="11" fill="#3c4a58">未量化</text>')
    # error feed-forward arc: from current column top to pending zone top
    cx, tx = x(ci) + cell/2, lx + 544
    s.append(f'<path d="M {cx},{ly-4} C {cx},96 {tx},96 {tx},{ly-6}" fill="none" stroke="#c05621" stroke-width="1.6" marker-end="url(#ah)"/>')
    s.append(f'<text x="556" y="102" font-size="11" text-anchor="middle" fill="#c05621">误差前馈：δ_i 摊到未量化列</text>')
    # row / column annotations
    s.append(f'<text x="{lx-14}" y="{ly+40}" font-size="10.5" text-anchor="middle" fill="#5b6b7d" transform="rotate(-90 {lx-14} {ly+40})">行 r（每行独立 s_r）</text>')
    s.append(f'<text x="{cx}" y="{ly + rows*pitch + 18}" font-size="10.5" text-anchor="middle" fill="#1f5e9e">列 i（正在量化）</text>')

    # formula box
    s.append(f'<rect x="40" y="272" width="860" height="142" rx="6" fill="#f4f7fb" stroke="#d5dfeb"/>')
    lines = [
        ('① 块入口（每 16 列）  s_r \u2190 E4M3( c \u00b7 amax_16( W\u2032[r, \u00b7] ) / 6 )     \u2190 从当前值重算，非初始权重', None),
        ('② 列量化              q_i = round_E2M1( W\u2032[:, i] / s_r ) \u00b7 s_r ,  \u03b4_i = W\u2032[:, i] \u2212 q_i', None),
        ('③ 误差前馈            W\u2032[:, i+1:] \u2190 W\u2032[:, i+1:] \u2212 \u03b4_i \u00b7 U[i, i+1:] / U[i,i]', '#0f6db3'),
        ('    U = chol(H\u207b\u00b9) 上三角因子：一次分解 = 每步对剩余子矩阵求逆的精确 OBS 条件解', '#5b6b7d'),
    ]
    for k, (txt, color) in enumerate(lines):
        s.append(f'<text x="58" y="{300 + k*27}" font-size="12" font-family="{M}" fill="{color or "#16324a"}">{txt}</text>')

    # clip search: label + chips +示意 MSE U-bars
    s.append(f'<text x="40" y="445" font-size="11.5" fill="#1a1a2e">逐层裁剪搜索（max 与 MSE 校准在此合流）：c 网格 \u2192</text>')
    grid = ["1.0", "0.95", "0.9", "0.85", "0.8", "0.7", "0.6", "0.5"]
    mse = [1.00, 0.86, 0.72, 0.78, 0.86, 0.96, 1.08, 1.22]
    for k, c in enumerate(grid):
        best = (k == 2)
        chx = 340 + k * 52
        fill, tc = ("#2f7fd1", "#ffffff") if best else ("#e8eef5", "#3c4a58")
        s.append(f'<rect x="{chx}" y="430" width="46" height="22" rx="4" fill="{fill}"/>')
        s.append(f'<text x="{chx+23}" y="445" font-size="11" text-anchor="middle" fill="{tc}">{c}</text>')
        bh = mse[k] * 30
        s.append(f'<rect x="{chx}" y="{512-bh:.0f}" width="46" height="{bh:.0f}" rx="2" fill="{"#2f7fd1" if best else "#b9cfe6"}"/>')
    s.append(f'<text x="{340+2*52+23}" y="474" font-size="10" text-anchor="middle" fill="#1f5e9e">c*（示意）</text>')
    s.append(f'<line x1="340" y1="512" x2="750" y2="512" stroke="#8fa3b8"/>')
    s.append(f'<text x="762" y="492" font-size="10.5" fill="#5b6b7d">逐层输出 MSE（示意，U 形）</text>')
    s.append(f'<text x="40" y="540" font-size="11" fill="#3c4a58">判据 argmin_c tr(\u0394W_c \u00b7 H \u00b7 \u0394W_c\u1d40)（Hessian 恒等式评估，无须跑网络）；求逆前阻尼 H \u2190 H + 0.01\u00b7mean(diag H)\u00b7I。</text>')
    s.append("</svg>")
    (FIGS / "gptq_block.svg").write_text("".join(s), encoding="utf-8")
    print("wrote gptq_block.svg")

if __name__ == "__main__":
    e1_chart()
    ladder_chart()
    swizzle_layout()
    gptq_block()
    print("figures done ->", FIGS)
