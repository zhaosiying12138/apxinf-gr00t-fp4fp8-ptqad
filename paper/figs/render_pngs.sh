#!/bin/bash
# Render all paper figure SVGs to 2x PNGs (cairosvg picks up Noto Sans CJK SC via fc).
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
for f in e1_latency ladder opbench_heatmap swizzle_layout; do
  uv run --with cairosvg python -c "import cairosvg; cairosvg.svg2png(url='$f.svg', write_to='$f.png', scale=2.0)"
  echo "ok $f"
done
