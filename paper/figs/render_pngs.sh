#!/bin/bash
# Render verified SVGs to 2x PNGs and bind raster outputs to their exact sources.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
uv run --with cairosvg==2.9.1 python - <<'PY'
import hashlib,json,pathlib
import cairosvg
root=pathlib.Path.cwd()
paper=root/'paper'
manifest=json.loads((paper/'validation/figure-inputs.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for row in manifest['svgs']:
    source=root/row['path']
    if sha(source)!=row['sha256']:raise RuntimeError('SVG changed after generation: '+str(source))
    target=source.with_suffix('.png')
    cairosvg.svg2png(url=str(source),write_to=str(target),scale=2.0)
    rows.append({'svg':row['path'],'svg_sha256':row['sha256'],
                 'png':str(target.relative_to(root)),'png_sha256':sha(target)})
    print('Rendered',target.name)
result={'status':'passed','renderer':'cairosvg','version':cairosvg.__version__,'scale':2.0,'figures':rows}
(paper/'validation/figure-renders.json').write_text(json.dumps(result,indent=2)+'\n')
PY
