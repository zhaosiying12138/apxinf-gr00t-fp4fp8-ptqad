"""CPU-only appendix A source/figure/link resolution; no model imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper'
APX=ROOT/'third_party/apxinf-robo/apxinf'

def listing(folder):
    return [folder/x for x in subprocess.check_output(
        ['git','ls-files','--cached','--others','--exclude-standard'],cwd=folder,text=True).splitlines()]

def main():
    appendix=next((PAPER/'sections').glob('14-*.md'));text=appendix.read_text()
    refs=json.loads((PAPER/'validation/source_refs.json').read_text())
    sources={row['path']:row for row in refs['files']}
    known={p.resolve() for p in [*listing(ROOT),*listing(APX)] if p.suffix in ('.py','.rs','.cu','.cuh') and p.is_file()}
    checked=[];missing=[]
    for token in re.findall(r'`([^`\n]+)`',text):
        match=re.fullmatch(r'([\w./-]+\.(?:py|rs|cu|cuh))(?:::(\w+))?',token)
        if not match:continue
        relative,symbol=match.groups()
        candidates=[p for p in known if p.as_posix().endswith('/'+relative)]
        if len(candidates)!=1:
            missing.append({'token':token,'candidate_count':len(candidates)});continue
        path=candidates[0];key=path.relative_to(ROOT).as_posix()
        if symbol:
            records=sources.get(key,{}).get('symbols',[])
            if not any(r['symbol']==symbol or r['symbol'].endswith('.'+symbol) for r in records):
                missing.append({'token':token,'reason':'symbol not in independently generated source_refs'});continue
        checked.append({'token':token,'path':key,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    figures=json.loads((PAPER/'figures.json').read_text())
    figure_names=re.findall(r'\{\{fig:([\w.-]+)\}\}',text)
    for name in figure_names:
        if name not in figures or not any((PAPER/'figs'/f'{name}{ext}').is_file() for ext in ('.svg','.png')):
            missing.append({'figure':name,'reason':'missing catalog/asset'})
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if target.startswith(('http:','https:','#')):continue
        target=target.strip('<>');plain=target.split('#')[0]
        candidates=[appendix.parent/plain,PAPER/plain,ROOT/plain]
        exists=[str(p.resolve().relative_to(ROOT)) for p in candidates if p.exists() and p.resolve().is_relative_to(ROOT)]
        if not exists:missing.append({'link':target,'reason':'local target missing'})
        links.append({'target':target,'resolved_paths':exists})
    for path in (PAPER/'validation/source_refs.json',PAPER/'validation/cpu-tests.json',PAPER/'validation/cpu-tests.log'):
        if not path.is_file():missing.append({'evidence_file':str(path),'reason':'missing'})
    report={'status':'passed' if not missing else 'failed','generated_utc':datetime.now(timezone.utc).isoformat(),
        'appendix':appendix.relative_to(ROOT).as_posix(),'appendix_sha256':hashlib.sha256(appendix.read_bytes()).hexdigest(),
        'scope':'Source filename/symbol navigation, figure catalog/assets and literal local Markdown links; not GPU/numerical validation.',
        'source_tokens':checked,'figures':figure_names,'literal_local_links':links,'unresolved':missing}
    (PAPER/'validation/appendix_a_refs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'source_tokens':len(checked),'figures':len(figure_names),
                      'literal_local_links':len(links),'unresolved':missing},ensure_ascii=False))
    if missing:raise SystemExit(1)

if __name__=='__main__':main()
