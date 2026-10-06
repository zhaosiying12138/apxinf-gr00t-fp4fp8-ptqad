// CPU-only browser layout QA, bound to the exact current article and figures.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {pathToFileURL} = require('url');
const P = __dirname;
const output = path.join(P, 'validation/browser-validation.json');
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const requireCheck = (ok, why) => { if (!ok) throw new Error(why); };
const SCREENSHOTS = ['shot_bake','shot_collect','shot_evalserver','shot_fp8probe','shot_gemm',
  'shot_gr00t','shot_nvfp4','shot_opbench','shot_opd','shot_opdcache','shot_packed','shot_pi05',
  'shot_probe','shot_qad','shot_qat','shot_rollout','shot_verify'].sort();
const DIAGRAMS = ['e1_latency','ladder','swizzle_layout','gptq_block','recovery_protocol',
  'budget_ladder','ptq_frontier'].sort();
const REVIEW_PENDING = ['shot_bake','shot_collect','shot_qad','shot_opdcache','shot_opd','shot_evalserver','shot_rollout'].sort();
function snapshot() {
  const registry = JSON.parse(fs.readFileSync(path.join(P,'figures.json'),'utf8'));
  const meta = JSON.parse(fs.readFileSync(path.join(P,'meta.json'),'utf8'));
  const names = Object.keys(registry).sort();
  requireCheck(JSON.stringify(names)===JSON.stringify([...SCREENSHOTS,...DIAGRAMS].sort()),
    'The article must retain exactly 17 screenshot slots and seven diagrams');
  const pending = names.filter(name=>registry[name].refresh_pending===true);
  requireCheck(pending.length===0 || String(meta.status||'').includes('审阅稿'),
    'Pending screenshots forbid a final publication');
  requireCheck(pending.every(name=>REVIEW_PENDING.includes(name)),
    'Pending screenshot set includes an unaudited review slot');
  return {names, screenshot_slots:SCREENSHOTS.length, screenshots:SCREENSHOTS.length-pending.length,
    screenshot_refresh_pending:pending, meta_sha256:hash(path.join(P,'meta.json')),
    html_sha256:hash(path.join(P,'paper.html')), qa_script_sha256:hash(__filename),
    figures_manifest_sha256:hash(path.join(P,'figures.json')),
    svg_sha256:Object.fromEntries(names.filter(n=>!n.startsWith('shot_')).map(n=>[n,hash(path.join(P,'figs',n+'.svg'))])),
    png_sha256:Object.fromEntries(names.map(n=>[n,hash(path.join(P,'figs',n+'.png'))]))};
}
function write(report) {
  fs.mkdirSync(path.dirname(output),{recursive:true});
  fs.writeFileSync(output+'.tmp',JSON.stringify(report,null,2)+'\n');
  fs.renameSync(output+'.tmp',output);
}
(async()=>{
  let browser;
  let report={pass:false,status:'running'};
  write(report); // Never preserve a previous pass if this invocation fails.
  try {
    const before=snapshot(); report={...report,...before}; write(report);
    const {chromium}=require('./_build/renderer/node_modules/playwright');
    browser=await chromium.launch({headless:true,args:['--disable-gpu','--no-sandbox']});
    const page=await browser.newPage({viewport:{width:1360,height:960},deviceScaleFactor:1});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('request',r=>{if(/^https?:/.test(r.url())) errors.push('Unexpected external runtime request: '+r.url());});
    page.on('requestfailed',r=>errors.push(r.url()+': '+r.failure()?.errorText));
    await page.goto(pathToFileURL(path.join(P,'paper.html')).href,{waitUntil:'load'});
    await page.addStyleTag({content:'html{scroll-behavior:auto!important}'});
    await page.evaluate(()=>document.fonts.ready);
    await page.waitForFunction(expected=>{
      const figures=[...document.querySelectorAll('figure[data-figure]')];
      return figures.length===expected.names.length && figures.every(f=>{
        if(expected.screenshot_refresh_pending.includes(f.dataset.figure)) {
          return f.dataset.refreshPending==='true' && f.querySelectorAll('img,svg').length===0 &&
            f.querySelectorAll('.pending-shot').length===1 && f.textContent.includes('待补本轮真实运行截图');
        }
        if(f.dataset.refreshPending==='true') return false;
        return f.dataset.figure.startsWith('shot_')
          ? f.querySelectorAll('img').length===1 && [...f.querySelectorAll('img')].every(i=>i.complete&&i.naturalWidth===3840&&i.naturalHeight===2280)
          : f.querySelectorAll(':scope > svg').length===1;
      });
    },before,{timeout:30000});
    const desktop=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,
      figures:document.querySelectorAll('figure').length,
      figureNames:[...document.querySelectorAll('figure')].map(f=>f.dataset.figure).sort(),
      pendingNames:[...document.querySelectorAll('figure[data-refresh-pending="true"]')].map(f=>f.dataset.figure).sort(),
      diagramCount:document.querySelectorAll('figure.diagram > svg').length,
      mathCount:document.querySelectorAll('.katex').length,
      mathErrors:document.querySelectorAll('.katex-error').length,
      svgOverflow:[...document.querySelectorAll('figure > svg text')].filter(e=>{
        const b=e.getBBox(),v=e.ownerSVGElement.viewBox.baseVal;
        return b.x<v.x-1||b.y<v.y-1||b.x+b.width>v.x+v.width+1||b.y+b.height>v.y+v.height+1;
      }).map(e=>({figure:e.closest('figure').dataset.figure,text:e.textContent})),
      decoded:[...document.querySelectorAll('figure img')].map(i=>({figure:i.closest('figure').dataset.figure,width:i.naturalWidth,height:i.naturalHeight}))}));
    requireCheck(JSON.stringify(desktop.figureNames)===JSON.stringify(before.names),'Unexpected or missing browser figure identities');
    requireCheck(desktop.mathCount>0,'No rendered equations');
    requireCheck(desktop.decoded.length===before.screenshots,'Missing screenshot images');
    requireCheck(JSON.stringify(desktop.decoded.map(row=>row.figure).sort())===
      JSON.stringify(SCREENSHOTS.filter(name=>!before.screenshot_refresh_pending.includes(name))),
      'Decoded screenshot identities differ');
    requireCheck(JSON.stringify(desktop.pendingNames)===JSON.stringify(before.screenshot_refresh_pending),
      'Pending screenshot slots differ');
    requireCheck(desktop.diagramCount===7,'Missing method or data diagrams');
    fs.mkdirSync(path.join(P,'_build'),{recursive:true});
    await page.screenshot({path:path.join(P,'_build/qa-desktop.png')});
    for(const name of before.names.filter(n=>!n.startsWith('shot_'))) {
      const figure=page.locator('#fig-'+name);
      await figure.evaluate(e=>e.scrollIntoView({behavior:'instant',block:'center'}));
      await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
      await figure.screenshot({path:path.join(P,'_build/qa-'+name+'.png')});
    }
    if(before.screenshot_refresh_pending.length) {
      await page.locator('#fig-'+before.screenshot_refresh_pending[0]).screenshot({path:path.join(P,'_build/qa-pending-screenshot.png')});
    }
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>scrollTo(0,0));
    const mobile=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
    await page.screenshot({path:path.join(P,'_build/qa-mobile.png')});
    requireCheck(JSON.stringify(before)===JSON.stringify(snapshot()),'QA inputs changed during inspection');
    report={...report,desktop,mobile,errors,pass:desktop.scroll<=desktop.width&&mobile.scroll<=mobile.width&&errors.length===0&&desktop.mathErrors===0&&desktop.svgOverflow.length===0};
    report.status=report.pass?'passed':'failed'; write(report);
    requireCheck(report.pass,'Browser layout or rendering checks failed');
    console.log(JSON.stringify(report));
  } catch(error) {
    report={...report,pass:false,status:'failed',failure:String(error.stack||error)};write(report);
    console.error(report.failure);process.exitCode=1;
  } finally {if(browser) await browser.close();}
})();
