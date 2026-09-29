// CPU-only browser layout QA, bound to the exact current article and figures.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {pathToFileURL} = require('url');
const P = __dirname;
const output = path.join(P, 'validation/browser-validation.json');
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const requireCheck = (ok, why) => { if (!ok) throw new Error(why); };
function snapshot() {
  const registry = JSON.parse(fs.readFileSync(path.join(P,'figures.json'),'utf8'));
  const names = Object.keys(registry).sort();
  requireCheck(names.length > 0, 'Empty figure registry');
  return {names, html_sha256:hash(path.join(P,'paper.html')), qa_script_sha256:hash(__filename),
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
      return figures.length===expected && figures.every(f=>f.dataset.figure.startsWith('shot_')
        ? f.querySelectorAll('img').length===1 && [...f.querySelectorAll('img')].every(i=>i.complete&&i.naturalWidth===3840&&i.naturalHeight===2280)
        : f.querySelectorAll(':scope > svg').length===1);
    },before.names.length,{timeout:30000});
    const desktop=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,
      figures:document.querySelectorAll('figure').length,
      figureNames:[...document.querySelectorAll('figure')].map(f=>f.dataset.figure).sort(),
      mathCount:document.querySelectorAll('.katex').length,
      mathErrors:document.querySelectorAll('.katex-error').length,
      svgOverflow:[...document.querySelectorAll('figure > svg text')].filter(e=>{
        const b=e.getBBox(),v=e.ownerSVGElement.viewBox.baseVal;
        return b.x<v.x-1||b.y<v.y-1||b.x+b.width>v.x+v.width+1||b.y+b.height>v.y+v.height+1;
      }).map(e=>({figure:e.closest('figure').dataset.figure,text:e.textContent})),
      decoded:[...document.querySelectorAll('figure img')].map(i=>({figure:i.closest('figure').dataset.figure,width:i.naturalWidth,height:i.naturalHeight}))}));
    requireCheck(JSON.stringify(desktop.figureNames)===JSON.stringify(before.names),'Unexpected or missing browser figure identities');
    requireCheck(desktop.mathCount>0,'No rendered equations');
    requireCheck(desktop.decoded.length===before.names.filter(n=>n.startsWith('shot_')).length,'Missing screenshot images');
    fs.mkdirSync(path.join(P,'_build'),{recursive:true});
    await page.screenshot({path:path.join(P,'_build/qa-desktop.png')});
    for(const name of before.names.filter(n=>!n.startsWith('shot_'))) {
      const figure=page.locator('#fig-'+name);
      await figure.evaluate(e=>e.scrollIntoView({behavior:'instant',block:'center'}));
      await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
      await figure.screenshot({path:path.join(P,'_build/qa-'+name+'.png')});
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
