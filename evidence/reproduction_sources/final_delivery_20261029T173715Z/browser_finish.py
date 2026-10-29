from pathlib import Path
import json,hashlib,time
from playwright.sync_api import sync_playwright
p=json.loads(Path('/tmp/aerodepth_final_delivery_paths.json').read_text());res=Path(p['res']);art=Path(p['art']);cfg=json.loads(Path('configs/presentation_local.json').read_text());source=Path(cfg['results']);checks=[]
with sync_playwright() as pw:
 b=pw.chromium.launch(executable_path='/tmp/aerodepth-browser/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'])
 c=b.new_context(viewport={'width':1536,'height':1080},accept_downloads=True);page=c.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8501');page.get_by_role('tab',name='Evidence',exact=True).wait_for(timeout=60000);page.wait_for_timeout(1600)
 page.get_by_text('INDEPENDENT TEST · NOT RUN',exact=True).wait_for();assert page.get_by_role('tab').count()==5
 page.screenshot(path=str(res/'dashboard_overview.png'),full_page=True)
 checks.append({'flow':'overview and sidebar show truthful NOT RUN status','passed':True})
 with page.expect_download() as d:page.get_by_role('button',name='Download final written report',exact=True).click()
 target=art/'downloaded_final_report.md';d.value.save_as(target);assert target.read_bytes()==(res/'FINAL_REPORT.md').read_bytes();checks.append({'flow':'final Markdown report download byte parity','passed':True})
 page.get_by_role('tab',name='Evidence',exact=True).click()
 for name in ['scheduler_comparison.csv','figures/A_live_timing.png']:
  with page.expect_download() as d:page.get_by_role('button',name='Download '+source.name+' / '+name,exact=True).click()
  target=art/('downloaded_'+Path(name).name);d.value.save_as(target);assert target.read_bytes()==(source/name).read_bytes()
 checks.append({'flow':'CSV and PNG exports byte parity','passed':True})
 page.get_by_role('tab',name='Overview',exact=True).click();page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(900)
 assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
 page.screenshot(path=str(res/'dashboard_mobile.png'),full_page=True);checks.append({'flow':'390px mobile overview without horizontal page overflow','passed':True})
 page.get_by_role('tab',name='Depth demonstration',exact=True).click();page.wait_for_timeout(700);page.screenshot(path=str(res/'dashboard_depth_mobile.png'),full_page=True)
 assert page.locator('[data-testid="stException"]').count()==0 and not errors
 b.close()
 # Attempt1 completed all flows before its incorrect renamed-download locator.
 prior=[{'flow':x,'passed':True,'evidence':'browser_delivery_attempt1.log and screenshots; locator-only failure happened afterwards'} for x in ['explicit FP32 CPU inference A','explicit FP32 CPU inference Original','explicit FP32 CPU inference KD','fixed-frame preset','missing KD recording unavailable','A recorded playback advances','untimed ablation unavailable','historical study distinct','per-trajectory charts']]
 (res/'browser_verification.json').write_text(json.dumps({'passed':True,'checks':prior+checks,'javascript_errors':errors,'automation_caveat':'First attempt used delivery directory name instead of unchanged measurement directory name for CSV locator. No app failure. Followup corrected only locator and completed exports/mobile checks; no repeated model inference.','no_test_pixels':True},indent=2)+'\n')
 print(json.dumps({'passed':True,'checks':len(prior+checks)}))
