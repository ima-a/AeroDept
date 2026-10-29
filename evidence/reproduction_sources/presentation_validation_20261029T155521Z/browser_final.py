from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
repo=Path('/home/Ima/work/Projects/DL-projects/AeroDept');c=json.loads((repo/'configs/presentation_local.json').read_text());r=Path(c['results']);a=repo/'artifacts'/r.name
with sync_playwright() as pw:
 browser=pw.chromium.launch(executable_path='/tmp/aerodepth-browser/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'])
 context=browser.new_context(viewport={'width':1536,'height':1080},accept_downloads=True);page=context.new_page();errors=[];page.on('pageerror',lambda x:errors.append(str(x)))
 page.goto('http://127.0.0.1:8501',wait_until='domcontentloaded');page.get_by_role('tab',name='Depth demonstration',exact=True).wait_for(timeout=60000);page.get_by_role('tab',name='Depth demonstration',exact=True).click();page.get_by_role('button',name='Run actual model inference',exact=True).wait_for(state='visible',timeout=60000)
 page.get_by_label('Actual inference device',exact=True).click();page.get_by_role('option',name='cuda',exact=True).click();page.wait_for_timeout(500);page.get_by_role('button',name='Run actual model inference',exact=True).click();page.get_by_text('Actual inference completed; model parameters and BN buffers unchanged.',exact=True).wait_for(timeout=60000);page.wait_for_timeout(800);page.screenshot(path=str(r/'dashboard_actual_A_CUDA.png'),full_page=True)
 page.get_by_role('tab',name='Scheduler',exact=True).click();page.get_by_text('Full-split quality is causal counterfactual replay over verified A predictions; policy calls are replay decisions. Live timings below execute the model on separate fixed windows.',exact=True).wait_for(timeout=30000);page.wait_for_timeout(500);page.screenshot(path=str(r/'dashboard_scheduler_final.png'),full_page=True)
 page.get_by_role('tab',name='Evidence',exact=True).click();button=page.get_by_role('button',name='Download '+r.name+' / figures/A_live_timing.png',exact=True)
 with page.expect_download() as item:button.click()
 downloaded=a/'downloaded_A_live_timing.png';item.value.save_as(downloaded);assert hashlib.file_digest(downloaded.open('rb'),'sha256').hexdigest()==hashlib.file_digest((r/'figures/A_live_timing.png').open('rb'),'sha256').hexdigest()
 page.get_by_role('tab',name='Overview',exact=True).click();page.wait_for_timeout(500);page.screenshot(path=str(r/'dashboard_overview_final.png'),full_page=True)
 assert not errors and page.locator('[data-testid="stException"]').count()==0
 (r/'browser_final_verification.json').write_text(json.dumps({'passed':True,'actual_A_CUDA_inference':True,'model_and_BN_unchanged':True,'latest_scope_caption_verified':True,'figure_download_exact_bytes':True,'javascript_errors':errors,'prior_full_flow_receipt':'browser_verification.json'},indent=2)+'\n')
 browser.close()
print('Final rendered UI, actual CUDA inference and PNG export verified.',flush=True)
