from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,os
repo=Path('/home/Ima/work/Projects/DL-projects/AeroDept');config=json.loads((repo/'configs/presentation_local.json').read_text());result=Path(config['results'])
def state(active,code=None):
 p=result/'timing_activity.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'active':active,'exit_code':code,'UTC':datetime.now(timezone.utc).isoformat(),'unit':os.environ.get('AERODEPTH_TIMING_UNIT')}));tmp.replace(p)
state(True)
try:
 cp=subprocess.run([sys.executable,'-u','-B','scripts/presentation.py','timing'],cwd=repo);code=cp.returncode
finally:state(False,locals().get('code',1))
raise SystemExit(code)
