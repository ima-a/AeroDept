"""One frozen reserved-test study; streaming inputs, seven actual policy traces.

This version does not modify the old blocked runner or its read/provenance guards.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import argparse, ast, csv, gzip, hashlib, json, math, os, platform, subprocess, sys, time
import numpy as np
import torch
from . import expanded as e
from .independent_data import EXPECTED, RESERVATIONS, TRAIN_VAL, write_new
from .inference import file_sha256
from .metrics import DepthMetricAccumulator
from .presentation import load_model, decode_pair, ValidationGuard, verify_file
from .presentation_scheduler import PresentationScheduler, fixed_positions
from .research import summarize, clean

ROOT=Path(__file__).resolve().parents[2]
STUDY='results/independent_test_20261030'
VALIDATION='results/presentation_validation_20261029T155521Z/PROTOCOL.json'
REFERENCE='fb1b4aedd15280abdb1126293e022d78cdcbf4e5'
POLICIES=['every','fixed2','fixed5','budget_fixed','v2']
MATRIX=[('A_'+p,'A',p) for p in POLICIES]+[('Original_every','Original','every'),('KD_every','KD','every')]
DATA={'image_size':[192,320],'min_depth':.1,'max_depth':1000.,'depth_encoding':'midair_float16_bits','depth_scale':1.}
SOURCES=['src/aerodepth_lite/'+n+'.py' for n in ['independent_test','independent_data','model','data','metrics','losses','expanded','repro','kaggle','inference','presentation','presentation_scheduler','scheduler_v2','scheduler','research']]+['scripts/independent_test.py','scripts/prepare_independent_test.py','tests/test_independent_test.py','configs/presentation.json']

def read(path):return json.loads(Path(path).read_text())
def source_hashes():return {name:file_sha256(ROOT/name) for name in SOURCES}
def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def model_entries(locations):
    config=read(ROOT/'configs/presentation.json');out={}
    for key in ['A','Original','KD']:
        entry=dict(config['models'][key])
        for field in ['weights','metadata']:
            if field in entry:entry[field]=str(Path(locations['kd_checkpoint']) if key=='KD' else ROOT/entry[field])
        out[key]=entry
    return out

# Compatibility
def compatibility(entries):
    checkpoint=torch.load(entries['KD']['weights'],map_location='cpu',weights_only=False)
    identity=checkpoint['extra']['identity']
    if checkpoint['epoch']!=8 or identity['arm']!='exploratory_kd_rejected_teacher_v1' or identity['original_gate_passed'] is not False:
        raise ValueError('KD is not the preselected rejected-teacher epoch-9 checkpoint')
    old={**identity['source_sha256'],**identity['exploratory_source_sha256']};rows={}
    for name,expected in old.items():
        prior=subprocess.check_output(['git','show',REFERENCE+':'+name],cwd=ROOT)
        current=(ROOT/name).read_bytes()
        if hashlib.sha256(prior).hexdigest()!=expected or ast.dump(ast.parse(prior))!=ast.dump(ast.parse(current)):
            raise ValueError('Historical checkpoint source has substantive incompatibility: '+name)
        rows[name]={'historical_sha256':expected,'current_sha256':hashlib.sha256(current).hexdigest(),'ast_equal':True}
    return {'scope':'Inference-only loading of the exact pinned immutable KD checkpoint; no resume, training or gate override.',
            'reference_commit':REFERENCE,'checkpoint_sha256':entries['KD']['sha256'],'selected_epoch':9,
            'original_teacher_gate_passed':False,'files':rows}

def verify_manifest(manifest,root,verify_bytes=False):
    if manifest.get('per_trajectory')!=EXPECTED or manifest.get('frame_counts')!={'test':8820}:raise ValueError('Wrong reserved test counts')
    if manifest['reservations_sha256']!=file_sha256(RESERVATIONS) or manifest['train_val_manifest_sha256']!=file_sha256(TRAIN_VAL):raise ValueError('Reference splits changed')
    rows=manifest['records'];keys=[(r['trajectory_id'],r['frame_index']) for r in rows]
    expected=[(tr,i) for tr,n in sorted(EXPECTED.items()) for i in range(n)]
    if keys!=expected:raise ValueError('Duplicate, missing, replaced or nonchronological frames')
    for r in rows:
        tr=r['trajectory_id'];parent,name=tr.rsplit('/',1)
        if set(r['navigation'])!=set(['ax','ay','az','wx','wy','wz','gps_vx','gps_vy','gps_vz']) or not all(math.isfinite(v) for v in r['navigation'].values()):
            raise ValueError('Navigation is not the sensor-only contract')
        if r['timestamp']!=r['frame_index']/25:raise ValueError('Camera timestamp changed')
        for field,stream,ext in [('rgb','color_left','JPEG'),('depth','depth','PNG')]:
            relative=Path(r[field+'_path']);correct=f'{parent}/{stream}/{name}/{r["frame_index"]:06d}.{ext}'
            if relative.is_absolute() or relative.as_posix()!=correct or '..' in relative.parts:raise ValueError('Unsafe or mismatched test path')
            p=root/relative
            for part in [p,*p.parents]:
                if part.is_symlink():raise ValueError('Test symlinks forbidden')
                if part==root:break
            if not p.is_file() or p.stat().st_size!=r[field+'_bytes']:raise ValueError('Input file absent or size changed')
            if verify_bytes:verify_file(p,r[field+'_sha256'])
    sensors={tr.rsplit('/',1)[0]+'/sensor_records.hdf5' for tr in EXPECTED}
    if set(manifest['sensor_sha256'])!=sensors:raise ValueError('Wrong sensor records')
    for name,digest in manifest['sensor_sha256'].items():verify_file(root/name,digest)

def freeze(locations_path,checks_path):
    directory=ROOT/STUDY
    if (directory/'PROTOCOL.json').exists() or (directory/'EXPOSURE.json').exists():raise FileExistsError('Never refreeze an existing study')
    locations=read(locations_path);root=Path(locations['dataset_root']).absolute();prep=directory/'prepared'
    receipt=read(prep/'PREPARED.json');checks=read(checks_path)
    if checks.get('passed') is not True or checks.get('source_sha256')!=source_hashes():raise ValueError('Focused checks are not current')
    verify_file(prep/'manifest.json.gz',receipt['manifest_sha256']);verify_file(directory/'INVENTORY.json',receipt['inventory_sha256'])
    if receipt['status']!='verified_not_exposed' or receipt['decoder_access'] is not False or receipt['leakage']['passed'] is not True:raise ValueError('Preparation not complete')
    manifest=json.loads(gzip.decompress((prep/'manifest.json.gz').read_bytes()));verify_manifest(manifest,root,True)
    entries=model_entries(locations)
    for entry in entries.values():verify_file(entry['weights'],entry['sha256'])
    record=compatibility(entries);write_new(directory/'COMPATIBILITY.json',record)
    validation=read(ROOT/VALIDATION)
    models={key:{k:v for k,v in entry.items() if k not in ('weights','metadata')} for key,entry in entries.items()}
    for key in ['A','Original']:
        for field in ['weights','metadata']:models[key][field]=Path(entries[key][field]).relative_to(ROOT).as_posix()
    models['KD']['weights']='external:kd_checkpoint'
    p={'version':'independent_reserved_test_v3','status':'frozen_ready_for_explicit_test','frozen_utc':now(),'study_path':STUDY,
       'authorization':'User attachment 45ce2504-1897-48d2-91b8-15d6eff476d1: ONE verified frozen final study; no training/search/automatic retry.',
       'reference_specification':'results/final_delivery_20261029T173715Z/FINAL_STUDY_SPECIFICATION.json',
       'manifest_path':STUDY+'/prepared/manifest.json.gz','manifest_sha256':receipt['manifest_sha256'],
       'prepared_sha256':file_sha256(prep/'PREPARED.json'),'inventory_sha256':file_sha256(directory/'INVENTORY.json'),
       'test_realizations':EXPECTED,'sensor_sha256':manifest['sensor_sha256'],'location_config_sha256':file_sha256(locations_path),
       'validation_protocol':VALIDATION,'validation_protocol_sha256':file_sha256(ROOT/VALIDATION),
       'v2':validation['v2'],'budget_fraction':validation['budget_fraction'],'budget_rule':validation['budget_rule'],
       'matrix':[list(x) for x in MATRIX],'primary':'A_v2 versus A_budget_fixed',
       'secondary':'A policies versus A_every; Original_every and KD_every versus A_every. No post-test selection.',
       'models':models,'compatibility_sha256':file_sha256(directory/'COMPATIBILITY.json'),
       'data':DATA,'device':'cpu','torch_threads':2,'precision':'FP32 eval, batch 1; no AMP/TF32; seed 42',
       'metrics':{'pooling':'Additive float64 valid-pixel sums, sqrt(pooled SSE/count); no frame-average RMSE',
                  'delta':'strict max(P/GT,GT/P) < 1.25**k for k=1,2,3','near':'valid GT <10 m',
                  'diagnostics':'signed P-GT bias, floor P<=0.101 on valid/near pixels, near P>=10 fraction',
                  'nonfinite_predictions':'fatal anywhere, never silently masked','undefined':'JSON null'},
       'events':{'definition':'valid GT p10 <8 m and (previous p10 >=8 or drop >=2 m); reset per trajectory',
                 'delay':'frames until next fresh inference in same trajectory; retain right censoring; no events -> null rates',
                 'scoring_only':True},
       'causal_inputs':'Current RGB versus last inferred RGB; IMU at floor(i*100/25), held GPS floor(i/25); last prediction risk and age. No GT/masks/future inputs.',
       'timing':{'execution':'single chronological traversal; rotate seven-policy execution order by global frame index modulo 7',
                 'warmup':'10 synthetic zero-image forwards per model before exposure, excluded and counted separately',
                 'inference':'synchronized transfer-in + model forward + transfer-out, actual calls for every policy; no prediction lookup cache',
                 'scheduler':'reset/decision and fresh risk-cache update; compute = inference + scheduler',
                 'shared':'RGB/depth file I/O and decode/resize measured once per frame; shared among policies',
                 'excluded':'model/setup/warmups, offline metrics/events and CSV writes excluded from per-policy compute',
                 'wall':'whole traversal wall time includes shared preparation, all seven policies, scoring and logging; not a single-policy end-to-end benchmark',
                 'repetitions':1,'limitations':'CPU-only available in this runtime; OS cache/load uncontrolled; no energy/battery measurement'},
       'known_limits':validation['known_limits']+['Four reserved simulation paths; shared scenery possible; one seed and one timing pass; independent test may reverse validation rankings.'],
       'source_sha256':source_hashes(),'checks_sha256':file_sha256(checks_path),'no_retry_or_resume':True}
    write_new(directory/'PROTOCOL.json',p)
    digest=file_sha256(directory/'PROTOCOL.json');(directory/'PROTOCOL.sha256').write_text(digest+'  PROTOCOL.json\n')
    return p,digest

def require_unused(directory):
    if (directory/'EXPOSURE.json').exists() or (directory/'run').exists():raise FileExistsError('Study already exposed or output exists; no retry/resume')

def validate_protocol(p,path,digest,locations_path):
    verify_file(path,digest)
    if p.get('status')!='frozen_ready_for_explicit_test' or p.get('version')!='independent_reserved_test_v3':raise ValueError('Blocked or unknown protocol')
    if p['study_path']!=STUDY or p['test_realizations']!=EXPECTED or p['matrix']!=[list(x) for x in MATRIX]:raise ValueError('Comparison or reserved identity changed')
    if p['data']!=DATA or p['device']!='cpu' or p['torch_threads']!=2:raise ValueError('Numerical contract changed')
    verify_file(locations_path,p['location_config_sha256']);verify_file(ROOT/p['validation_protocol'],p['validation_protocol_sha256'])
    validation=read(ROOT/p['validation_protocol'])
    if p['v2']!=validation['v2'] or p['budget_fraction']!=validation['budget_fraction']:raise ValueError('Policy retuning prohibited')
    for name,h in p['source_sha256'].items():verify_file(ROOT/name,h)
    if set(p['source_sha256'])!=set(SOURCES):raise ValueError('Incomplete source identity')
    verify_file(ROOT/p['manifest_path'],p['manifest_sha256'])
    directory=ROOT/STUDY
    for name,key in [('COMPATIBILITY.json','compatibility_sha256'),('INVENTORY.json','inventory_sha256'),('prepared/PREPARED.json','prepared_sha256'),('CHECKS.json','checks_sha256')]:verify_file(directory/name,p[key])
    expected=model_entries(read(locations_path))
    for key,entry in expected.items():
        if p['models'][key]['sha256']!=entry['sha256'] or p['models'][key]['selected_epoch']!=entry['selected_epoch']:raise ValueError('Model selection changed')

# Metrics
class Scores:
    def __init__(self):
        self.depth=DepthMetricAccumulator();self.bias=0.;self.near_bias=0.;self.floor=0;self.near_floor=0;self.near_far=0
    def update(self,p,g,v):
        if p.shape!=g.shape or not torch.isfinite(p).all() or not (p>=.1).all():raise ValueError('Invalid full prediction; study stops')
        self.depth.update(p,g,v);pred=p[v].double();gt=g[v].double();near=gt<10
        self.bias+=float((pred-gt).sum());self.near_bias+=float((pred[near]-gt[near]).sum())
        self.floor+=int((pred<=.101).sum());self.near_floor+=int((pred[near]<=.101).sum());self.near_far+=int((pred[near]>=10).sum())
    def compute(self):
        n=self.depth.valid_count;k=self.depth.near_count
        return clean(self.depth.compute()|{'near_valid_pixels':k,'bias_m':self.bias/n if n else None,'near_bias_m':self.near_bias/k if k else None,
                     'valid_floor_count':self.floor,'valid_floor_rate':self.floor/n if n else None,'near_floor_rate':self.near_floor/k if k else None,
                     'near_prediction_ge_10m_rate':self.near_far/k if k else None})

def sync(device):
    if device=='cuda':torch.cuda.synchronize()

# Inference
def forward(model,x,device):
    sync(device);start=time.perf_counter();xx=x[None].to(device);sync(device);incoming=time.perf_counter()-start
    start=time.perf_counter()
    with torch.amp.autocast(device,enabled=False):pred=model(xx)
    sync(device);model_s=time.perf_counter()-start;start=time.perf_counter();pred=pred[0].cpu();sync(device);outgoing=time.perf_counter()-start
    if not torch.isfinite(pred).all() or not (pred>=.1).all():raise ValueError('Nonfinite or below-floor model output')
    return pred,incoming,model_s,outgoing

def update_progress(out,obj):
    tmp=out/'progress.tmp';tmp.write_text(json.dumps(obj,allow_nan=False));tmp.replace(out/'progress.json')

# Evaluation
def evaluate_stream(records,read_sample,models,protocol,out):
    states={name:{'scheduler':PresentationScheduler(protocol['v2']),'previous':None,'source':None,'scores':Scores(),'per':defaultdict(Scores),'logs':[]} for name,_,_ in MATRIX}
    lengths=CounterLike(records);fixed={tr:{p:fixed_positions(p,n,**protocol['budget_fraction']) for p in ['every','fixed2','fixed5','budget_fixed']} for tr,n in lengths.items()}
    handles={};writers={};shared_io=shared_decode=0.;lasttr=None;lastgt=math.inf;started=time.perf_counter()
    try:
        with torch.inference_mode():
            for i,r in enumerate(records):
                x,g,v,io_s,decode_s=read_sample(r);shared_io+=io_s;shared_decode+=decode_s
                tr=r['trajectory_id'];new=tr!=lasttr
                if new:lasttr=tr;lastgt=math.inf
                values=g[v];p10=float(torch.quantile(values,.1)) if values.numel() else math.inf
                event=p10<8 and (lastgt>=8 or lastgt-p10>=2);lastgt=p10
                order=MATRIX[i%len(MATRIX):]+MATRIX[:i%len(MATRIX)]
                for name,model_key,policy in order:
                    state=states[name];start=time.perf_counter()
                    if new:state['scheduler'].reset();state['previous']=None;state['source']=None
                    if policy=='v2':decision=state['scheduler'].decide(x,r['navigation'],state['previous']);infer=decision.infer;reason=decision.reason
                    else:infer=r['frame_index'] in fixed[tr][policy];reason='first_frame' if r['frame_index']==0 else 'fixed_interval' if infer else 'fixed_skip'
                    scheduler_s=time.perf_counter()-start;tin=model_s=tout=0.
                    if infer:
                        state['previous'],tin,model_s,tout=forward(models[model_key],x,protocol['device']);state['source']=i
                        start=time.perf_counter()
                        if policy=='v2':state['scheduler'].update_prediction(state['previous'],i)
                        scheduler_s+=time.perf_counter()-start
                    if state['source'] is None:raise AssertionError('No first prediction')
                    age=i-state['source']
                    if policy=='v2' and age!=decision.age_after:raise AssertionError('Scheduler/source age mismatch')
                    if age>({'every':0,'fixed2':1,'budget_fixed':1,'fixed5':4,'v2':4}[policy]):raise AssertionError('Prediction-age bound')
                    pred=state['previous'];state['scores'].update(pred,g,v);state['per'][tr].update(pred,g,v)
                    pp=pred[v];predp10=float(torch.quantile(pp,.1)) if event and pp.numel() else None
                    row={'index':i,'trajectory':tr,'position':r['frame_index'],'frame_index':r['frame_index'],'infer':infer,'reason':reason,
                         'source_index':state['source'],'prediction_age':age,'critical_event':event,'gt_p10':p10 if math.isfinite(p10) else None,
                         'offline_predicted_p10_on_event':predp10,'valid_pixels':int(v.sum()),'near_valid_pixels':int((v&(g<10)).sum()),
                         'scheduler_ms':scheduler_s*1000,'inference_ms':(tin+model_s+tout)*1000,'model_ms':model_s*1000,
                         'transfer_in_ms':tin*1000,'transfer_out_ms':tout*1000,'shared_io_ms':io_s*1000,'shared_decode_ms':decode_s*1000}
                    if name not in handles:
                        handles[name]=(out/(name+'_frames.csv')).open('x',newline='');writers[name]=csv.DictWriter(handles[name],fieldnames=list(row));writers[name].writeheader()
                    writers[name].writerow(row);state['logs'].append(row)
                if (i+1)%200==0 or i+1==len(records):
                    for f in handles.values():f.flush()
                    update_progress(out,{'frames_completed':i+1,'frames_total':len(records),'trajectory':tr,'elapsed_s':time.perf_counter()-started})
                    print(json.dumps({'frames':i+1,'total':len(records),'elapsed_s':round(time.perf_counter()-started,1)}),flush=True)
    finally:
        for f in handles.values():f.close()
    results={}
    for name,model_key,policy in MATRIX:
        state=states[name];result=summarize(state['logs'],{tr:a.compute() for tr,a in state['per'].items()});result['overall']['metrics']=state['scores'].compute()
        for scope,part in [('overall',result['overall']),*result['per_trajectory'].items()]:
            rows=state['logs'] if scope=='overall' else [r for r in state['logs'] if r['trajectory']==scope]
            for key in ['model','transfer_in','transfer_out']:part[key+'_seconds']=sum(r[key+'_ms'] for r in rows)/1000
            part['compute_seconds']=part['inference_seconds']+part['scheduler_seconds']
            times=[r['model_ms'] for r in rows if r['infer']];part['model_call_median_ms']=float(np.median(times)) if times else None
        results[name]=result;write_new(out/(name+'.json'),result)
    counts={(v['overall']['metrics']['valid_pixels'],v['overall']['metrics']['near_valid_pixels']) for v in results.values()}
    if len(counts)!=1:raise AssertionError('Models/policies have unequal evaluation counts')
    for tr in lengths:
        if len({(v['per_trajectory'][tr]['metrics']['valid_pixels'],v['per_trajectory'][tr]['metrics']['near_valid_pixels']) for v in results.values()})!=1:raise AssertionError('Per-path count mismatch')
    return results,{'shared_io_seconds':shared_io,'shared_decode_seconds':shared_decode,'matrix_wall_seconds':time.perf_counter()-started}

def CounterLike(records):
    from collections import Counter
    return dict(Counter(r['trajectory_id'] for r in records))

def run(locations_path,protocol_sha,opt_in=False):
    if not opt_in:raise PermissionError('Explicit --confirm-new-test-exposure required before any test access')
    directory=ROOT/STUDY;require_unused(directory);path=directory/'PROTOCOL.json';p=read(path)
    validate_protocol(p,path,protocol_sha,locations_path)
    locations=read(locations_path);root=Path(locations['dataset_root']).absolute();m=json.loads(gzip.decompress((ROOT/p['manifest_path']).read_bytes()))
    verify_manifest(m,root,False)
    if m['sensor_sha256']!=p['sensor_sha256']:raise ValueError('Manifest and protocol sensors differ')
    entries=model_entries(locations);record=read(directory/'COMPATIBILITY.json')
    for name,item in record['files'].items():verify_file(ROOT/name,item['current_sha256'])
    torch.set_num_threads(2);e.seed_everything(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={};digests={}
    for key,entry in entries.items():
        model,cfg=load_model(entry,p['device'])
        if any(cfg['data'].get(k)!=v for k,v in DATA.items()):raise ValueError('Model preprocessing differs')
        models[key]=model;digests[key]=e.state_digest(model)
    with torch.inference_mode():
        for model in models.values():
            for _ in range(10):forward(model,torch.zeros(3,192,320),p['device'])
    sensors=[root/s for s in m['sensor_sha256']]
    guard=ValidationGuard(root,[SimpleNamespace(**r) for r in m['records']],sensors);sys.addaudithook(guard.check)
    require_unused(directory)
    # Exposure
    write_new(directory/'EXPOSURE.json',{'status':'exposure_started','started_utc':now(),'protocol_sha256':protocol_sha,
              'manifest_sha256':p['manifest_sha256'],'models':{k:v['sha256'] for k,v in entries.items()},'no_retry_or_resume':True})
    out=directory/'run';out.mkdir()
    def sample(r):
        start=time.perf_counter();rgb=(root/r['rgb_path']).read_bytes();depth=(root/r['depth_path']).read_bytes();io_s=time.perf_counter()-start
        if hashlib.sha256(rgb).hexdigest()!=r['rgb_sha256'] or hashlib.sha256(depth).hexdigest()!=r['depth_sha256']:raise ValueError('Test input changed since freeze')
        start=time.perf_counter();x,g,v=decode_pair(rgb,depth,DATA);decode_s=time.perf_counter()-start
        return x,g,v,io_s,decode_s
    try:
        results,timing=evaluate_stream(m['records'],sample,models,p,out)
        if {k:e.state_digest(v) for k,v in models.items()}!=digests:raise AssertionError('Model weights or BN state changed')
        if source_hashes()!=p['source_sha256']:raise AssertionError('Source changed during study')
        rows=[]
        for name,result in results.items():
            for scope,part in [('pooled',result['overall']),*result['per_trajectory'].items()]:
                rows.append({'scope':scope,'comparison':name,**part['metrics'],'frames':part['frames'],'calls':part['inference_calls'],
                    'skip_rate':part['skip_rate'],'mean_age':part['mean_prediction_age'],'max_age':part['max_prediction_age'],
                    'events':part['critical_events']['count'],'max_delay':part['critical_events']['max_observed_delay_frames'],
                    'censored':part['critical_events']['right_censored_count'],'inference_seconds':part['inference_seconds'],
                    'scheduler_seconds':part['scheduler_seconds'],'compute_seconds':part['compute_seconds'],'model_call_median_ms':part['model_call_median_ms']})
        with (directory/'comparison.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        write_new(directory/'results.json',results)
        write_new(directory/'timing.json',{'shared':timing,'scope':p['timing'],'hardware':{'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'device':p['device'],'torch_threads':torch.get_num_threads()}})
        artifacts={x.relative_to(directory).as_posix():file_sha256(x) for x in [directory/'comparison.csv',directory/'results.json',directory/'timing.json',*sorted(out.glob('*.csv')),*sorted(out.glob('A_*.json')),out/'Original_every.json',out/'KD_every.json']}
        write_new(directory/'COMPLETION.json',{'status':'COMPLETED','completed_utc':now(),'protocol_sha256':protocol_sha,'manifest_sha256':p['manifest_sha256'],
                  'frames':len(m['records']),'comparison_rows':len(rows),'policies':list(results),'equal_counts':True,'model_state_before_and_after':digests,
                  'warmup_calls_per_model':10,'artifacts_sha256':artifacts,'no_training_or_selection':True})
    except BaseException as error:
        write_new(directory/'FAILED.json',{'status':'INCOMPLETE','failed_utc':now(),'error_type':type(error).__name__,'message':str(error),'protocol_sha256':protocol_sha,'no_retry_or_resume':True})
        raise

def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['freeze','run']);a.add_argument('--locations',type=Path,required=True)
    a.add_argument('--checks',type=Path);a.add_argument('--protocol-sha256');a.add_argument('--confirm-new-test-exposure',action='store_true');p=a.parse_args(argv)
    if p.stage=='freeze':
        if not p.checks:a.error('--checks required')
        _,digest=freeze(p.locations,p.checks);print('Frozen protocol SHA-256:',digest)
    else:
        if not p.protocol_sha256:a.error('--protocol-sha256 required')
        run(p.locations,p.protocol_sha256,p.confirm_new_test_exposure)

if __name__=='__main__':main()
