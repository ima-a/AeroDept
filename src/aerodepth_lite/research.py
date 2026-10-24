"""Portable frozen validation/scheduler reproduction; no training or test evaluation."""
from __future__ import annotations
import argparse,csv,json,math,os,sys,hashlib,time,platform
from pathlib import Path
from collections import Counter
from dataclasses import asdict,replace
from unittest.mock import patch
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
from aerodepth_lite.data import MidAirDataset,collate_samples,load_manifest
from aerodepth_lite.model import build_model,load_checkpoint
from aerodepth_lite.metrics import DepthMetricAccumulator
import aerodepth_lite.scheduler as sm
from aerodepth_lite.replay import _critical_depth
from aerodepth_lite.repro import seed_everything
from aerodepth_lite.inference import load_inference_model
from aerodepth_lite.scheduler_v2 import SchedulerV2,thumbnail
OUT=PROJECT=ROOT=RAW_MANIFEST=CONFIG=WEIGHTS=METADATA=PUBLISHED=None
DEVICE=torch.device('cpu')
STUDY='validation'

def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def save(name,obj):
 (OUT/name).parent.mkdir(parents=True,exist_ok=True)
 (OUT/name).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')
def read(name):return json.loads((OUT/name).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def protected():
 files=[CONFIG,RAW_MANIFEST,WEIGHTS,METADATA]+list(Path(__file__).parent.glob('*.py'))
 return {str(p):sha(p) for p in files}

def csvsave(name,rows):
 if not rows:return
 (OUT/name).parent.mkdir(parents=True,exist_ok=True)
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
def setup():
 torch.set_num_threads(2);seed_everything(42)
 cfg=yaml.safe_load(CONFIG.read_text());info=json.loads(METADATA.read_text())
 if sha(RAW_MANIFEST)!=info['manifest_sha256']:raise ValueError('Frozen manifest identity mismatch')
 if sha(CONFIG)!=info['training_config_sha256']:raise ValueError('Frozen training configuration mismatch')
 records,_=load_manifest(RAW_MANIFEST,'val')
 assert len(records)==4410 and len(set(r.trajectory_id for r in records))==2
 if STUDY=='scheduler-v2':
  import h5py
  transformed=[]
  for tr in sorted(set(r.trajectory_id for r in records)):
   selected=[r for r in records if r.trajectory_id==tr];parent,group=tr.rsplit('/',1)
   with h5py.File(ROOT/parent/'sensor_records.hdf5','r') as h:
    gps=h[group+'/gps/velocity'];frequency=float(gps.attrs['sampling_frequency']);values=np.array(gps[:])
   for ordinal,r in enumerate(selected):
    sample=min(int(ordinal*frequency/25),len(values)-1)
    nav={k:r.navigation[k] for k in ['ax','ay','az','wx','wy','wz'] if k in r.navigation}
    nav.update({k:float(v) for k,v in zip(['gps_vx','gps_vy','gps_vz'],values[sample]) if np.isfinite(v)})
    transformed.append(replace(r,navigation=nav))
  records=transformed
 ds=MidAirDataset(ROOT,records,cfg['data'],False)
 return cfg,records,ds

def model_load(cfg):
 if DEVICE.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
 model,info=load_inference_model(WEIGHTS,METADATA,DEVICE)
 return model

def sync():
 if DEVICE.type=='cuda':torch.cuda.synchronize()

def loader(ds):return DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate_samples)

def prepare():
 cfg,records,ds=setup()
 save('preserved_before.json',protected())
 save('protocol.json',{'split':'val','frames':4410,'precision':'FP32 eval, batch1; unchanged preprocessing/masks, near GT<10m; no test access',
 'audit_tolerances_absolute':{'absrel':1e-5,'rmse':1e-3,'delta1':1e-5,'delta2':1e-5,'delta3':1e-5,'near_absrel':1e-5,'near_rmse':1e-3,'valid_pixels':0,'near_valid_pixels':0},
 'timing':'Actual synchronized device inference includes H2D/model/D2H; 10 excluded warmups each policy; CPU scheduler overhead included. Search cache time is OFFLINE ONLY. Single pass; no energy/battery measurement.',
 'causality':'V1 uses original privileged simulated GT velocity plus simulated IMU and consecutive RGB. V2 uses held causal simulated GPS velocity plus separately normalized simulated IMU proxies, current RGB vs last inference, previous predicted risk and age. GT/masks only offline scoring. Sensor latency not modeled.',
 'critical_event':'Offline valid GT p10<8m entry or >=2m drop within near zone, reset per trajectory. No fresh inference on event = missed update, not missed hazard detection. Terminal delays censored; zero-event rates undefined.'})
 save('environment.json',{'python':platform.python_version(),'torch':torch.__version__,'device':str(DEVICE),'cuda_build':torch.version.cuda,'seed':42})
 print('Prepared frozen validation reproduction',flush=True)

@torch.inference_mode()
def live(name,cfg_override=None,make_cache=False,matched=None):
 cfg,records,ds=setup();model=model_load(cfg);scfg=cfg_override or cfg['scheduler']
 if (OUT/f'{name}.json').exists():raise RuntimeError('Refusing to overwrite completed policy '+name)
 warm=ds[0]['rgb'].unsqueeze(0).to(DEVICE)
 for _ in range(10):model(warm)
 sync();del warm
 N=len(records);shape=(N,1,*cfg['data']['image_size'])
 if make_cache and STUDY=='scheduler-v2':graycache=np.lib.format.open_memmap(OUT/'gray.npy',mode='w+',dtype='float32',shape=(N,36,64))
 if make_cache:
  predcache=np.lib.format.open_memmap(OUT/'predictions.npy',mode='w+',dtype='float32',shape=shape)
  gtcache=np.lib.format.open_memmap(OUT/'targets.npy',mode='w+',dtype='float32',shape=shape)
  maskcache=np.lib.format.open_memmap(OUT/'valid.npy',mode='w+',dtype='bool',shape=shape)
 else:predcache=np.load(OUT/'predictions.npy',mmap_mode='r')
 logs=[];features=[];scoring=[];accs={};lasttr=None;previous=None;src=None;age=0;pos=0;previous_rgb=None;last_gt=math.inf
 sched=sm.AdaptiveScheduler(scfg);wall=time.perf_counter();maxdiff=0
 for i,b in enumerate(loader(ds)):
  r=b['record'][0];tr=r['trajectory_id'];rgb=b['rgb'][0];g=b['depth'][0];v=b['valid'][0]
  if tr!=lasttr:
   sched.reset();previous=None;src=None;age=0;pos=0;previous_rgb=None;last_gt=math.inf;lasttr=tr;accs[tr]=DepthMetricAccumulator()
  t=time.perf_counter()
  if name in ['adaptive_existing','adaptive_selected']:
   d=asdict(sched.decide(rgb,r.get('navigation',{}),previous));infer=d['infer']
  else:
   infer=pos in matched[tr] if matched is not None else pos % {'every':1,'fixed2':2,'fixed5':5}[name]==0
   d={'infer':infer,'reason':'first_frame' if pos==0 else 'fixed_interval' if infer else 'fixed_skip','age_before':age,'age_after':0 if infer else age+1}
  overhead=time.perf_counter()-t;latency=0
  if infer:
   sync();t=time.perf_counter();previous=model(rgb.unsqueeze(0).to(DEVICE)).cpu()[0];sync();latency=time.perf_counter()-t;src=i
   assert torch.isfinite(previous).all() and (previous>=.1).all()
   if not make_cache:maxdiff=max(maxdiff,float(np.max(np.abs(previous.numpy()-predcache[i]))))
  assert previous is not None
  age=i-src
  assert age==d['age_after']
  accs[tr].update(previous,g,v,10.)
  critical_depth=_critical_depth(g,v);event=critical_depth<8 and (last_gt>=8 or last_gt-critical_depth>=2);last_gt=critical_depth
  pred_p10=_critical_depth(previous,v) if event else math.nan # offline score ONLY after decision
  logs.append({'index':i,'trajectory':tr,'position':pos,'frame_index':r['frame_index'],'timestamp':r.get('timestamp'),'source_index':src,**d,'prediction_age':age,'scheduler_ms':overhead*1000,'inference_ms':latency*1000,'critical_event':event,'offline_predicted_p10_on_event':pred_p10})
  if make_cache:
   predcache[i]=previous.numpy();gtcache[i]=g.numpy();maskcache[i]=v.numpy()
   if STUDY=='scheduler-v2':graycache[i]=thumbnail(rgb)
   risk,available=sm.near_field_risk(previous,10)
   p=previous.squeeze();roi=p[p.shape[0]//3:,p.shape[1]//4:3*p.shape[1]//4]
   features.append({'index':i,'trajectory':tr,'position':pos,'frame_index':r['frame_index'],'navigation':r.get('navigation',{}),'visual_raw':sm.visual_change(previous_rgb,rgb),'fresh_risk':risk,'risk_available':available,'floor_all_fraction':float((p<=.101).float().mean()),'floor_roi_fraction':float((roi<=.101).float().mean()),'motion_frozen':sm.motion_score(r.get('navigation',{}),5)[0]})
   scoring.append({'index':i,'trajectory':tr,'position':pos,'gt_p10':critical_depth,'critical_event':event})
   previous_rgb=rgb.clone()
  pos+=1
  if (i+1)%500==0:print(f'{name}: {i+1}/{N} frames; elapsed {time.perf_counter()-wall:.1f}s',flush=True)
 wall=time.perf_counter()-wall
 if make_cache:
  predcache.flush();gtcache.flush();maskcache.flush();
  if STUDY=='scheduler-v2':graycache.flush()
  save('decision_features.json',features);save('offline_scoring_features.json',scoring)
 result=summarize(logs,{tr: a.compute()|{'near_valid_pixels':a.near_count} for tr,a in accs.items()})
 result.update({'mode':'actual FP32 inference','wall_seconds_including_IO_scoring_instrumentation':wall,'max_prediction_difference_from_FP32_cache':maxdiff,'scheduler_config':scfg,'warmup_calls_excluded':10})
 # Global from exact additive accumulator fields, not means of trajectory metrics.
 total=DepthMetricAccumulator()
 for a in accs.values():
  for key in vars(total):setattr(total,key,getattr(total,key)+getattr(a,key))
 result['overall']['metrics']=total.compute()|{'near_valid_pixels':total.near_count}
 csvsave(f'{name}_frames.csv',logs);save(f'{name}.json',result)
 if make_cache:
  audit=json.loads((PUBLISHED/'expanded_validation/validation_metrics.json').read_text());tol=read('protocol.json')['audit_tolerances_absolute'];diff={};okay=True
  for scope,actual,expected in [('overall',result['overall']['metrics'],audit['metrics'])]+[(tr,result['per_trajectory'][tr]['metrics'],v) for tr,v in audit['per_trajectory'].items()]:
   diff[scope]={k:abs(actual[k]-expected[k]) for k in tol};okay &= all(diff[scope][k]<=tol[k] for k in tol)
  save('reference_agreement.json',{'passed':bool(okay),'absolute_differences':diff,'tolerances':tol,'audit_batch_size':4,'replay_batch_size':1})
  assert okay,'Material disagreement: stop before other policies/search'
 else:assert maxdiff<1e-4,'Actual model output differs materially from cache'
 print(name,json.dumps(clean(result['overall'])),flush=True)

def event_stats(rows):
 indices=[i for i,r in enumerate(rows) if r['critical_event']];delays=[];censored=[];correct=0
 for ix in indices:
  nxt=next((j for j in range(ix,len(rows)) if rows[j]['infer']),None)
  if nxt is None:censored.append({'position':rows[ix]['position'],'observed_wait_frames':len(rows)-1-ix})
  else:delays.append(nxt-ix)
  p=rows[ix].get('offline_predicted_p10_on_event')
  if p is not None and np.isfinite(p) and p<8:correct+=1
 n=len(indices);on=sum(rows[i]['infer'] for i in indices)
 return {'count':n,'fresh_on_event':on,'missed_update_rate':(n-on)/n if n else None,'observed_delays_frames':delays,'mean_observed_delay_frames':float(np.mean(delays)) if delays else None,'max_observed_delay_frames':max(delays) if delays else None,'right_censored_count':len(censored),'censored':censored,'offline_near_estimated_on_event':correct,'offline_near_estimation_rate':correct/n if n else None}

def summarize(logs,metrics):
 groups={tr:[r for r in logs if r['trajectory']==tr] for tr in metrics}
 def one(rows,m):
  n=len(rows);calls=sum(r['infer'] for r in rows)
  return {'frames':n,'inference_calls':calls,'skip_rate':1-calls/n,'mean_prediction_age':float(np.mean([r['prediction_age'] for r in rows])),'max_prediction_age':max(r['prediction_age'] for r in rows),'inference_reasons':dict(Counter(r['reason'] for r in rows if r['infer'])),'all_reasons':dict(Counter(r['reason'] for r in rows)),'scheduler_seconds':sum(r.get('scheduler_ms',0) for r in rows)/1000,'inference_seconds':sum(r.get('inference_ms',0) for r in rows)/1000,'critical_events':event_stats(rows),'metrics':m}
 per={tr:one(rows,metrics[tr]) for tr,rows in groups.items()}
 overall=one(logs,{})
 # Never allow critical-update matching to cross trajectory boundaries.
 es=[v['critical_events'] for v in per.values()];n=sum(e['count'] for e in es);delays=sum([e['observed_delays_frames'] for e in es],[])
 overall['critical_events']={'count':n,'fresh_on_event':sum(e['fresh_on_event'] for e in es),'missed_update_rate':sum(e['count']-e['fresh_on_event'] for e in es)/n if n else None,'observed_delays_frames':delays,'mean_observed_delay_frames':float(np.mean(delays)) if delays else None,'max_observed_delay_frames':max(delays) if delays else None,'right_censored_count':sum(e['right_censored_count'] for e in es),'offline_near_estimated_on_event':sum(e['offline_near_estimated_on_event'] for e in es),'offline_near_estimation_rate':sum(e['offline_near_estimated_on_event'] for e in es)/n if n else None}
 return {'overall':overall,'per_trajectory':per}

def cached_decisions(features,cfg):
 """ONLY decision inputs. Risk is looked up at the last inferred source, not current frame."""
 if STUDY=='scheduler-v2':
  scheduler=SchedulerV2(cfg);gray=np.load(OUT/'gray.npy',mmap_mode='r');source=None;lasttr=None;logs=[]
  for f in features:
   if f['trajectory']!=lasttr:scheduler.reset();source=None;lasttr=f['trajectory']
   risk,available=(features[source]['fresh_risk'],features[source]['risk_available']) if source is not None else (0.,False)
   d=asdict(scheduler.decide_features(gray[f['index']],f['navigation'],risk,available))
   if d['infer']:source=f['index']
   logs.append({'index':f['index'],'trajectory':f['trajectory'],'position':f['position'],'frame_index':f['frame_index'],'source_index':source,**d,'prediction_age':f['index']-source})
  return logs
 sched=sm.AdaptiveScheduler(cfg);logs=[];lasttr=None;source=None;dummy=torch.zeros(3,2,2)
 for f in features:
  if f['trajectory']!=lasttr:sched.reset();source=None;lasttr=f['trajectory']
  risk=(features[source]['fresh_risk'],features[source]['risk_available']) if source is not None else (0.,False)
  with patch.object(sm,'visual_change',return_value=f['visual_raw']),patch.object(sm,'near_field_risk',return_value=risk):
   d=asdict(sched.decide(dummy,f['navigation'],None if source is None else dummy))
  if d['infer']:source=f['index']
  logs.append({'index':f['index'],'trajectory':f['trajectory'],'position':f['position'],'frame_index':f['frame_index'],'source_index':source,**d,'prediction_age':f['index']-source})
 return logs

@torch.inference_mode()
def score_cached(logs):
 p=np.load(OUT/'predictions.npy',mmap_mode='r');g=np.load(OUT/'targets.npy',mmap_mode='r');v=np.load(OUT/'valid.npy',mmap_mode='r');features=read('offline_scoring_features.json');accs={}
 for start in range(0,len(logs),32):
  rr=logs[start:start+32];ids=[r['index'] for r in rr];sources=[r['source_index'] for r in rr]
  pp=torch.from_numpy(np.array(p[sources])).to(DEVICE);gg=torch.from_numpy(np.array(g[ids])).to(DEVICE);vv=torch.from_numpy(np.array(v[ids])).to(DEVICE)
  for j,r in enumerate(rr):
   tr=r['trajectory'];accs.setdefault(tr,DepthMetricAccumulator()).update(pp[j],gg[j],vv[j],10.)
   event=features[r['index']]['critical_event'];r['critical_event']=event
   r['offline_predicted_p10_on_event']=_critical_depth(pp[j],vv[j]) if event else math.nan
 result=summarize(logs,{tr:a.compute()|{'near_valid_pixels':a.near_count} for tr,a in accs.items()})
 total=DepthMetricAccumulator()
 for a in accs.values():
  for k in vars(total):setattr(total,k,getattr(total,k)+getattr(a,k))
 result['overall']['metrics']=total.compute()|{'near_valid_pixels':total.near_count}
 return result

def declare():
 path=PUBLISHED/('scheduler_v2/plan.json' if STUDY=='scheduler-v2' else 'scheduler_v1/search_plan.json')
 plan=json.loads(path.read_text());plan['reproduction_note']='Published bounded configurations loaded unchanged, before search; no new tuning.'
 save('search_plan.json',plan)
 print('Loaded',len(plan['configurations']),'frozen configurations',flush=True)

def search():
 features=read('decision_features.json');plan=read('search_plan.json');reference=read('every.json');results=[]
 assert not (OUT/'selection.json').exists()
 for item in plan['configurations']:
  t=time.perf_counter();logs=cached_decisions(features,item['config']);r=score_cached(logs);r.update({'mode':'cached predictions, OFFLINE SEARCH ONLY','config':item['config'],'cached_search_seconds':time.perf_counter()-t})
  violations=[];worst=0;callfractions=[];delays=[]
  for tr,x in r['per_trajectory'].items():
   ref=reference['per_trajectory'][tr]['metrics']
   for key,bound in [('absrel',1.05),('rmse',1.05),('near_absrel',1.10),('near_rmse',1.10)]:
    ratio=x['metrics'][key]/ref[key];worst=max(worst,ratio);violations.append(max(0,ratio/bound-1))
   e=x['critical_events'];violations.extend([e['right_censored_count'],max(0,(e['max_observed_delay_frames'] or 0)-2)])
   callfractions.append(x['inference_calls']/x['frames']);delays.extend(e['observed_delays_frames'])
  r.update({'id':item['id'],'eligible':max(violations)==0,'violation_sum':sum(violations),'selection_rank':[max(callfractions),worst,float(np.mean(delays)) if delays else 0,item['id']]})
  csvsave(f"search/{item['id']}_frames.csv",logs);save(f"search/{item['id']}.json",r);results.append(r)
  print(item['id'],'calls',r['overall']['inference_calls'],'eligible',r['eligible'],'seconds',round(r['cached_search_seconds'],1),flush=True)
 eligible=[r for r in results if r['eligible']];chosen=min(eligible,key=lambda r:r['selection_rank']) if eligible else min(results,key=lambda r:(r['violation_sum'],r['selection_rank']))
 save('selection.json',{'id':chosen['id'],'eligible':chosen['eligible'],'config':chosen['config'],'selection_rank':chosen['selection_rank'],'exploratory_validation_only':True})
 matches={}
 for tr,x in chosen['per_trajectory'].items():
  N=x['frames'];K=x['inference_calls'];matches[tr]=[(k*N+K-1)//K for k in range(K)]
 save('matched_fixed_positions.json',matches)
 save('search_results.json',results)
 print('SELECTED',chosen['id'],chosen['selection_rank'],flush=True)

def finish():
 before=read('preserved_before.json');after=protected();assert before==after,'Input artifacts changed during reproduction'
 save('preservation.json',{'unchanged':True,'files_checked':len(before)})
 save('hashes.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='hashes.json'})

def report():
 rows=[];results={}
 for name in ['every','fixed2','fixed5','adaptive_existing','adaptive_selected','fixed_matched']:
  if not (OUT/f'{name}.json').exists():continue
  results[name]=read(name+'.json')
  for tr,x in [('Overall',results[name]['overall'])]+list(results[name]['per_trajectory'].items()):
   ev=x['critical_events'];rows.append({'policy':name,'scope':tr,**x['metrics'],'calls':x['inference_calls'],'skip_rate':x['skip_rate'],'mean_age':x['mean_prediction_age'],'max_age':x['max_prediction_age'],'critical_events':ev['count'],'fresh_on_event':ev['fresh_on_event'],'missed_update_rate':ev['missed_update_rate'],'max_delay':ev['max_observed_delay_frames'],'inference_seconds':x['inference_seconds'],'scheduler_seconds':x['scheduler_seconds']})
 csvsave('comparison.csv',rows)
 os.environ.setdefault('MPLCONFIGDIR','/tmp/aerodepth-mpl')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,5))
 for ax,tr in zip(axes,results['every']['per_trajectory']):
  for name,r in results.items():
   x=r['per_trajectory'][tr];ax.scatter(x['inference_calls'],x['metrics']['near_rmse'],label=name)
  ax.set(title=tr,xlabel='Inference calls',ylabel='Near-field RMSE m');ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'quality_vs_calls.png',dpi=150);plt.close(fig)
 (OUT/'REPORT.md').write_text('# Validation reproduction\n\nSee comparison.csv, per-policy JSON and full per-frame logs. All selection is exploratory validation only. Cached search times are not live inference performance. No test data, energy or battery evaluation.\n')

def main(argv=None):
 global OUT,PROJECT,ROOT,RAW_MANIFEST,CONFIG,WEIGHTS,METADATA,PUBLISHED,DEVICE,STUDY
 parser=argparse.ArgumentParser(description='Reproduce frozen full validation and bounded scheduler experiments with published weights. Never evaluates test data.')
 parser.add_argument('--dataset-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--study',choices=['validation','scheduler-v1','scheduler-v2'],default='validation');parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
 parser.add_argument('--config',type=Path,default=Path('models/training_config.yaml'));parser.add_argument('--manifest',type=Path,default=Path('results/expanded_validation/manifest.json.gz'));parser.add_argument('--weights',type=Path,default=Path('models/mobilenet_v3_small_epoch4.state_dict.pt'));parser.add_argument('--metadata',type=Path,default=Path('models/mobilenet_v3_small_epoch4.json'));parser.add_argument('--results-root',type=Path,default=Path('results'))
 a=parser.parse_args(argv);OUT=a.output.resolve();ROOT=a.dataset_root.resolve();PROJECT=Path.cwd();CONFIG=a.config.resolve();WEIGHTS=a.weights.resolve();METADATA=a.metadata.resolve();PUBLISHED=a.results_root.resolve();DEVICE=torch.device(a.device);STUDY=a.study
 if OUT==ROOT or ROOT in OUT.parents:raise ValueError('Outputs cannot be inside the read-only dataset')
 if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Choose a new output directory; existing evidence is preserved')
 OUT.mkdir(parents=True,exist_ok=True)
 import gzip
 payload=gzip.decompress(a.manifest.read_bytes()) if a.manifest.suffix=='.gz' else a.manifest.read_bytes();RAW_MANIFEST=OUT/'manifest.json';RAW_MANIFEST.write_bytes(payload)
 if STUDY=='scheduler-v2':sm.AdaptiveScheduler=SchedulerV2
 prepare();live('every',make_cache=True)
 if STUDY!='validation':
  live('fixed2');live('fixed5')
  if STUDY=='scheduler-v1':live('adaptive_existing')
  declare();search();live('adaptive_selected',read('selection.json')['config']);live('fixed_matched',matched={k:set(v) for k,v in read('matched_fixed_positions.json').items()})
 report();finish()
 print('Completed',STUDY,'at',OUT,flush=True)

if __name__=='__main__':main()
