"""Missing KD validation diagnostics; cached original/A results, no training."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv,gc,hashlib,json,math,os,sys,time
ART=Path(__file__).resolve().parent;REPO=ART.parents[1];RES=REPO/'results'/ART.name
sys.path.insert(0,str(REPO/'src'));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from aerodepth_lite import expanded as e
from aerodepth_lite.metrics import DepthMetricAccumulator
from aerodepth_lite.model import model_contract
from reused_statistics import Stats
P=json.loads((RES/'PLAN.json').read_text());ROOT=Path(P['dataset_root']);OLD=Path(P['reuse_root']);B=Path(P['KD_bundle']);START=time.monotonic()

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def save(name,x):
 f=RES/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');t.replace(f)
def log(stage,**kw):
 x={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-START,**kw};save('status.json',x);print(json.dumps(x),flush=True)
class Guard(e.ImageReadGuard):
 def __init__(self,root,records):
  super().__init__(root,records);self.fd=os.open(ART/'validation_image_opens_v2.jsonl',os.O_CREAT|os.O_EXCL|os.O_APPEND|os.O_WRONLY,0o600)
 def check(self,event,args):
  if event not in ('open','os.listdir','os.scandir') or not args or isinstance(args[0],int):return
  path=os.path.abspath(os.fsdecode(args[0]))
  if path=='/run/media/Ima/Low storage' or path.startswith('/run/media/Ima/Low storage/'):raise PermissionError('Failed HDD reads forbidden')
  super().check(event,args)
  if event=='open' and path in self.allowed:os.write(self.fd,(json.dumps({'pid':os.getpid(),'path':path})+'\n').encode())
def standard(acc):return clean(acc.compute()|{'near_valid_pixels':acc.near_count})
def checkmetrics(actual,expected):
 diffs={}
 for k in ['absrel','rmse','delta1','delta2','delta3','near_absrel','near_rmse','valid_pixels','near_valid_pixels']:
  x,y=actual[k],expected[k]
  if x is None or y is None:assert x is y;diffs[k]=None;continue
  d=x-y;assert abs(d)<=(0 if k.endswith('pixels') else 1e-8),(k,x,y,d);diffs[k]=d
 return diffs

def run():
 seal=json.loads((RES/'FREEZE_V2.json').read_text())
 for name,h in seal['files_sha256'].items():assert sha(REPO/name)==h,name
 assert e.source_hashes()==P['parent_sources']
 for v in P['checkpoints'].values():assert sha(v['path'])==v['sha256']
 for name,h in P['reuse_sha256'].items():assert sha(OLD/name)==h,name
 torch.set_num_threads(2);e.seed_everything(42);assert torch.cuda.is_available()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.hub.download_url_to_file=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Downloads forbidden'))
 records,manifest=e.load_inputs(REPO/P['manifest'],REPO/P['reservations'],ROOT);val=records['val'];assert len(val)==4410
 guard=Guard(ROOT,val);sys.addaudithook(guard.check)
 for rel in ['Kite_training/sunny/color_left/trajectory_0007/000000.JPEG','Kite_training/sunny/color_left/trajectory_0004/000000.JPEG']:
  try:guard.check('open',(str(ROOT/rel),'rb',0))
  except PermissionError:pass
  else:raise AssertionError('Test read guard ineffective')
 prior=json.loads((OLD/'metrics.json').read_text());oldids=json.loads((OLD/'identities.json').read_text());completion=json.loads((B/'completion.json').read_text())
 assert completion['epochs']==10 and completion['global_step']==220490 and completion['original_gate_passed'] is False
 hist=json.loads((B/'run/history.json').read_text());assert min(hist,key=lambda h:h['validation']['overall']['absrel'])['epoch']==9
 selection=json.loads((RES/'frame_selection.json').read_text())['examples'];pos={(r.trajectory_id,r.frame_index):i for i,r in enumerate(val)}
 selected_positions={pos[(s['trajectory'],s['frame_index'])] for s in selection};pairs=sorted({i//2*2 for i in selected_positions});pair_positions={i for start in pairs for i in range(start,start+2)}
 models={};cfgs={};ids={};expected={}
 for name,info in P['checkpoints'].items():
  cp=torch.load(info['path'],map_location='cpu',weights_only=name=='epoch4')
  if name=='epoch4':
   config=json.loads((REPO/'models/mobilenet_v3_small_epoch4.json').read_text())['training_config'];state=cp;expected[name]=prior[name]
  else:
   state=cp['model'];extra=cp['extra'];config=extra['config'];identity=extra['identity']
   assert cp['epoch']+1==9 and extra['global_step']==198441
   assert identity['manifest_sha256']==e.MANIFEST_SHA and identity['source_sha256']==e.source_hashes()
   assert identity['starting_weights']['sha256']==e.INITIAL_SHA
   assert extra['model_contract']==model_contract(config)
   expected[name]=extra['validation'];assert expected[name]['precision']=='FP32'
   assert expected[name]['overall']['absrel']==extra['best_validation_absrel']
   if name=='student_A':assert identity['arm']=='supervised' and identity['scope']=='full'
   else:
    assert identity==completion['identity'] and identity['arm']=='exploratory_kd_rejected_teacher_v1'
    assert identity['original_gate_passed'] is False and identity['rejected_gate_sha256']==json.loads((RES/'source_inputs.json').read_text())['failed_gate_sha256']
    for rel,h in identity['exploratory_source_sha256'].items():assert sha(REPO/rel)==h
  config=dict(config);config['device']='cuda';cfgs[name]=config
  model=e.build_model(config['model']|{'pretrained':False,'min_depth':.1});model.load_state_dict(state,strict=True);model.cuda().requires_grad_(False).eval()
  assert all(torch.isfinite(v).all() for v in model.state_dict().values())
  assert all(not m.training for m in model.modules())
  before=e.state_digest(model)
  if name in oldids:assert before==oldids[name]['state_digest_after'] and model_contract(config)==oldids[name]['contract']
  models[name]=model;ids[name]=dict(info,state_digest_before=before,contract=model_contract(config))
  del cp,state;gc.collect()
 assert len({json.dumps(model_contract(c),sort_keys=True) for c in cfgs.values()})==1
 assert all(c['data']==cfgs['student_A']['data'] for c in cfgs.values())
 assert json.loads((OLD/'evaluation_completion.json').read_text())['passed']
 assert json.loads((OLD/'contract_checks.json').read_text())['all_model_and_bn_buffers_unchanged']
 for name in ['epoch4','student_A']:
  assert oldids[name]['sha256']==ids[name]['sha256']
  if name=='student_A':
   for scope,actual in [('overall',prior[name]['overall']),*prior[name]['per_trajectory'].items()]:checkmetrics(actual,expected[name]['overall'] if scope=='overall' else expected[name]['per_trajectory'][scope])
 save('identities.json',ids);save('configurations.json',cfgs)
 oldruntime=json.loads((OLD/'runtime.json').read_text())
 runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(),'threads':2,'batch_size':2,'autocast':False,'TF32':False,'model_input_prediction_precision':'FP32','metric_sums':'existing FP64 CPU accumulator and reused FP64 numpy Stats','mode':'eval/inference_mode; no gradients or optimizer','source_sha256':e.source_hashes(),'reused_runtime':oldruntime,'runtime_differences':{k:{'old':oldruntime[k],'new':v} for k,v in [('torch',torch.__version__),('cuda',torch.version.cuda),('cudnn',torch.backends.cudnn.version()),('gpu',torch.cuda.get_device_name()),('threads',2),('batch_size',2),('autocast',False)] if oldruntime[k]!=v},'pid':os.getpid()}
 save('runtime.json',runtime)
 ref=json.loads((e.PROJECT/'results/expanded_validation/validation_metrics.json').read_text());trs=list(ref['per_trajectory']);assert ref['metrics']['valid_pixels']==212903814 and ref['metrics']['near_valid_pixels']==32267840
 per={tr:DepthMetricAccumulator() for tr in trs};overall=DepthMetricAccumulator();full={tr:Stats() for tr in trs};near={tr:Stats() for tr in trs};bands={tr:{label:Stats() for label,_,_ in P['contract']['depth_bands']} for tr in trs}
 dl=e.loader(ROOT,val,cfgs['exploratory_KD'],'val',False);dl.multiprocessing_context='fork';cached={};frames=[];normcheck={};current_rgb=[None]
 conv=next(m for m in models['exploratory_KD'].encoder.modules() if isinstance(m,torch.nn.Conv2d))
 def normalization(module,args):
  expected=(current_rgb[0]-models['exploratory_KD'].rgb_mean)/models['exploratory_KD'].rgb_std
  assert torch.equal(args[0],expected);normcheck.update(checked=True,max_abs_error=0.)
 handle=conv.register_forward_pre_hook(normalization)
 with torch.inference_mode():
  for bi,batch in enumerate(dl):
   rgb=batch['rgb'].cuda();gt=batch['depth'];valid=batch['valid'];current_rgb[0]=rgb
   assert rgb.dtype==gt.dtype==torch.float32 and valid.dtype==torch.bool
   assert torch.isfinite(gt[valid]).all() and (gt[valid]>=.1).all() and (gt[valid]<=1000).all()
   with torch.amp.autocast('cuda',enabled=False):pred=models['exploratory_KD'](rgb.float()).cpu()
   if handle:handle.remove();handle=None
   assert pred.dtype==torch.float32 and pred.shape==gt.shape and torch.isfinite(pred).all() and (pred>=.1).all()
   overall.update(pred,gt,valid)
   for j,record in enumerate(batch['record']):
    idx=bi*2+j;tr=record['trajectory_id'];assert tr==val[idx].trajectory_id and record['frame_index']==val[idx].frame_index
    per[tr].update(pred[j:j+1],gt[j:j+1],valid[j:j+1]);mask=valid[j,0].numpy();g=gt[j,0].numpy()[mask].astype(np.float64);pp=pred[j,0].numpy()[mask].astype(np.float64);n=g<10
    full[tr].add(pp,g);near[tr].add(pp[n],g[n]);fm=Stats();fm.add(pp,g);nn=Stats();nn.add(pp[n],g[n])
    for label,lo,hi in P['contract']['depth_bands']:
     use=(g>=lo)&((g<hi) if hi!=1000 else (g<=hi));bands[tr][label].add(pp[use],g[use])
    frames.append({'trajectory':tr,'frame_index':record['frame_index'],**fm.result(),'near_valid_pixels':nn.n,'near_absrel':nn.result()['absrel'],'near_rmse':nn.result()['rmse'],'near_bias_m':nn.result()['bias_m']})
    if idx in pair_positions:cached[idx]={'rgb_sha256':hashlib.sha256(batch['rgb'][j].numpy().tobytes()).hexdigest(),'pred':pred[j,0].numpy()}
   if (bi+1)%100==0:log('KD_validation',frames=min((bi+1)*2,4410),total=4410)
 assert len(frames)==4410 and normcheck['checked']
 actual={'overall':standard(overall),'per_trajectory':{tr:standard(acc) for tr,acc in per.items()},'supplementary':{tr:{'all_valid':full[tr].result(),'near':near[tr].result(),'depth_bands':{label:st.result() for label,st in bands[tr].items()}} for tr in trs}}
 save('KD_aggregates_before_reference_checks.json',actual)
 save('KD_model_state_after_full_pass.json',{'before':ids['exploratory_KD']['state_digest_before'],'after':e.state_digest(models['exploratory_KD'])})
 differences={}
 for scope,d in [('overall',actual['overall']),*actual['per_trajectory'].items()]:
  expected_scope=expected['exploratory_KD']['overall'] if scope=='overall' else expected['exploratory_KD']['per_trajectory'][scope]
  differences[scope]=checkmetrics(d,expected_scope)
  countref=ref['metrics'] if scope=='overall' else ref['per_trajectory'][scope]
  assert d['valid_pixels']==countref['valid_pixels'] and d['near_valid_pixels']==countref['near_valid_pixels']
 for tr in trs:assert sum(x.n for x in bands[tr].values())==full[tr].n
 save('KD_computed_metrics.json',actual);save('KD_saved_reference_differences.json',differences)
 with (RES/'KD_per_frame.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(clean(frames))
 ds=e.GuardedDataset(ROOT,val,cfgs['student_A']['data'],'val',False)
 oldframes={}
 with (OLD/'per_frame.csv').open() as f:
  for row in csv.DictReader(f):oldframes[(row['model'],row['trajectory'],int(row['frame_index']))]=row
 examples=[]
 with torch.inference_mode():
  for pos0 in pairs:
   batch=e.collate_samples([ds[i] for i in range(pos0,pos0+2)]);rgb=batch['rgb'].cuda();preds={}
   for name in ['epoch4','student_A']:
    with torch.amp.autocast('cuda',enabled=False):preds[name]=models[name](rgb.float()).cpu()
   for j in range(2):
    idx=pos0+j;assert hashlib.sha256(batch['rgb'][j].numpy().tobytes()).hexdigest()==cached[idx]['rgb_sha256']
    if idx not in selected_positions:continue
    record=val[idx];arrays={'rgb':batch['rgb'][j].permute(1,2,0).numpy(),'gt':batch['depth'][j,0].numpy(),'valid':batch['valid'][j,0].numpy(),'exploratory_KD':cached[idx]['pred']};check={}
    for name in ['epoch4','student_A']:
     arrays[name]=preds[name][j,0].numpy();acc=DepthMetricAccumulator();acc.update(preds[name][j:j+1],batch['depth'][j:j+1],batch['valid'][j:j+1]);v=acc.compute();row=oldframes[(name,record.trajectory_id,record.frame_index)]
     check[name]={k:v[k]-float(row[k]) for k in ['absrel','rmse','near_absrel','near_rmse']}
     assert all(abs(d)<=1e-8 for d in check[name].values()),check
    filename=record.trajectory_id.replace('/','__')+f'__{record.frame_index:06d}.npz';np.savez_compressed(ART/filename,**arrays)
    examples.append({'trajectory':record.trajectory_id,'frame_index':record.frame_index,'arrays':filename,'old_per_frame_differences':check,'batch_partner_indices':[pos0,pos0+1],'input_sha256':cached[idx]['rgb_sha256']})
 save('example_verification.json',{'passed':True,'examples':examples,'original_A_replayed_batches':len(pairs),'new_KD_selection':False})
 for name,model in models.items():
  ids[name]['state_digest_after']=e.state_digest(model);assert ids[name]['state_digest_after']==ids[name]['state_digest_before'];assert all(not m.training for m in model.modules())
 save('identities.json',ids)
 os.fsync(guard.fd);opens=defaultdict(int)
 with (ART/'validation_image_opens_v2.jsonl').open() as f:
  for line in f:
   row=json.loads(line);assert row['path'] in guard.allowed;opens[row['path']]+=1
 assert len(opens)==8820
 for name,v in P['checkpoints'].items():assert sha(v['path'])==v['sha256']
 assert e.source_hashes()==P['parent_sources']
 save('completion.json',{'passed':True,'KD_validation_frames':4410,'original_A_visualization_frames_only':len(pairs)*2,'teacher_forward_calls':0,'unique_validation_image_paths':len(opens),'total_image_opens':sum(opens.values()),'test_image_opens':0,'failed_HDD_reads':0,'all_models_and_BN_unchanged':True,'normalization_first_batch':normcheck,'saved_KD_metrics_reproduced':True,'expected_pixel_counts_exact':True,'optimizer_instances':0,'seconds':time.monotonic()-START})
 log('complete',KD_frames=4410,visualization_examples=len(examples))

if __name__=='__main__':
 try:run()
 except BaseException as ex:log('failed',error=repr(ex));raise
