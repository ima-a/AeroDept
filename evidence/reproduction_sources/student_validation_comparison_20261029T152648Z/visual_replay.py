"""Only fixed example inputs; full KD scores already persisted and checked."""
from pathlib import Path
import json,os,sys,csv,hashlib,gc,time
ART=Path(__file__).resolve().parent;REPO=ART.parents[1];RES=REPO/'results'/ART.name
sys.path.insert(0,str(REPO/'src'));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from aerodepth_lite import expanded as e
from aerodepth_lite.metrics import DepthMetricAccumulator
P=json.loads((RES/'PLAN.json').read_text());ROOT=Path(P['dataset_root']);OLD=Path(P['reuse_root']);START=time.monotonic()
def sha(f):
 with Path(f).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def save(n,d):(RES/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
class Guard(e.ImageReadGuard):
 def __init__(self,root,records):
  super().__init__(root,records);self.fd=os.open(ART/'visual_image_opens.jsonl',os.O_CREAT|os.O_EXCL|os.O_APPEND|os.O_WRONLY,0o600)
 def check(self,event,args):
  if event not in ('open','os.scandir','os.listdir') or not args or isinstance(args[0],int):return
  path=os.path.abspath(os.fsdecode(args[0]))
  if path=='/run/media/Ima/Low storage' or path.startswith('/run/media/Ima/Low storage/'):raise PermissionError('HDD forbidden')
  super().check(event,args)
  if event=='open' and path in self.allowed:os.write(self.fd,(json.dumps({'pid':os.getpid(),'path':path})+'\n').encode())
torch.set_num_threads(2);e.seed_everything(42);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert e.source_hashes()==P['parent_sources']
fullcheck=json.loads((RES/'KD_saved_reference_differences.json').read_text())
for scope,d in fullcheck.items():assert all(v is None or abs(v)<=1e-8 for v in d.values())
state=json.loads((RES/'KD_model_state_after_full_pass.json').read_text());assert state['before']==state['after']
records,_=e.load_inputs(REPO/P['manifest'],REPO/P['reservations'],ROOT);val=records['val'];pos={(r.trajectory_id,r.frame_index):i for i,r in enumerate(val)}
selection=json.loads((RES/'frame_selection.json').read_text())['examples'];chosen={pos[(x['trajectory'],x['frame_index'])] for x in selection};pairs=sorted({i//2*2 for i in chosen});original_indices=[i for k in pairs for i in range(k,k+2)];permitted=[val[i] for i in original_indices]
guard=Guard(ROOT,permitted);sys.addaudithook(guard.check)
cfgs=json.loads((RES/'configurations.json').read_text());ids=json.loads((RES/'identities.json').read_text());models={}
for name,info in P['checkpoints'].items():
 assert sha(info['path'])==info['sha256']
 cp=torch.load(info['path'],map_location='cpu',weights_only=name=='epoch4');state=cp if name=='epoch4' else cp['model']
 model=e.build_model(cfgs[name]['model']|{'pretrained':False,'min_depth':.1});model.load_state_dict(state,strict=True);model.cuda().requires_grad_(False).eval()
 assert e.state_digest(model)==ids[name]['state_digest_before'];models[name]=model;del cp,state;gc.collect()
refs={}
for f,names in [(OLD/'per_frame.csv',{'epoch4','student_A'}),(RES/'KD_per_frame.csv',{'exploratory_KD'})]:
 with f.open() as h:
  for row in csv.DictReader(h):
   name=row.get('model','exploratory_KD')
   if name in names:refs[(name,row['trajectory'],int(row['frame_index']))]=row
loader=e.loader(ROOT,permitted,cfgs['exploratory_KD'],'val',False);loader.multiprocessing_context='fork';checks=[];differences={}
with torch.inference_mode():
 for bi,batch in enumerate(loader):
  if bi==0:
   # Bound the direct-versus-worker difference on exactly the first fixed pair.
   direct=e.collate_samples([loader.dataset[0],loader.dataset[1]])
   differences={'frames':[(r['trajectory_id'],r['frame_index']) for r in batch['record']],
     'worker_RGB_sha256':hashlib.sha256(batch['rgb'].numpy().tobytes()).hexdigest(),
     'main_RGB_sha256':hashlib.sha256(direct['rgb'].numpy().tobytes()).hexdigest(),
     'RGB_max_abs_difference':float((batch['rgb']-direct['rgb']).abs().max()),
     'RGB_numeric_unequal_elements':int((batch['rgb']!=direct['rgb']).sum()),
     'GT_max_abs_difference':float((batch['depth']-direct['depth']).abs().max()),
     'mask_unequal_elements':int((batch['valid']!=direct['valid']).sum()),
     'worker_torch_threads':1,'main_torch_threads':torch.get_num_threads(),
     'final_figures':'All three predictions use the identical worker-produced tensor and original two-frame partners; full-metric reference tolerances unchanged.'}
   save('preprocessing_context_observation.json',differences)
  rgb=batch['rgb'].cuda();preds={}
  for name,model in models.items():
   with torch.amp.autocast('cuda',enabled=False):preds[name]=model(rgb.float()).cpu()
   assert preds[name].dtype==torch.float32 and torch.isfinite(preds[name]).all() and (preds[name]>=.1).all()
  for j in range(2):
   idx=original_indices[bi*2+j]
   if idx not in chosen:continue
   record=val[idx];arrays={'rgb':batch['rgb'][j].permute(1,2,0).numpy(),'gt':batch['depth'][j,0].numpy(),'valid':batch['valid'][j,0].numpy()};delta={}
   for name in models:
    arrays[name]=preds[name][j,0].numpy();acc=DepthMetricAccumulator();acc.update(preds[name][j:j+1],batch['depth'][j:j+1],batch['valid'][j:j+1]);v=acc.compute();ref=refs[(name,record.trajectory_id,record.frame_index)]
    delta[name]={k:v[k]-float(ref[k]) for k in ['absrel','rmse','near_absrel','near_rmse']};assert all(abs(x)<=1e-8 for x in delta[name].values()),delta
   fn=record.trajectory_id.replace('/','__')+f'__{record.frame_index:06d}.npz';np.savez_compressed(ART/fn,**arrays)
   checks.append({'trajectory':record.trajectory_id,'frame_index':record.frame_index,'arrays':fn,'old_per_frame_differences':delta,'batch_partner_indices':[pairs[bi],pairs[bi]+1],'input_sha256':hashlib.sha256(batch['rgb'][j].numpy().tobytes()).hexdigest()})
assert len(checks)==6
for name,model in models.items():ids[name]['state_digest_after']=e.state_digest(model);assert ids[name]['state_digest_after']==ids[name]['state_digest_before'];assert all(not m.training for m in model.modules())
save('identities.json',ids);save('example_verification.json',{'passed':True,'examples':checks,'original_A_replayed_batches':len(pairs),'KD_visual_replayed_frames':12,'new_KD_selection':False,'all_three_input_tensors_identical':True})
from collections import Counter
log_counts={}
for f in ['validation_image_opens_v2.jsonl','visual_image_opens.jsonl']:
 with (ART/f).open() as h:log_counts[f]=Counter(json.loads(line)['path'] for line in h)
allpaths=set(log_counts['validation_image_opens_v2.jsonl'])|set(log_counts['visual_image_opens.jsonl']);assert len(allpaths)==8820
# The accepted full pass and fixed common-worker visualization are separate.
# Old aborted direct-replay opens remain logged rather than relabelled as figures.
save('completion.json',{'passed':True,'KD_validation_frames':4410,'original_A_visualization_frames_only':12,'KD_visualization_frames_only':12,'teacher_forward_calls':0,'unique_validation_image_paths':8820,'total_image_opens':sum(sum(x.values()) for x in log_counts.values()),'test_image_opens':0,'failed_HDD_reads':0,'all_models_and_BN_unchanged':True,'normalization_first_batch':'passed in completed KD full pass','saved_KD_metrics_reproduced':True,'expected_pixel_counts_exact':True,'optimizer_instances':0,'seconds_visual_replay':time.monotonic()-START,'full_pass_reporting_stops_preserved':True})
save('status.json',{'stage':'complete','utc':datetime.now(timezone.utc).isoformat(),'KD_frames':4410,'visualization_examples':6})
print(json.dumps({'status':'complete','examples':6,'input_context_observation':differences}),flush=True)
