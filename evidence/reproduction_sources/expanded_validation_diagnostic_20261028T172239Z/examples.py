"""Predeclared validation examples, retaining original FP32 batch partners."""
from pathlib import Path
import os,sys,json,csv,hashlib,gc
ART=Path(__file__).resolve().parent
REPO=ART.parents[1]
RES=REPO/'results'/ART.name
ROOT=Path('/run/media/Ima/AERODEPTH/AeroDepth-rescue/MidAir')
os.environ['MPLCONFIGDIR']=str(ART/'mpl-cache')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from aerodepth_lite import expanded as e
from aerodepth_lite.model import build_model
from aerodepth_lite.metrics import DepthMetricAccumulator

torch.set_num_threads(2);e.seed_everything(42)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
selection=json.loads((RES/'example_selection.json').read_text())['examples']
identities=json.loads((RES/'identities.json').read_text())
records,m=e.load_inputs(e.PREP/'manifest_train_val.json.gz',e.PREP/'reservations.json',ROOT)
val=records['val'];positions={(r.trajectory_id,r.frame_index):i for i,r in enumerate(val)}
pairs=sorted({positions[(r['trajectory'],r['frame_index'])]//2*2 for r in selection})
permitted=[r for pos in pairs for r in val[pos:pos+2]]
guard=e.ImageReadGuard(ROOT,permitted);sys.addaudithook(guard.check)
models={};data=None
for name in ['epoch4','student_A','teacher_B']:
 p=Path(identities[name]['checkpoint'])
 assert e.sha256_file(p)==identities[name]['sha256']
 if name=='epoch4':
  payload=torch.load(p,map_location='cpu',weights_only=True);state=payload
  cfg=json.loads((REPO/'models/mobilenet_v3_small_epoch4.json').read_text())['training_config']
 else:
  payload=torch.load(p,map_location='cpu',weights_only=False);state=payload['model'];cfg=payload['extra']['config']
 model=build_model(cfg['model']|{'pretrained':False,'min_depth':.1})
 model.load_state_dict(state,strict=True);model.cuda().requires_grad_(False).eval();models[name]=model
 data=cfg['data'];del payload,state;gc.collect()
before={n:e.state_digest(model) for n,model in models.items()}
ds=e.GuardedDataset(ROOT,val,data,'val',False)
figdir=RES/'examples';figdir.mkdir()
saved={}
with (RES/'per_frame.csv').open() as f:
 for r in csv.DictReader(f):saved[(r['model'],r['trajectory'],int(r['frame_index']))]=r
checks=[]
for pos in pairs:
 batch=e.collate_samples([ds[i] for i in range(pos,min(pos+2,len(val)))])
 rgb=batch['rgb'].cuda();preds={}
 with torch.inference_mode(),torch.amp.autocast('cuda',enabled=False):
  for name,model in models.items():preds[name]=model(rgb.float()).cpu()
 for chosen in selection:
  index=positions[(chosen['trajectory'],chosen['frame_index'])]
  if not pos<=index<pos+2:continue
  j=index-pos;gt=batch['depth'][j,0].numpy();valid=batch['valid'][j,0].numpy();near=valid&(gt<10)
  compare={}
  for name,pred in preds.items():
   a=DepthMetricAccumulator();a.update(pred[j:j+1],batch['depth'][j:j+1],batch['valid'][j:j+1]);d=a.compute()
   ref=saved[(name,chosen['trajectory'],chosen['frame_index'])]
   compare[name]={}
   for key in ['absrel','rmse','near_absrel','near_rmse']:
    delta=d[key]-float(ref[key]);assert abs(delta)<(1e-3 if key.endswith('rmse') else 1e-5)
    compare[name][key]=delta
  fig,axes=plt.subplots(2,5,figsize=(20,8),layout='constrained')
  image=batch['rgb'][j].permute(1,2,0).numpy();axes[0,0].imshow(image);axes[0,0].set_title('Resized validation RGB')
  cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#bbbbbb')
  depth_img=axes[0,1].imshow(np.ma.masked_where(~valid,gt),cmap=cmap,norm=LogNorm(vmin=.1,vmax=1000))
  axes[0,1].set_title('GT Euclidean range, m')
  for col,(name,label) in enumerate([('epoch4','Original epoch 4'),('student_A','Supervised A'),('teacher_B','Teacher B')],2):
   pp=preds[name][j,0].numpy();r=saved[(name,chosen['trajectory'],chosen['frame_index'])]
   axes[0,col].imshow(np.ma.masked_where(~valid,pp),cmap=cmap,norm=LogNorm(vmin=.1,vmax=1000))
   axes[0,col].set_title(f'{label}\nNear RMSE {float(r["near_rmse"]):.2f} m; bias {float(r["near_bias_m"]):+.2f} m')
  fig.colorbar(depth_img,ax=list(axes[0,1:]),fraction=.025,label='Range, m (log); invalid GT grey')
  axes[1,0].imshow(image);axes[1,0].imshow(np.ma.masked_where(~near,near),cmap='autumn',vmin=0,vmax=1,alpha=.65)
  axes[1,0].set_title(f'GT<10m overlay; {near.sum():,} pixels')
  for col,name in enumerate(['epoch4','student_A','teacher_B'],1):
   err=preds[name][j,0].numpy()-gt
   error_img=axes[1,col].imshow(np.ma.masked_where(~near,err),cmap='RdBu_r',vmin=-10,vmax=10)
   axes[1,col].set_title(name+' signed error on GT<10m')
  delta=np.abs(preds['teacher_B'][j,0].numpy()-gt)-np.abs(preds['student_A'][j,0].numpy()-gt)
  axes[1,4].imshow(np.ma.masked_where(~near,delta),cmap='RdBu_r',vmin=-10,vmax=10)
  axes[1,4].set_title('Teacher − A absolute error\nRed = teacher worse')
  fig.colorbar(error_img,ax=list(axes[1,1:]),fraction=.025,label='Metres; colours clipped to ±10')
  for ax in axes.ravel():ax.set_axis_off()
  fig.suptitle(f'{chosen["trajectory"]} / frame {chosen["frame_index"]} / deterministic {chosen["reason"]}\n'
               'Validation only; all metrics use unclipped predictions. MidAir: Fonder & Van Droogenbroeck, CC BY-NC-SA 4.0',fontsize=12)
  filename=chosen['trajectory'].replace('/','__')+f'__{chosen["frame_index"]:06d}__{chosen["reason"]}.png'
  fig.savefig(figdir/filename,dpi=140);plt.close(fig)
  checks.append(dict(trajectory=chosen['trajectory'],frame_index=chosen['frame_index'],reason=chosen['reason'],figure='examples/'+filename,metric_differences=compare,batch_partner_frames=[(r['trajectory_id'],r['frame_index']) for r in batch['record']]))
  print('EXAMPLE',filename,flush=True)
for n,model in models.items():assert e.state_digest(model)==before[n]
(RES/'example_verification.json').write_text(json.dumps({'passed':True,'examples':checks,'permitted_val_frames':len(permitted),'training_or_test_reads':0,'all_model_buffers_unchanged':True},indent=2)+'\n')
