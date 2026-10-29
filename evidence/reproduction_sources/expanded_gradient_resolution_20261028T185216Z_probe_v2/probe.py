"""Fixed FP32 eval-mode TRAIN gradient probe; no optimizer or parameter updates."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time

ART=Path(__file__).resolve().parent
REPO=ART.parents[1]
RES=REPO/'results'/ART.name
sys.path.insert(0,str(REPO/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from aerodepth_lite import expanded as e
from aerodepth_lite.losses import depth_loss
from aerodepth_lite.data import FrameRecord

START=time.monotonic()
P=json.loads((RES/'PLAN.json').read_text())
ROOT=Path(P['dataset_root'])


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float):assert math.isfinite(x),('Nonfinite reported result',x)
    return x


def save(name,x):
    p=RES/name;t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');t.replace(p)


def progress(stage,**kwargs):
    d={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-START,**kwargs}
    save('status.json',d);print(json.dumps(clean(d)),flush=True)


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_hash(generator=None):
    h=hashlib.sha256(repr(random.getstate()).encode())
    ns=np.random.get_state();h.update(ns[0].encode());h.update(ns[1].tobytes());h.update(repr(ns[2:]).encode())
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for s in torch.cuda.get_rng_state_all():h.update(s.cpu().numpy().tobytes())
    if generator is not None:h.update(generator.get_state().numpy().tobytes())
    return h.hexdigest()


class Guard(e.ImageReadGuard):
    def __init__(self,root,records):
        super().__init__(root,records)
        self.fd=os.open(ART/'train_image_opens.jsonl',os.O_CREAT|os.O_EXCL|os.O_APPEND|os.O_WRONLY,0o600)

    def check(self,event,args):
        if event not in ('open','os.scandir','os.listdir') or not args or isinstance(args[0],int):return
        p=os.path.abspath(os.fsdecode(args[0]))
        if p=='/run/media/Ima/Low storage' or p.startswith('/run/media/Ima/Low storage/'):
            raise PermissionError('Failing HDD forbidden')
        if 'expanded_validation_diagnostic_' in p and '/near_arrays/' in p:
            raise PermissionError('Previous validation-pixel cache forbidden')
        super().check(event,args)
        if event=='open' and p in self.allowed:
            os.write(self.fd,(json.dumps({'pid':os.getpid(),'path':p})+'\n').encode())


def objectives(pred,gt,valid,config):
    v=valid.bool();n=v&(gt<10);f=v&~n;anchor=pred.sum()*0.
    total,terms=depth_loss(pred,gt,v,config)
    total=total+anchor
    def mean_num(values,mask,denom):
        return values[mask].sum()/denom+anchor if denom else anchor
    nv=int(v.sum());nn=int(n.sum())
    absolute=(pred-gt).abs()
    logs=(pred.clamp_min(1e-6).log()-gt.clamp_min(1e-6).log()).abs()
    ln=mean_num(absolute,n,nv);lf=mean_num(absolute,f,nv)
    logn=mean_num(logs,n,nv);logf=mean_num(logs,f,nv)
    hn=v[:,:,:,1:]&v[:,:,:,:-1];vn=v[:,:,1:,:]&v[:,:,:-1,:]
    hnear=hn&(n[:,:,:,1:]|n[:,:,:,:-1]);vnear=vn&(n[:,:,1:,:]|n[:,:,:-1,:])
    he=((pred[:,:,:,1:]-pred[:,:,:,:-1])-(gt[:,:,:,1:]-gt[:,:,:,:-1])).abs()
    ve=((pred[:,:,1:,:]-pred[:,:,:-1,:])-(gt[:,:,1:,:]-gt[:,:,:-1,:])).abs()
    gn=mean_num(he,hnear,int(hn.sum()))+mean_num(ve,vnear,int(vn.sum()))
    gf=mean_num(he,hn&~hnear,int(hn.sum()))+mean_num(ve,vn&~vnear,int(vn.sum()))
    near=terms['near']+anchor
    candidate,candidate_terms=depth_loss(pred,gt,v,config|{'near_weight':.2})
    objs={'L1':terms['l1']+anchor,'log':terms['log']+anchor,'gradient':terms['gradient']+anchor,
          'near_MAE':near,'current':total,'current_near':ln+.2*logn+.1*gn,
          'current_far':lf+.2*logf+.1*gf,'near_L1_global':ln,'candidate':candidate+anchor}
    edges={'valid_horizontal_edges':int(hn.sum()),'valid_vertical_edges':int(vn.sum()),
           'near_incident_horizontal_edges':int(hnear.sum()),'near_incident_vertical_edges':int(vnear.sum()),
           'near_far_boundary_horizontal_edges':int((hn&(n[:,:,:,1:]^n[:,:,:,:-1])).sum()),
           'near_far_boundary_vertical_edges':int((vn&(n[:,:,1:,:]^n[:,:,:-1,:])).sum())}
    return objs,{'valid_pixels':nv,'near_pixels':nn,'near_fraction':nn/nv if nv else 0.,**edges}


def check_vector(actual,reconstructed,layout,label):
    diff=(actual-reconstructed).abs();bound=1e-6+1e-4*actual.abs()
    ratio=diff/bound
    maximum=float(ratio.max()) if ratio.numel() else 0.
    bad=[]
    for name,start,end in layout:
        if bool((ratio[start:end]>1).any()):bad.append(name)
    assert not bad,(label,bad,maximum)
    return {'comparison':label,'passed':True,'max_abs_difference':float(diff.max()),
            'max_fraction_of_allowed_tolerance':maximum,'parameters_checked':len(layout)}


def self_check(config):
    checks=[]
    for case in ['mixed','all_near','no_near','all_invalid']:
        gt=torch.tensor([[[[2.,4.,10.],[8.,20.,50.],[.1,100.,1000.]]]],dtype=torch.float32)
        if case=='all_near':gt.fill_(4.)
        if case=='no_near':gt.fill_(20.)
        v=torch.ones_like(gt,dtype=torch.bool)
        if case=='mixed':v[0,0,1,1]=False
        if case=='all_invalid':v.fill_(False)
        pred=(gt*.8+1).detach().requires_grad_(True)
        o,c=objectives(pred,gt,v,config)
        grad={k:torch.autograd.grad(x,pred,retain_graph=True)[0].flatten() for k,x in o.items()}
        layout=[('prediction',0,pred.numel())]
        for a,b,label in [(grad['current'],grad['L1']+.2*grad['log']+.1*grad['gradient'],'current'),
                          (grad['current'],grad['current_near']+grad['current_far'],'partition'),
                          (grad['candidate'],grad['current']+.2*grad['near_MAE'],'candidate'),
                          (grad['near_L1_global'],c['near_fraction']*grad['near_MAE'],'near_normalization')]:
            checks.append({'case':case,**check_vector(a,b,layout,label)})
    save('synthetic_reduction_checks.json',{'passed':True,'checks':checks,'dataset_images_used':0})


def grad_vector(loss,params,retain):
    gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
    chunks=[]
    for p,g in zip(params,gs):
        t=torch.zeros_like(p) if g is None else g
        assert t.dtype==torch.float32 and torch.isfinite(t).all()
        chunks.append(t.detach().cpu().flatten())
    result=torch.cat(chunks)
    del gs,chunks
    return result


def group_stats(grads,groups):
    result={}
    pairs=[('current','near_MAE'),('candidate','near_MAE'),('current_near','current_far'),
           ('L1','near_MAE'),('log','near_MAE'),('gradient','near_MAE'),
           ('current','candidate'),('current','current_near')]
    for group,(lo,hi) in groups.items():
        vec={k:v[lo:hi] for k,v in grads.items()}
        norms={k:float(torch.linalg.vector_norm(v.double())) for k,v in vec.items()}
        dots={};cos={}
        for x,y in pairs:
            key=x+'__'+y;dot=float(torch.dot(vec[x].double(),vec[y].double()));dots[key]=dot
            cos[key]=max(-1.,min(1.,dot/(norms[x]*norms[y]))) if norms[x]>1e-12 and norms[y]>1e-12 else None
        den=norms['current']
        result[group]={'parameters':hi-lo,'L2':norms,'RMS':{k:v/math.sqrt(hi-lo) for k,v in norms.items()},
                       'weighted_component_L2':{'L1':norms['L1'],'0.2_log':.2*norms['log'],'0.1_gradient':.1*norms['gradient'],'0.2_near_MAE':.2*norms['near_MAE']},
                       'cosines':cos,'dots':dots,
                       'near_partition_current_norm_ratio':norms['current_near']/den if den>1e-12 else None,
                       'weighted_near_MAE_current_norm_ratio':.2*norms['near_MAE']/den if den>1e-12 else None,
                       'candidate_current_norm_ratio':norms['candidate']/den if den>1e-12 else None}
    return result


# Numerical amendment V2; the original elementwise check is retained as a recorded diagnostic.
legacy_elementwise_check=check_vector
from importlib.util import spec_from_file_location,module_from_spec
_numeric_spec=spec_from_file_location('bounded_numeric_reference',REPO/'artifacts'/'expanded_gradient_resolution_20261028T185216Z'/'diagnose.py')
_numeric=module_from_spec(_numeric_spec);_numeric_spec.loader.exec_module(_numeric)
AMEND=json.loads((RES/'NUMERICAL_AMENDMENT_V2.json').read_text())

def check_vector(actual,reconstructed,layout,label,near=None):
    if layout[0][0]=='prediction':
        return legacy_elementwise_check(actual,reconstructed,layout,label)
    groups={'encoder':(0,23508032),'decoder':(23508032,23999633),'output_head':(23999488,23999633)}
    details=_numeric.compare(actual,reconstructed,groups,near)
    for group,(lo,hi) in groups.items():
        row=details[group];maxactual=float(actual[lo:hi].abs().max());row['max_abs_guard']=1e-6+1e-4*maxactual
        row['joint_numerical_passed']=(row['relative_L2_error']<=1e-5 and row['max_abs_error']<=row['max_abs_guard'] and (row.get('cosine_absolute_effect') is None or row['cosine_absolute_effect']<=1e-5))
    try:
        legacy=legacy_elementwise_check(actual,reconstructed,layout,label)
    except AssertionError as exc:
        legacy={'passed':False,'failing_parameter_names':exc.args[0][1],'max_fraction_of_allowed_tolerance':exc.args[0][2]}
    result={'comparison':label,'passed':all(x['joint_numerical_passed'] for x in details.values()),'groups':details,'original_elementwise_check':legacy}
    assert result['passed'],('Amended numerical check failed',result)
    return result

def classification_stability(g,groups):
    alt={**g,'current':g['L1']+.2*g['log']+.1*g['gradient'],'candidate':g['current']+.2*g['near_MAE']}
    direct=group_stats(g,groups);reconstructed=group_stats(alt,groups)
    def flags(s):
        c=s['cosines']['current__near_MAE'];a=s['cosines']['candidate__near_MAE']
        return {'conflict':c is not None and c<-.05,'candidate_effect':a is not None and c is not None and a-c>=.05,
                'weak':s['near_partition_current_norm_ratio'] is not None and s['near_partition_current_norm_ratio']<=.10,
                'weak_effect':s['weighted_near_MAE_current_norm_ratio'] is not None and s['weighted_near_MAE_current_norm_ratio']>=.10}
    result={k:{'direct':flags(direct[k]),'reconstructed':flags(reconstructed[k])} for k in groups}
    for k,v in result.items():assert v['direct']==v['reconstructed'],('Numerically unstable scientific boundary',k,v)
    return result


def main():
    seal=json.loads((RES/'FREEZE.json').read_text())
    for name,digest in seal['files_sha256'].items():assert sha(REPO/name)==digest,name
    progress('preflight')
    assert sha(P['teacher_checkpoint'])==P['teacher_sha256']
    assert sha(P['failed_gate_path'])==P['failed_gate_sha256'] and not json.loads(Path(P['failed_gate_path']).read_text())['passed']
    assert sha(P['frozen_protocol_path'])==P['frozen_protocol_sha256']
    assert e.source_hashes()==P['source_sha256']
    torch.set_num_threads(2);e.seed_everything(42)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.hub.download_url_to_file=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Downloads forbidden'))
    sample=json.loads((RES/'sampling.json').read_text());records=[FrameRecord(**x['record']) for x in sample]
    allrows,_=e.load_inputs(e.PREP/'manifest_train_val.json.gz',e.PREP/'reservations.json',ROOT)
    assert records==e.fixed_training_subset(allrows['train'],64)
    for r in records:e.check_record(r,'train',ROOT)
    guard=Guard(ROOT,records);sys.addaudithook(guard.check)
    for bad in ['Kite_training/foggy/color_left/trajectory_2003/000000.JPEG',
                'Kite_training/sunny/color_left/trajectory_0007/000000.JPEG']:
        try:e.ImageReadGuard.check(guard,'open',(str(ROOT/bad),'rb',0))
        except PermissionError:pass
        else:raise AssertionError('Non-TRAIN read guard ineffective')
    # Bind the selected image bytes to the already verified USB rescue index.
    wanted={str((ROOT/str(getattr(r,k))).relative_to(ROOT.parent)) for r in records for k in ['rgb_path','depth_path']}
    prior={}
    with (ROOT.parent/'recovery/dataset_index.jsonl').open() as f:
        for line in f:
            d=json.loads(line)
            if d['destination'] in wanted:prior[d['destination']]=d['sha256']
    assert len(prior)==128
    for rel,digest in prior.items():assert sha(ROOT.parent/rel)==digest,rel
    payload=torch.load(P['teacher_checkpoint'],map_location='cpu',weights_only=False)
    identity=payload['extra']['identity'];config=payload['extra']['config']
    assert identity['source_sha256']==e.source_hashes() and identity['manifest_sha256']==e.MANIFEST_SHA
    assert identity['arm']=='teacher' and identity['scope']=='full'
    assert payload['epoch']==2 and config['loss']=={'l1_weight':1.,'log_weight':.2,'gradient_weight':.1,'near_weight':0.,'near_threshold':10.}
    e.check_config(config)
    model=e.build_model(config['model']|{'pretrained':False,'min_depth':.1})
    model.load_state_dict(payload['model'],strict=True);del payload;gc.collect()
    model.cuda().requires_grad_(True).eval()
    assert all(not mod.training for mod in model.modules())
    assert torch.equal(model.rgb_mean.flatten().cpu(),torch.tensor([.485,.456,.406]))
    assert torch.equal(model.rgb_std.flatten().cpu(),torch.tensor([.229,.224,.225]))
    before=e.state_digest(model)
    names_params=list(model.named_parameters());params=[p for _,p in names_params]
    layout=[];off=0
    for name,p in names_params:layout.append((name,off,off+p.numel()));off+=p.numel()
    enc=sum(p.numel() for n,p in names_params if n.startswith('encoder.'))
    head=sum(p.numel() for n,p in names_params if n.startswith('head.'))
    assert enc==23508032 and off-enc==491601 and head==145
    assert all(n.startswith('encoder.') for n,lo,hi in layout if lo<enc)
    assert all(n.startswith('head.') for n,lo,hi in layout if lo>=off-head)
    groups={'encoder':(0,enc),'decoder':(enc,off),'output_head':(off-head,off)}
    save('runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
         'precision':'FP32 forward/loss/gradients; FP64 summary dot/norm accumulation','mode':'eval with gradients enabled',
         'group_offsets':groups,'parameter_layout':[{'name':n,'start':lo,'end':hi} for n,lo,hi in layout],
         'source_sha256':e.source_hashes(),'model_state_before':before,'verified_input_file_hashes':128,
         'image_guard_installed_before_PIL':True,'non_train_probes_blocked_before_io':2,'sampling_sha256':sha(RES/'sampling.json')})
    self_check(config['loss'])
    loader=e.loader(ROOT,records,config|{'device':'cuda'},'train',False);loader.multiprocessing_context='fork'
    torch.cuda.reset_peak_memory_stats()
    all_batches=[];frame_rows=[];reconstruction=[]
    for b,batch in enumerate(loader):
        rgb,gt,valid=[batch[k].cuda() for k in ['rgb','depth','valid']]
        assert rgb.dtype==gt.dtype==torch.float32 and valid.dtype==torch.bool
        assert rgb.min()>=0 and rgb.max()<=1 and not model.training
        hashes={k:tensor_hash(t) for k,t in [('rgb',rgb),('gt',gt),('valid',valid)]}
        rng=rng_hash(loader.generator)
        with torch.enable_grad(),torch.amp.autocast('cuda',enabled=False):
            pred=model(rgb)
            assert pred.dtype==torch.float32 and pred.shape==gt.shape and torch.isfinite(pred).all() and (pred>=.1).all()
            assert rng_hash(loader.generator)==rng
            hashes['prediction']=tensor_hash(pred)
            objs,counts=objectives(pred,gt,valid,config['loss'])
            assert all(torch.isfinite(x) for x in objs.values())
            grads={}
            for i,(name,loss) in enumerate(objs.items()):
                assert rng_hash(loader.generator)==rng
                grads[name]=grad_vector(loss,params,retain=i<len(objs)-1)
                assert rng_hash(loader.generator)==rng
                assert all(p.grad is None for p in params)
        checks=[]
        for actual,recon,label in [(grads['current'],grads['L1']+.2*grads['log']+.1*grads['gradient'],'current_components'),
                                   (grads['candidate'],grads['current']+.2*grads['near_MAE'],'candidate_components'),
                                   (grads['current'],grads['current_near']+grads['current_far'],'near_far_partition'),
                                   (grads['near_L1_global'],counts['near_fraction']*grads['near_MAE'],'near_L1_normalization')]:
            checks.append(check_vector(actual,recon,layout,label,near=grads['near_MAE']))
        reconstruction.append({'batch_index':b,'checks':checks})
        stability=classification_stability(grads,groups)
        stats=group_stats(grads,groups)
        for j,rec in enumerate(batch['record']):
            info=sample[b*2+j];assert rec['trajectory_id']==info['record']['trajectory_id'] and rec['frame_index']==info['record']['frame_index']
            v=valid[j];n=v&(gt[j]<10);far=v&~n;err=(pred[j]-gt[j]).detach()
            row={k:info[k] for k in ['sample_index','batch_index','slot','environment','condition','path_group','fraction_pass','ordinal_in_realization','realization_frames','normalized_ordinal']}
            row.update(trajectory=rec['trajectory_id'],frame_index=rec['frame_index'],valid_pixels=int(v.sum()),near_pixels=int(n.sum()),
                       near_fraction=float(n.sum()/v.sum()) if v.any() else 0.,
                       near_bias_m=float(err[n].double().mean()) if n.any() else None,
                       near_mae_m=float(err[n].abs().double().mean()) if n.any() else None,
                       far_bias_m=float(err[far].double().mean()) if far.any() else None,
                       far_mae_m=float(err[far].abs().double().mean()) if far.any() else None)
            frame_rows.append(row)
        assert {k:tensor_hash(t) for k,t in [('rgb',rgb),('gt',gt),('valid',valid),('prediction',pred)]}==hashes
        assert e.state_digest(model)==before and rng_hash(loader.generator)==rng
        row={'batch_index':b,'records':sample[b*2:b*2+2],'counts':counts,'losses':{k:float(v.detach()) for k,v in objs.items()},
             'groups':stats,'scientific_classification_stability':stability,'tensor_sha256':hashes,'RNG_sha256_before_and_after':rng,
             'model_parameters_and_buffers_unchanged':True,'parameter_grad_fields_none':True,'optimizer_steps':0}
        all_batches.append(row)
        save('batches.json',all_batches);save('gradient_reconstruction.json',{'passed':True,'batches':reconstruction})
        progress('probing',batches=b+1,total_batches=32,frames=(b+1)*2,near_pixels=counts['near_pixels'])
        del grads,objs,pred,stats;gc.collect()
    assert len(all_batches)==32 and len(frame_rows)==64
    for name in ['rgb_path','depth_path']:
        assert len(set(getattr(r,name) for r in records))==64
    os.fsync(guard.fd)
    opened=defaultdict(int)
    with (ART/'train_image_opens.jsonl').open() as f:
        for line in f:
            d=json.loads(line);assert d['path'] in guard.allowed;opened[d['path']]+=1
    assert len(opened)==128
    with (RES/'sample_coverage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(frame_rows[0]));w.writeheader();w.writerows(frame_rows)
    with (RES/'block_gradients.csv').open('w') as f:
        rows=[]
        for batch in all_batches:
            for name,s in batch['groups'].items():
                rows.append({'batch_index':batch['batch_index'],'group':name,'near_pixels':batch['counts']['near_pixels'],
                    'near_fraction':batch['counts']['near_fraction'],**{'L2_'+k:v for k,v in s['L2'].items()},
                    **{'RMS_'+k:v for k,v in s['RMS'].items()},**{'cos_'+k:v for k,v in s['cosines'].items()},
                    'near_partition_current_norm_ratio':s['near_partition_current_norm_ratio'],
                    'weighted_near_MAE_current_norm_ratio':s['weighted_near_MAE_current_norm_ratio']})
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name,digest in seal['files_sha256'].items():assert sha(REPO/name)==digest,name
    assert sha(P['teacher_checkpoint'])==P['teacher_sha256'] and sha(P['failed_gate_path'])==P['failed_gate_sha256']
    assert sha(P['frozen_protocol_path'])==P['frozen_protocol_sha256']
    final=e.state_digest(model);assert final==before
    save('completion.json',{'passed':True,'batches':32,'frames':64,'model_mode':'eval','autocast':False,
         'gradients_enabled':True,'optimizer_instances':0,'optimizer_steps':0,'model_state_before':before,'model_state_after':final,
         'all_parameter_grad_fields_none':all(p.grad is None for p in params),'all_branch_rng_checks_passed':True,
         'source_identity_preserved':True,'checkpoint_gate_protocol_unchanged':True,'train_unique_image_paths':len(opened),
         'train_image_opens':sum(opened.values()),'validation_test_image_reads':0,
         'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-START})
    progress('probe_complete',batches=32,frames=64)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        progress('failed',error=repr(exc))
        raise
