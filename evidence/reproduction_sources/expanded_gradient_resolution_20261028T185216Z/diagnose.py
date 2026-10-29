"""Bounded first-two TRAIN-batch numerical resolution; no optimizer."""
from pathlib import Path
import importlib.util,json,os,sys,time,gc,hashlib,math,traceback
ART=Path(__file__).resolve().parent; REPO=ART.parents[1]; RES=REPO/'results'/ART.name
SPEC=json.loads((RES/'DIAGNOSTIC_PLAN.json').read_text())
sp=importlib.util.spec_from_file_location('original_probe',SPEC['original_probe']); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.ART=ART;m.RES=RES
np=m.np;torch=m.torch;e=m.e;P=SPEC['identity'];ROOT=Path(P['dataset_root'])
START=time.monotonic(); MODEL=None;BEFORE=None

def save(name,x): m.save(name,x)
def log(stage,**kw):
    d={'stage':stage,'elapsed_seconds':time.monotonic()-START,**kw};print(json.dumps(d),flush=True);save('status.json',d)
    assert d['elapsed_seconds']<1800,'Numerical budget exceeded'

def flatgrad(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    out=torch.cat([(torch.zeros_like(p) if g is None else g).detach().cpu().reshape(-1) for p,g in zip(params,gs)])
    assert out.dtype==params[0].dtype and torch.isfinite(out).all()
    assert all(p.grad is None for p in params)
    return out

def cos(a,b):
    dot=aa=bb=0.
    for i in range(0,len(a),131072):
        x=a[i:i+131072].double();y=b[i:i+131072].double()
        dot+=float(torch.dot(x,y));aa+=float(torch.dot(x,x));bb+=float(torch.dot(y,y))
    return dot/math.sqrt(aa*bb) if aa>1e-24 and bb>1e-24 else None

def compare(actual,recon,groups,near=None,candidate=None):
    result={}
    for block,(lo,hi) in groups.items():
        aa=dd=mx=0.;bad=total=0; bins={str(t):0 for t in [1e-6,1e-5,1e-4,1e-3,1e-2,1.]};minbad=None;maxbad=0.; maxratio=0.
        for i in range(lo,hi,131072):
            a=actual[i:min(i+131072,hi)].double();b=recon[i:min(i+131072,hi)].double();d=(a-b).abs();allow=1e-6+1e-4*a.abs();mask=d>allow
            aa+=float(torch.dot(a,a));dd+=float(torch.dot(d,d));mx=max(mx,float(d.max()));maxratio=max(maxratio,float((d/allow).max()))
            bad+=int(mask.sum());total+=len(a)
            for t in bins:bins[t]+=int((mask&(a.abs()<=float(t))).sum())
            if mask.any():minbad=min(minbad if minbad is not None else math.inf,float(a[mask].abs().min()));maxbad=max(maxbad,float(a[mask].abs().max()))
        rms=math.sqrt(aa/total);nzbad=0
        for i in range(lo,hi,131072):
            a=actual[i:min(i+131072,hi)].double();b=recon[i:min(i+131072,hi)].double();nzbad+=int((((a-b).abs()>1e-6+1e-4*a.abs())&(a.abs()<=.001*rms)).sum())
        row={'elements':total,'actual_L2':math.sqrt(aa),'actual_RMS':rms,'max_abs_error':mx,'relative_L2_error':math.sqrt(dd/aa) if aa else math.sqrt(dd),'original_failing_elements':bad,'original_max_allowance_ratio':maxratio,'failures_by_abs_actual_le':bins,'failed_abs_actual_min':minbad,'failed_abs_actual_max':maxbad,'near_zero_cutoff_0_001_RMS':.001*rms,'near_zero_failing_elements':nzbad}
        if near is not None:
            row.update(cos_direct_near=cos(actual[lo:hi],near[lo:hi]),cos_reconstructed_near=cos(recon[lo:hi],near[lo:hi]))
            x,y=row['cos_direct_near'],row['cos_reconstructed_near'];row['cosine_absolute_effect']=abs(x-y) if x is not None and y is not None else None
        result[block]=row
    return result

def relations(g,fraction):
    return [('current_components',g['current'],g['L1']+.2*g['log']+.1*g['gradient']),('candidate_components',g['candidate'],g['current']+.2*g['near_MAE']),('near_far_partition',g['current'],g['current_near']+g['current_far']),('near_L1_normalization',g['near_L1_global'],fraction*g['near_MAE'])]

def hook_dtypes(model,dtype):
    seen={}; handles=[]
    def visit(name):
        def check(module,args,out):
            tensors=[x for x in args if torch.is_tensor(x)]+([out] if torch.is_tensor(out) else [])
            assert all(x.dtype==dtype for x in tensors if x.is_floating_point()),(name,[(x.dtype,x.shape) for x in tensors])
            seen[name]=str(dtype)
        return check
    for name,mod in model.named_modules():handles.append(mod.register_forward_hook(visit(name)))
    return seen,handles

def inputs(records,config):
    dl=e.loader(ROOT,records,config|{'device':'cuda'},'train',False);dl.multiprocessing_context='fork'
    result=[]
    for batch in dl:result.append({k:batch[k] for k in ['rgb','depth','valid']})
    assert len(result)==2
    return result

def run():
    global MODEL,BEFORE
    seal=json.loads((RES/'DIAGNOSTIC_FREEZE.json').read_text())
    for path,h in seal['files_sha256'].items():assert m.sha(REPO/path)==h,path
    assert e.source_hashes()==P['source_sha256']
    for key,hkey in [('teacher_checkpoint','teacher_sha256'),('failed_gate_path','failed_gate_sha256'),('frozen_protocol_path','frozen_protocol_sha256')]:assert m.sha(P[key])==P[hkey]
    assert not json.loads(Path(P['failed_gate_path']).read_text())['passed']
    assert torch.cuda.is_available();torch.set_num_threads(2);e.seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.hub.download_url_to_file=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Downloads forbidden'))
    samples=json.loads(Path(SPEC['original_sampling']).read_text()); records=[m.FrameRecord(**x['record']) for x in samples[:4]]
    allrows,_=e.load_inputs(e.PREP/'manifest_train_val.json.gz',e.PREP/'reservations.json',ROOT)
    assert [m.FrameRecord(**x['record']) for x in samples]==e.fixed_training_subset(allrows['train'],64)
    del allrows
    guard=m.Guard(ROOT,records);sys.addaudithook(guard.check)
    # Verify explicit guard blocks held-out and broader discovery without opening them.
    for event,args in [('open',(str(ROOT/'Kite_training/foggy/color_left/trajectory_2003/000000.JPEG'),'rb',0)),('os.listdir',(str(ROOT),))]:
        try:guard.check(event,args)
        except PermissionError:pass
        else:raise AssertionError('Read guard failure')
    wanted={str((ROOT/getattr(r,k)).relative_to(ROOT.parent)) for r in records for k in ['rgb_path','depth_path']}; prior={}
    with (ROOT.parent/'recovery/dataset_index.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['destination'] in wanted:prior[row['destination']]=row['sha256']
    assert len(prior)==8
    for rel,h in prior.items():assert m.sha(ROOT.parent/rel)==h
    payload=torch.load(P['teacher_checkpoint'],map_location='cpu',weights_only=False);config=payload['extra']['config'];identity=payload['extra']['identity']
    assert identity['source_sha256']==e.source_hashes() and identity['manifest_sha256']==e.MANIFEST_SHA and identity['arm']=='teacher' and identity['scope']=='full'
    assert payload['epoch']==2; e.check_config(config)
    assert config['loss']=={'l1_weight':1.,'log_weight':.2,'gradient_weight':.1,'near_weight':0.,'near_threshold':10.}
    MODEL=e.build_model(config['model']|{'pretrained':False,'min_depth':.1});MODEL.load_state_dict(payload['model'],strict=True)
    del payload;gc.collect();MODEL.cuda().requires_grad_(True).eval();BEFORE=e.state_digest(MODEL)
    expected=json.loads((Path(SPEC['original_plan']).parent/'runtime.json').read_text())
    assert BEFORE==expected['model_state_before']
    layout=expected['parameter_layout'];groups=expected['group_offsets']
    assert [(n,p.numel()) for n,p in MODEL.named_parameters()]==[(x['name'],x['end']-x['start']) for x in layout]
    batches=inputs(records,config);report=[]
    save('runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'state_initial':BEFORE,'groups':groups,'parameter_layout':layout,'first_two_samples':samples[:4],'verified_input_hashes':prior,'original_guard_checks':True,'pid':os.getpid()})
    seen,handles=hook_dtypes(MODEL,torch.float32)
    for bi,batch in enumerate(batches):
        log('FP32_batch',batch=bi)
        rgb,gt,v=[batch[k].cuda() for k in ['rgb','depth','valid']];params=list(MODEL.parameters());rng=m.rng_hash();hashes={k:m.tensor_hash(x) for k,x in [('rgb',rgb),('gt',gt),('valid',v)]}
        with torch.enable_grad(),torch.amp.autocast('cuda',enabled=False):
            pred=MODEL(rgb);o,c=m.objectives(pred,gt,v,config['loss']);assert all(x.dtype==torch.float32 for x in o.values())
            assert pred.dtype==torch.float32 and torch.isfinite(pred).all() and (pred>=.1).all()
            pg={k:torch.autograd.grad(x,pred,retain_graph=True)[0].detach() for k,x in o.items()}
            checks_out={name:compare(a.cpu().flatten(),b.cpu().flatten(),{'prediction':(0,pred.numel())},pg['near_MAE'].cpu().flatten()) for name,a,b in relations(pg,c['near_fraction'])}
            # Loss-only double reference starts from exactly the same prediction values, not a fresh forward.
            pd=pred.detach().double().requires_grad_(True);od,cd=m.objectives(pd,gt.double(),v,config['loss']);assert all(x.dtype==torch.float64 for x in od.values())
            gd={k:torch.autograd.grad(x,pd,retain_graph=True)[0].detach().cpu().flatten() for k,x in od.items()}
            out64={name:compare(a,b,{'prediction':(0,pred.numel())},gd['near_MAE']) for name,a,b in relations(gd,c['near_fraction'])}
            del gd,od,pd
            g={k:flatgrad(loss,params) for k,loss in o.items()}
            checks={name:compare(a,b,groups,g['near_MAE']) for name,a,b in relations(g,c['near_fraction'])}
            # Independent repeat on the same graph. autograd.grad never accumulates .grad buffers.
            repeats={k:flatgrad(o[k],params) for k in ['L1','log','gradient','current']}
            repeat={k:bool(torch.equal(g[k],repeats[k])) for k in repeats}
            del repeats
            shared=torch.autograd.grad(pred,params,grad_outputs=pg['L1']+.2*pg['log']+.1*pg['gradient'],retain_graph=True,allow_unused=True)
            shared=torch.cat([(torch.zeros_like(p) if x is None else x).detach().cpu().flatten() for p,x in zip(params,shared)])
            jac=compare(g['current'],shared,groups,g['near_MAE']);del shared
            aggregate64=compare(g['current'],g['L1'].double()+.2*g['log'].double()+.1*g['gradient'].double(),groups,g['near_MAE'])
        assert e.state_digest(MODEL)==BEFORE and m.rng_hash()==rng and all(p.grad is None for p in params)
        assert hashes=={k:m.tensor_hash(x) for k,x in [('rgb',rgb),('gt',gt),('valid',v)]}
        row={'batch_index':bi,'counts':c,'losses':{k:float(x.detach()) for k,x in o.items()},'input_hashes':hashes,'prediction_sha256':m.tensor_hash(pred),'output_FP32':checks_out,'output_FP64_same_prediction':out64,'parameters_FP32':checks,'FP64_sum_of_FP32_parameter_components':aggregate64,'shared_Jacobian_weighted_output_VJP':jac,'same_graph_repeat_bitwise':repeat,'state_and_RNG_unchanged':True,'all_grad_fields_none':True,'groups':m.group_stats(g,groups),'module_io_dtypes':seen.copy()}
        report.append(row);save('diagnostic_batches.json',report)
        del g,pg,o,pred,params,rgb,gt,v;gc.collect();torch.cuda.empty_cache()
    for h in handles:h.remove()
    assert e.state_digest(MODEL)==BEFORE
    del MODEL;MODEL=None;gc.collect();torch.cuda.empty_cache()
    # Independent genuine FP64 model, from identical checkpoint values; no FP32 wrapper is called.
    payload=torch.load(P['teacher_checkpoint'],map_location='cpu',weights_only=False)
    MODEL=e.build_model(config['model']|{'pretrained':False,'min_depth':.1});MODEL.load_state_dict(payload['model'],strict=True);del payload
    MODEL=MODEL.double().cuda().requires_grad_(True).eval();BEFORE=e.state_digest(MODEL); seen,handles=hook_dtypes(MODEL,torch.float64)
    assert all(x.dtype==torch.float64 for x in MODEL.parameters())
    assert all(x.dtype==torch.float64 for x in MODEL.buffers() if x.is_floating_point())
    ref=[]
    for bi,batch in enumerate(batches):
        log('full_FP64_batch',batch=bi)
        rgb,gt,v=[batch[k].cuda().to(torch.float64) if k!='valid' else batch[k].cuda() for k in ['rgb','depth','valid']];params=list(MODEL.parameters());rng=m.rng_hash()
        with torch.enable_grad(),torch.amp.autocast('cuda',enabled=False):
            pred=MODEL(rgb);o,c=m.objectives(pred,gt,v,config['loss'])
            assert pred.dtype==gt.dtype==rgb.dtype==torch.float64 and all(x.dtype==torch.float64 for x in o.values())
            keys=['L1','log','gradient','near_MAE','current','candidate'];g={k:flatgrad(o[k],params) for k in keys}
        checks={name:compare(a,b,groups,g['near_MAE']) for name,a,b in [('current_components',g['current'],g['L1']+.2*g['log']+.1*g['gradient']),('candidate_components',g['candidate'],g['current']+.2*g['near_MAE'])]}
        assert all(r['relative_L2_error']<=1e-10 for check in checks.values() for r in check.values())
        assert e.state_digest(MODEL)==BEFORE and m.rng_hash()==rng and all(p.grad is None for p in params)
        fp64cos={block:{'current_near':cos(g['current'][lo:hi],g['near_MAE'][lo:hi]),'candidate_near':cos(g['candidate'][lo:hi],g['near_MAE'][lo:hi])} for block,(lo,hi) in groups.items()}
        ref.append({'batch_index':bi,'counts':c,'all_forward_module_io_dtypes':seen.copy(),'RGB_GT_prediction_loss_parameter_gradient_dtype':'torch.float64','model_floating_buffer_dtype':'torch.float64','losses':{k:float(o[k].detach()) for k in keys},'checks':checks,'cosines':fp64cos,'state_and_RNG_unchanged':True,'all_grad_fields_none':True})
        save('full_FP64_reference.json',ref)
        del g,o,pred,params,rgb,gt,v;gc.collect();torch.cuda.empty_cache()
    for h in handles:h.remove()
    assert e.state_digest(MODEL)==BEFORE
    for key,hkey in [('teacher_checkpoint','teacher_sha256'),('failed_gate_path','failed_gate_sha256'),('frozen_protocol_path','frozen_protocol_sha256')]:assert m.sha(P[key])==P[hkey]
    assert e.source_hashes()==P['source_sha256']
    save('diagnostic_completion.json',{'passed':True,'numerical_seconds':time.monotonic()-START,'batches':2,'unique_TRAIN_frames':4,'optimizer_instances':0,'optimizer_steps':0,'FP32_state_and_BN_unchanged':True,'FP64_state_and_BN_unchanged':True,'all_RNG_checks_passed':True,'persistent_checkpoint_gate_protocol_source_unchanged':True,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    log('diagnostic_complete')

if __name__=='__main__':
    try:run()
    except BaseException as ex:
        save('failure.json',{'error':repr(ex),'traceback':traceback.format_exc(),'numerical_seconds':time.monotonic()-START,'state_digest':e.state_digest(MODEL) if MODEL is not None else None,'expected_digest':BEFORE})
        raise
    finally:
        save('numerical_budget_diagnostic.json',{'seconds_used':time.monotonic()-START,'limit_seconds':1800})
