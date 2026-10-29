"""Guarded local presentation artifacts; historical research runners are unchanged."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse, io, json, math, os, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from . import expanded as e
from .inference import file_sha256, load_inference_model
from .metrics import DepthMetricAccumulator
from .model import build_model, model_contract, model_stats
from .presentation_scheduler import PresentationScheduler, POLICIES, DISABLED, fixed_positions
from .scheduler import near_field_risk
from .scheduler_v2 import SchedulerV2, thumbnail
from .research import summarize

VERSION = 'presentation_validation_v1'


def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    e.write_json(path,value)

def validation_inputs(config):
    root = Path(config['dataset_root'])
    records, manifest = e.load_inputs(Path(config['manifest']),Path(config['reservations']),root)
    return root,records['val'],manifest


class ValidationGuard(e.ImageReadGuard):
    """No discovery, test pixels, archives or arbitrary dataset metadata opens."""
    def __init__(self,root,records,sensors=()):
        super().__init__(root,records)
        self.sensors = {os.path.abspath(p) for p in sensors}
    def check(self,event,args):
        if event not in ('open','os.scandir','os.listdir') or not args or isinstance(args[0],int):return
        p=os.path.abspath(os.fsdecode(args[0]))
        if p=='/run/media/Ima/Low storage' or p.startswith('/run/media/Ima/Low storage/'):
            raise PermissionError('Failed HDD is not an input source')
        if event=='open' and p in self.sensors:
            flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
            if flags & 3:raise PermissionError('Sensor writes forbidden')
            return
        super().check(event,args)


def verify_file(path,sha):
    if not Path(path).is_file() or file_sha256(path)!=sha:
        raise ValueError('Missing or changed pinned input: '+str(path))


def load_model(entry,device='cpu'):
    verify_file(entry['weights'],entry['sha256'])
    if entry['kind']=='export':
        verify_file(entry['metadata'],entry['metadata_sha256'])
        model,info=load_inference_model(entry['weights'],entry['metadata'],device)
        config=info['training_config']
    else:
        cp=torch.load(entry['weights'],map_location='cpu',weights_only=False)
        if cp['epoch']+1!=entry['selected_epoch']:raise ValueError('Selected epoch mismatch')
        config=cp['extra']['config']
        model=build_model(config['model']|{'pretrained':False,'min_depth':.1})
        model.load_state_dict(cp['model'],strict=True);model=model.to(device).eval()
    model.requires_grad_(False)
    return model,config


def export_a(config,out):
    """Separate inference artifact; refuses to overwrite any earlier export."""
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    item=config['source_A'];verify_file(item['path'],item['sha256'])
    cp=torch.load(item['path'],map_location='cpu',weights_only=False)
    assert cp['epoch']==8 and cp['extra']['identity']['arm']=='supervised'
    assert cp['extra']['identity']['source_sha256']==e.source_hashes()
    cfg=cp['extra']['config'];original=build_model(cfg['model']|{'pretrained':False,'min_depth':.1}).eval()
    original.load_state_dict(cp['model'],strict=True);before=e.state_digest(original)
    out.mkdir(parents=True);weights=out/'supervised_A_epoch9.state_dict.pt';meta=out/'supervised_A_epoch9.json'
    torch.save({k:v.detach().cpu().contiguous() for k,v in original.state_dict().items()},weights)
    info={'format':'plain state_dict including all BN buffers; strict loading','architecture':'mobilenet_v3_small_depth',
          'parameters':model_stats(original)['parameters'],'original_checkpoint_sha256':item['sha256'],
          'export_sha256':file_sha256(weights),'epoch_human':9,'selection':'minimum pooled FP32 validation AbsRel',
          'training_config':cfg,'contract':model_contract(cfg),'parent_source_sha256':e.source_hashes(),
          'designation':'General-depth presentation candidate; near-field regression versus original; no automatic published-default change'}
    write(meta,info);restored,_=load_inference_model(weights,meta)
    generator=torch.Generator().manual_seed(42);checks=[]
    root,records,_=validation_inputs(config)
    guard=ValidationGuard(root,records);__import__('sys').addaudithook(guard.check)
    real=e.GuardedDataset(root,records,cfg['data'],'val',False)[0]['rgb'][None]
    with torch.inference_mode():
        for value in [torch.zeros(1,3,192,320),torch.ones(1,3,192,320),torch.rand(1,3,192,320,generator=generator),real]:
            x=original(value);y=restored(value);assert torch.equal(x,y);checks.append(float((x-y).abs().max()))
    assert e.state_digest(original)==before==e.state_digest(restored)
    info['strict_export_parity']={'passed':True,'precision':'CPU FP32 eval','inputs':'black, white, seeded random and first allowed validation RGB','max_abs_errors':checks,'state_digest':before}
    write(meta,info)
    verify_file(item['path'],item['sha256'])
    return {'kind':'export','weights':str(weights.absolute()),'metadata':str(meta.absolute()),'sha256':file_sha256(weights),'metadata_sha256':file_sha256(meta),'selected_epoch':9,'label':'Supervised A · epoch 9'}


def decode_pair(rgb_bytes,depth_bytes,data):
    rgb=np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert('RGB'),dtype=np.float32)/255.
    raw=np.asarray(Image.open(io.BytesIO(depth_bytes))).astype(np.uint16,copy=False)
    depth=raw.view(np.float16).astype(np.float32)
    if depth.ndim==3:depth=depth[...,0]
    if rgb.shape[:2]!=depth.shape[:2]:raise ValueError('RGB/depth alignment mismatch')
    x=torch.from_numpy(rgb).permute(2,0,1);y=torch.from_numpy(depth).unsqueeze(0)
    valid=torch.isfinite(y)&(y>=data['min_depth'])&(y<=data['max_depth'])
    y=torch.where(valid,y,torch.zeros_like(y));size=data['image_size']
    return (F.interpolate(x[None],size,mode='bilinear',align_corners=False)[0],
            F.interpolate(y[None],size,mode='nearest')[0],
            F.interpolate(valid.float()[None],size,mode='nearest')[0].bool())


def read_sample(root,record,data):
    e.check_record(record,'val',root)
    start=time.perf_counter();rgb=(root/record.rgb_path).read_bytes();depth=(root/record.depth_path).read_bytes();io_s=time.perf_counter()-start
    start=time.perf_counter();x,y,v=decode_pair(rgb,depth,data);prep=time.perf_counter()-start
    return x,y,v,io_s,prep


def causal_navigation(root,records):
    import h5py
    result=[];provenance={};grouped=defaultdict(list)
    for r in records:grouped[r.trajectory_id].append(r)
    for tr,rs in grouped.items():
        parent,group=tr.rsplit('/',1);path=root/parent/'sensor_records.hdf5'
        with h5py.File(path,'r') as h:
            gps=h[group+'/gps/velocity'];frequency=float(gps.attrs['sampling_frequency']);values=np.asarray(gps[:])
        provenance[tr]={'sensor':str(path),'sha256':file_sha256(path),'gps_frequency_Hz':frequency,'image_frequency_Hz':25.,'mapping':'floor(ordinal*GPS_Hz/25), held causal sample; no future interpolation'}
        for ordinal,r in enumerate(rs):
            nav={k:r.navigation[k] for k in ['ax','ay','az','wx','wy','wz'] if k in r.navigation}
            sample=min(int(ordinal*frequency/25),len(values)-1)
            nav.update({k:float(v) for k,v in zip(['gps_vx','gps_vy','gps_vz'],values[sample]) if np.isfinite(v)})
            result.append(nav)
    return result,provenance


def cache_validation(config,protocol,result):
    root,records,_=validation_inputs(config);cache=Path(config['cache']);cache.mkdir(parents=True,exist_ok=False)
    sensors=[root/Path(r.trajectory_id).parent/'sensor_records.hdf5' for r in records]
    guard=ValidationGuard(root,records,sensors);__import__('sys').addaudithook(guard.check)
    nav,sensors=causal_navigation(root,records)
    model,cfg=load_model(config['models']['A'],'cuda');before=e.state_digest(model);n=len(records);shape=(n,1,192,320)
    pred=np.lib.format.open_memmap(cache/'predictions.npy',mode='w+',dtype='float32',shape=shape)
    gt=np.lib.format.open_memmap(cache/'targets.npy',mode='w+',dtype='float32',shape=shape)
    valid=np.lib.format.open_memmap(cache/'valid.npy',mode='w+',dtype='bool',shape=shape)
    displays=np.lib.format.open_memmap(cache/'display_RGB_uint8.npy',mode='w+',dtype='uint8',shape=(n,192,320,3))
    gray=np.lib.format.open_memmap(cache/'gray.npy',mode='w+',dtype='float32',shape=(n,36,64))
    dl=e.loader(root,records,cfg|{'device':'cuda','train':cfg['train']|{'batch_size':1}},'val',False);dl.multiprocessing_context='fork'
    overall=DepthMetricAccumulator();per=defaultdict(DepthMetricAccumulator);features=[];lasttr=None;lastgt=math.inf;positions=defaultdict(int)
    start=time.perf_counter()
    with torch.inference_mode():
        for i,batch in enumerate(dl):
            rgb=batch['rgb'];g=batch['depth'];v=batch['valid'];tr=records[i].trajectory_id
            with torch.amp.autocast('cuda',enabled=False):p=model(rgb.cuda()).cpu()
            if not torch.isfinite(p).all() or not (p>=.1).all():raise ValueError('Invalid fresh prediction')
            overall.update(p,g,v);per[tr].update(p,g,v);pred[i]=p.numpy();gt[i]=g.numpy();valid[i]=v.numpy()
            displays[i]=np.rint(rgb[0].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');gray[i]=thumbnail(rgb)
            risk,available=near_field_risk(p,10.)
            values=g[v];p10=float(torch.quantile(values,.1)) if values.numel() else math.inf
            if tr!=lasttr:lastgt=math.inf;lasttr=tr
            event=p10<8 and (lastgt>=8 or lastgt-p10>=2);lastgt=p10
            features.append({'index':i,'trajectory':tr,'frame_index':records[i].frame_index,'position':positions[tr],'navigation':nav[i],
                             'fresh_risk':risk,'risk_available':available,'gt_p10':p10,'critical_event':event})
            positions[tr]+=1
            if (i+1)%250==0:print(json.dumps({'stage':'A_batch1_cache','frames':i+1,'total':n}),flush=True)
    reference=read(Path(config['comparison'])/'metrics.json')['students']['student_A']['scopes']
    differences={}
    for scope,acc in [('pooled',overall),*per.items()]:
        actual=acc.compute()|{'near_valid_pixels':acc.near_count};expected=reference[scope];differences[scope]={}
        for k,value in actual.items():
            delta=value-expected[k];limit=0 if k.endswith('pixels') else .001 if k.endswith('rmse') else 1e-5
            if abs(delta)>limit:raise ValueError(f'Batch1 reference mismatch {scope} {k}: {delta}')
            differences[scope][k]=delta
    assert e.state_digest(model)==before
    for x in [pred,gt,valid,displays,gray]:x.flush()
    write(cache/'features.json',features);write(result/'causal_sensor_provenance.json',sensors)
    write(result/'cache_reference.json',{'passed':True,'differences':differences,'frame_count':n,'batch_size':1,'model_state_before_and_after':before,'seconds_including_IO_scoring_cache':time.perf_counter()-start,'display_RGB_quantized_only_for_playback':True,'recorded_playback_is_not_live_inference':True})
    write(cache/'completion.json',{'passed':True,'frames':n,'export_sha256':config['models']['A']['sha256'],'manifest_sha256':e.MANIFEST_SHA,'artifacts':{f.name:{'bytes':f.stat().st_size,'sha256':file_sha256(f)} for f in cache.iterdir() if f.is_file()}})


def replay_quality(config,protocol,result):
    cache=Path(config['cache']);assert read(cache/'completion.json')['passed']
    p=np.load(cache/'predictions.npy',mmap_mode='r');g=np.load(cache/'targets.npy',mmap_mode='r');v=np.load(cache/'valid.npy',mmap_mode='r');gray=np.load(cache/'gray.npy',mmap_mode='r');features=read(cache/'features.json')
    lengths=Counter(x['trajectory'] for x in features);results={};summaryrows=[];equivalence={}
    for policy in POLICIES:
        logs=[];acc=defaultdict(DepthMetricAccumulator);total=DepthMetricAccumulator();previous=None;source=None;lasttr=None;age=0
        sched=PresentationScheduler(protocol['v2'],DISABLED.get(policy,()))
        original=SchedulerV2(protocol['v2']);risk=0.;available=False;eq=0
        for i,f in enumerate(features):
            tr=f['trajectory'];pos=f['position']
            if tr!=lasttr:
                sched.reset();original.reset();source=None;previous=None;age=0;risk=0.;available=False;lasttr=tr
            if policy in DISABLED:
                d=asdict(sched.decide_features(gray[i],f['navigation'],risk,available));infer=d['infer']
                if policy=='v2':assert d==asdict(original.decide_features(gray[i],f['navigation'],risk,available));eq+=1
            else:
                infer=pos in fixed_positions(policy,lengths[tr],**protocol['budget_fraction'])
                d={'infer':infer,'reason':'first_frame' if pos==0 else 'fixed_interval' if infer else 'fixed_skip','age_before':age,'age_after':0 if infer else age+1,'max_skip_forced':False}
            if infer:source=i;previous=torch.tensor(np.array(p[i]));risk=f['fresh_risk'];available=f['risk_available']
            assert previous is not None
            age=i-source;assert age==d['age_after'];assert age<=4 if policy!='budget_fixed' else True
            target=torch.tensor(np.array(g[i]));mask=torch.tensor(np.array(v[i]));acc[tr].update(previous,target,mask);total.update(previous,target,mask)
            values=previous[mask];p10=float(torch.quantile(values,.1)) if f['critical_event'] and values.numel() else None
            logs.append({'index':i,'trajectory':tr,'position':pos,'frame_index':f['frame_index'],'source_index':source,'prediction_age':age,**d,'critical_event':f['critical_event'],'offline_predicted_p10_on_event':p10})
        metrics={tr:a.compute()|{'near_valid_pixels':a.near_count} for tr,a in acc.items()};out=summarize(logs,metrics);out['overall']['metrics']=total.compute()|{'near_valid_pixels':total.near_count}
        out.update({'scope':'A validation; counterfactual frozen per-frame prediction cache, NOT live timing','policy':policy,'removed':list(DISABLED.get(policy,())),'surviving_weights_renormalized':False})
        results[policy]=out;write(result/'policies'/f'{policy}.json',out);write(result/'policies'/f'{policy}_frames.json',logs)
        for scope,row in [('pooled',out['overall']),*out['per_trajectory'].items()]:
            event=row['critical_events'];summaryrows.append({'policy':policy,'scope':scope,**row['metrics'],'calls':row['inference_calls'],'skip_rate':row['skip_rate'],'mean_age':row['mean_prediction_age'],'max_age':row['max_prediction_age'],'events':event['count'],'missed_update_rate':event['missed_update_rate'],'max_event_delay':event['max_observed_delay_frames'],'censored_events':event['right_censored_count']})
        equivalence[policy]={'historical_v2_feature_decisions_equal':eq if policy=='v2' else None}
        print(json.dumps({'stage':'quality','policy':policy,'calls':out['overall']['inference_calls']}),flush=True)
    write(result/'scheduler_results.json',results);write(result/'historical_v2_equivalence.json',equivalence)
    import csv
    with (result/'scheduler_comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summaryrows[0]));w.writeheader();w.writerows(e.clean_json(summaryrows))


def verify_live_cache(config,protocol,result):
    """Recompute risk on actual full-resolution predictions for uncached reference."""
    cache=Path(config['cache']);features=read(cache/'features.json');gray=np.load(cache/'gray.npy',mmap_mode='r');p=np.load(cache/'predictions.npy',mmap_mode='r');report={}
    for policy,disabled in DISABLED.items():
        cached=PresentationScheduler(protocol['v2'],disabled,True);uncached=PresentationScheduler(protocol['v2'],disabled,False)
        previous=None;lasttr=None;source_c=source_u=None;updates=0
        for i,f in enumerate(features):
            if f['trajectory']!=lasttr:
                cached.reset();uncached.reset();previous=None;source_c=source_u=None;lasttr=f['trajectory']
            risk,available=near_field_risk(previous,10.) if 'risk' not in disabled else (0.,False)
            a=asdict(cached.decide_features(gray[i],f['navigation'],cached.risk_value,cached.risk_available))
            b=asdict(uncached.decide_features(gray[i],f['navigation'],risk,available))
            if a!=b:raise AssertionError(f'Live risk-cache decision differs at {policy}:{i}')
            if a['infer']:
                source_c=source_u=i;previous=torch.tensor(np.array(p[i]));cached.update_prediction(previous,i);updates+=int('risk' not in disabled)
            assert source_c==source_u and i-source_c==a['age_after']
        report[policy]={'passed':True,'frames':len(features),'compared':'all Decision fields including reasons/scores, source indices, ages; uncached full-prediction risk recomputation','risk_updates':updates}
    write(result/'live_cache_equivalence.json',report)


def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


def actual_timing(config,protocol,result):
    root,allrecords,_=validation_inputs(config);windows=protocol['timing']['windows'];records=[]
    for tr,w in windows.items():
        rows=[r for r in allrecords if r.trajectory_id==tr]
        records.extend(rows[w['start']:w['start']+w['frames']])
    guard=ValidationGuard(root,records);__import__('sys').addaudithook(guard.check)
    allfeatures=read(Path(config['cache'])/'features.json');nav={(f['trajectory'],f['frame_index']):f['navigation'] for f in allfeatures}
    devices=['cpu']+(['cuda'] if torch.cuda.is_available() else []);policies=protocol['timing']['policies'];rows=[];frame_rows=[]
    for repetition in range(3):
        for device in devices if repetition%2==0 else list(reversed(devices)):
            model,cfg=load_model(config['models']['A'],device);before=e.state_digest(model)
            order=policies if repetition==0 else list(reversed(policies)) if repetition==1 else policies[2:]+policies[:2]
            for policy in order:
                warm=read_sample(root,records[0],cfg['data'])[0][None].to(device)
                with torch.inference_mode():
                    for _ in range(10):model(warm)
                synchronize(device)
                sched=PresentationScheduler(protocol['v2'],cache_risk=policy!='v2_uncached')
                previous=None;source=None;lasttr=None;age=0;position=0;calls=0;frames=[];lengths=Counter(r.trajectory_id for r in records)
                with torch.inference_mode():
                    for index,record in enumerate(records):
                        beginning=time.perf_counter();x,g,v,io_s,prep_s=read_sample(root,record,cfg['data'])
                        if record.trajectory_id!=lasttr:sched.reset();previous=None;source=None;age=0;position=0;lasttr=record.trajectory_id
                        t=time.perf_counter()
                        if policy.startswith('v2'):
                            d=asdict(sched.decide(x,nav[(record.trajectory_id,record.frame_index)],previous));infer=d['infer']
                        else:
                            infer=position in fixed_positions(policy,lengths[record.trajectory_id],**protocol['budget_fraction'])
                            d={'infer':infer,'reason':'first_frame' if position==0 else 'fixed_interval' if infer else 'fixed_skip','age_before':age,'age_after':0 if infer else age+1}
                        scheduler_s=time.perf_counter()-t;model_s=transfer_s=update_s=0.
                        if infer:
                            synchronize(device);t=time.perf_counter();inp=x[None].to(device);synchronize(device);transfer_s=time.perf_counter()-t
                            t=time.perf_counter();out=model(inp);synchronize(device);model_s=time.perf_counter()-t
                            t=time.perf_counter();previous=out.cpu()[0];synchronize(device);transfer_s+=time.perf_counter()-t
                            if not torch.isfinite(previous).all() or not (previous>=.1).all():raise ValueError('Nonfinite timing output')
                            t=time.perf_counter()
                            if policy.startswith('v2'):sched.update_prediction(previous,index)
                            update_s=time.perf_counter()-t;scheduler_s+=update_s;source=index;calls+=1
                        age=index-source;assert age==d['age_after'];position+=1
                        frames.append({'rep':repetition+1,'device':device,'policy':policy,'trajectory':record.trajectory_id,'frame_index':record.frame_index,'infer':infer,'reason':d['reason'],'source_index':source,'age':age,'model_s':model_s,'scheduler_s':scheduler_s,'risk_update_s':update_s,'transfer_s':transfer_s,'io_s':io_s,'preprocess_s':prep_s,'end_to_end_s':time.perf_counter()-beginning})
                model.assert_teacher_frozen() if hasattr(model,'assert_teacher_frozen') else None
                assert e.state_digest(model)==before
                row={'rep':repetition+1,'device':device,'policy':policy,'frames':len(frames),'calls':calls,'excluded_warmup_calls':10,'batch_size':1,'threads':torch.get_num_threads(),**{key:sum(f[key] for f in frames) for key in ['model_s','scheduler_s','risk_update_s','transfer_s','io_s','preprocess_s','end_to_end_s']}}
                lat=[f['model_s']*1000 for f in frames if f['infer']];row.update(model_call_median_ms=float(np.median(lat)),model_call_p95_ms=float(np.quantile(lat,.95)))
                rows.append(row);frame_rows.extend(frames);write(result/'timing_individual.json',rows)
                print(json.dumps({'stage':'actual_timing',**row}),flush=True)
            del model
            if device=='cuda':torch.cuda.empty_cache()
    summaries=[]
    for device in devices:
        for policy in policies:
            selected=[r for r in rows if r['device']==device and r['policy']==policy]
            summary={'device':device,'policy':policy,'repetitions':len(selected),'actual_calls':[r['calls'] for r in selected]}
            for key in ['model_s','scheduler_s','risk_update_s','transfer_s','io_s','preprocess_s','end_to_end_s','model_call_median_ms']:
                values=[r[key] for r in selected];summary[key]={'median':float(np.median(values)),'min':min(values),'max':max(values)}
            summaries.append(summary)
    write(result/'timing_summary.json',{'scope':'Bounded actual batch1 A inference on fixed128-frame windows per trajectory; reset at each window. Not full-path latency, not offline replay cache speed.','windows':windows,'records':len(records),'devices':devices,'warmups':10,'repetitions':3,'OS_cache':'uncontrolled/warm from prior validation; alternating orders recorded','end_to_end':'Includes actual JPEG+depth PNG file reads, decoding/resize, transfers, model, scheduler/cache update and instrumentation. GT only for validation display/pipeline, never a decision input. Excludes offline quality scoring and final file writes.','rows':summaries})
    import csv
    for name,data in [('timing_individual.csv',rows),('timing_frames.csv',frame_rows)]:
        with (result/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # Actual cached/uncached traces must be identical for each hardware repetition.
    equivalence=[]
    for device in devices:
        for rep in range(1,4):
            a=[x for x in frame_rows if x['device']==device and x['rep']==rep and x['policy']=='v2_cached']
            b=[x for x in frame_rows if x['device']==device and x['rep']==rep and x['policy']=='v2_uncached']
            assert [{k:x[k] for k in ['trajectory','frame_index','infer','reason','source_index','age']} for x in a]==[{k:x[k] for k in ['trajectory','frame_index','infer','reason','source_index','age']} for x in b]
            equivalence.append({'device':device,'rep':rep,'passed':True,'frames':len(a)})
    write(result/'timed_live_cache_equivalence.json',equivalence)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['export','cache','quality','cache-check','timing']);parser.add_argument('--config',type=Path,default=Path('configs/presentation_local.json'));args=parser.parse_args(argv)
    config=read(args.config);result=Path(config['results']);result.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);e.seed_everything(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage=='export':
        entry=export_a(config,config['export_directory']);config['models']['A']=entry;write(args.config,config);write(result/'export.json',entry);print(json.dumps(entry));return
    protocol=read(result/'PROTOCOL.json')
    if protocol['model_export_sha256']!=config['models']['A']['sha256']:raise ValueError('Protocol/model identity changed')
    for name,sha in protocol['source_sha256'].items():verify_file(e.PROJECT/name,sha)
    if args.stage=='cache':cache_validation(config,protocol,result)
    elif args.stage=='quality':replay_quality(config,protocol,result)
    elif args.stage=='cache-check':verify_live_cache(config,protocol,result)
    elif args.stage=='timing':actual_timing(config,protocol,result)


if __name__=='__main__':main()
