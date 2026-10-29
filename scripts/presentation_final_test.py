"""Explicit opt-in future test exposure; proposed/blocked protocols cannot run.

Never imported by the dashboard. No discovery, overwrite, resume or retry.
"""
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import argparse,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from aerodepth_lite import expanded as e
from aerodepth_lite.data import FrameRecord,MidAirDataset
from aerodepth_lite.presentation import read,write,verify_file,load_model,ValidationGuard,causal_navigation,synchronize
from aerodepth_lite.presentation_scheduler import PresentationScheduler,fixed_positions
from aerodepth_lite.metrics import DepthMetricAccumulator
from aerodepth_lite.research import summarize

EXPECTED={'Kite_training/foggy/trajectory_2007','Kite_training/sunny/trajectory_0008',
          'PLE_training/spring/trajectory_5008','PLE_training/winter/trajectory_6009'}


def ready(protocol,path,expected_sha,manifest):
    verify_file(path,expected_sha)
    if protocol.get('status')!='frozen_ready_for_explicit_test' or not protocol.get('test_manifest_sha256'):
        raise ValueError('No test access: proposal is blocked pending healthy reserved data, verified manifest and a separately frozen ready protocol.')
    if protocol.get('test_realizations')!=sorted(EXPECTED):raise ValueError('Reserved test identity changed')
    if not protocol.get('dataset_root') or set(protocol.get('verified_frame_counts',{}))!=EXPECTED or not protocol.get('sensor_sha256'):
        raise ValueError('No test access: verified root/count/sensor identities missing')
    verify_file(protocol['validation_protocol'],protocol['validation_protocol_sha256'])
    validation=read(protocol['validation_protocol'])
    if protocol['v2']!=validation['v2'] or protocol['budget_fraction']!=validation['budget_fraction']:
        raise ValueError('No retuning of the validation-frozen policy at test time')
    if protocol['policies']!=['every','fixed2','fixed5','budget_fixed','v2']:
        raise ValueError('Final comparison matrix changed')
    verify_file(manifest,protocol['test_manifest_sha256'])
    for name,sha in protocol['source_sha256'].items():verify_file(ROOT/name,sha)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--confirm-new-test-exposure',action='store_true');a=p.parse_args(argv)
    if not a.confirm_new_test_exposure:p.error('Explicit --confirm-new-test-exposure required; no inputs opened')
    protocol=read(a.protocol);ready(protocol,a.protocol,a.protocol_sha256,a.manifest)
    m=read(a.manifest);records=[FrameRecord(**r) for r in m['records']]
    if set(r.trajectory_id for r in records)!=EXPECTED or len(records)!=m['frame_counts']['test']:
        raise ValueError('Manifest scope/count differs from reserved metadata')
    if str(a.dataset_root.absolute())!=protocol['dataset_root']:raise ValueError('Test dataset root differs from frozen proposal')
    for r in records:
        env,climate,tr=r.trajectory_id.split('/')
        for field,stream,ext in [('rgb_path','color_left','JPEG'),('depth_path','depth','PNG')]:
            relative=Path(getattr(r,field))
            if relative.is_absolute() or '..' in relative.parts or relative.parts[-5:]!=(env,climate,stream,tr,f'{r.frame_index:06d}.{ext}'):
                raise ValueError('Unsafe/mismatched test path')
            for q in (a.dataset_root/relative,*(a.dataset_root/relative).parents):
                if q.is_symlink():raise ValueError('Test symlinks forbidden')
                if q==a.dataset_root:break
    keys=[(r.trajectory_id,r.frame_index) for r in records]
    if len(set(keys))!=len(records) or keys!=sorted(keys):raise ValueError('Duplicate/nonchronological test frames')
    for tr,count in protocol['verified_frame_counts'].items():
        indices=[r.frame_index for r in records if r.trajectory_id==tr]
        if indices!=list(range(count)):raise ValueError('Verified full-trajectory counts/indices mismatch')
    if a.output.exists():raise FileExistsError('Existing exposure/output is never reused')
    if a.output.absolute()==a.dataset_root.absolute() or a.dataset_root.absolute() in a.output.absolute().parents:raise ValueError('Output cannot be inside dataset')
    sensors=[a.dataset_root/Path(r.trajectory_id).parent/'sensor_records.hdf5' for r in records]
    guard=ValidationGuard(a.dataset_root,records,sensors);sys.addaudithook(guard.check)
    if set(protocol['sensor_sha256'])!={str(p.absolute()) for p in sensors}:raise ValueError('Exact sensor identity list required')
    for path,sha in protocol['sensor_sha256'].items():verify_file(path,sha)
    navigation,_=causal_navigation(a.dataset_root,records)
    torch.set_num_threads(2);e.seed_everything(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=protocol['device'];model,cfg=load_model(protocol['model'],device);before=e.state_digest(model)
    a.output.mkdir(parents=True)
    # The exclusive ledger precedes ANY Image.open/decoder access to test pixels.
    with (a.output/'EXPOSURE.json').open('x') as f:
        json.dump({'protocol_sha256':a.protocol_sha256,'manifest_sha256':protocol['test_manifest_sha256'],'model_sha256':protocol['model']['sha256'],'status':'exposure_started','no_retry_or_resume':True},f,indent=2);f.flush();os.fsync(f.fileno())
    ds=MidAirDataset(a.dataset_root,records,cfg['data'],False);lengths={tr:sum(r.trajectory_id==tr for r in records) for tr in EXPECTED};all_results={}
    with torch.inference_mode():
        for policy in protocol['policies']:
            scheduler=PresentationScheduler(protocol['v2']);previous=None;lasttr=None;source=None;position=age=0;lastgt=math.inf;logs=[];acc=defaultdict(DepthMetricAccumulator);overall=DepthMetricAccumulator()
            for i,r in enumerate(records):
                sample=ds[i];x,g,v=sample['rgb'],sample['depth'],sample['valid']
                if r.trajectory_id!=lasttr:scheduler.reset();previous=None;lasttr=r.trajectory_id;position=age=0;lastgt=math.inf
                if policy=='v2':d=asdict(scheduler.decide(x,navigation[i],previous));infer=d['infer']
                else:
                    infer=position in fixed_positions(policy,lengths[r.trajectory_id],**protocol['budget_fraction']);d={'infer':infer,'reason':'first_frame' if position==0 else 'fixed_interval' if infer else 'fixed_skip','age_before':age,'age_after':0 if infer else age+1}
                if infer:
                    with torch.amp.autocast(device,enabled=False):previous=model(x[None].to(device)).cpu()[0]
                    if not torch.isfinite(previous).all() or not (previous>=.1).all():raise ValueError('Bad test prediction')
                    scheduler.update_prediction(previous,i);source=i
                age=i-source;assert age==d['age_after'];overall.update(previous,g,v);acc[r.trajectory_id].update(previous,g,v)
                values=g[v];p10=float(torch.quantile(values,.1)) if values.numel() else math.inf;event=p10<8 and (lastgt>=8 or lastgt-p10>=2);lastgt=p10
                pp=previous[v];predp10=float(torch.quantile(pp,.1)) if event and pp.numel() else None
                logs.append({'index':i,'trajectory':r.trajectory_id,'position':position,'frame_index':r.frame_index,'source_index':source,'prediction_age':age,**d,'critical_event':event,'offline_predicted_p10_on_event':predp10});position+=1
            result=summarize(logs,{tr:a.compute()|{'near_valid_pixels':a.near_count} for tr,a in acc.items()});result['overall']['metrics']=overall.compute()|{'near_valid_pixels':overall.near_count};all_results[policy]=result;write(a.output/f'{policy}.json',result);write(a.output/f'{policy}_frames.json',logs)
    assert e.state_digest(model)==before;write(a.output/'completion.json',{'passed':True,'results':all_results,'model_state_before_and_after':before,'protocol_sha256':a.protocol_sha256,'test_frames':len(records),'no_selection_or_training':True})

if __name__=='__main__':main()
