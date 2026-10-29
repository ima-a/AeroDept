import copy
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from aerodepth_lite.presentation_scheduler import PresentationScheduler, fixed_positions
from aerodepth_lite.scheduler_v2 import SchedulerV2
from aerodepth_lite.scheduler import near_field_risk
from aerodepth_lite.presentation import ValidationGuard
from aerodepth_lite import expanded as e

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'results/final_test/protocol.json').read_text())['v2']['config']


def test_full_v2_identical_and_threshold_boundaries():
    # Exact thresholds and their immediate floating neighbours; no tuning.
    for threshold in ['critical_risk','decision_threshold']:
        for value in [np.nextafter(CFG[threshold],-np.inf),CFG[threshold],np.nextafter(CFG[threshold],np.inf)]:
            a=SchedulerV2(CFG);b=PresentationScheduler(CFG)
            for i in range(12):
                gray=np.full((36,64),i*.008,dtype='float32')
                nav={'ax':i*.1,'ay':0.,'az':9.8,'wx':0.,'wy':0.,'wz':.1,'gps_vx':1.,'gps_vy':0.,'gps_vz':0.}
                assert asdict(a.decide_features(gray,nav,float(value),True))==asdict(b.decide_features(gray,nav,float(value),True))


@pytest.mark.parametrize('removed,reason,nav,gray,risk',[
 ('motion','extreme_motion',{'gps_vx':100.,'gps_vy':0.,'gps_vz':0.},0.,0.),
 ('visual','extreme_visual',{},1.,0.),('risk','critical_risk',{},0.,1.)])
def test_ablation_removes_weight_and_forced_trigger(removed,reason,nav,gray,risk):
    config=copy.deepcopy(CFG);config['decision_threshold']=100.
    full=PresentationScheduler(config);off=PresentationScheduler(config,[removed]);zeros=np.zeros((36,64),dtype='float32')
    full.decide_features(zeros,{},0.,False);off.decide_features(zeros,{},0.,False)
    a=full.decide_features(zeros+gray,nav,risk,True);b=off.decide_features(zeros+gray,nav,risk,True)
    assert a.infer and a.reason==reason
    assert not b.infer and b.reason=='stable'
    assert getattr(b,'risk' if removed=='risk' else removed)==0.
    assert config==full.config==off.config


def test_age_only_is_fixed5_and_hard_bound_survives_no_age():
    config=CFG|{'decision_threshold':100.}
    s=PresentationScheduler(config,['motion','visual','risk','age']);calls=[]
    for i in range(16):
        d=s.decide_features(np.zeros((36,64)),{},1.,True)
        if d.infer:calls.append(i)
        assert d.age_after<=4
    assert calls==sorted(fixed_positions('fixed5',16))
    assert len(fixed_positions('budget_fixed',2205))==1228


def test_risk_cache_actual_prediction_and_reset():
    cached=PresentationScheduler(CFG,cache_risk=True);uncached=PresentationScheduler(CFG,cache_risk=False)
    source1=source2=None;previous=None;rgb=torch.zeros(3,192,320)
    for i in range(25):
        x=cached.decide(rgb,{},previous);y=uncached.decide(rgb,{},previous)
        assert asdict(x)==asdict(y)
        if x.infer:
            source1=source2=i;previous=torch.full((1,192,320),.1 if i%2 else 100.)
            cached.update_prediction(previous,i)
            old=cached.risk_updates;cached.update_prediction(previous,i);assert old==cached.risk_updates
        assert source1==source2 and i-source1==x.age_after
    cached.reset();assert cached.risk_value==0 and not cached.risk_available and cached.prediction_token is None and cached.age==0
    for depth in [None,torch.full((1,8,8),float('nan')),torch.zeros(1,8,8),torch.full((1,8,8),10.)]:
        a=PresentationScheduler(CFG);a.update_prediction(depth,0)
        assert (a.risk_value,a.risk_available)==near_field_risk(depth,10.)


def test_no_privileged_navigation_or_test_discovery(tmp_path):
    a=PresentationScheduler(CFG);b=PresentationScheduler(CFG)
    for i in range(8):
        x=a.decide_features(np.zeros((36,64)),{'vx':1e20,'groundtruth_depth':.1},0,False)
        y=b.decide_features(np.zeros((36,64)),{},0,False)
        assert asdict(x)==asdict(y)
    records,_=e.load_inputs(e.PREP/'manifest_train_val.json.gz',e.PREP/'reservations.json',tmp_path)
    guard=ValidationGuard(tmp_path,records['val'])
    for event,args in [('os.listdir',(str(tmp_path),)),('open',(str(tmp_path/'Kite_training/foggy/color_left/trajectory_2007/000000.JPEG'),'rb',0))]:
        with pytest.raises(PermissionError):guard.check(event,args)


def test_score_exact_threshold_and_forced_boundary_neighbours():
    c=CFG|{'motion_weight':0.,'visual_weight':0.,'risk_weight':1.,'age_weight':0.,'critical_risk':2.}
    for value,expected in [(np.nextafter(c['decision_threshold'],-np.inf),False),(c['decision_threshold'],True),(np.nextafter(c['decision_threshold'],np.inf),True)]:
        s=PresentationScheduler(c);s.decide_features(np.zeros((36,64)),{},0.,False)
        d=s.decide_features(np.zeros((36,64)),{},float(value),True)
        assert d.infer==expected
    c=CFG|{'decision_threshold':100.}
    for value,expected in [(np.nextafter(c['critical_risk'],-np.inf),False),(c['critical_risk'],True),(np.nextafter(c['critical_risk'],np.inf),True)]:
        s=PresentationScheduler(c);s.decide_features(np.zeros((36,64)),{},0.,False)
        assert s.decide_features(np.zeros((36,64)),{},float(value),True).infer==expected


def test_future_test_requires_optin_and_blocked_proposal_opens_no_pixels(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('future_test',ROOT/'scripts/presentation_final_test.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from PIL import Image
    monkeypatch.setattr(Image,'open',lambda *a,**kw:pytest.fail('Test pixel opened'))
    protocol=tmp_path/'proposal.json';protocol.write_text(json.dumps({'status':'proposed_blocked_missing_reserved_manifest'}))
    from aerodepth_lite.inference import file_sha256
    args=['--protocol',str(protocol),'--protocol-sha256',file_sha256(protocol),'--manifest',str(tmp_path/'absent.json'),'--dataset-root',str(tmp_path),'--output',str(tmp_path/'output')]
    with pytest.raises(SystemExit):module.main(args)
    with pytest.raises(ValueError,match='No test access'):module.main(args+['--confirm-new-test-exposure'])
    assert not (tmp_path/'output').exists()


def test_demonstration_byte_decoder_matches_original_preprocessing(tmp_path):
    from PIL import Image
    from aerodepth_lite.data import FrameRecord,MidAirDataset
    from aerodepth_lite.presentation import decode_pair
    rgb=np.arange(8*12*3,dtype='uint8').reshape(8,12,3)
    depth=np.array([0.,.05,.1,1.,9.99,10.,100.,1000.,1001.,np.nan,np.inf,3.],dtype='float16')[None,:].repeat(8,axis=0)
    Image.fromarray(rgb).save(tmp_path/'rgb.JPEG');Image.fromarray(depth.view('uint16')).save(tmp_path/'depth.PNG')
    record=FrameRecord('Kite_training/foggy/trajectory_2003',0,'rgb.JPEG','depth.PNG')
    config={'image_size':[192,320],'min_depth':.1,'max_depth':1000.,'depth_scale':1.,'depth_encoding':'midair_float16_bits'}
    expected=MidAirDataset(tmp_path,[record],config,False)[0]
    actual=decode_pair((tmp_path/'rgb.JPEG').read_bytes(),(tmp_path/'depth.PNG').read_bytes(),config)
    assert all(torch.equal(actual[i],expected[key]) for i,key in enumerate(['rgb','depth','valid']))


def test_zero_event_and_censored_end_do_not_cross_trajectory():
    from aerodepth_lite.research import summarize
    rows=[{'trajectory':'A','position':0,'infer':True,'reason':'first_frame','prediction_age':0,'critical_event':False,'offline_predicted_p10_on_event':None},
          {'trajectory':'A','position':1,'infer':False,'reason':'stable','prediction_age':1,'critical_event':True,'offline_predicted_p10_on_event':9.},
          {'trajectory':'B','position':0,'infer':True,'reason':'first_frame','prediction_age':0,'critical_event':False,'offline_predicted_p10_on_event':None}]
    result=summarize(rows,{'A':{},'B':{}})
    assert result['per_trajectory']['A']['critical_events']['right_censored_count']==1
    assert result['per_trajectory']['A']['critical_events']['max_observed_delay_frames'] is None
    assert result['per_trajectory']['B']['critical_events']['missed_update_rate'] is None
    assert result['overall']['critical_events']['right_censored_count']==1
