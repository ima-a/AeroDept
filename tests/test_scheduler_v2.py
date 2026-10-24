import importlib.util
from pathlib import Path
import numpy as np
import torch
from aerodepth_lite import scheduler_v2 as v

def cfg():return {'max_skip':2,'accel_change_scale':2.,'gyro_scale':.5,'visual_scale':.1,'critical_risk':.9,'extreme_motion':.9,'extreme_visual':.9,'decision_threshold':2.,'motion_weight':.25,'visual_weight':.30,'risk_weight':.30,'age_weight':.15}
def test_no_privileged_velocity_and_static_gravity_not_extreme_motion():
 a=v.SchedulerV2(cfg());gray=np.zeros((36,64),np.float32);nav={'vx':9999.,'ax':0.,'ay':0.,'az':9.81,'wx':0.,'wy':0.,'wz':0.}
 a.decide_features(gray,nav,0,False);d=a.decide_features(gray,nav,0,True)
 assert not d.infer and d.motion==0
 nav['vx']=-9999.;assert a.decide_features(gray,nav,0,True).motion==0
def test_change_since_last_inference_accumulates_and_reset():
 a=v.SchedulerV2(cfg()|{'max_skip':5})
 a.decide_features(np.zeros((36,64),np.float32),{},0,False)
 d=a.decide_features(np.full((36,64),.05,np.float32),{},0,True);assert not d.infer and abs(d.visual-.5)<1e-6
 d=a.decide_features(np.full((36,64),.1,np.float32),{},0,True);assert d.infer and d.reason=='extreme_visual'
 a.reset();assert a.decide_features(np.ones((36,64),np.float32),{},0,False).reason=='first_frame'
def test_imu_scales_and_future_independence():
 a=v.SchedulerV2(cfg());g=np.zeros((36,64),np.float32)
 a.decide_features(g,{'ax':0.,'ay':0.,'az':9.81},0,False)
 d=a.decide_features(g,{'ax':2.,'ay':0.,'az':9.81},0,True);assert d.infer and d.motion==1 and d.reason=='extreme_motion'
 b=v.SchedulerV2(cfg());assert b.decide_features(g,{},0,False).risk_available is False
 d=b.decide_features(g,{},0,True);assert not d.motion_available
def test_causal_gps_component_separate_from_privileged_velocity():
 a=v.SchedulerV2(cfg()|{'gps_speed_scale':20.});gray=np.zeros((36,64),np.float32)
 a.decide_features(gray,{},0,False)
 d=a.decide_features(gray,{'gps_vx':10.,'gps_vy':0.,'gps_vz':0.,'vx':9999.},0,True)
 assert d.motion==.5 and d.motion_available and not d.infer
