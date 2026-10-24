"""Causal IMU/visual scheduler revision; original scheduler remains unchanged."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from aerodepth_lite.scheduler import Decision,near_field_risk

def thumbnail(rgb):
 x=torch.as_tensor(rgb,dtype=torch.float32)
 if x.ndim==4:x=x[0]
 return F.interpolate(x.mean(0,keepdim=True).unsqueeze(0),size=(36,64),mode='bilinear',align_corners=False)[0,0].cpu().numpy()

class SchedulerV2:
 def __init__(self,config):self.config=config;self.reset()
 def reset(self):
  self.age=0;self.frames=0;self.last_inferred_gray=None;self.previous_acceleration=None
 def decide(self,rgb,navigation,previous_depth):
  risk,available=near_field_risk(previous_depth,10.)
  return self.decide_features(thumbnail(rgb),navigation,risk,available)
 def decide_features(self,gray,navigation,risk,risk_available):
  c=self.config
  def vector(keys):
   values=[navigation.get(k) for k in keys]
   return np.array(values,dtype=float) if all(v is not None and np.isfinite(v) for v in values) else None
  acceleration=vector(('ax','ay','az'));gyro=vector(('wx','wy','wz'));gps=vector(('gps_vx','gps_vy','gps_vz'))
  accel_change=float(np.linalg.norm(acceleration-self.previous_acceleration)) if acceleration is not None and self.previous_acceleration is not None else 0.
  # Separate dimensionless scales; no privileged velocity or GT navigation enters this calculation.
  motion=max(min(accel_change/c['accel_change_scale'],1.),min(float(np.linalg.norm(gyro))/c['gyro_scale'],1.) if gyro is not None else 0.)
  motion=max(motion,min(float(np.linalg.norm(gps))/c.get('gps_speed_scale',20.),1.) if gps is not None else 0.)
  motion_available=acceleration is not None or gyro is not None or gps is not None
  visual_raw=float(np.abs(gray-self.last_inferred_gray).mean()) if self.last_inferred_gray is not None else 0.
  visual=min(visual_raw/c['visual_scale'],1.)
  age=self.age;score=c['motion_weight']*motion+c['visual_weight']*visual+c['risk_weight']*risk+c['age_weight']*min(age/max(c['max_skip'],1),1.)
  forced=False
  if self.frames==0:infer,reason=True,'first_frame'
  elif age>=c['max_skip']:infer,reason,forced=True,'max_skip',True
  elif risk_available and risk>=c['critical_risk']:infer,reason=True,'critical_risk'
  elif motion_available and motion>=c['extreme_motion']:infer,reason=True,'extreme_motion'
  elif visual>=c['extreme_visual']:infer,reason=True,'extreme_visual'
  elif score>=c['decision_threshold']:infer,reason=True,'combined_score'
  else:infer,reason=False,'stable'
  self.age=0 if infer else age+1;self.frames+=1;self.previous_acceleration=acceleration
  if infer:self.last_inferred_gray=np.array(gray,copy=True)
  return Decision(infer,reason,score,motion,motion_available,visual,risk,risk_available,age,self.age,forced)
