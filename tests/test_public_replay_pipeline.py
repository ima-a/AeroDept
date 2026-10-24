"""Public v2 cache/search inputs and actual replay must agree on synthetic data."""
from dataclasses import asdict
import csv,json
import numpy as np
import torch
from aerodepth_lite import research as r
from aerodepth_lite.scheduler_v2 import SchedulerV2
from aerodepth_lite.metrics import DepthMetricAccumulator

def test_portable_v2_cached_and_actual_replay_agree(tmp_path,monkeypatch):
 cfg={'data':{'image_size':[16,24]},'scheduler':{}}
 samples=[]
 for i in range(8):
  samples.append({'rgb':torch.full((3,16,24),i*.03),'depth':torch.full((1,16,24),5.),'valid':torch.ones((1,16,24),dtype=torch.bool),'record':{'trajectory_id':'A' if i<4 else 'B','frame_index':i%4,'navigation':{'gps_vx':i,'gps_vy':0.,'gps_vz':0.,'ax':0.,'ay':0.,'az':9.81}}})
 class Model(torch.nn.Module):
  def forward(self,x):return x.mean(1,keepdim=True)+6
 model=Model().eval();expected={};total=DepthMetricAccumulator()
 for sample in samples:
  p=model(sample['rgb'][None])[0];tr=sample['record']['trajectory_id'];expected.setdefault(tr,DepthMetricAccumulator()).update(p,sample['depth'],sample['valid']);total.update(p,sample['depth'],sample['valid'])
 def metrics(a):return a.compute()|{'near_valid_pixels':a.near_count}
 published=tmp_path/'published';(published/'expanded_validation').mkdir(parents=True)
 (published/'expanded_validation/validation_metrics.json').write_text(json.dumps({'metrics':metrics(total),'per_trajectory':{tr:metrics(a) for tr,a in expected.items()}}))
 monkeypatch.setattr(r,'OUT',tmp_path);monkeypatch.setattr(r,'PUBLISHED',published);monkeypatch.setattr(r,'DEVICE',torch.device('cpu'));monkeypatch.setattr(r,'STUDY','scheduler-v2')
 monkeypatch.setattr(r,'setup',lambda:(cfg,[s['record'] for s in samples],samples));monkeypatch.setattr(r,'model_load',lambda cfg:model)
 monkeypatch.setattr(r,'loader',lambda ds:({'rgb':s['rgb'][None],'depth':s['depth'][None],'valid':s['valid'][None],'record':[s['record']]} for s in ds))
 monkeypatch.setattr(r.sm,'AdaptiveScheduler',SchedulerV2)
 tol={k:1e-8 for k in metrics(total)};r.save('protocol.json',{'audit_tolerances_absolute':tol})
 r.live('every',make_cache=True)
 config={'max_skip':2,'accel_change_scale':2.,'gyro_scale':.5,'gps_speed_scale':20.,'visual_scale':.08,'critical_risk':.9,'extreme_motion':.9,'extreme_visual':.9,'decision_threshold':.65,'motion_weight':.25,'visual_weight':.3,'risk_weight':.3,'age_weight':.15}
 rows=r.cached_decisions(r.read('decision_features.json'),config);cached=r.score_cached(rows)
 r.live('adaptive_selected',config)
 actual=r.read('adaptive_selected.json')
 for k in metrics(total):assert np.isclose(actual['overall']['metrics'][k],cached['overall']['metrics'][k])
 with (tmp_path/'adaptive_selected_frames.csv').open() as f:logged=list(csv.DictReader(f))
 for a,b in zip(logged,rows):
  for k in ['infer','source_index','reason','prediction_age']:assert a[k]==str(b[k])
