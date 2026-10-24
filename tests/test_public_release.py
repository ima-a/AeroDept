from pathlib import Path
import json,gzip
import numpy as np
from PIL import Image
import pytest,torch
from aerodepth_lite.inference import file_sha256,load_inference_model,infer_image
from aerodepth_lite import research as r
ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=ROOT/'models/mobilenet_v3_small_epoch4.state_dict.pt'
META=ROOT/'models/mobilenet_v3_small_epoch4.json'

def test_published_weights_strict_offline_loading_buffers_and_inference(tmp_path,monkeypatch):
 def no_download(*a,**kw):raise AssertionError('Inference attempted a network download')
 monkeypatch.setattr(torch.hub,'download_url_to_file',no_download)
 model,info=load_inference_model(WEIGHTS,META)
 state=torch.load(WEIGHTS,weights_only=True)
 assert all(torch.is_tensor(v) for v in state.values())
 assert 'rgb_mean' in state and any(k.endswith('running_var') for k in state)
 assert info['parameters']==1_027_489 and not model.training
 image=tmp_path/'rgb.png';Image.fromarray(np.full((37,53,3),128,np.uint8)).save(image)
 out=tmp_path/'pred.npy';res=infer_image(WEIGHTS,META,image,out)
 assert np.load(out).shape==(192,320) and np.isfinite(np.load(out)).all()
 assert res['weights_sha256']==file_sha256(WEIGHTS)
 tampered=tmp_path/'wrong.pt';tampered.write_bytes(b'not valid weights')
 with pytest.raises(ValueError,match='SHA-256'):load_inference_model(tampered,META)

def test_frozen_manifest_identity_and_relative_paths():
 info=json.loads(META.read_text());raw=gzip.decompress((ROOT/'results/expanded_validation/manifest.json.gz').read_bytes())
 import hashlib
 assert hashlib.sha256(raw).hexdigest()==info['manifest_sha256']
 m=json.loads(raw);assert m['frame_counts']['val']==4410
 assert all(not Path(x['rgb_path']).is_absolute() and not Path(x['depth_path']).is_absolute() for x in m['records'])

def test_public_event_semantics_and_no_boundary_crossing():
 rows=[{'trajectory':'A','position':0,'critical_event':True,'infer':False,'prediction_age':1,'reason':'stable'},{'trajectory':'B','position':0,'critical_event':False,'infer':True,'prediction_age':0,'reason':'first_frame'}]
 result=r.summarize(rows,{'A':{},'B':{}})
 assert result['overall']['critical_events']['right_censored_count']==1
 assert result['overall']['critical_events']['observed_delays_frames']==[]
 assert result['per_trajectory']['B']['critical_events']['missed_update_rate'] is None

def test_cached_v1_uses_previous_selected_prediction_and_is_causal(monkeypatch):
 monkeypatch.setattr(r,'STUDY','scheduler-v1')
 cfg={'max_skip':3,'decision_threshold':2,'critical_risk':.9,'extreme_motion':2,'extreme_visual':2}
 fs=[{'index':i,'trajectory':'A','position':i,'frame_index':i,'navigation':{},'visual_raw':0.,'fresh_risk':0. if i==0 else 1.,'risk_available':True} for i in range(4)]
 rows=r.cached_decisions(fs,cfg)
 assert [x['infer'] for x in rows]==[True,False,False,False]
 changed=[dict(f) for f in fs];changed[-1]['navigation']={'vx':9999.}
 assert rows[:3]==r.cached_decisions(changed,cfg)[:3]

def test_csv_output_directory_created(tmp_path,monkeypatch):
 monkeypatch.setattr(r,'OUT',tmp_path);r.csvsave('nested/frame.csv',[{'index':0,'infer':True}]);assert (tmp_path/'nested/frame.csv').read_text().splitlines()==['index,infer','0,True']
