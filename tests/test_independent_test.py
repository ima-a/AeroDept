"""Synthetic/metadata-only guards and accounting; never decode reserved pixels."""
from pathlib import Path
import csv, io, json, math, zipfile
import pytest
import torch
from aerodepth_lite import independent_test as t
from aerodepth_lite import independent_data as d

def zipped(names):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        for name in names:z.writestr(name,b'fixture')
    buffer.seek(0)
    return zipfile.ZipFile(buffer)

@pytest.mark.parametrize('names',[['../000000.PNG'],['/000000.PNG'],['000001.PNG'],['000000.PNG','000000.PNG']])
def test_archive_rejects_unsafe_missing_and_duplicate_members(names):
    with zipped(names) as z,pytest.raises(ValueError):d.safe_members(z,1,'PNG')

def test_archive_member_order_is_not_identity():
    with zipped(['000001.PNG','000000.PNG']) as z:
        assert [v.filename for v in d.safe_members(z,2,'PNG')]==['000000.PNG','000001.PNG']

def test_opt_in_precedes_protocol_and_input_reads(monkeypatch,tmp_path):
    monkeypatch.setattr(t,'read',lambda *a:pytest.fail('Read before opt-in'))
    with pytest.raises(PermissionError):t.run(tmp_path/'absent.json','0'*64,False)

@pytest.mark.parametrize('name',['EXPOSURE.json','run'])
def test_exposure_or_partial_output_prevents_retry(tmp_path,name):
    (tmp_path/name).touch()
    with pytest.raises(FileExistsError):t.require_unused(tmp_path)

def test_blocked_protocol_cannot_be_forced(tmp_path):
    p={'version':'independent_reserved_test_v3','status':'proposed_blocked'};path=tmp_path/'protocol.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Blocked'):t.validate_protocol(p,path,t.file_sha256(path),tmp_path/'locations.json')

def test_matrix_change_rejected_before_dataset_access(tmp_path):
    p={'version':'independent_reserved_test_v3','status':'frozen_ready_for_explicit_test','study_path':t.STUDY,'test_realizations':d.EXPECTED,'matrix':[['KD_every','KD','every']]}
    path=tmp_path/'protocol.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Comparison'):t.validate_protocol(p,path,t.file_sha256(path),tmp_path/'locations.json')

def test_protocol_digest_tampering_rejected(tmp_path):
    p=tmp_path/'protocol.json';p.write_text('{}')
    with pytest.raises(ValueError,match='pinned'):t.validate_protocol({},p,'0'*64,tmp_path/'locations.json')

def test_reserved_position_alias_is_leakage():
    r=json.loads(d.RESERVATIONS.read_text());positions={row['trajectory']:row['position_sha256'] for row in r['position_identity_audit'] if row['trajectory'] in d.EXPECTED}
    first=next(iter(positions));r['position_identity_audit'].append({'trajectory':'Kite_training/sunny/trajectory_0000','group':'Kite/0000','position_sha256':positions[first]})
    with pytest.raises(ValueError,match='another path'):d.verify_isolation(positions,r,{'path_groups':{}})

def test_pooled_metrics_strict_delta_and_undefined_near():
    a=t.Scores();p=torch.full((1,2,2),12.5);g=torch.full_like(p,10.);v=torch.ones_like(p,dtype=torch.bool)
    a.update(p,g,v);r=a.compute()
    assert r['absrel']==.25 and r['rmse']==2.5 and r['delta1']==0 and r['delta2']==1
    assert r['valid_pixels']==4 and r['near_valid_pixels']==0 and r['near_rmse'] is None and r['near_bias_m'] is None

def test_nonfinite_prediction_rejected_even_outside_mask():
    p=torch.ones(1,2,2);p[0,0,0]=float('nan')
    with pytest.raises(ValueError):t.Scores().update(p,torch.ones_like(p),torch.zeros_like(p,dtype=torch.bool))

def test_streaming_matrix_scores_current_gt_counts_calls_and_censoring(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self,offset):super().__init__();self.offset=offset;self.calls=0
        def forward(self,x):self.calls+=1;return x[:,:1]+self.offset
    models={k:Model(offset) for k,offset in [('A',1),('Original',2),('KD',3)]}
    records=[{'trajectory_id':tr,'frame_index':j,'navigation':{'ax':0.,'ay':0.,'az':0.,'wx':0.,'wy':0.,'wz':0.,'gps_vx':0.,'gps_vy':0.,'gps_vz':0.},'index':i,'gt':gt}
             for i,(tr,j,gt) in enumerate([('path1',0,20),('path1',1,5),('path1',2,3),('path2',0,5),('path2',1,12)])]
    def sample(r):
        x=torch.full((3,4,6),float(r['index']+1));g=torch.full((1,4,6),float(r['gt']));return x,g,torch.ones_like(g,dtype=torch.bool),.001,.002
    p=json.loads((t.ROOT/t.VALIDATION).read_text());p['device']='cpu'
    result,timing=t.evaluate_stream(records,sample,models,p,tmp_path)
    assert set(result)=={name for name,_,_ in t.MATRIX}
    assert {x['overall']['metrics']['valid_pixels'] for x in result.values()}=={120}
    assert {x['overall']['metrics']['near_valid_pixels'] for x in result.values()}=={72}
    assert models['A'].calls==sum(result['A_'+policy]['overall']['inference_calls'] for policy in t.POLICIES)
    assert models['Original'].calls==models['KD'].calls==5
    fixed=result['A_fixed5']
    assert fixed['overall']['critical_events']['right_censored_count']==2
    assert fixed['overall']['critical_events']['observed_delays_frames']==[0]
    assert fixed['per_trajectory']['path1']['metrics']['rmse']==pytest.approx(math.sqrt((18**2+3**2+1)/3))
    rows=list(csv.DictReader((tmp_path/'A_fixed5_frames.csv').open()))
    assert [int(r['source_index']) for r in rows]==[0,0,0,3,3]
    assert [int(r['prediction_age']) for r in rows]==[0,1,2,0,1]
    assert timing['shared_io_seconds']==pytest.approx(.005)
    assert fixed['overall']['compute_seconds']>=fixed['overall']['inference_seconds']>0
