"""Final exposure gates, grouping, budget independence, and current-frame scoring."""
import copy
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aerodepth_lite import final_study as f
from aerodepth_lite.data import FrameRecord

ROOT = Path(__file__).resolve().parents[1]


def published():
    m = json.loads(gzip.decompress((ROOT/'results/expanded_validation/manifest.json.gz').read_bytes()))
    pilot = f.read(ROOT/'evidence/real_midair_pilot/metadata.json')
    return m, pilot


def test_all_weather_variants_and_original_pilot_excluded():
    m, pilot = published()
    assert f.isolation(m, pilot)['passed']
    leaked = copy.deepcopy(m)
    leaked['splits']['test'].append('Kite_training/sunset/trajectory_1003')
    audit = f.isolation(leaked, pilot)
    assert not audit['passed']
    assert any('equivalent_paths' in p for p in audit['problems'])
    assert any('pilot_test_overlap' in p for p in audit['problems'])


def test_frozen_budget_positions_do_not_use_test_predictions():
    p = f.read(ROOT/'results/final_test/protocol.json')
    for n in [1, 2, 7, 2205, 2232]:
        got = sorted(f.budget_positions(n, p))
        k = (n*2455+4409)//4410
        assert len(got) == k and got[0] == 0 and got[-1] < n
        assert got == [int(np.ceil(i*n/k)) for i in range(k)]


def test_identical_position_stream_catches_different_numbered_paths(tmp_path):
    import h5py
    m = {'splits':{'train':['Kite_training/foggy/trajectory_2001'],
                   'val':['Kite_training/sunny/trajectory_0002'],
                   'test':['Kite_training/sunny/trajectory_0004']}}
    for climate, names in [('foggy',['trajectory_2001']),('sunny',['trajectory_0002','trajectory_0004'])]:
        sensor=tmp_path/'Kite_training'/climate/'sensor_records.hdf5';sensor.parent.mkdir(parents=True)
        with h5py.File(sensor,'w') as h:
            for name in names:
                value=np.arange(12).reshape(4,3)+(20 if name.endswith('0002') else 0)
                h.create_dataset(name+'/groundtruth/position',data=value)
    audit=f.sensor_path_audit(tmp_path,m)
    assert not audit['passed']
    assert 'identical_position_stream_cross_split' in audit['problems'][0]


def test_protocol_tampering_rejected(tmp_path):
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    protocol.with_suffix('.sha256').write_text('incorrect  protocol.json\n')
    with pytest.raises(ValueError,match='protocol hash mismatch'):
        f.protocol_load(protocol)


def test_opt_in_required_before_any_test_access(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError('Test data access before opt-in')
    monkeypatch.setattr(f, 'records_for', forbidden)
    with pytest.raises(ValueError, match='allow-final-test'):
        f.final_test({}, {}, 'hash', tmp_path, tmp_path/'test', tmp_path/'timing', False)


def test_final_synthetic_once_scores_reused_predictions_on_current_gt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = f.read(ROOT/'results/final_test/protocol.json')
    m, pilot = published()
    f.write('evidence/real_midair_pilot/metadata.json', pilot)
    f.write('models/mobilenet_v3_small_epoch4.json', {'training_config':{'data':{}}})
    timing = tmp_path/'timing';timing.mkdir()
    (timing/'individual_results.csv').write_text('synthetic timing receipt\n')
    f.write(timing/'completed.json', {'passed':True,'protocol_sha256':'synthetic', 'repetitions_per_policy':3,
                                    'individual_results_sha256':f.file_sha256(timing/'individual_results.csv'), 'frame_logs_sha256':{}})
    records, samples = [], []
    for tr in m['splits']['test']:
        for i in range(6):
            records.append(FrameRecord(tr,i,'never-opened','never-opened',navigation={}))
            target = 6. if i == 0 else 3. if 'Kite' in tr else 20.
            samples.append({'rgb':torch.full((3,8,8),float(i+1)), 'depth':torch.full((1,8,8),target),
                            'valid':torch.ones((1,8,8),dtype=torch.bool)})
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=0
        def forward(self,x):
            self.calls+=1
            return x.mean(1,keepdim=True)+5
    model=Model()
    monkeypatch.setattr(f,'records_for',lambda *a:(records,[]))
    monkeypatch.setattr(f,'sensor_path_audit',lambda *a:{'passed':True})
    monkeypatch.setattr(f,'MidAirDataset',lambda *a:samples)
    monkeypatch.setattr(f,'load_inference_model',lambda *a:(model,{}))
    # Keep this test focused on execution/scoring; plotting is verified on the actual artifacts.
    monkeypatch.setattr(f,'final_report',lambda *a:None)
    out=tmp_path/'test'
    f.final_test(p,m,'synthetic',tmp_path,out,timing,True)
    completion=f.read(out/'completed.json')
    assert model.calls == completion['actual_measured_model_calls']+50
    fixed=f.read(out/'fixed5.json')
    with (out/'fixed5_frames.csv').open() as handle:rows=list(csv.DictReader(handle))
    assert [r['source_index'] for r in rows[:6]]==['0','0','0','0','0','5']
    assert fixed['overall']['frames']==12
    # First trajectory targets [6,3,3,3,3,3], predictions [6,6,6,6,6,11].
    kite=m['splits']['test'][0]
    expected=(0+1+1+1+1+8/3)/6
    assert fixed['per_trajectory'][kite]['metrics']['absrel']==pytest.approx(expected)
    assert fixed['overall']['metrics']['valid_pixels']==12*64
    assert fixed['per_trajectory'][kite]['critical_events']['max_observed_delay_frames']==4
    # Exposure ledger prevents changing the output directory and silently running again.
    with pytest.raises(FileExistsError):
        f.final_test(p,m,'synthetic',tmp_path,tmp_path/'retry',timing,True)
