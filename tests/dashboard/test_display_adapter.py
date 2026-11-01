"""Display projections must agree with saved evidence; no test image reads."""
from pathlib import Path
import json
import pytest
from aerodepth_lite import dashboard_data as d
from aerodepth_lite.presentation_ui import load_dashboard_config
ROOT=Path(__file__).resolve().parents[2]


def cfg():return load_dashboard_config(ROOT/'configs/dashboard.json',ROOT)


def test_independent_projection_matches_every_saved_metric_and_call():
    raw=json.loads((ROOT/d.INDEPENDENT/'results.json').read_text());bundle=d.study_bundle(ROOT,cfg(),'Independent test')
    assert bundle['status']=='COMPLETED'
    for name,result in raw.items():
        model,policy=name.split('_',1)
        for scope,part in [('pooled',result['overall']),*result['per_trajectory'].items()]:
            projected=d.part(bundle,model,policy,scope)
            assert projected==part
    rows=d.metric_rows(bundle,'pooled')
    assert {r['model'] for r in rows}=={'Original','A','KD'}
    assert next(r for r in rows if r['model']=='Original')['absrel']==pytest.approx(.390815,abs=.0000005)
    assert next(r for r in rows if r['model']=='A')['near_rmse']==pytest.approx(4.144905,abs=.0000005)
    assert next(r for r in rows if r['model']=='KD')['absrel']==pytest.approx(.383202,abs=.0000005)
    assert d.part(bundle,'A','v2','pooled')['inference_calls']!=d.part(bundle,'A','budget_fixed','pooled')['inference_calls']


def test_validation_does_not_fabricate_model_scheduler_accounting():
    bundle=d.study_bundle(ROOT,cfg(),'Validation')
    assert d.part(bundle,'Original','every','pooled').get('inference_calls') is None
    raw=json.loads((ROOT/'results/student_validation_comparison_20261029T152648Z/metrics.json').read_text())
    for source,model in [('epoch4','Original'),('student_A','A'),('exploratory_KD','KD')]:
        assert d.part(bundle,model,'every','pooled')['metrics']==raw['students'][source]['scopes']['pooled']


def test_historical_supports_original_only_and_no_new_models():
    bundle=d.study_bundle(ROOT,cfg(),'Historical exposed test')
    assert d.models(bundle)==['Original']
    assert not d.part(bundle,'A','every','pooled')
    assert d.part(bundle,'Original','every','pooled')['frames']==4436


def test_zero_events_and_undefined_values_are_retained():
    bundle=d.study_bundle(ROOT,cfg(),'Independent test')
    events=d.part(bundle,'A','v2','Kite_training/sunny/trajectory_0008')['critical_events']
    assert events['count']==0
    assert events['mean_observed_delay_frames'] is None and events['missed_update_rate'] is None
    assert d.relative_change(None,3) is None and d.relative_change(3,0) is None
    assert d.relative_change(9,10)==pytest.approx(-10)


def test_cache_reuses_integrity_check_and_invalidates_on_file_change(tmp_path,monkeypatch):
    folder=tmp_path/d.INDEPENDENT;folder.mkdir(parents=True);path=folder/'EXPOSURE.json';path.write_text('{}')
    calls=[]
    monkeypatch.setattr(d.backend,'independent_study',lambda root:calls.append(root) or {'status':'NOT RUN'})
    d.verified_independent(tmp_path);d.verified_independent(tmp_path);assert len(calls)==1
    path.write_text('{"changed":true}');d.verified_independent(tmp_path);assert len(calls)==2


def test_trace_projection_keeps_types_and_available_signals_only():
    independent=d.study_bundle(ROOT,cfg(),'Independent test');val=d.study_bundle(ROOT,cfg(),'Validation')
    test_rows=d.trace_rows(independent,'A','v2');val_rows=d.trace_rows(val,'A','v2')
    assert len(test_rows)==8820 and isinstance(test_rows[0]['infer'],bool)
    assert 'motion' not in test_rows[0] and all(k in val_rows[0] for k in ['motion','visual','risk','score'])
    assert test_rows[0]['prediction_age']==0
    assert json.loads((ROOT/'results/presentation_validation_20261029T155521Z/policies/v2_frames.json').read_text())[0]==d.json_file(ROOT/'results/presentation_validation_20261029T155521Z/policies/v2_frames.json')[0]


def test_fixed_example_cannot_load_non_validation_path():
    with pytest.raises(ValueError,match='Validation'):d.fixed_frame(cfg(),'Kite_training/sunny/trajectory_0008',0,'A')


def test_teacher_reuses_exact_saved_validation_and_no_test_or_call_data():
    config=cfg();bundle=d.study_bundle(ROOT,config,'Validation')
    raw=json.loads((Path(config['comparison'])/'metrics.json').read_text())['teacher_context_reused_no_inference']
    for scope,metrics in raw['scopes'].items():
        assert d.part(bundle,'Teacher','every',scope)['metrics']==metrics
        row=next(r for r in d.metric_rows(bundle,scope) if r['model']=='Teacher')
        assert row['Calls'] is None and row['Skip rate'] is None
    assert bundle['teacher_gate']['teacher_gate_passed'] is False
    assert d.policies(bundle,'Teacher')==['every'] and d.trace_rows(bundle,'Teacher','every')==[]
    for study in ['Independent test','Historical exposed test']:
        other=d.study_bundle(ROOT,config,study)
        assert 'Teacher' not in d.models(other) and not d.part(other,'Teacher','every','pooled')
        assert other['teacher_gate'] is None
