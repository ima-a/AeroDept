"""Optional UI checks. Normal CLI installation does not require Streamlit."""
from pathlib import Path
import json
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from aerodepth_lite import presentation_ui
ROOT=Path(__file__).resolve().parents[1]


def app(monkeypatch,portable=False):
    monkeypatch.setattr(presentation_ui,'actual_inference',lambda *a,**k:pytest.fail('Automatic inference'))
    if portable:monkeypatch.setenv('AERODEPTH_PRESENTATION_CONFIG',str(ROOT/'configs/dashboard.json'))
    return AppTest.from_file(str(ROOT/'scripts/dashboard.py')).run(timeout=60)


def nav(a,page):return a.radio[0].set_value(page).run(timeout=60)
def select(a,label,value):return next(s for s in a.selectbox if s.label==label).select(value).run(timeout=60)


def test_navigation_defaults_and_no_automatic_inference(monkeypatch):
    a=app(monkeypatch)
    assert a.radio[0].options==['Results','Model comparison','Depth viewer','Scheduler','Artifacts']
    assert next(s for s in a.selectbox if s.label=='Study / split').value=='Independent test'
    assert next(s for s in a.selectbox if s.label=='Model').value=='A'
    assert next(m for m in a.metric if m.label=='AbsRel ↓').value=='0.383690'
    for page in a.radio[0].options:
        nav(a,page);assert not a.exception


def test_context_and_nulls_never_show_stale_values(monkeypatch):
    a=app(monkeypatch);select(a,'Policy','v2');nav(a,'Model comparison')
    assert next(s for s in a.selectbox if s.label=='Policy').value=='every'
    select(a,'Study / split','Historical exposed test')
    assert next(s for s in a.selectbox if s.label=='Model').value=='Original'
    assert next(s for s in a.selectbox if s.label=='Policy').disabled
    select(a,'Study / split','Independent test');select(a,'Model','A');nav(a,'Scheduler')
    select(a,'Trajectory','Kite_training/sunny/trajectory_0008')
    assert next(m for m in a.metric if m.label=='Max event delay (frames)').value=='—'
    assert not a.exception


def test_study_specific_model_comparisons_and_reference(monkeypatch):
    a=app(monkeypatch,True);nav(a,'Model comparison')
    for study in ['Independent test','Validation','Historical exposed test']:
        select(a,'Study / split',study);assert not a.exception
        assert len(a.dataframe[0].value)=={'Independent test':3,'Validation':4,'Historical exposed test':1}[study]
    select(a,'Study / split','Independent test');select(a,'Reference model','KD')
    assert not a.exception


@pytest.mark.parametrize('missing',['config','comparison','cache'])
def test_missing_inputs_are_explicit(tmp_path,monkeypatch,missing):
    cfg=json.loads((ROOT/'configs/dashboard.json').read_text())
    if missing=='config':path=tmp_path/'absent.json'
    else:
        cfg[missing]=str(tmp_path/'absent');path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    monkeypatch.setenv('AERODEPTH_PRESENTATION_CONFIG',str(path));a=app(monkeypatch)
    if missing=='comparison':nav(a,'Model comparison');select(a,'Study / split','Validation')
    if missing=='cache':nav(a,'Depth viewer')
    assert not a.exception
    assert any('unavailable' in str(x.value).lower() for x in [*a.info,*a.caption,*a.warning])


def test_portable_configuration_and_published_models(monkeypatch):
    monkeypatch.delenv('MIDAIR_ROOT',raising=False);monkeypatch.delenv('AERODEPTH_KD_CHECKPOINT',raising=False)
    config=presentation_ui.load_dashboard_config(ROOT/'configs/dashboard.json',ROOT)
    assert config['dataset_root'].startswith(str(ROOT/'.unavailable'))
    for name in ['Original','A','KD','Teacher']:
        entry=config['models'][name]
        assert entry['kind']=='export' and Path(entry['weights']).is_file() and Path(entry['metadata']).is_file()
    a=app(monkeypatch,True);nav(a,'Depth viewer')
    assert not a.exception
    assert any('unavailable' in str(x.value).lower() for x in a.info)


def test_teacher_validation_gate_and_context_switch(monkeypatch):
    a=app(monkeypatch,True);nav(a,'Model comparison');select(a,'Study / split','Validation')
    assert 'ResNet-50 teacher' in a.dataframe[0].value['Model'].tolist()
    assert any('failed its acceptance gate' in str(w.value) for w in a.warning)
    select(a,'Model','Teacher');select(a,'Trajectory','Kite_training/foggy/trajectory_2003')
    teacher=a.dataframe[0].value.query('Model == "ResNet-50 teacher"').iloc[0]
    assert teacher['near_rmse']==pytest.approx(6.636089534161115)
    select(a,'Reference model','Teacher')
    assert not a.exception
    select(a,'Study / split','Independent test')
    assert 'ResNet-50 teacher' not in next(s for s in a.selectbox if s.label=='Model').options
    assert 'ResNet-50 teacher' not in a.dataframe[0].value['Model'].tolist()
    assert not any('failed its acceptance gate' in str(w.value) for w in a.warning)
    assert not a.exception


def test_artifacts_accepts_configured_model_without_display_label(monkeypatch):
    from aerodepth_lite import dashboard_data
    monkeypatch.delitem(dashboard_data.MODELS,'Teacher')
    a=app(monkeypatch,True);nav(a,'Artifacts')
    assert not a.exception
    config=json.loads((ROOT/'configs/dashboard.json').read_text())
    inspector=json.loads(a.json[0].value)
    assert config['models']['Teacher']['label'] in inspector['models']


def test_frame_presets_do_not_trigger_inference(monkeypatch):
    a=app(monkeypatch);nav(a,'Depth viewer')
    next(b for b in a.button if b.label=='Poor · 1534').click().run(timeout=60)
    assert not a.exception
    sliders=[s for s in a.slider if s.label=='Validation frame']
    if sliders:assert sliders[0].value==1534
    else:assert next(s for s in a.selectbox if s.label=='Recorded validation frame').value==1534


def test_independent_loader_distinguishes_missing_exposed_and_failed(tmp_path):
    folder=tmp_path/'results/independent_test_20261030';folder.mkdir(parents=True)
    assert presentation_ui.independent_study(tmp_path)['status']=='NOT RUN'
    (folder/'EXPOSURE.json').write_text('{}')
    assert presentation_ui.independent_study(tmp_path)['status']=='IN PROGRESS'
    (folder/'EXPOSURE.json').write_text('{"status":"exposure_started"}')
    assert presentation_ui.independent_study(tmp_path)['status']=='IN PROGRESS'
    (folder/'FAILED.json').write_text('{"status":"INCOMPLETE"}')
    assert presentation_ui.independent_study(tmp_path)['status']=='INCOMPLETE'


def test_independent_loader_rejects_artifact_tampering(tmp_path):
    from aerodepth_lite.inference import file_sha256
    folder=tmp_path/'results/independent_test_20261030';folder.mkdir(parents=True)
    (folder/'PROTOCOL.json').write_text('{}');sha=file_sha256(folder/'PROTOCOL.json')
    (folder/'EXPOSURE.json').write_text(json.dumps({'protocol_sha256':sha}))
    for name in ['results.json','comparison.csv','timing.json']:(folder/name).write_text('{}')
    complete={'status':'COMPLETED','frames':8820,'equal_counts':True,'protocol_sha256':sha,'artifacts_sha256':{name:file_sha256(folder/name) for name in ['results.json','comparison.csv','timing.json']}}
    (folder/'COMPLETION.json').write_text(json.dumps(complete))
    assert presentation_ui.independent_study(tmp_path)['status']=='COMPLETED'
    (folder/'results.json').write_text('{"altered":true}')
    result=presentation_ui.independent_study(tmp_path)
    assert result['status']=='UNAVAILABLE · integrity check failed' and result['results'] is None


def test_missing_private_kd_disables_forward_preserves_recorded_view(tmp_path,monkeypatch):
    cfg=json.loads((ROOT/'configs/dashboard.json').read_text())
    cfg['models']['KD']['weights']=str(tmp_path/'missing_KD.pt')
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    monkeypatch.setenv('AERODEPTH_PRESENTATION_CONFIG',str(path))
    a=app(monkeypatch);nav(a,'Depth viewer');select(a,'Model','KD')
    assert not a.exception
    assert next(b for b in a.button if b.label=='Run actual inference').disabled
    assert any('checkpoint unavailable' in str(c.value) for c in a.caption)


@pytest.mark.parametrize('cache_mode',['continuous','examples'])
def test_viewer_frame_survives_navigation_and_widget_cleanup(monkeypatch,cache_mode):
    if cache_mode=='examples':
        from aerodepth_lite import dashboard_data
        monkeypatch.setattr(dashboard_data,'viewer_features',lambda cfg:[])
    a=app(monkeypatch);nav(a,'Depth viewer')
    next(b for b in a.button if b.label=='Typical · 1567').click().run(timeout=60)
    assert not a.exception
    for page in ['Results','Model comparison','Scheduler','Artifacts']:
        nav(a,page)
        nav(a,'Depth viewer')
        assert not a.exception
        frames=[s for s in a.slider if s.label=='Validation frame']
        frame=frames[0].value if frames else next(s for s in a.selectbox if s.label=='Recorded validation frame').value
        assert frame==1567
