"""Optional UI checks. Normal CLI installation does not require Streamlit."""
from pathlib import Path
import json
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from aerodepth_lite import presentation_ui
ROOT=Path(__file__).resolve().parents[1]


def test_dashboard_tabs_selectors_and_no_automatic_inference(monkeypatch):
    monkeypatch.setattr(presentation_ui,'actual_inference',lambda *a,**k:pytest.fail('Automatic inference'))
    app=AppTest.from_file(str(ROOT/'scripts/dashboard.py')).run(timeout=60)
    assert not app.exception
    assert [x.label for x in app.tabs]==['Overview','Model comparison','Depth demonstration','Scheduler','Evidence']
    selector=next(x for x in app.selectbox if x.label=='Student model')
    assert selector.value=='A'
    selector.select('Original').run(timeout=60);assert not app.exception
    next(x for x in app.selectbox if x.label=='Student model').select('KD').run(timeout=60);assert not app.exception
    assert any('Run actual model inference' in b.label for b in app.button)


@pytest.mark.parametrize('missing',['config','comparison','cache'])
def test_missing_inputs_are_unavailable_not_invented(tmp_path,monkeypatch,missing):
    if missing=='config':path=tmp_path/'absent.json'
    else:
        cfg=json.loads((ROOT/'configs/presentation_local.json').read_text());cfg[missing]=str(tmp_path/'absent');path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    monkeypatch.setenv('AERODEPTH_PRESENTATION_CONFIG',str(path))
    app=AppTest.from_file(str(ROOT/'scripts/dashboard.py')).run(timeout=60)
    assert not app.exception
    assert any('unavailable' in str(x.value).lower() or 'pending' in str(x.value).lower() for x in [*app.warning,*app.info,*app.caption])


def test_portable_config_resolves_only_explicit_locations(tmp_path,monkeypatch):
    monkeypatch.delenv('MIDAIR_ROOT',raising=False)
    monkeypatch.setenv('AERODEPTH_KD_CHECKPOINT','/explicit/private/best.pt')
    config=presentation_ui.load_dashboard_config(ROOT/'configs/presentation.json',tmp_path)
    assert config['results'].startswith(str(tmp_path/'results'))
    assert config['models']['A']['weights'].startswith(str(tmp_path/'models'))
    assert config['dataset_root'].startswith(str(tmp_path/'.unavailable'))
    assert config['models']['KD']['weights']=='/explicit/private/best.pt'
    assert config['models']['A']['sha256']=='6820e7050dd5b6278ecdaf7055d9d69dbbe246a9644ebbdff77056da4278ae20'


def test_portable_dashboard_without_usb_loads_saved_artifacts(monkeypatch):
    for name in ['MIDAIR_ROOT','AERODEPTH_PRESENTATION_CACHE','AERODEPTH_COMPARISON_CACHE','AERODEPTH_KD_CHECKPOINT']:
        monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('AERODEPTH_PRESENTATION_CONFIG',str(ROOT/'configs/presentation.json'))
    monkeypatch.setattr(presentation_ui,'actual_inference',lambda *a,**k:pytest.fail('Automatic inference'))
    app=AppTest.from_file(str(ROOT/'scripts/dashboard.py')).run(timeout=60)
    assert not app.exception
    assert len(app.tabs)==5
    assert any('cache unavailable' in str(x.value).lower() for x in app.info)
    assert any('INDEPENDENT TEST' in str(x.value) for x in app.markdown)
    assert presentation_ui.independent_study(ROOT)['status'] in ['COMPLETED','IN PROGRESS','INCOMPLETE','NOT RUN']


def test_frame_presets_do_not_run_inference(monkeypatch):
    monkeypatch.setattr(presentation_ui,'actual_inference',lambda *a,**k:pytest.fail('Automatic inference'))
    app=AppTest.from_file(str(ROOT/'scripts/dashboard.py')).run(timeout=60)
    preset=next(b for b in app.button if b.label=='Poor · frame 1534')
    preset.click().run(timeout=60)
    assert not app.exception
    assert next(s for s in app.slider if s.label=='Validation frame').value==1534


def test_independent_and_historical_selectors_are_artifact_only(monkeypatch):
    monkeypatch.setenv('AERODEPTH_PRESENTATION_CONFIG',str(ROOT/'configs/presentation.json'))
    monkeypatch.setattr(presentation_ui,'actual_inference',lambda *a,**k:pytest.fail('Automatic inference'))
    app=AppTest.from_file(str(ROOT/'scripts/dashboard.py')).run(timeout=60)
    for split in ['New independent test','Historical exposed test','Validation']:
        next(s for s in app.selectbox if s.label=='Evaluation split').select(split).run(timeout=60)
        assert not app.exception
    for study in ['New independent test · A','Historical exposed test · Original']:
        next(s for s in app.selectbox if s.label=='Scheduler study').select(study).run(timeout=60)
        assert not app.exception
    assert next(s for s in app.selectbox if s.label=='Student model').value=='A'


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
