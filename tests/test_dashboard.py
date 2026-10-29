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
    assert any('NOT RUN' in str(x.value) for x in app.markdown)


def test_frame_presets_do_not_run_inference(monkeypatch):
    monkeypatch.setattr(presentation_ui,'actual_inference',lambda *a,**k:pytest.fail('Automatic inference'))
    app=AppTest.from_file(str(ROOT/'scripts/dashboard.py')).run(timeout=60)
    preset=next(b for b in app.button if b.label=='Poor · frame 1534')
    preset.click().run(timeout=60)
    assert not app.exception
    assert next(s for s in app.slider if s.label=='Validation frame').value==1534
