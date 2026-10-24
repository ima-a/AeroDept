"""Guardrails for the laptop launcher; no real GPU or dataset is needed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aerodepth_lite.repro import sha256_file


def launcher():
    path = Path(__file__).resolve().parents[1] / 'scripts/local_training.py'
    spec = importlib.util.spec_from_file_location('local_training', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_refuses_prolonged_cpu_training(monkeypatch, tmp_path):
    module = launcher()
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='prolonged training is disabled'):
        module.run(SimpleNamespace(mode='main', output=tmp_path / 'main'))
    assert not (tmp_path / 'main').exists()


def test_undefined_sample_metric_is_null_not_zero(tmp_path):
    path = tmp_path / 'measurement.json'
    launcher().write_json(path, {'epoch': {'near_rmse': float('nan'), 'val_loss': 2.0}})
    assert json.loads(path.read_text()) == {'epoch': {'near_rmse': None, 'val_loss': 2.0}}


def test_explicitly_authorized_duration_override_preserves_manifest_guard():
    module = launcher()
    gate = {'estimated_50_epoch_hours_with_20_percent_allowance': 22.9,
            'manifest_sha256': 'frozen', 'config': {'seed': 42}}
    with pytest.raises(ValueError, match='exceeds 12 hours'):
        module.validate_benchmark_gate(gate, {'seed': 42}, 'frozen')
    module.validate_benchmark_gate(gate, {'seed': 42}, 'frozen', max_hours=24)
    with pytest.raises(ValueError, match='does not match'):
        module.validate_benchmark_gate(gate, {'seed': 42}, 'different', max_hours=24)


@pytest.mark.parametrize('hours,wrong_manifest', [(12.01, False), (1, True)])
def test_main_refuses_over_budget_or_wrong_manifest(monkeypatch, tmp_path, fixture_dataset, hours, wrong_manifest):
    module = launcher()
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: True)
    class FakeCudaResult:
        def sum(self): return self
        def item(self): return 4
    monkeypatch.setattr(module.torch, 'ones', lambda *a, **kw: FakeCudaResult())
    root, manifest = fixture_dataset
    config_path = Path(__file__).resolve().parents[1] / 'configs/mobilenet_local.yaml'
    benchmark = tmp_path / 'measurement.json'
    benchmark.write_text(json.dumps({'estimated_50_epoch_hours_with_20_percent_allowance': hours,
        'manifest_sha256': 'wrong' if wrong_manifest else sha256_file(manifest),
        'config': module.load_config(config_path)}))
    args = SimpleNamespace(config=config_path, output=tmp_path / 'main', resume=None,
        mode='main', benchmark=benchmark, manifest=manifest, dataset_root=root)
    with pytest.raises(ValueError, match='exceeds 12 hours or does not match'):
        module.run(args)
    assert not list(args.output.glob('*.pt'))
