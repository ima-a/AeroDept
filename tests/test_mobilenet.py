from copy import deepcopy
import csv

import pytest
import torch

from aerodepth_lite.data import MidAirDataset, load_manifest
from aerodepth_lite.losses import depth_loss
from aerodepth_lite.model import build_model, load_checkpoint, model_contract, model_stats, save_checkpoint
from aerodepth_lite.replay import replay_policy
from aerodepth_lite.train import train_model
from aerodepth_lite.kaggle import validation_outputs, estimate_training_seconds


def mobile_config(smoke_config):
    config = deepcopy(smoke_config)
    config['model'] = {'name': 'mobilenet_v3_small_depth', 'pretrained': False}
    config['data']['image_size'] = [33, 49]
    config['train']['lr_schedule'] = {'name': 'cosine', 't_max': 2, 'eta_min': 1e-5}
    return config


@pytest.mark.parametrize('size', [(64, 96), (65, 97), (192, 320)])
def test_mobile_shape_range_and_normalization(size):
    model = build_model({'name': 'mobilenet_v3_small_depth', 'pretrained': False}).eval()
    rgb = torch.rand(1, 3, *size)
    observed = []
    hook = model.encoder[0].register_forward_pre_hook(lambda module, args: observed.append(args[0]))
    with torch.no_grad():
        result = model(rgb)
    hook.remove()
    assert result.shape == (1, 1, *size)
    assert torch.isfinite(result).all() and (result > 0.1).all()
    assert torch.equal(observed[0], (rgb - model.rgb_mean) / model.rgb_std)
    assert model_stats(model)['parameters'] == 1_027_489
    assert not hasattr(model, 'classifier')


def test_pretrained_weight_selection_without_download(monkeypatch):
    import torchvision.models as models
    original = models.mobilenet_v3_small
    requested = []
    def fake_factory(*, weights):
        requested.append(weights)
        return original(weights=None)
    monkeypatch.setattr(models, 'mobilenet_v3_small', fake_factory)
    build_model({'name': 'mobilenet_v3_small_depth', 'pretrained': True})
    build_model({'name': 'mobilenet_v3_small_depth', 'pretrained': False})
    assert requested == [models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, None]


def test_mobile_masked_backward_and_checkpoint(tmp_path, smoke_config):
    config = mobile_config(smoke_config)
    model = build_model(config['model'])
    rgb = torch.rand(2, 3, 33, 49)
    prediction = model(rgb)
    target = torch.full_like(prediction, 5.0)
    valid = torch.ones_like(target, dtype=torch.bool)
    target[..., 0, 0] = float('nan')
    target[..., 1, 1] = 0
    valid[..., 1, 1] = False
    loss, _ = depth_loss(prediction, target, valid, config['loss'])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder[0][0].weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(rgb)
    path = tmp_path / 'mobile.pt'
    save_checkpoint(path, model, optimizer, 0, {'model_contract': model_contract(config)})
    restored = build_model(config['model']).eval()
    checkpoint = load_checkpoint(path, restored, torch.optim.AdamW(restored.parameters()))
    assert checkpoint['extra']['model_contract']['preprocessing']['crop'] is None
    with torch.no_grad():
        assert torch.equal(expected, restored(rgb))


def test_mobile_smoke_resume_equivalent_and_validation(fixture_dataset, smoke_config, tmp_path):
    root, manifest = fixture_dataset
    config = mobile_config(smoke_config)
    config['train']['epochs'] = 2
    continuous = tmp_path / 'continuous'
    train_model(config, root, manifest, continuous)
    stopped = deepcopy(config)
    stopped['train']['stop_after_epochs'] = 1
    interrupted = tmp_path / 'interrupted'
    train_model(stopped, root, manifest, interrupted)
    resumed = tmp_path / 'resumed'
    summary = train_model(config, root, manifest, resumed, interrupted / 'last.pt')
    a = torch.load(continuous / 'last.pt', weights_only=False)
    b = torch.load(resumed / 'last.pt', weights_only=False)
    assert all(torch.equal(a['model'][key], b['model'][key]) for key in a['model'])
    assert a['extra']['scheduler'] == b['extra']['scheduler']
    assert b['epoch'] == 1 and len(b['extra']['history']) == 2
    assert a['extra']['global_step'] == b['extra']['global_step'] == 2
    assert b['extra']['model_contract'] == model_contract(config)
    assert {'rng', 'scaler', 'manifest_sha256', 'config'} <= b['extra'].keys()
    assert all((resumed / name).exists() for name in ('best.pt', 'last.pt', 'history.csv', 'config.yaml', 'manifest.json', 'metadata.json', 'run.json'))
    with (resumed / 'history.csv').open() as handle:
        assert [r['epoch'] for r in csv.DictReader(handle)] == ['0', '1']
    result = validation_outputs(config, root, manifest, resumed / 'best.pt', resumed / 'validation')
    assert result['split'] == 'val'
    assert (resumed / 'validation/prediction_00.png').exists()
    estimate = estimate_training_seconds(summary, {'train': 100, 'val': 20}, 50)
    assert estimate['estimated_training_hours'] > 0
    wrong = deepcopy(config)
    wrong['data']['image_size'] = [32, 48]
    with pytest.raises(ValueError, match='preprocessing'):
        train_model(wrong, root, manifest, tmp_path / 'wrong', resumed / 'last.pt')


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    model = build_model({'name': 'aerodepth_tiny', 'base_channels': 4})
    path = tmp_path / 'last.pt'
    save_checkpoint(path, model, None, 0)
    original = path.read_bytes()
    def interrupted_save(payload, handle):
        handle.write(b'partial checkpoint')
        raise OSError('simulated interrupted write')
    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError, match='interrupted'):
        save_checkpoint(path, model, None, 1)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('policy', ['every', 'fixed2', 'adaptive'])
def test_mobile_replay_contract(policy, fixture_dataset, smoke_config, tmp_path):
    root, manifest = fixture_dataset
    config = mobile_config(smoke_config)
    records, _ = load_manifest(manifest, 'val')
    dataset = MidAirDataset(root, records, config['data'])
    assert 0 <= dataset[0]['rgb'].min() <= dataset[0]['rgb'].max() <= 1
    result = replay_policy(build_model(config['model']), dataset, policy,
        config['scheduler'], config['replay'], torch.device('cpu'), tmp_path / f'{policy}.csv')
    assert result['total_frames'] == 4
    assert result['metrics']['valid_pixels'] == 4 * 33 * 49
