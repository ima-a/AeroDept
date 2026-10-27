from pathlib import Path
import copy
import json
import hashlib

import pytest
import torch
import yaml

from aerodepth_lite.data import FrameRecord
from aerodepth_lite.expanded import (MANIFEST_SHA, INITIAL_SHA, PREP, GuardedDataset, ImageReadGuard,
    check_record, load_inputs, initialize_student, require_identity, strict_supervised,
    teacher_gate, check_config, fixed_training_subset, ensure_prior_best, backoff_scaler,
    GRADIENT_POLICY)
from aerodepth_lite.losses import masked_log_range_consistency
from aerodepth_lite.model import build_model, model_contract, model_stats

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('shape', [(64, 96), (65, 97), (193, 321)])
def test_teacher_alignment_normalizes_once_and_offline(shape, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError('Offline model attempted a download')
    monkeypatch.setattr(torch.hub, 'download_url_to_file', no_network)
    torch.set_num_threads(2)
    model = build_model({'name': 'resnet50_depth', 'pretrained': False}).eval()
    rgb = torch.rand(1, 3, *shape)
    observed = []
    handle = model.encoder['conv1'].register_forward_pre_hook(lambda m, args: observed.append(args[0].detach()))
    with torch.no_grad():
        output = model(rgb)
    handle.remove()
    assert output.shape == (1, 1, *shape)
    assert torch.isfinite(output).all() and (output >= .1).all()
    assert torch.equal(observed[0], (rgb - model.rgb_mean) / model.rgb_std)
    stats = model_stats(model)
    assert stats['encoder_parameters'] + stats['decoder_parameters'] == stats['parameters']
    assert not any('fc' in k for k in model.state_dict())
    state = model.state_dict()
    restored = build_model({'name': 'resnet50_depth', 'pretrained': False}).eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        assert torch.equal(output, restored(rgb))
    contract = model_contract({'model': {'name': 'resnet50_depth'}, 'data': {}})
    assert contract['preprocessing']['normalization_location'] == 'model'
    assert contract['preprocessing']['crop'] is None


def test_kd_mask_formula_detach_and_empty():
    student = torch.tensor([[[[2., 4.], [8., 16.]]]], requires_grad=True)
    teacher = torch.ones_like(student, requires_grad=True)
    mask = torch.tensor([[[[True, False], [True, False]]]])
    loss = masked_log_range_consistency(student, teacher, mask)
    assert loss.item() == pytest.approx((torch.log(torch.tensor(2.)) + torch.log(torch.tensor(8.))).item() / 2)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[~mask].eq(0).all()
    student.grad = None
    zero = masked_log_range_consistency(student, teacher, torch.zeros_like(mask))
    zero.backward()
    assert zero.item() == 0 and student.grad.eq(0).all()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0., -1.])
def test_kd_rejects_invalid_prediction_even_outside_mask(bad):
    x = torch.ones(1, 1, 2, 2, requires_grad=True)
    t = x.detach().clone()
    t[0, 0, 0, 0] = bad
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, 0, 0, 0] = False
    with pytest.raises(FloatingPointError):
        masked_log_range_consistency(x, t, mask)


def test_supervised_checks_and_empty_mask():
    x = torch.ones(1, 1, 2, 2, requires_grad=True)
    mask = torch.zeros_like(x, dtype=torch.bool)
    loss, _ = strict_supervised(x, torch.zeros_like(x), mask, {})
    loss.backward()
    assert loss.item() == 0 and x.grad.eq(0).all()
    bad = x.detach().clone()
    bad[0, 0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        strict_supervised(bad, x, mask, {})
    with pytest.raises(ValueError):
        masked_log_range_consistency(x, x[..., :1], mask)


def test_initialization_loads_all_epoch4_buffers_with_fresh_optimizer(tmp_path):
    model = build_model({'name': 'mobilenet_v3_small_depth', 'pretrained': False})
    path = ROOT / 'models/mobilenet_v3_small_epoch4.state_dict.pt'
    initialize_student(model, path)
    expected = torch.load(path, weights_only=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == INITIAL_SHA
    assert any(k.endswith('running_var') for k in expected)
    assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items())
    optimizer = torch.optim.AdamW(model.parameters())
    assert not optimizer.state
    wrong = tmp_path / 'wrong.pt'
    wrong.write_bytes(b'wrong weights')
    with pytest.raises(ValueError, match='SHA-256'):
        initialize_student(model, wrong)


@pytest.mark.parametrize('key', ['teacher', 'starting_weights', 'config', 'manifest_sha256', 'arm', 'scope', 'source_sha256', 'gradient_policy'])
def test_resume_identity_is_strict(key):
    original = {key: 'frozen', 'format': 1}
    changed = original | {key: 'different'}
    with pytest.raises(ValueError, match='identity'):
        require_identity(changed, original)
    with pytest.raises(ValueError):
        require_identity({}, original)


def test_frozen_manifest_and_read_guards_before_image_open(tmp_path, monkeypatch):
    rows, manifest = load_inputs(PREP / 'manifest_train_val.json.gz', PREP / 'reservations.json', tmp_path)
    assert len(rows['train']) == 44098 and len(rows['val']) == 4410
    assert not manifest['splits']['test']
    assert len(fixed_training_subset(rows['train'])) == 64
    changed = tmp_path / 'bad.json'
    changed.write_text('{}')
    with pytest.raises(ValueError, match='manifest SHA'):
        load_inputs(changed, PREP / 'reservations.json', tmp_path)
    guard = ImageReadGuard(tmp_path, rows['train'][:1])
    with pytest.raises(PermissionError):
        guard.check('os.scandir', (str(tmp_path),))
    forbidden = FrameRecord('Kite_training/foggy/trajectory_2007', 0,
        'Kite_training/foggy/color_left/trajectory_2007/000000.JPEG',
        'Kite_training/foggy/depth/trajectory_2007/000000.PNG')
    monkeypatch.setattr('PIL.Image.open', lambda *a, **k: pytest.fail('Forbidden image opened'))
    with pytest.raises(ValueError, match='Forbidden'):
        GuardedDataset(tmp_path, [forbidden], {}, 'train', True)
    with pytest.raises(PermissionError):
        guard.check('open', (str(tmp_path / forbidden.rgb_path), 'rb', 0))
    escaped = FrameRecord('Kite_training/sunny/trajectory_0005', 0, '../bad.JPEG', '../bad.PNG')
    with pytest.raises(ValueError, match='Unsafe'):
        check_record(escaped, 'train', tmp_path)


def test_matched_student_configs():
    a, c = [yaml.safe_load((ROOT / f'configs/expanded_{s}.yaml').read_text()) for s in ('supervised', 'kd')]
    for section in ('seed', 'data', 'model', 'train', 'loss'):
        assert a[section] == c[section]
    assert a['kd_weight'] == 0 and c['kd_weight'] == .2
    for arm in ('supervised', 'teacher', 'kd'):
        check_config(yaml.safe_load((ROOT / f'configs/expanded_{arm}.yaml').read_text()))


def test_gate_rejects_near_regression_despite_better_pooled_error():
    metric = dict(absrel=.3, rmse=100., near_absrel=.2, near_rmse=3., valid_pixels=100, near_valid_pixels=10)
    baseline = {'overall': metric.copy(), 'per_trajectory': {'Kite': metric.copy(), 'PLE': metric.copy()}}
    teacher = copy.deepcopy(baseline)
    teacher['overall']['absrel'] = .2
    assert teacher_gate(baseline, teacher)['passed']
    teacher['per_trajectory']['PLE']['near_rmse'] = 3.01
    assert not teacher_gate(baseline, teacher)['passed']
    teacher['per_trajectory']['PLE']['near_rmse'] = None
    assert not teacher_gate(baseline, teacher)['passed']


def test_atomic_best_last_interruption_recovers_only_pinned_copy(tmp_path):
    prior = b'exact prior checkpoint serialization'
    expected = hashlib.sha256(prior).hexdigest()
    best, previous = tmp_path / 'best.pt', tmp_path / 'previous_best.pt'
    best.write_bytes(b'new best promoted before last writer was interrupted')
    previous.write_bytes(b'unrelated checkpoint')
    with pytest.raises(ValueError, match='exact prior'):
        ensure_prior_best(tmp_path, expected)
    assert best.read_bytes().startswith(b'new best')
    previous.write_bytes(prior)
    ensure_prior_best(tmp_path, expected)
    assert best.read_bytes() == previous.read_bytes() == prior
    previous.write_bytes(b'older recovery copy')
    ensure_prior_best(tmp_path, expected)  # Correct best wins; do not use stale backup.
    assert best.read_bytes() == prior


def test_amp_backoff_resets_unscale_state_without_optimizer_update():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=1024.)
    scaler.scale(parameter * 2).backward()
    scaler.unscale_(optimizer)
    state = scaler.state_dict()
    state['_growth_tracker'] = 37
    scaler.load_state_dict(state)
    assert backoff_scaler(scaler) == 512.
    assert scaler.state_dict()['_growth_tracker'] == 0
    assert parameter.item() == 1. and not optimizer.state
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter * 2).backward()
    scaler.unscale_(optimizer)  # Previous UNSCALED bookkeeping must be gone.
    assert parameter.grad.item() == 2.
    assert GRADIENT_POLICY['max_overflow_retries'] == 8


def test_amp_backoff_rejects_float32_scale_underflow():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    scaler = torch.amp.GradScaler('cpu', init_scale=2. ** -149)
    scaler.scale(parameter).backward()
    with pytest.raises(FloatingPointError, match='cannot be reduced'):
        backoff_scaler(scaler)
