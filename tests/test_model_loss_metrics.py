from __future__ import annotations

import math
import torch

from aerodepth_lite.losses import depth_loss
from aerodepth_lite.metrics import depth_metrics
from aerodepth_lite.model import build_model, load_checkpoint, model_stats, save_checkpoint


def test_model_shape_positive_and_lightweight():
    model = build_model({"name": "aerodepth_tiny", "base_channels": 8, "min_depth": 0.1})
    output = model(torch.rand(2, 3, 64, 96))
    assert output.shape == (2, 1, 64, 96)
    assert torch.all(output >= 0.1)
    assert model_stats(model)["parameters"] < 5_000_000


def test_checkpoint_round_trip(tmp_path):
    model = build_model({"base_channels": 4})
    original = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, None, 3, {"metric": 1.0})
    for parameter in model.parameters():
        parameter.data.zero_()
    state = load_checkpoint(path, model)
    assert state["epoch"] == 3
    assert all(torch.equal(model.state_dict()[k], value) for k, value in original.items())


def test_loss_is_finite_masked_and_differentiable():
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, float("nan")], [0.0, 8.0]]]])
    valid = torch.tensor([[[[True, True], [False, True]]]])
    total, parts = depth_loss(pred, target, valid, {"l1_weight": 1, "log_weight": 1, "gradient_weight": 1, "near_weight": 1, "near_threshold": 10})
    assert torch.isfinite(total)
    assert all(torch.isfinite(v) for v in parts.values())
    total.backward()
    assert torch.isfinite(pred.grad).all()


def test_hand_calculated_metrics_and_invalid_masking():
    pred = torch.tensor([1.0, 4.0, 99.0])
    target = torch.tensor([1.0, 2.0, 0.0])
    valid = torch.tensor([True, True, False])
    result = depth_metrics(pred, target, valid, near_threshold=3.0)
    assert math.isclose(result["absrel"], 0.5)
    assert math.isclose(result["rmse"], math.sqrt(2.0))
    assert result["delta1"] == 0.5
    assert result["delta2"] == 0.5
    assert result["delta3"] == 0.5
    assert result["valid_pixels"] == 2

