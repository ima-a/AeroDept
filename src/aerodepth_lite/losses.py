from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        # Do not derive zero from values: an invalid neighborhood may contain
        # NaN/Inf, and NaN * 0 remains NaN.
        return values.new_zeros(())
    return values[mask].mean()


# Loss
def depth_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = valid.bool() & torch.isfinite(pred) & torch.isfinite(target) & (target > 0)
    # L1
    l1 = _masked_mean(torch.abs(pred - target), valid)
    eps = 1e-6
    # Log
    log = _masked_mean(torch.abs(torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps))), valid)
    # Gradient
    dx_valid = valid[..., :, 1:] & valid[..., :, :-1]
    dy_valid = valid[..., 1:, :] & valid[..., :-1, :]
    pred_dx, target_dx = pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    pred_dy, target_dy = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    gradient = _masked_mean(torch.abs(pred_dx - target_dx), dx_valid) + _masked_mean(torch.abs(pred_dy - target_dy), dy_valid)
    # Near
    near_mask = valid & (target < float(config.get("near_threshold", 10.0)))
    near = _masked_mean(torch.abs(pred - target), near_mask)
    components = {"l1": l1, "log": log, "gradient": gradient, "near": near}
    # Weights
    total = (
        float(config.get("l1_weight", 1.0)) * l1
        + float(config.get("log_weight", 0.0)) * log
        + float(config.get("gradient_weight", 0.0)) * gradient
        + float(config.get("near_weight", 0.0)) * near
    )
    return total, components


# Distillation
def masked_log_range_consistency(student: torch.Tensor, teacher: torch.Tensor,
                                 valid: torch.Tensor) -> torch.Tensor:
    """Mean |log(S)-log(stop_gradient(T))| on valid training-GT pixels.

    Reject bad predictions globally, even outside the GT mask. Empty masks have
    a differentiable zero so an all-invalid batch can safely call backward().
    """
    if student.shape != teacher.shape or student.shape != valid.shape:
        raise ValueError('Student, teacher and mask shapes must match')
    for name, value in [('student', student), ('teacher', teacher)]:
        if not torch.isfinite(value).all() or not (value > 0).all():
            raise FloatingPointError(f'Nonfinite/nonpositive {name} prediction')
    error = (student.float().log() - teacher.detach().float().log()).abs()
    mask = valid.bool()
    return error[mask].mean() if mask.any() else student.float().sum() * 0.
