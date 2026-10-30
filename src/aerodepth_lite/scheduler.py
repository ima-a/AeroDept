from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


def visual_change(previous: torch.Tensor | np.ndarray | None, current: torch.Tensor | np.ndarray, size: tuple[int, int] = (36, 64)) -> float:
    if previous is None:
        return 0.0

    def gray(value: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.mean(dim=0, keepdim=True)
        elif tensor.ndim == 3:
            tensor = tensor.mean(dim=-1).unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze()

    return float(torch.mean(torch.abs(gray(previous) - gray(current))).clamp(0, 1))


def motion_score(navigation: dict[str, float], scale: float = 5.0) -> tuple[float, bool]:
    groups = (
        ("vx", "vy", "vz"), ("velocity_x", "velocity_y", "velocity_z"),
        ("ax", "ay", "az"), ("acceleration_x", "acceleration_y", "acceleration_z"),
        ("wx", "wy", "wz"), ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
    )
    magnitudes = []
    for keys in groups:
        values = [navigation.get(k) for k in keys]
        present = [float(v) for v in values if v is not None and np.isfinite(v)]
        if present:
            magnitudes.append(float(np.linalg.norm(present)))
    if not magnitudes:
        return 0.0, False
    raw = max(magnitudes)
    return float(np.clip(raw / max(scale, 1e-6), 0.0, 1.0)), True


# Risk
def near_field_risk(depth: torch.Tensor | np.ndarray | None, threshold: float = 10.0) -> tuple[float, bool]:
    if depth is None:
        return 0.0, False
    values = torch.as_tensor(depth, dtype=torch.float32).squeeze()
    valid = torch.isfinite(values) & (values > 0)
    if not torch.any(valid):
        return 0.0, False
    height, width = values.shape[-2:]
    roi = values[height // 3:, width // 4: 3 * width // 4]
    roi_valid = torch.isfinite(roi) & (roi > 0)
    selected = roi[roi_valid] if torch.any(roi_valid) else values[valid]
    proportion = torch.mean((selected < threshold).float())
    percentile = torch.quantile(selected, 0.10)
    proximity = torch.clamp((threshold - percentile) / max(threshold, 1e-6), 0, 1)
    return float(torch.clamp(0.6 * proportion + 0.4 * proximity, 0, 1)), True


@dataclass
class Decision:
    infer: bool
    reason: str
    score: float
    motion: float
    motion_available: bool
    visual: float
    risk: float
    risk_available: bool
    age_before: int
    age_after: int
    max_skip_forced: bool


class AdaptiveScheduler:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.age = 0
        self.frame_count = 0
        self.previous_rgb: torch.Tensor | np.ndarray | None = None

    def decide(self, rgb: torch.Tensor | np.ndarray, navigation: dict[str, float], previous_depth: torch.Tensor | np.ndarray | None) -> Decision:
        cfg = self.config
        age_before = self.age
        motion, motion_available = motion_score(navigation, float(cfg.get("motion_scale", 5.0)))
        visual_raw = visual_change(self.previous_rgb, rgb, tuple(cfg.get("visual_size", [36, 64])))
        visual = float(np.clip(visual_raw / max(float(cfg.get("visual_scale", 0.15)), 1e-6), 0, 1))
        risk, risk_available = near_field_risk(previous_depth, float(cfg.get("near_threshold", 10.0)))
        max_skip = int(cfg.get("max_skip", 4))
        normalized_age = min(age_before / max(max_skip, 1), 1.0)
        score = (
            float(cfg.get("motion_weight", 0.25)) * motion
            + float(cfg.get("visual_weight", 0.30)) * visual
            + float(cfg.get("risk_weight", 0.30)) * risk
            + float(cfg.get("age_weight", 0.15)) * normalized_age
        )
        forced = False
        if self.frame_count == 0:
            infer, reason = True, "first_frame"
        elif age_before >= max_skip:
            infer, reason, forced = True, "max_skip", True
        elif risk_available and risk >= float(cfg.get("critical_risk", 0.75)):
            infer, reason = True, "critical_risk"
        elif motion_available and motion >= float(cfg.get("extreme_motion", 0.9)):
            infer, reason = True, "extreme_motion"
        elif visual >= float(cfg.get("extreme_visual", 0.9)):
            infer, reason = True, "extreme_visual"
        elif score >= float(cfg.get("decision_threshold", 0.62)):
            infer, reason = True, "combined_score"
        else:
            infer, reason = False, "stable"
        self.age = 0 if infer else age_before + 1
        self.frame_count += 1
        self.previous_rgb = torch.as_tensor(rgb).detach().cpu().clone()
        return Decision(infer, reason, score, motion, motion_available, visual, risk, risk_available, age_before, self.age, forced)

