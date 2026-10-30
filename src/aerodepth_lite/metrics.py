from __future__ import annotations

from dataclasses import dataclass
import math
import torch


# Metrics
@dataclass
class DepthMetricAccumulator:
    absrel_sum: float = 0.0
    squared_sum: float = 0.0
    delta1_count: int = 0
    delta2_count: int = 0
    delta3_count: int = 0
    valid_count: int = 0
    near_absrel_sum: float = 0.0
    near_squared_sum: float = 0.0
    near_count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, near_threshold: float = 10.0) -> None:
        # Mask
        mask = valid.bool() & torch.isfinite(pred) & torch.isfinite(target) & (target > 0)
        if not torch.any(mask):
            return
        p = pred[mask].double().clamp_min(1e-6)
        g = target[mask].double().clamp_min(1e-6)
        error = p - g
        ratio = torch.maximum(p / g, g / p)
        # AbsRel
        self.absrel_sum += torch.sum(torch.abs(error) / g).item()
        # RMSE
        self.squared_sum += torch.sum(error.square()).item()
        # Delta
        self.delta1_count += torch.sum(ratio < 1.25).item()
        self.delta2_count += torch.sum(ratio < 1.25**2).item()
        self.delta3_count += torch.sum(ratio < 1.25**3).item()
        self.valid_count += p.numel()
        # Near
        near = g < near_threshold
        if torch.any(near):
            near_error = p[near] - g[near]
            self.near_absrel_sum += torch.sum(torch.abs(near_error) / g[near]).item()
            self.near_squared_sum += torch.sum(near_error.square()).item()
            self.near_count += int(torch.sum(near).item())

    # Pooling
    def compute(self) -> dict[str, float | int]:
        if self.valid_count == 0:
            return {k: math.nan for k in ("absrel", "rmse", "delta1", "delta2", "delta3", "near_absrel", "near_rmse")} | {"valid_pixels": 0}
        n = self.valid_count
        return {
            "absrel": self.absrel_sum / n,
            "rmse": math.sqrt(self.squared_sum / n),
            "delta1": self.delta1_count / n,
            "delta2": self.delta2_count / n,
            "delta3": self.delta3_count / n,
            "near_absrel": self.near_absrel_sum / self.near_count if self.near_count else math.nan,
            "near_rmse": math.sqrt(self.near_squared_sum / self.near_count) if self.near_count else math.nan,
            "valid_pixels": n,
        }


def depth_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, near_threshold: float = 10.0) -> dict[str, float | int]:
    accumulator = DepthMetricAccumulator()
    accumulator.update(pred, target, valid, near_threshold)
    return accumulator.compute()
