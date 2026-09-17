from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(),
        )


class DSBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Hardswish(),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DSBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((x, skip), dim=1))


class AeroDepthTiny(nn.Module):
    """Compact depth network using depthwise-separable encoder/decoder blocks."""
    def __init__(self, base_channels: int = 24, min_depth: float = 0.1):
        super().__init__()
        b = base_channels
        self.min_depth = min_depth
        self.stem = ConvBNAct(3, b, 2)
        self.enc2 = DSBlock(b, b * 2, 2)
        self.enc3 = DSBlock(b * 2, b * 4, 2)
        self.enc4 = DSBlock(b * 4, b * 8, 2)
        self.bottleneck = DSBlock(b * 8, b * 8)
        self.up3 = UpBlock(b * 8, b * 4, b * 4)
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)
        self.head = nn.Conv2d(b, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        s1 = self.stem(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.bottleneck(self.enc4(s3))
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return F.softplus(self.head(x)) + self.min_depth


MODEL_REGISTRY = {"aerodepth_tiny": AeroDepthTiny}


def build_model(config: dict[str, Any]) -> nn.Module:
    name = config.get("name", "aerodepth_tiny")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choices: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        base_channels=int(config.get("base_channels", 24)),
        min_depth=float(config.get("min_depth", 0.1)),
    )


def model_stats(model: nn.Module) -> dict[str, int | float]:
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": parameters, "trainable_parameters": trainable, "fp32_size_mb": parameters * 4 / 1_000_000}


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None, epoch: int, extra: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

