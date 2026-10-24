from __future__ import annotations

from pathlib import Path
import os
import tempfile
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


class MobileNetV3SmallDepth(nn.Module):
    """ImageNet encoder, skip decoder, and positive camera range in metres.

    Input is paired/resized RGB in [0, 1]. Normalizing here keeps the dataset
    and replay scheduler's visual-change contract unchanged. No centre crop.
    """
    def __init__(self, pretrained: bool = True, min_depth: float = 0.1):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self.min_depth = min_depth
        self.encoder = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        ).features  # Neither classifier nor global pooling retained.
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.project = DSBlock(576, 128)
        self.up4 = UpBlock(128, 48, 64)
        self.up3 = UpBlock(64, 24, 32)
        self.up2 = UpBlock(32, 16, 24)
        self.up1 = UpBlock(24, 16, 16)
        self.head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        x = (x - self.rgb_mean) / self.rgb_std
        skips = {}
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in (0, 1, 3, 8):
                skips[index] = x
        x = self.project(x)
        x = self.up4(x, skips[8])
        x = self.up3(x, skips[3])
        x = self.up2(x, skips[1])
        x = self.up1(x, skips[0])
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return F.softplus(self.head(x)) + self.min_depth


MODEL_REGISTRY = {"aerodepth_tiny": AeroDepthTiny, "mobilenet_v3_small_depth": MobileNetV3SmallDepth}


def build_model(config: dict[str, Any]) -> nn.Module:
    name = config.get("name", "aerodepth_tiny")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choices: {sorted(MODEL_REGISTRY)}")
    if name == "mobilenet_v3_small_depth":
        return MobileNetV3SmallDepth(pretrained=bool(config.get("pretrained", True)),
                                    min_depth=float(config.get("min_depth", 0.1)))
    return MODEL_REGISTRY[name](
        base_channels=int(config.get("base_channels", 24)),
        min_depth=float(config.get("min_depth", 0.1)),
    )


def model_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Serializable inference contract, independent of weight initialization."""
    mobile = config["model"].get("name", "aerodepth_tiny") == "mobilenet_v3_small_depth"
    return {
        "architecture": config["model"].get("name", "aerodepth_tiny"),
        "model_options": {k: v for k, v in config["model"].items() if k != "pretrained"},
        "preprocessing": {
            "input": "RGB float32 [0,1]", "image_size": config["data"].get("image_size", [192, 320]),
            "rgb_resize": "bilinear align_corners=False", "depth_resize": "nearest",
            "crop": None, "paired_training_horizontal_flip": True,
            "normalization_location": "model" if mobile else None,
            "mean": [0.485, 0.456, 0.406] if mobile else None,
            "std": [0.229, 0.224, 0.225] if mobile else None,
            "depth_encoding": config["data"].get("depth_encoding", "midair_float16_bits"),
            "depth_scale": config["data"].get("depth_scale", 1.0),
            "valid_depth_metres": [config["data"].get("min_depth", 0.1), config["data"].get("max_depth", 1000.0)],
            "invalid_depth": "nonfinite or outside valid range; masked before nearest resize",
        },
        "output": "single-channel positive Euclidean camera range in metres; input spatial dimensions",
    }


def model_stats(model: nn.Module) -> dict[str, int | float]:
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats = {"parameters": parameters, "trainable_parameters": trainable, "fp32_size_mb": parameters * 4 / 1_000_000}
    if isinstance(model, MobileNetV3SmallDepth):
        encoder_parameters = sum(p.numel() for p in model.encoder.parameters())
        stats.update(encoder_parameters=encoder_parameters, decoder_parameters=parameters - encoder_parameters)
    return stats


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None, epoch: int, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed writer must leave the previous completed checkpoint readable.
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "epoch": epoch,
                "extra": extra or {},
            }, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

