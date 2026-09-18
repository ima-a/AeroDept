from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
import csv
import json

import torch
from torch.utils.data import DataLoader

from .data import MidAirDataset, collate_samples, load_manifest
from .losses import depth_loss
from .metrics import DepthMetricAccumulator
from .model import build_model, load_checkpoint, model_stats, save_checkpoint
from .repro import run_metadata, seed_everything, seed_worker


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _loader(root: str | Path, manifest: str | Path, split: str, config: dict[str, Any], training: bool) -> DataLoader:
    records, _ = load_manifest(manifest, split)
    dataset = MidAirDataset(root, records, config["data"], training=training)
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 8)),
        shuffle=training,
        num_workers=int(config["data"].get("workers", 0)),
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate_samples,
    )


def _epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device, loss_config: dict[str, Any], optimizer: torch.optim.Optimizer | None, scaler: Any, max_batches: int | None, amp: bool, grad_clip: float = 1.0) -> tuple[float, dict[str, float], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    components_total: dict[str, float] = {"l1": 0.0, "log": 0.0, "gradient": 0.0, "near": 0.0}
    metrics = DepthMetricAccumulator()
    batches = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            rgb, target, valid = (batch[k].to(device) for k in ("rgb", "depth", "valid"))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                prediction = model(rgb)
                loss, components = depth_loss(prediction, target, valid, loss_config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {batch_index}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            loss_total += float(loss.detach())
            for key, value in components.items():
                components_total[key] += float(value.detach())
            metrics.update(prediction.detach(), target, valid, float(loss_config.get("near_threshold", 10.0)))
            batches += 1
    if not batches:
        raise ValueError("No batches available for training/evaluation")
    return loss_total / batches, {k: v / batches for k, v in components_total.items()}, metrics.compute()


def train_model(config: dict[str, Any], dataset_root: str | Path, manifest: str | Path, output: str | Path, resume: str | Path | None = None) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 42)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"] | {"min_depth": config["data"].get("min_depth", 0.1)}
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"].get("learning_rate", 3e-4)), weight_decay=float(config["train"].get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["train"].get("epochs", 1)))
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    prior_duration = 0.0
    best_absrel = float("inf")
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        extra = checkpoint.get("extra", {})
        if "scheduler" in extra:
            scheduler.load_state_dict(extra["scheduler"])
        else:
            # Compatibility with checkpoints written before scheduler state was
            # persisted. The optimizer contains the LR used for the saved epoch.
            scheduler.last_epoch = int(checkpoint["epoch"])
            scheduler.step()
        if isinstance(extra.get("history"), list):
            history = list(extra["history"])
            prior_duration = float(extra.get("duration_seconds", 0.0))
            best_absrel = float(extra.get("best_validation_absrel", float("inf")))
        elif "epoch" in extra:
            history = [extra]
            best_absrel = float(extra.get("val_absrel", float("inf")))
    train_loader = _loader(dataset_root, manifest, "train", config, True)
    val_loader = _loader(dataset_root, manifest, "val", config, False)
    started = perf_counter()
    max_train = config["train"].get("max_train_batches")
    max_val = config["train"].get("max_val_batches")
    for epoch in range(start_epoch, int(config["train"].get("epochs", 1))):
        grad_clip = float(config["train"].get("grad_clip", 1.0))
        train_loss, train_components, _ = _epoch(model, train_loader, device, config["loss"], optimizer, scaler, max_train, amp, grad_clip)
        val_loss, val_components, val_metrics = _epoch(model, val_loader, device, config["loss"], None, scaler, max_val, amp)
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_components.items()},
            **{f"val_{k}": v for k, v in val_components.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = float(val_metrics["absrel"]) < best_absrel
        if improved:
            best_absrel = float(val_metrics["absrel"])
        scheduler.step()
        duration = prior_duration + perf_counter() - started
        checkpoint_extra = {
            "history": history,
            "scheduler": scheduler.state_dict(),
            "duration_seconds": duration,
            "best_validation_absrel": best_absrel,
        }
        save_checkpoint(output / "last.pt", model, optimizer, epoch, checkpoint_extra)
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, epoch, checkpoint_extra)
        _write_history(output / "history.csv", history)
    if not history:
        raise ValueError("No epochs available for training")
    duration = prior_duration + perf_counter() - started
    summary = {
        **run_metadata(config, Path(manifest)),
        "duration_seconds": duration,
        "epochs_completed": len(history),
        "checkpoint": str(output / "best.pt"),
        "checkpoint_size_mb": (output / "best.pt").stat().st_size / 1_000_000,
        "model": model_stats(model),
        "best_validation_absrel": best_absrel,
        "final": history[-1],
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return summary
