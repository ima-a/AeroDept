from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
import csv
import json
import os
import random
import shutil
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader

from .data import MidAirDataset, collate_samples, load_manifest
from .losses import depth_loss
from .metrics import DepthMetricAccumulator
from .model import build_model, load_checkpoint, model_contract, model_stats, save_checkpoint
from .repro import run_metadata, seed_everything, seed_worker, sha256_file


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


# Loader
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


# Epoch
def _epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device, loss_config: dict[str, Any], optimizer: torch.optim.Optimizer | None, scaler: Any, max_batches: int | None, amp: bool, grad_clip: float = 1.0, progress=None, timing: dict | None = None) -> tuple[float, dict[str, float], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    components_total: dict[str, float] = {"l1": 0.0, "log": 0.0, "gradient": 0.0, "near": 0.0}
    metrics = DepthMetricAccumulator()
    batches = 0
    waiting = 0.0
    last_finished = perf_counter()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            waiting += perf_counter() - last_finished
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
            if progress is not None:
                progress(batches, min(len(loader), max_batches) if max_batches is not None else len(loader))
            last_finished = perf_counter()
    if not batches:
        raise ValueError("No batches available for training/evaluation")
    if timing is not None:
        timing.update(loader_wait_seconds=waiting, batches=batches)
    return loss_total / batches, {k: v / batches for k, v in components_total.items()}, metrics.compute()


# Training
def train_model(config: dict[str, Any], dataset_root: str | Path, manifest: str | Path, output: str | Path, resume: str | Path | None = None) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 42)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"] | {"min_depth": config["data"].get("min_depth", 0.1)}
    if resume:
        model_config["pretrained"] = False  # Resume never needs an ImageNet download.
    model = build_model(model_config).to(device)
    # AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"].get("learning_rate", 3e-4)), weight_decay=float(config["train"].get("weight_decay", 1e-4)))
    schedule = config["train"].get("lr_schedule", {})
    if schedule.get("name", "cosine") != "cosine":
        raise ValueError("Only the cosine learning-rate schedule is supported")
    # Cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(schedule.get("t_max", config["train"].get("epochs", 1))), eta_min=float(schedule.get("eta_min", 0.0)))
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    # AMP
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    prior_duration = 0.0
    best_absrel = float("inf")
    contract = model_contract(config)
    extra: dict[str, Any] = {}
    if resume:
        # Resume
        checkpoint = load_checkpoint(resume, model, optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        extra = checkpoint.get("extra", {})
        if "model_contract" in extra and extra["model_contract"] != contract:
            raise ValueError("Checkpoint architecture/preprocessing does not match this configuration")
        if "manifest_sha256" in extra and extra["manifest_sha256"] != sha256_file(manifest):
            raise ValueError("Checkpoint split manifest does not match")
        if "config" in extra:
            previous = extra["config"]
            for section in ("loss", "seed"):
                if previous.get(section) != config.get(section):
                    raise ValueError(f"Resume {section} settings differ from checkpoint")
            for option in ("batch_size", "learning_rate", "weight_decay", "lr_schedule", "grad_clip", "amp"):
                if previous["train"].get(option) != config["train"].get(option):
                    raise ValueError(f"Resume training option {option} differs from checkpoint")
            if previous["data"].get("workers", 0) != config["data"].get("workers", 0):
                raise ValueError("Resume worker count differs from checkpoint")
        if "scaler" in extra:
            scaler.load_state_dict(extra["scaler"])
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
        if not (output / "best.pt").exists():
            prior_best = Path(resume).parent / "best.pt"
            if not prior_best.exists():
                raise ValueError("Resume requires the prior best.pt alongside the resume checkpoint")
            shutil.copyfile(prior_best, output / "best.pt")
    train_loader = _loader(dataset_root, manifest, "train", config, True)
    val_loader = _loader(dataset_root, manifest, "val", config, False)
    if "rng" in extra:
        rng = extra["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        train_loader.generator.set_state(rng["train_loader"].cpu())
        val_loader.generator.set_state(rng["val_loader"].cpu())
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if Path(manifest).resolve() != (output / "manifest.json").resolve():
        shutil.copyfile(manifest, output / "manifest.json")
    metadata = run_metadata(config, Path(manifest)) | {
        "model_contract": contract, "model": model_stats(model),
        "dataset_root": str(Path(dataset_root).resolve()),
        "resume_checkpoint": {"path": str(Path(resume).resolve()), "sha256": sha256_file(resume)} if resume else None,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    started = perf_counter()
    max_train = config["train"].get("max_train_batches")
    max_val = config["train"].get("max_val_batches")
    end_epoch = int(config["train"].get("epochs", 1))
    if config["train"].get("stop_after_epochs") is not None:
        end_epoch = min(end_epoch, start_epoch + int(config["train"]["stop_after_epochs"]))
    global_step = int(extra.get("global_step", sum(int(r.get("train_samples", 0)) // train_loader.batch_size for r in history)))
    for epoch in range(start_epoch, end_epoch):
        grad_clip = float(config["train"].get("grad_clip", 1.0))
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        def progress(stage):
            def update(done, total):
                if done == 1 or done % 100 == 0 or done == total:
                    state = {"epoch": epoch + 1, "epoch_budget": int(config["train"]["epochs"]),
                             "stage": stage, "batches_completed": done, "batches_total": total,
                             "global_step_at_epoch_start": global_step, "pid": os.getpid()}
                    temporary = output / "progress.json.tmp"
                    temporary.write_text(json.dumps(state) + "\n", encoding="utf-8")
                    temporary.replace(output / "progress.json")
                    print(json.dumps(state), flush=True)
            return update
        train_timing, val_timing = {}, {}
        epoch_started = perf_counter()
        # Training
        train_loss, train_components, _ = _epoch(model, train_loader, device, config["loss"], optimizer, scaler, max_train, amp, grad_clip, progress("train"), train_timing)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = perf_counter() - epoch_started
        validation_started = perf_counter()
        # Validation
        val_loss, val_components, val_metrics = _epoch(model, val_loader, device, config["loss"], None, scaler, max_val, amp, progress=progress("val"), timing=val_timing)
        if device.type == "cuda":
            torch.cuda.synchronize()
        val_seconds = perf_counter() - validation_started
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_components.items()},
            **{f"val_{k}": v for k, v in val_components.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_seconds": train_seconds, "val_seconds": val_seconds,
            "train_samples": min(len(train_loader.dataset), (max_train if max_train is not None else len(train_loader)) * train_loader.batch_size),
            "val_samples": min(len(val_loader.dataset), (max_val if max_val is not None else len(val_loader)) * val_loader.batch_size),
            "train_loader_wait_seconds": train_timing["loader_wait_seconds"],
            "val_loader_wait_seconds": val_timing["loader_wait_seconds"],
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / 1e6 if device.type == "cuda" else 0.0,
        }
        global_step += train_timing["batches"]
        history.append(row)
        # Selection
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
            "global_step": global_step,
            "model_contract": contract, "config": config,
            "manifest_sha256": sha256_file(manifest),
            "scaler": scaler.state_dict(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "train_loader": train_loader.generator.get_state(), "val_loader": val_loader.generator.get_state()},
        }
        # Write best first: after interruption, last.pt must never refer to a
        # selected best model that has not been persisted yet.
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, epoch, checkpoint_extra)
        save_checkpoint(output / "last.pt", model, optimizer, epoch, checkpoint_extra)
        _write_history(output / "history.csv", history)
        print(json.dumps({"completed_epoch": epoch + 1, **row}), flush=True)
    if not history:
        raise ValueError("No epochs available for training")
    duration = prior_duration + perf_counter() - started
    summary = {
        **metadata,
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
