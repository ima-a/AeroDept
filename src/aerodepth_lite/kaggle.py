"""Reusable Kaggle preparation; no dataset copying or final-test execution."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import os
import random
import re

from PIL import Image
import numpy as np
import torch

from .cli import _validate_hdf5_sync
from .data import (FrameRecord, MidAirDataset, assert_no_leakage, dataset_fingerprint,
                   discover_midair, load_manifest)
from .model import build_model, load_checkpoint
from .train import _epoch, _loader
from .artifacts import plot_history, visualize_sample


PILOT_PATHS = {"Kite/trajectory_0003", "Kite/trajectory_0021"}


def path_group(trajectory: str) -> str:
    """Drop MidAir's climate digit: 0003/1003/2003/3003 share path 003.

    Include environment to avoid joining unrelated environments. Unknown layouts
    conservatively group the same numeric path together to prevent leakage.
    """
    parts = Path(trajectory).parts
    environment = next((re.sub(r"_(training|test)$", "", p, flags=re.I)
                        for p in parts if re.match(r"^(Kite|PLE)(_|$)", p, re.I)), "unknown")
    environment = {"kite": "Kite", "ple": "PLE"}.get(environment.lower(), environment)
    match = re.fullmatch(r"trajectory_(\d{4})", parts[-1])
    path = f"trajectory_{int(match[1]) % 1000:04d}" if match else parts[-1]
    return f"{environment}/{path}"


def _inspected(group: str, exclusions: set[str]) -> bool:
    return group in exclusions or (group.startswith("unknown/") and
        any(group.split("/")[-1] == excluded.split("/")[-1] for excluded in exclusions))


def discover_dataset_root(input_dir: str | Path) -> Path:
    """Find a single MidAir tree through arbitrary archive wrapper folders."""
    base = Path(input_dir).resolve()
    sensors = sorted(base.rglob("sensor_records.hdf5"))
    if not sensors:
        raise ValueError("No sensor_records.hdf5 found; attach the extracted MidAir dataset")
    root = Path(os.path.commonpath([str(p.parent) for p in sensors]))
    # One-weather extraction: retain environment in relative trajectory IDs.
    while root != base and not any(re.match(r"^(Kite|PLE)(_|$)", p.name, re.I)
                                   for p in root.iterdir() if p.is_dir()):
        if re.match(r"^(Kite|PLE)(_|$)", root.name, re.I):
            root = root.parent
            break
        root = root.parent
    if root == base and len({p.relative_to(base).parts[0] for p in sensors}) > 1:
        raise ValueError("Multiple attached dataset trees; set DATASET_ROOT explicitly")
    return root


def validate_dataset(root: Path, expected_trajectories: list[str] | None = None,
                     expected_count: int | None = None) -> tuple[list[FrameRecord], dict[str, Any]]:
    records, report = discover_midair(root)
    trajectories = sorted({r.trajectory_id for r in records})
    if expected_count is not None and len(trajectories) != expected_count:
        raise ValueError(f"Expected {expected_count} trajectories; found {len(trajectories)}")
    if expected_trajectories is not None and set(trajectories) != set(expected_trajectories):
        raise ValueError("Discovered trajectories differ from the expected inventory")
    if not (report["paired_frames"] == report["rgb_files"] == report["depth_files"]):
        raise ValueError("RGB/depth counts or pairing differ")
    identities = [(r.trajectory_id, r.frame_index) for r in records]
    if len(identities) != len(set(identities)) or len({r.depth_path for r in records}) != len(records):
        raise ValueError("Duplicate or ambiguously paired frames")
    # Header-level spatial validation of every pair, without test-depth statistics.
    for record in records:
        with Image.open(root / record.rgb_path) as image:
            rgb_size = image.size
        if Path(record.depth_path).suffix == ".npy":
            shape = np.load(root / record.depth_path, mmap_mode="r").shape
            depth_size = (shape[1], shape[0])
        else:
            with Image.open(root / record.depth_path) as image:
                depth_size = image.size
        if rgb_size != depth_size:
            raise ValueError(f"Spatial alignment mismatch: {record.rgb_path}")
    hdf5 = _validate_hdf5_sync(root, records)
    if (not hdf5["available"] or hdf5["unmatched_trajectories"] or
            hdf5["total_path_mismatches"] or hdf5["validation_errors"] or
            hdf5["maximum_navigation_resampling_error"] != 0):
        raise ValueError(f"HDF5 synchronization validation failed: {hdf5}")
    report.update(trajectories=trajectories, hdf5_validation=hdf5,
                  expected_inventory_verified=expected_trajectories is not None or expected_count is not None)
    return records, report


def grouped_manifest(records: list[FrameRecord], output: Path, seed: int = 42,
                     pilot_paths: set[str] | None = None) -> dict[str, Any]:
    records = sorted(records, key=lambda r: (r.trajectory_id, r.frame_index))
    excluded = PILOT_PATHS | {path_group(p) for p in pilot_paths or set()}
    groups: dict[str, list[str]] = {}
    for trajectory in sorted({r.trajectory_id for r in records}):
        groups.setdefault(path_group(trajectory), []).append(trajectory)
    eligible = sorted(g for g in groups if not _inspected(g, excluded))
    if len(groups) < 3 or not eligible:
        raise ValueError("Need at least three independent paths and an uninspected test path")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n_test = max(1, round(0.15 * len(groups)))
    n_test = min(n_test, len(eligible), len(groups) - 2)
    test = eligible[:n_test]
    remaining = sorted(set(groups) - set(test))
    rng.shuffle(remaining)
    n_val = min(max(1, round(0.15 * len(groups))), len(remaining) - 1)
    split_groups = {"train": remaining[n_val:], "val": remaining[:n_val], "test": test}
    splits = {name: sorted(t for group in selected for t in groups[group])
              for name, selected in split_groups.items()}
    payload = {
        "version": 2, "seed": seed, "generation_method": "deterministic path-grouped 70/15/15; pilot groups excluded from test; nonempty splits",
        "dataset_fingerprint": dataset_fingerprint(records), "trajectory_count": len(groups),
        "realization_count": sum(len(v) for v in groups.values()), "frame_count": len(records),
        "splits": splits, "path_groups": groups, "pilot_paths_excluded_from_test": sorted(excluded),
        "path_grouping": "environment + numeric trajectory ID modulo 1000; climate digit removed",
        "frame_counts": {name: sum(r.trajectory_id in selected for r in records) for name, selected in splits.items()},
        "records": [asdict(r) for r in records],
    }
    validate_manifest(payload, records, excluded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_manifest(payload: dict[str, Any], records: list[FrameRecord],
                      pilot_paths: set[str] | None = None) -> None:
    splits = payload["splits"]
    assert_no_leakage(splits)
    if set(splits) != {"train", "val", "test"} or not all(splits.values()):
        raise ValueError("All three splits must be nonempty")
    if set(sum(splits.values(), [])) != {r.trajectory_id for r in records}:
        raise ValueError("Manifest trajectory inventory differs")
    saved = [FrameRecord(**r) for r in payload["records"]]
    if (dataset_fingerprint(saved) != dataset_fingerprint(records) or
            payload["dataset_fingerprint"] != dataset_fingerprint(records)):
        raise ValueError("Manifest dataset fingerprint differs")
    owners = {}
    for name, trajectories in splits.items():
        for trajectory in trajectories:
            group = path_group(trajectory)
            if group in owners and owners[group] != name:
                raise ValueError(f"Weather/path group leakage: {group}")
            owners[group] = name
    excluded = PILOT_PATHS | {path_group(p) for p in pilot_paths or set()}
    if any(_inspected(path_group(t), excluded) for t in splits["test"]):
        raise ValueError("Previously inspected pilot path in final test")


def load_or_create_manifest(records: list[FrameRecord], output: Path, seed: int = 42,
                            existing: Path | None = None, pilot_paths: set[str] | None = None) -> dict[str, Any]:
    source = existing or (output if output.exists() else None)
    if source is None:
        return grouped_manifest(records, output, seed, pilot_paths)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_manifest(payload, records, pilot_paths)
    if payload.get("seed") != seed:
        raise ValueError("Manifest seed differs from configured seed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def estimate_training_seconds(smoke: dict[str, Any], counts: dict[str, int], epochs: int) -> dict[str, float]:
    row = smoke["final"]
    train_rate = row["train_samples"] / row["train_seconds"]
    val_rate = row["val_samples"] / row["val_seconds"]
    return {"train_samples_per_second": train_rate, "validation_samples_per_second": val_rate,
            "estimated_training_hours": epochs * (counts["train"] / train_rate + counts["val"] / val_rate) / 3600}


def validation_outputs(config: dict[str, Any], root: Path, manifest: Path,
                       checkpoint: Path, output: Path) -> dict[str, Any]:
    """Only validation: existing loss, metrics and plotting interfaces."""
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"] | {"pretrained": False, "min_depth": config["data"].get("min_depth", 0.1)}).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    loader = _loader(root, manifest, "val", config, False)
    loss, components, metrics = _epoch(model, loader, device, config["loss"], None, None, None, False)
    result = {"split": "val", "loss": loss, "components": components, "metrics": metrics}
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    records, _ = load_manifest(manifest, "val")
    dataset = MidAirDataset(root, records, config["data"])
    with torch.inference_mode():
        for index in range(min(3, len(dataset))):
            prediction = model(dataset[index]["rgb"].unsqueeze(0).to(device)).cpu().squeeze(0)
            visualize_sample(dataset, output / f"prediction_{index:02d}.png", index, prediction)
    plot_history(output.parent / "history.csv", output / "loss_curve.png")
    return result
