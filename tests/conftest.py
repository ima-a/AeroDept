from __future__ import annotations

from pathlib import Path
import pytest

from aerodepth_lite.cli import make_fixture
from aerodepth_lite.data import discover_midair, write_manifest


@pytest.fixture
def fixture_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "fixture"
    make_fixture(root, trajectories=6, frames=4)
    records, _ = discover_midair(root)
    manifest = root / "splits.json"
    write_manifest(records, manifest)
    return root, manifest


@pytest.fixture
def smoke_config() -> dict:
    return {
        "seed": 42,
        "data": {"image_size": [32, 48], "min_depth": 0.1, "max_depth": 1000.0, "depth_encoding": "midair_float16_bits", "depth_scale": 1.0, "workers": 0},
        "model": {"name": "aerodepth_tiny", "base_channels": 4},
        "train": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-3, "weight_decay": 1e-4, "amp": False, "max_train_batches": 1, "max_val_batches": 1},
        "loss": {"l1_weight": 1.0, "log_weight": 0.2, "gradient_weight": 0.1, "near_weight": 0.1, "near_threshold": 10.0},
        "scheduler": {"max_skip": 2, "decision_threshold": 0.9, "motion_weight": 0.25, "visual_weight": 0.3, "risk_weight": 0.3, "age_weight": 0.15, "motion_scale": 5.0, "visual_scale": 0.15, "near_threshold": 10.0, "critical_risk": 0.99, "extreme_motion": 0.99, "extreme_visual": 0.99, "visual_size": [18, 32]},
        "replay": {"policies": ["every", "fixed2", "fixed5", "adaptive"], "critical_near_threshold": 8.0, "critical_change_threshold": 2.0},
    }
