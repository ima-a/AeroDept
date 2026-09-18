from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from aerodepth_lite.cli import _validate_hdf5_sync
from aerodepth_lite.data import discover_midair, load_depth


@pytest.mark.integration
def test_real_midair_discovery_depth_and_navigation():
    configured_root = os.environ.get("MIDAIR_ROOT")
    if not configured_root:
        pytest.skip("set MIDAIR_ROOT to run the real MidAir integration check")
    root = Path(configured_root).expanduser().resolve()
    records, report = discover_midair(root)
    assert report["paired_frames"] == report["rgb_files"] == report["depth_files"]
    assert report["missing_depth_count"] == 0
    assert report["navigation_frames"] == len(records)

    rng = np.random.default_rng(42)
    selected = rng.choice(len(records), size=min(10, len(records)), replace=False)
    for index in selected:
        record = records[int(index)]
        with Image.open(root / record.rgb_path) as rgb:
            rgb_shape = rgb.size[::-1]
        depth = load_depth(root / record.depth_path)
        assert depth.shape == rgb_shape
        assert np.isfinite(depth).any()
        assert (depth[np.isfinite(depth)] > 0).all()

    hdf5 = _validate_hdf5_sync(root, records)
    assert hdf5["available"]
    assert hdf5["unmatched_trajectories"] == []
    assert hdf5["total_path_mismatches"] == 0
    assert hdf5["maximum_navigation_resampling_error"] == 0.0
    assert hdf5["validation_errors"] == []
