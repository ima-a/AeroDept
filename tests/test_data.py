from __future__ import annotations

import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import pytest

from aerodepth_lite.cli import _validate_hdf5_sync
from aerodepth_lite.data import MidAirDataset, assert_no_leakage, discover_midair, load_depth, load_manifest, split_trajectories, write_manifest


def test_path_parsing_and_pairing(fixture_dataset):
    root, _ = fixture_dataset
    records, report = discover_midair(root)
    assert len(records) == 24
    assert report["trajectory_count"] == 6
    assert records[0].rgb_path.endswith(".JPEG")
    assert records[0].depth_path.endswith(".PNG")


def test_deterministic_trajectory_split(fixture_dataset, tmp_path):
    root, _ = fixture_dataset
    records, _ = discover_midair(root)
    assert split_trajectories(records, 42) == split_trajectories(list(reversed(records)), 42)
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    write_manifest(records, one, 42)
    write_manifest(records, two, 42)
    assert one.read_bytes() == two.read_bytes()


def test_split_has_no_leakage(fixture_dataset):
    _, manifest = fixture_dataset
    payload = json.loads(manifest.read_text())
    assert_no_leakage(payload["splits"])
    all_ids = sum((ids for ids in payload["splits"].values()), [])
    assert len(all_ids) == len(set(all_ids)) == 6
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage({"train": ["x"], "val": ["x"], "test": ["z"]})


def test_invalid_depth_mask_and_alignment(fixture_dataset, smoke_config):
    root, manifest = fixture_dataset
    records, _ = load_manifest(manifest, "train")
    depth_path = root / records[0].depth_path
    raw = np.asarray(Image.open(depth_path)).copy()
    raw[0, 0] = 0
    Image.fromarray(raw).save(depth_path)
    sample = MidAirDataset(root, records, smoke_config["data"])[0]
    assert sample["rgb"].shape[-2:] == sample["depth"].shape[-2:] == sample["valid"].shape[-2:]
    assert not sample["valid"].all()
    Image.fromarray(np.ones((10, 10), dtype=np.uint16)).save(depth_path)
    with pytest.raises(ValueError, match="alignment mismatch"):
        MidAirDataset(root, records, smoke_config["data"])[0]


def test_hdf5_navigation_is_attached_when_available(fixture_dataset):
    root, _ = fixture_dataset
    with h5py.File(root / "environment_0" / "sensor_records.hdf5", "w") as handle:
        groundtruth = handle.create_group("trajectory_0000/groundtruth")
        groundtruth.create_dataset("velocity", data=np.array([[3.0, 4.0, 0.0]] * 4))
        groundtruth.create_dataset("angular_velocity", data=np.array([[0.1, 0.2, 0.3]] * 4))
    records, report = discover_midair(root)
    selected = [r for r in records if r.trajectory_id.endswith("trajectory_0000")]
    assert selected[0].navigation["vx"] == 3.0
    assert selected[0].navigation["wy"] == 0.2
    assert report["navigation_frames"] == 4


def test_hdf5_paths_and_navigation_resampling_are_validated(tmp_path):
    root = tmp_path / "MidAir"
    rgb_dir = root / "Kite_training" / "foggy" / "color_left" / "trajectory_2003"
    depth_dir = root / "Kite_training" / "foggy" / "depth" / "trajectory_2003"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    rgb_paths, depth_paths = [], []
    for frame in range(3):
        rgb_path = rgb_dir / f"{frame:06d}.JPEG"
        depth_path = depth_dir / f"{frame:06d}.PNG"
        Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(rgb_path)
        Image.fromarray(np.full((4, 6), 5.0, dtype=np.float16).view(np.uint16)).save(depth_path)
        rgb_paths.append(rgb_path.relative_to(root).as_posix())
        depth_paths.append(depth_path.relative_to(root).as_posix())
    with h5py.File(root / "Kite_training" / "foggy" / "sensor_records.hdf5", "w") as handle:
        group = handle.create_group("trajectory_2003")
        camera = group.create_group("camera_data")
        string_type = h5py.string_dtype(encoding="utf-8")
        color = camera.create_dataset("color_left", data=rgb_paths, dtype=string_type)
        depth = camera.create_dataset("depth", data=depth_paths, dtype=string_type)
        color.attrs["local_position"] = [0.0, 0.0, 0.0]
        depth.attrs["local_position"] = [0.0, 0.0, 0.0]
        for name in ("groundtruth/velocity", "imu/accelerometer", "imu/gyroscope"):
            dataset = group.create_dataset(name, data=np.arange(36, dtype=np.float64).reshape(12, 3))
            dataset.attrs["sampling_frequency"] = 100.0
            dataset.attrs["axis_ordering"] = "xyz"
    records, _ = discover_midair(root)
    result = _validate_hdf5_sync(root, records)
    assert result["available"]
    assert result["total_path_mismatches"] == 0
    assert result["maximum_navigation_resampling_error"] == 0.0
    assert result["validation_errors"] == []
    assert result["trajectories"][0]["navigation_values_checked"] == 27
    assert result["trajectories"][0]["camera_metadata"]["camera_data/color_left"]["local_position"] == [0.0, 0.0, 0.0]


def test_midair_float16_bit_decoding(tmp_path):
    expected = np.array([[1.5, 42.25]], dtype=np.float16)
    path = tmp_path / "depth.png"
    Image.fromarray(expected.view(np.uint16)).save(path)
    decoded = load_depth(path, encoding="midair_float16_bits")
    np.testing.assert_allclose(decoded, expected.astype(np.float32))


def test_full_layout_prefers_left_camera_and_normalizes_trajectory(tmp_path):
    root = tmp_path / "MidAir"
    for modality in ("color_left", "color_right", "color_down", "depth", "normals"):
        directory = root / "sunny" / modality / "trajectory_0000"
        directory.mkdir(parents=True)
        if modality.startswith("color"):
            Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(directory / "0000.JPEG")
        else:
            bits = np.full((4, 6), 5.0, dtype=np.float16).view(np.uint16)
            Image.fromarray(bits).save(directory / "0000.PNG")
    records, report = discover_midair(root)
    assert len(records) == 1
    assert records[0].rgb_path.startswith("sunny/color_left/")
    assert records[0].trajectory_id == "sunny/trajectory_0000"
    assert report["rgb_selection"] == "color_left"
