from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from .artifacts import generate_report, plot_history, visualize_sample
from .config import load_config
from .data import MidAirDataset, discover_midair, load_depth, load_manifest, write_manifest
from .model import build_model, load_checkpoint, model_stats
from .replay import replay_policy, save_replay_results
from .train import train_model


def _decode_hdf5_path(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_value(value: object) -> object:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _camera_path_matches(expected: str, actual: str) -> bool:
    expected_parts = Path(expected).parts
    actual_parts = Path(actual).parts
    length = min(len(expected_parts), len(actual_parts))
    return length >= 3 and expected_parts[-length:] == actual_parts[-length:]


def _validate_hdf5_sync(root: Path, records: list) -> dict:
    """Cross-check discovered frames and resampled navigation against MidAir HDF5."""
    try:
        import h5py
    except ImportError:
        return {"available": False, "reason": "h5py is not installed"}
    by_trajectory: dict[str, list] = {}
    for record in records:
        by_trajectory.setdefault(record.trajectory_id, []).append(record)
    details = []
    total_path_mismatches = 0
    maximum_navigation_error = 0.0
    matched_trajectories = set()
    validation_errors = []
    field_sources = {
        "groundtruth/velocity": ("vx", "vy", "vz"),
        "imu/accelerometer": ("ax", "ay", "az"),
        "imu/gyroscope": ("wx", "wy", "wz"),
    }
    for h5_path in sorted((*root.rglob("*.hdf5"), *root.rglob("*.h5"))):
        with h5py.File(h5_path, "r") as handle:
            for trajectory, trajectory_records in by_trajectory.items():
                parent = h5_path.parent.relative_to(root).as_posix()
                if parent != "." and not trajectory.startswith(parent + "/"):
                    continue
                group_name = trajectory.split("/")[-1]
                if group_name not in handle:
                    continue
                matched_trajectories.add(trajectory)
                group = handle[group_name]
                required_datasets = ["camera_data/color_left", "camera_data/depth", *field_sources]
                missing_datasets = [name for name in required_datasets if name not in group]
                if missing_datasets:
                    validation_errors.append({
                        "trajectory": trajectory,
                        "hdf5_path": h5_path.relative_to(root).as_posix(),
                        "missing_datasets": missing_datasets,
                    })
                    continue
                rgb_paths = [_decode_hdf5_path(value) for value in group["camera_data/color_left"]]
                depth_paths = [_decode_hdf5_path(value) for value in group["camera_data/depth"]]
                rgb_matches = sum(_camera_path_matches(expected, record.rgb_path) for expected, record in zip(rgb_paths, trajectory_records))
                depth_matches = sum(_camera_path_matches(expected, record.depth_path) for expected, record in zip(depth_paths, trajectory_records))
                path_mismatches = (
                    abs(len(rgb_paths) - len(trajectory_records))
                    + abs(len(depth_paths) - len(trajectory_records))
                    + len(trajectory_records) - rgb_matches
                    + len(trajectory_records) - depth_matches
                )
                total_path_mismatches += path_mismatches
                checked_indices = sorted({0, len(trajectory_records) // 2, len(trajectory_records) - 1})
                navigation_checks = 0
                trajectory_max_error = 0.0
                for frame_ordinal in checked_indices:
                    record = trajectory_records[frame_ordinal]
                    for dataset_name, keys in field_sources.items():
                        dataset = group[dataset_name]
                        frequency = float(dataset.attrs["sampling_frequency"])
                        sensor_index = int(frame_ordinal * frequency / 25.0)
                        if sensor_index >= dataset.shape[0]:
                            validation_errors.append({"trajectory": trajectory, "dataset": dataset_name, "reason": "navigation shorter than camera sequence"})
                            continue
                        expected = np.asarray(dataset[sensor_index]).reshape(-1)
                        for key, expected_value in zip(keys, expected[:3]):
                            error = abs(float(record.navigation[key]) - float(expected_value))
                            trajectory_max_error = max(trajectory_max_error, error)
                            maximum_navigation_error = max(maximum_navigation_error, error)
                            navigation_checks += 1
                details.append({
                    "trajectory": trajectory,
                    "hdf5_path": h5_path.relative_to(root).as_posix(),
                    "record_count": len(trajectory_records),
                    "hdf5_rgb_count": len(rgb_paths),
                    "hdf5_depth_count": len(depth_paths),
                    "rgb_order_matches": rgb_matches,
                    "depth_order_matches": depth_matches,
                    "path_mismatches": path_mismatches,
                    "navigation_values_checked": navigation_checks,
                    "maximum_navigation_resampling_error": trajectory_max_error,
                    "camera_metadata": {
                        name: {key: _json_value(value) for key, value in group[name].attrs.items()}
                        for name in ("camera_data/color_left", "camera_data/depth")
                    },
                    "navigation_metadata": {
                        name: {key: _json_value(value) for key, value in group[name].attrs.items()}
                        for name in field_sources
                    },
                })
    return {
        "available": bool(details),
        "matched_trajectory_count": len(matched_trajectories),
        "unmatched_trajectories": sorted(set(by_trajectory) - matched_trajectories),
        "total_path_mismatches": total_path_mismatches,
        "maximum_navigation_resampling_error": maximum_navigation_error,
        "validation_errors": validation_errors,
        "trajectories": details,
    }


def inspect_dataset(
    root: Path,
    output: Path,
    config: dict,
    scan_limit: int | None = None,
    sample_count: int = 10,
    stats_sample_count: int = 32,
    seed: int = 42,
) -> dict:
    records, report = discover_midair(root)
    invalid = []
    depth_min, depth_max, valid_values = float("inf"), float("-inf"), 0
    selected = records if scan_limit is None else records[:scan_limit]
    rng = np.random.default_rng(seed)
    stats_indices = sorted(rng.choice(len(selected), size=min(stats_sample_count, len(selected)), replace=False).tolist())
    stats_index_set = set(stats_indices)
    sampled_pixels = []
    sampled_frames = []
    total_pixels = 0
    invalid_pixels = 0
    below_minimum_pixels = 0
    above_maximum_pixels = 0
    nonfinite_pixels = 0
    raw_finite_min, raw_finite_max = float("inf"), float("-inf")
    for selected_index, record in enumerate(selected):
        try:
            with Image.open(root / record.rgb_path) as rgb_image:
                rgb_size = rgb_image.size[::-1]
            depth = load_depth(root / record.depth_path, float(config["data"].get("depth_scale", 1.0)), str(config["data"].get("depth_encoding", "midair_float16_bits")))
            if rgb_size != depth.shape[:2]:
                invalid.append({"rgb": record.rgb_path, "reason": f"shape mismatch {rgb_size} vs {depth.shape[:2]}"})
                continue
            valid_mask = np.isfinite(depth) & (depth >= config["data"]["min_depth"]) & (depth <= config["data"]["max_depth"])
            valid = depth[valid_mask]
            finite_mask = np.isfinite(depth)
            finite = depth[finite_mask]
            below_minimum = int(np.count_nonzero(finite_mask & (depth < config["data"]["min_depth"])))
            above_maximum = int(np.count_nonzero(finite_mask & (depth > config["data"]["max_depth"])))
            nonfinite = int(depth.size - finite.size)
            total_pixels += int(depth.size)
            invalid_pixels += int(depth.size - valid.size)
            below_minimum_pixels += below_minimum
            above_maximum_pixels += above_maximum
            nonfinite_pixels += nonfinite
            if finite.size:
                raw_finite_min = min(raw_finite_min, float(finite.min()))
                raw_finite_max = max(raw_finite_max, float(finite.max()))
            if valid.size:
                depth_min, depth_max = min(depth_min, float(valid.min())), max(depth_max, float(valid.max()))
                valid_values += int(valid.size)
            if selected_index in stats_index_set:
                frame_invalid = int(depth.size - valid.size)
                sampled_frames.append({
                    "trajectory": record.trajectory_id,
                    "frame_index": record.frame_index,
                    "rgb_path": record.rgb_path,
                    "depth_path": record.depth_path,
                    "minimum": float(valid.min()) if valid.size else None,
                    "median": float(np.median(valid)) if valid.size else None,
                    "p05": float(np.percentile(valid, 5)) if valid.size else None,
                    "p95": float(np.percentile(valid, 95)) if valid.size else None,
                    "maximum": float(valid.max()) if valid.size else None,
                    "raw_finite_minimum": float(finite.min()) if finite.size else None,
                    "raw_finite_maximum": float(finite.max()) if finite.size else None,
                    "below_minimum_pixel_percentage": 100.0 * below_minimum / depth.size,
                    "above_maximum_pixel_percentage": 100.0 * above_maximum / depth.size,
                    "invalid_pixel_percentage": 100.0 * frame_invalid / depth.size,
                    "nonfinite_pixel_percentage": 100.0 * nonfinite / depth.size,
                })
                if valid.size:
                    stride = max(1, valid.size // 200_000)
                    sampled_pixels.append(valid[::stride][:200_000])
        except Exception as error:
            invalid.append({"rgb": record.rgb_path, "reason": f"{type(error).__name__}: {error}"})
    percentile_values = np.concatenate(sampled_pixels) if sampled_pixels else np.array([], dtype=np.float32)
    percentiles = {}
    if percentile_values.size:
        for percentile in (0, 1, 5, 25, 50, 75, 95, 99, 100):
            percentiles[f"p{percentile:02d}"] = float(np.percentile(percentile_values, percentile))
    trajectory_counts = Counter(record.trajectory_id for record in records)
    navigation_keys = Counter(key for record in records for key in record.navigation)
    report.update({
        "scanned_pairs": len(selected), "invalid_or_misaligned_count": len(invalid),
        "invalid_examples": invalid[:20], "valid_depth_pixels": valid_values,
        "observed_metric_depth_min": depth_min if valid_values else None,
        "observed_metric_depth_max": depth_max if valid_values else None,
        "invalid_depth_pixels": invalid_pixels,
        "total_depth_pixels": total_pixels,
        "invalid_depth_percentage": 100.0 * invalid_pixels / total_pixels if total_pixels else None,
        "below_minimum_depth_pixels": below_minimum_pixels,
        "above_maximum_depth_pixels": above_maximum_pixels,
        "nonfinite_depth_pixels": nonfinite_pixels,
        "raw_finite_depth_min": raw_finite_min if raw_finite_min != float("inf") else None,
        "raw_finite_depth_max": raw_finite_max if raw_finite_max != float("-inf") else None,
        "trajectory_frame_counts": dict(trajectory_counts),
        "navigation_field_coverage": dict(navigation_keys),
        "depth_encoding_used": config["data"].get("depth_encoding", "midair_float16_bits"),
        "depth_scale_used": config["data"].get("depth_scale", 1.0),
        "local_depth_verification": "Local float16 decoding, alignment, and HDF5 cross-check completed; review the saved visualizations for semantic plausibility.",
        "depth_note": "Numeric and synchronization evidence is local; Euclidean camera-range semantics come from official MidAir documentation and require visual confirmation.",
        "documented_depth_semantics": "Euclidean distance from camera focal point in metres, encoded as float16 bit patterns in 16-bit PNG.",
        "depth_semantics_verified_on_local_files": None,
        "depth_semantics_status": "Decoded local evidence generated; spatial plausibility requires review of the saved visualizations.",
        "random_depth_statistics": {
            "seed": seed,
            "frame_count": len(sampled_frames),
            "sampled_pixel_count": int(percentile_values.size),
            "valid_pixel_percentiles_metres": percentiles,
            "frames": sampled_frames,
        },
        "hdf5_validation": _validate_hdf5_sync(root, records),
    })
    output.mkdir(parents=True, exist_ok=True)
    dataset = MidAirDataset(root, records, config["data"])
    visualization_indices = sorted(rng.choice(len(records), size=min(sample_count, len(records)), replace=False).tolist())
    visualization_dir = output / "samples"
    visualization_paths = []
    for number, record_index in enumerate(visualization_indices):
        path = visualization_dir / f"sample_{number:02d}.png"
        visualize_sample(dataset, path, index=record_index)
        visualization_paths.append(path.relative_to(output).as_posix())
    report["visualizations"] = visualization_paths
    (output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def make_fixture(output: Path, trajectories: int = 6, frames: int = 5) -> None:
    """Create a deterministic synthetic software fixture, never research results."""
    rng = np.random.default_rng(42)
    for trajectory in range(trajectories):
        rgb_dir = output / f"environment_{trajectory % 2}" / f"trajectory_{trajectory:04d}" / "rgb"
        depth_dir = output / f"environment_{trajectory % 2}" / f"trajectory_{trajectory:04d}" / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(frames):
            y, x = np.mgrid[:72, :128]
            rgb = np.stack(((x + frame * 8) % 256, (y * 3 + trajectory * 9) % 256, ((x + y) // 2 + rng.integers(0, 4, (72, 128))) % 256), axis=-1).astype(np.uint8)
            depth_m = (5.0 + trajectory + x / 25.0 + y / 40.0 - frame * 0.1).astype(np.float32)
            depth_raw = depth_m.astype(np.float16).view(np.uint16)
            Image.fromarray(rgb).save(rgb_dir / f"frame_{frame:06d}.JPEG")
            Image.fromarray(depth_raw).save(depth_dir / f"depth_{frame:06d}.PNG")
    (output / "SYNTHETIC_FIXTURE.txt").write_text("Synthetic deterministic fixture for software tests only. This is not MidAir data or experimental evidence.\n", encoding="utf-8")


def _config_from_args(args: argparse.Namespace) -> dict:
    return load_config(getattr(args, "config", "configs/smoke.yaml"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aerodepth")
    sub = parser.add_subparsers(dest="command", required=True)

    inference = sub.add_parser("infer", help="offline inference with published trained weights; no dataset required")
    inference.add_argument("--weights", type=Path, default=Path("models/mobilenet_v3_small_epoch4.state_dict.pt"))
    inference.add_argument("--metadata", type=Path, default=Path("models/mobilenet_v3_small_epoch4.json"))
    inference.add_argument("--image", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    fixture = sub.add_parser("make-fixture", help="create a clearly labelled synthetic test fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--trajectories", type=int, default=6)
    fixture.add_argument("--frames", type=int, default=5)

    inspect = sub.add_parser("inspect", help="validate a MidAir extraction and generate a synchronized sample")
    inspect.add_argument("--dataset-root", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--config", default="configs/preliminary.yaml")
    inspect.add_argument("--scan-limit", type=int)
    inspect.add_argument("--sample-count", type=int, default=10)
    inspect.add_argument("--stats-sample-count", type=int, default=32)
    inspect.add_argument("--seed", type=int, default=42)

    split = sub.add_parser("split", help="write deterministic trajectory-disjoint manifests")
    split.add_argument("--dataset-root", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)

    visual = sub.add_parser("visualize", help="visualize synchronized RGB and depth")
    visual.add_argument("--dataset-root", type=Path, required=True)
    visual.add_argument("--manifest", type=Path, required=True)
    visual.add_argument("--split", choices=("train", "val", "test"), default="val")
    visual.add_argument("--output", type=Path, required=True)
    visual.add_argument("--config", default="configs/preliminary.yaml")

    train = sub.add_parser("train", help="train and validate the lightweight depth model")
    train.add_argument("--config", required=True)
    train.add_argument("--dataset-root", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", type=Path)

    replay = sub.add_parser("replay", help="replay complete validation trajectories")
    replay.add_argument("--config", required=True)
    replay.add_argument("--dataset-root", type=Path, required=True)
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--checkpoint", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--split", choices=("val", "test"), default="val")
    replay.add_argument("--policies", nargs="+", choices=("every", "fixed2", "fixed5", "adaptive"))

    report = sub.add_parser("report", help="generate figures only from saved results")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "infer":
        from .inference import infer_image
        print(json.dumps(infer_image(args.weights, args.metadata, args.image, args.output, args.device), indent=2))
    elif args.command == "make-fixture":
        make_fixture(args.output, args.trajectories, args.frames)
        print(json.dumps({"fixture": str(args.output), "synthetic": True}))
    elif args.command == "inspect":
        result = inspect_dataset(args.dataset_root, args.output, _config_from_args(args), args.scan_limit, args.sample_count, args.stats_sample_count, args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "split":
        records, _ = discover_midair(args.dataset_root)
        print(json.dumps(write_manifest(records, args.output, args.seed), indent=2))
    elif args.command == "visualize":
        config = _config_from_args(args)
        records, _ = load_manifest(args.manifest, args.split)
        visualize_sample(MidAirDataset(args.dataset_root, records, config["data"]), args.output)
        print(args.output)
    elif args.command == "train":
        config = _config_from_args(args)
        summary = train_model(config, args.dataset_root, args.manifest, args.output, args.resume)
        plot_history(args.output / "history.csv", args.output / "loss_curve.png")
        model = build_model(config["model"] | {"min_depth": config["data"].get("min_depth", 0.1), "pretrained": False})
        load_checkpoint(args.output / "best.pt", model)
        records, _ = load_manifest(args.manifest, "val")
        dataset = MidAirDataset(args.dataset_root, records, config["data"])
        model.eval()
        with torch.inference_mode():
            prediction = model(dataset[0]["rgb"].unsqueeze(0)).squeeze(0)
        visualize_sample(dataset, args.output / "prediction_comparison.png", prediction=prediction)
        print(json.dumps(summary, indent=2, allow_nan=True))
    elif args.command == "replay":
        config = _config_from_args(args)
        records, _ = load_manifest(args.manifest, args.split)
        dataset = MidAirDataset(args.dataset_root, records, config["data"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(config["model"] | {"min_depth": config["data"].get("min_depth", 0.1), "pretrained": False}).to(device)
        load_checkpoint(args.checkpoint, model, map_location=device)
        args.output.mkdir(parents=True, exist_ok=True)
        policies = args.policies or config["replay"].get("policies", ["every", "fixed2", "fixed5", "adaptive"])
        results = [replay_policy(model, dataset, policy, config["scheduler"], config["replay"], device, args.output / f"{policy}_frames.csv") for policy in policies]
        save_replay_results(results, args.output / "results.json")
        print(json.dumps({"results": results}, indent=2, allow_nan=True))
    elif args.command == "report":
        print(json.dumps({"artifacts": generate_report(args.input, args.output)}, indent=2))


if __name__ == "__main__":
    main()
