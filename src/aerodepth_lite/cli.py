from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
import shutil

import numpy as np
from PIL import Image
import torch

from .artifacts import generate_report, plot_history, visualize_sample
from .config import load_config
from .data import MidAirDataset, discover_midair, load_depth, load_manifest, write_manifest
from .model import build_model, load_checkpoint, model_stats
from .replay import replay_policy, save_replay_results
from .train import train_model


def inspect_dataset(root: Path, output: Path, config: dict, scan_limit: int | None = None) -> dict:
    records, report = discover_midair(root)
    invalid = []
    depth_min, depth_max, valid_values = float("inf"), float("-inf"), 0
    selected = records if scan_limit is None else records[:scan_limit]
    for record in selected:
        try:
            rgb = Image.open(root / record.rgb_path)
            depth = load_depth(root / record.depth_path, float(config["data"].get("depth_scale", 1.0)), str(config["data"].get("depth_encoding", "midair_float16_bits")))
            if rgb.size[::-1] != depth.shape[:2]:
                invalid.append({"rgb": record.rgb_path, "reason": f"shape mismatch {rgb.size[::-1]} vs {depth.shape[:2]}"})
                continue
            valid = depth[np.isfinite(depth) & (depth >= config["data"]["min_depth"]) & (depth <= config["data"]["max_depth"])]
            if valid.size:
                depth_min, depth_max = min(depth_min, float(valid.min())), max(depth_max, float(valid.max()))
                valid_values += int(valid.size)
        except Exception as error:
            invalid.append({"rgb": record.rgb_path, "reason": f"{type(error).__name__}: {error}"})
    report.update({
        "scanned_pairs": len(selected), "invalid_or_misaligned_count": len(invalid),
        "invalid_examples": invalid[:20], "valid_depth_pixels": valid_values,
        "observed_metric_depth_min": depth_min if valid_values else None,
        "observed_metric_depth_max": depth_max if valid_values else None,
        "depth_encoding_used": config["data"].get("depth_encoding", "midair_float16_bits"),
        "depth_scale_used": config["data"].get("depth_scale", 1.0),
        "documented_depth_semantics": "Euclidean distance from camera focal point in metres, encoded as float16 bit patterns in 16-bit PNG.",
        "depth_semantics_verified_on_local_files": False,
        "verification_blocker": "Documentation is verified, but actual MidAir files are required for local metadata/value/visual confirmation.",
    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dataset = MidAirDataset(root, records, config["data"])
    visualize_sample(dataset, output / "synchronized_sample.png")
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

    fixture = sub.add_parser("make-fixture", help="create a clearly labelled synthetic test fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--trajectories", type=int, default=6)
    fixture.add_argument("--frames", type=int, default=5)

    inspect = sub.add_parser("inspect", help="validate a MidAir extraction and generate a synchronized sample")
    inspect.add_argument("--dataset-root", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--config", default="configs/preliminary.yaml")
    inspect.add_argument("--scan-limit", type=int)

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
    if args.command == "make-fixture":
        make_fixture(args.output, args.trajectories, args.frames)
        print(json.dumps({"fixture": str(args.output), "synthetic": True}))
    elif args.command == "inspect":
        result = inspect_dataset(args.dataset_root, args.output, _config_from_args(args), args.scan_limit)
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
        model = build_model(config["model"] | {"min_depth": config["data"].get("min_depth", 0.1)})
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
        model = build_model(config["model"] | {"min_depth": config["data"].get("min_depth", 0.1)}).to(device)
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
