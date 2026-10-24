"""Local MidAir preparation and measured training, without final-test inference.

Use the existing trainer for all optimization and resume behavior. Dataset inputs
are read-only; generated artifacts always go under the selected project output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from time import perf_counter

import h5py
import numpy as np
import torch

from aerodepth_lite.artifacts import plot_history, visualize_sample
from aerodepth_lite.config import load_config
from aerodepth_lite.data import MidAirDataset, load_depth, load_manifest
from aerodepth_lite.kaggle import (load_or_create_manifest, path_group,
                                  validate_dataset, validation_outputs)
from aerodepth_lite.model import build_model, load_checkpoint, model_stats
from aerodepth_lite.repro import run_metadata, sha256_file
from aerodepth_lite.train import train_model


def write_json(path, value):
    # Near-field metrics are undefined when the sampled batch has no near pixels.
    # Preserve that distinction as JSON null, never as a fabricated zero.
    def finite_json(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: finite_json(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(v) for v in item]
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(args):
    config = load_config(args.config)
    previous = json.loads(args.verification_report.read_text())
    if Path(previous["source_root"]).resolve() != args.dataset_root.resolve():
        raise ValueError("Verification report refers to a different dataset root")
    if previous["phase"] != "complete" or any(previous[key] for key in (
            "missing_files", "corrupt_archives", "extracted_file_mismatches",
            "rgb_depth_count_mismatches", "hdf5_path_mismatches", "hdf5_errors")):
        raise ValueError("Previous archive verification was not successful")
    expected = {t["combination"]: t["paired_frames"] for t in previous["trajectories"]}
    print("Checking current inventory, image headers and HDF5 alignment...", flush=True)
    records, report = validate_dataset(args.dataset_root, expected_trajectories=list(expected))
    if dict(Counter(r.trajectory_id for r in records)) != expected:
        raise ValueError("Per-trajectory counts differ from the verified extraction")
    manifest_path = args.output / "manifest.json"
    payload = load_or_create_manifest(records, manifest_path, existing=args.existing_manifest)
    write_json(args.output / "dataset_verification.json", report | {
        "archive_verification_reused": str(args.verification_report.resolve()),
        "archive_report_sha256": sha256_file(args.verification_report),
        "archive_crc_rerun": False,
        "dataset_identity_note": "Path inventory fingerprint plus all HDF5 hashes; prior archive CRC/size evidence reused, not a new full image-content audit",
        "hdf5_sha256": {str(p.relative_to(args.dataset_root)): sha256_file(p)
                         for p in sorted(args.dataset_root.rglob("*.hdf5"))},
        "manifest_sha256": sha256_file(manifest_path),
        "splits": payload["splits"], "frame_counts": payload["frame_counts"],
    })
    # Only train/validation labels are decoded for numerical range review.
    samples = []
    for split in ("train", "val"):
        for trajectory in payload["splits"][split]:
            selected = [r for r in records if r.trajectory_id == trajectory]
            for index in sorted({0, len(selected) // 2, len(selected) - 1}):
                record = selected[index]
                depth = load_depth(args.dataset_root / record.depth_path)
                finite = np.isfinite(depth)
                low = finite & (depth < config["data"]["min_depth"])
                high = finite & (depth > config["data"]["max_depth"])
                valid = finite & ~low & ~high
                samples.append({"split": split, "trajectory": trajectory, "frame": record.frame_index,
                    "pixels": depth.size, "nonfinite": int((~finite).sum()),
                    "below_minimum": int(low.sum()), "above_maximum": int(high.sum()),
                    "valid": int(valid.sum()),
                    "valid_percentiles": np.percentile(depth[valid], [0, 5, 50, 95, 100]).tolist() if valid.any() else [],
                    "near_pixels": int((valid & (depth < config["loss"]["near_threshold"])).sum())})
    write_json(args.output / "depth_range_review.json", {
        "splits": ["train", "val"], "selection": "first, middle, last per train/validation trajectory",
        "valid_range_metres": [config["data"]["min_depth"], config["data"]["max_depth"]],
        "samples": samples, "range_changed": False,
        "loss": config["loss"],
        "interpretation": "Retain the existing 0.1–1000m mask and L1/log/gradient loss; report masked sky/over-range pixels explicitly. This sample is range review, not full-split statistics or evidence of model quality.",
    })
    # Reconcile website base-path IDs with actual release HDF5 IDs, including
    # the pilot foggy paths: compare positional streams with the sunny variant.
    mapping = []
    with h5py.File(args.dataset_root / "Kite_training/sunny/sensor_records.hdf5", "r") as sunny:
        for trajectory in expected:
            parent, name = trajectory.rsplit("/", 1)
            if not trajectory.startswith("Kite_training/"):
                continue
            base = path_group(trajectory).split("/")[-1]
            with h5py.File(args.dataset_root / parent / "sensor_records.hdf5", "r") as handle:
                a, b = np.asarray(sunny[base + "/groundtruth/position"]), np.asarray(handle[name + "/groundtruth/position"])
                mapping.append({"website_base_id": base, "actual_trajectory": trajectory,
                    "sunny_position_shape": list(a.shape), "position_shape": list(b.shape),
                    "position_stream_identical": bool(np.array_equal(a, b)),
                    "position_stream_max_absolute_difference": float(np.max(np.abs(a-b))) if a.shape == b.shape else None})
    write_json(args.output / "path_id_evidence.json", mapping)
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256_file(manifest_path),
                      "splits": payload["splits"], "counts": payload["frame_counts"]}), flush=True)


class ResourceMonitor:
    """Sample process-tree RSS and NVIDIA telemetry while the external loader runs."""
    def __init__(self, output):
        self.output, self.stop = output, threading.Event()
        self.peak_rss = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with (self.output / "resources.jsonl").open("w") as handle:
            while not self.stop.is_set():
                processes = {}
                for p in Path("/proc").glob("[0-9]*/status"):
                    try:
                        fields = dict(line.split(":", 1) for line in p.read_text().splitlines() if ":" in line)
                        processes[int(p.parent.name)] = (int(fields["PPid"]), int(fields.get("VmRSS", "0 kB").split()[0]) * 1024)
                    except (OSError, ValueError, KeyError):
                        continue
                tree = {os.getpid()}
                for _ in range(4):
                    tree |= {pid for pid, (parent, _) in processes.items() if parent in tree}
                rss = sum(processes.get(pid, (0, 0))[1] for pid in tree)
                self.peak_rss = max(self.peak_rss, rss)
                try:
                    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm,clocks_event_reasons.sw_thermal_slowdown", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL).strip()
                except subprocess.SubprocessError:
                    gpu = None
                handle.write(json.dumps({"elapsed_seconds": perf_counter() - self.started,
                    "process_tree_rss_bytes": rss, "gpu_csv": gpu,
                    "gpu_fields": "temperature_C,utilization_percent,memory_MiB,power_W,SM_MHz,software_thermal_slowdown"}) + "\n")
                handle.flush()
                self.stop.wait(10)

    def __enter__(self):
        self.started = perf_counter()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def sample_predictions(config, args, checkpoint, output):
    model = build_model(config["model"] | {"pretrained": False, "min_depth": config["data"].get("min_depth", 0.1)}).cuda().eval()
    load_checkpoint(checkpoint, model, map_location="cuda")
    with torch.inference_mode():
        for split in ("train", "val"):
            records, _ = load_manifest(args.manifest, split)
            dataset = MidAirDataset(args.dataset_root, records, config["data"])
            for j, index in enumerate((0, len(dataset) // 2)):
                prediction = model(dataset[index]["rgb"].unsqueeze(0).cuda()).cpu().squeeze(0)
                visualize_sample(dataset, output / f"{split}_prediction_{j:02}.png", index, prediction)


def validate_benchmark_gate(gate, config, manifest_sha256, max_hours=12.0):
    estimate = gate["estimated_50_epoch_hours_with_20_percent_allowance"]
    if (not math.isfinite(estimate) or estimate <= 0 or not math.isfinite(max_hours) or max_hours <= 0 or
            estimate > max_hours or gate["manifest_sha256"] != manifest_sha256 or gate["config"] != config):
        raise ValueError(f"Benchmark exceeds {max_hours:g} hours or does not match the main configuration/manifest")


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use a bounded CPU sample separately; prolonged training is disabled")
    torch.ones(4, device="cuda").sum().item()
    config = load_config(args.config)
    original_config = deepcopy(config)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists() and not args.resume:
        raise ValueError("Run already contains a checkpoint; choose a distinct output or explicitly resume")
    if args.mode == "smoke":
        config["train"].update(max_train_batches=3, max_val_batches=3, stop_after_epochs=1)
    elif args.mode == "benchmark":
        config["train"]["stop_after_epochs"] = 1
    elif args.mode == "main":
        gate = json.loads(args.benchmark.read_text())
        max_hours = getattr(args, "max_hours", 12.0)
        validate_benchmark_gate(gate, config, sha256_file(args.manifest), max_hours)
        if not config.get("review", {}).get("depth_range_approved"):
            raise ValueError("Review training/validation depth evidence first")
        if not args.resume and list(args.output.glob("*.pt")):
            raise ValueError("A fresh final run must not contain smoke/benchmark weights")
        if shutil.disk_usage(args.output).free < 2 * 1024**3:
            raise ValueError("Less than 2GiB free for run artifacts")
        write_json(args.output / "launch_decision.json", {
            "benchmark": str(args.benchmark.resolve()), "benchmark_sha256": sha256_file(args.benchmark),
            "estimated_hours_with_allowance": gate["estimated_50_epoch_hours_with_20_percent_allowance"],
            "authorized_max_hours": max_hours, "default_limit_hours": 12,
            "duration_override": max_hours > 12,
            "fresh_seeded_pretrained_start": args.resume is None,
            "resume": str(args.resume.resolve()) if args.resume else None})
    monitor = ResourceMonitor(args.output)
    started = perf_counter()
    with monitor:
        summary = train_model(config, args.dataset_root, args.manifest, args.output, args.resume)
    wall_seconds = perf_counter() - started
    plot_history(args.output / "history.csv", args.output / "loss_curve.png")
    row = summary["final"]
    manifest = json.loads(args.manifest.read_text())
    measurement = {"mode": args.mode, "wall_seconds_including_setup_and_checkpoints": wall_seconds,
        "train_samples_per_second": row["train_samples"] / row["train_seconds"],
        "validation_samples_per_second": row["val_samples"] / row["val_seconds"],
        "peak_process_tree_rss_bytes_sampled": monitor.peak_rss,
        "peak_cuda_allocated_mb": row["peak_cuda_allocated_mb"],
        "peak_cuda_reserved_mb": row["peak_cuda_reserved_mb"],
        "manifest_sha256": sha256_file(args.manifest), "config": original_config,
        "dataset_root": str(args.dataset_root.resolve()),
        "epoch": row, "software": run_metadata(config, args.manifest)}
    if args.mode == "benchmark":
        if row["train_samples"] != manifest["frame_counts"]["train"] or row["val_samples"] != manifest["frame_counts"]["val"]:
            raise ValueError("Benchmark did not cover the complete train and validation splits")
        measurement["estimated_50_epoch_hours"] = wall_seconds * 50 / 3600
        measurement["estimated_50_epoch_hours_with_20_percent_allowance"] = wall_seconds * 50 * 1.2 / 3600
    write_json(args.output / "measurement.json", measurement)
    if args.mode == "smoke" and args.resume:
        checkpoint = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 1 and checkpoint["extra"]["global_step"] == 6
        assert len(checkpoint["extra"]["history"]) == 2 and checkpoint["extra"]["scheduler"]["T_max"] == 50
        write_json(args.output / "resume_verified.json", {"epochs": 2, "global_step": 6,
            "scheduler_horizon": 50, "optimizer_state_present": bool(checkpoint["optimizer"]["state"]),
            "scaler_state_present": bool(checkpoint["extra"]["scaler"]),
            "rng_states": sorted(checkpoint["extra"]["rng"]), "manifest_sha256": checkpoint["extra"]["manifest_sha256"]})
    sample_predictions(config, args, args.output / "best.pt", args.output)
    if args.mode == "main":
        validation_outputs(config, args.dataset_root, args.manifest, args.output / "best.pt", args.output / "validation")
    print(json.dumps(measurement), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "benchmark", "main"))
    parser.add_argument("--config", type=Path, default=Path("configs/mobilenet_local.yaml"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--existing-manifest", type=Path)
    parser.add_argument("--verification-report", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-hours", type=float, default=12.0,
                        help="main-run duration limit; raise only with explicit user authorization")
    parser.add_argument("--detach", action="store_true", help="launch a new session with sleep inhibition and a saved log")
    args = parser.parse_args()
    if args.detach:
        args.output.mkdir(parents=True, exist_ok=True)
        session_path = args.output / "session.json"
        if session_path.exists():
            prior_pid = json.loads(session_path.read_text())["wrapper_pid"]
            try:
                os.kill(prior_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Prior session PID {prior_pid} is still alive")
        command = ["systemd-inhibit", "--what=sleep", "--who=AeroDepth-Lite",
                   "--why=Local depth training", "--mode=block", sys.executable,
                   str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != "--detach"]]
        log = args.output / "training.log"
        with log.open("a") as handle:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        session = {"wrapper_pid": process.pid, "session_id": process.pid,
                   "started_utc": datetime.now(timezone.utc).isoformat(),
                   "command": command, "cwd": str(Path.cwd()), "log": str(log.resolve()),
                   "environment": {k: os.environ.get(k) for k in ("TORCH_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
        write_json(session_path, session)
        print(json.dumps(session), flush=True)
        return
    if args.mode == "prepare":
        if not args.verification_report:
            parser.error("prepare requires --verification-report")
        prepare(args)
    else:
        if not args.manifest or (args.mode == "main" and not args.benchmark):
            parser.error("training requires --manifest; main also requires --benchmark")
        run(args)


if __name__ == "__main__":
    main()
