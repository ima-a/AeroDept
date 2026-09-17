from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import random
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


RGB_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEPTH_EXTENSIONS = {".png", ".tif", ".tiff", ".npy"}
TRAJECTORY_PATTERN = re.compile(r"(?:trajectory|traj|sequence|seq)[_-]?(\d+)", re.I)


@dataclass(frozen=True)
class FrameRecord:
    trajectory_id: str
    frame_index: int
    rgb_path: str
    depth_path: str
    timestamp: float | None = None
    navigation: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path)))


def _trajectory_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    for part in reversed(relative.parts[:-1]):
        if TRAJECTORY_PATTERN.search(part):
            modality_parts = {"rgb", "color", "color_left", "color_right", "color_down", "left", "right", "down", "depth", "depth_left"}
            prefix = "/".join(p for p in relative.parts[: relative.parts.index(part) + 1] if p.lower() not in modality_parts)
            return prefix
    # Common fixture/layout: root/trajectory/RGB/file or root/RGB/trajectory/file.
    if len(relative.parts) >= 3:
        ignored = {"rgb", "color", "color_left", "left", "depth", "depth_left"}
        candidates = [p for p in relative.parts[:-1] if p.lower() not in ignored]
        if candidates:
            return "/".join(candidates)
    return relative.parent.as_posix()


def _frame_token(path: Path) -> str:
    numbers = re.findall(r"\d+", path.stem)
    return numbers[-1].lstrip("0") or "0" if numbers else path.stem.lower()


def _is_depth(path: Path) -> bool:
    return any("depth" in p.lower() for p in path.parts[:-1]) or "depth" in path.stem.lower()


def _attach_hdf5_navigation(root: Path, records: list[FrameRecord]) -> tuple[list[FrameRecord], list[str]]:
    """Attach MidAir navigation arrays when ``sensor_records.hdf5`` is present.

    MidAir releases have used nested HDF5 groups. Matching is therefore by the
    trajectory group's basename and semantic dataset suffix, not a fixed root.
    Missing fields remain explicitly absent in ``navigation``.
    """
    try:
        import h5py
    except ImportError:
        return records, []
    files = sorted((*root.rglob("*.hdf5"), *root.rglob("*.h5")))
    if not files:
        return records, []
    by_trajectory: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_trajectory.setdefault(record.trajectory_id, []).append(index)
    updated = list(records)
    used: list[str] = []
    vector_fields = {
        "velocity": ("vx", "vy", "vz"),
        "linear_velocity": ("vx", "vy", "vz"),
        "acceleration": ("ax", "ay", "az"),
        "linear_acceleration": ("ax", "ay", "az"),
        "angular_velocity": ("wx", "wy", "wz"),
        "accelerometer": ("ax", "ay", "az"),
        "gyroscope": ("wx", "wy", "wz"),
    }
    for h5_path in files:
        try:
            with h5py.File(h5_path, "r") as handle:
                groups: dict[str, Any] = {}
                def collect_groups(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Group):
                        groups[name] = obj

                handle.visititems(collect_groups)
                for trajectory, indices in by_trajectory.items():
                    h5_parent = h5_path.parent.relative_to(root).as_posix()
                    if h5_parent != "." and not trajectory.startswith(h5_parent + "/"):
                        continue
                    basename = trajectory.split("/")[-1]
                    candidates = [(name, group) for name, group in groups.items() if name.split("/")[-1] == basename]
                    if not candidates:
                        continue
                    _, group = candidates[0]
                    datasets: dict[str, Any] = {}
                    def collect_datasets(name: str, obj: Any) -> None:
                        if isinstance(obj, h5py.Dataset):
                            datasets[name] = obj

                    group.visititems(collect_datasets)
                    matched = False
                    for ordinal, record_index in enumerate(indices):
                        nav = dict(updated[record_index].navigation)
                        timestamp = updated[record_index].timestamp
                        for name, dataset in datasets.items():
                            leaf = name.lower().split("/")[-1]
                            if not dataset.shape or dataset.shape[0] == 0:
                                continue
                            frequency = float(dataset.attrs.get("sampling_frequency", 100.0 if name.lower().startswith(("groundtruth/", "imu/")) else 1.0 if name.lower().startswith("gps/") else 25.0))
                            sample_index = min(int(ordinal * frequency / 25.0), dataset.shape[0] - 1)
                            if leaf in vector_fields:
                                value = np.asarray(dataset[sample_index]).reshape(-1)
                                for key, item in zip(vector_fields[leaf], value[:3]):
                                    if np.isfinite(item):
                                        nav[key] = float(item)
                                        matched = True
                            elif leaf in {"timestamp", "timestamps", "time"}:
                                value = np.asarray(dataset[sample_index]).reshape(-1)
                                if value.size and np.isfinite(value[0]):
                                    timestamp = float(value[0])
                        updated[record_index] = replace(updated[record_index], navigation=nav, timestamp=timestamp)
                    if matched:
                        used.append(h5_path.relative_to(root).as_posix())
        except (OSError, ValueError):
            continue
    return updated, sorted(set(used))


def discover_midair(root: str | Path, strict: bool = True) -> tuple[list[FrameRecord], dict[str, Any]]:
    """Discover synchronized samples without assuming a fixed extracted layout.

    Pairing uses trajectory identity and the last numeric frame token. MidAir layouts
    with ``color_left/trajectory_x`` and ``depth/trajectory_x`` are both supported.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    image_files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in RGB_EXTENSIONS | DEPTH_EXTENSIONS),
        key=_natural_key,
    )
    # Classify only dataset-relative components. Absolute ancestor names (for
    # example a temporary test directory containing "depth") are irrelevant.
    depth_files = [p for p in image_files if _is_depth(p.relative_to(root)) and p.suffix.lower() in DEPTH_EXTENSIONS]
    rgb_files = [p for p in image_files if not _is_depth(p.relative_to(root)) and p.suffix.lower() in RGB_EXTENSIONS]
    # A full MidAir extraction contains three RGB cameras and several PNG map
    # modalities. The supervised baseline deliberately uses the left camera.
    if any("color_left" in (part.lower() for part in p.relative_to(root).parts[:-1]) for p in rgb_files):
        rgb_files = [p for p in rgb_files if "color_left" in (part.lower() for part in p.relative_to(root).parts[:-1])]
        rgb_selection = "color_left"
    elif any("rgb" in (part.lower() for part in p.relative_to(root).parts[:-1]) for p in rgb_files):
        rgb_files = [p for p in rgb_files if "rgb" in (part.lower() for part in p.relative_to(root).parts[:-1])]
        rgb_selection = "rgb"
    else:
        excluded = {"normals", "normal", "segmentation", "stereo_disparity", "disparity", "stereo_occlusion", "occlusion"}
        rgb_files = [p for p in rgb_files if not excluded.intersection(part.lower() for part in p.relative_to(root).parts[:-1])]
        rgb_selection = "non-map image candidates"
    depths: dict[tuple[str, str], Path] = {}
    for path in depth_files:
        depths[(_trajectory_id(path, root), _frame_token(path))] = path
    records: list[FrameRecord] = []
    missing_depth: list[str] = []
    for rgb in rgb_files:
        trajectory = _trajectory_id(rgb, root)
        token = _frame_token(rgb)
        depth = depths.get((trajectory, token))
        if depth is None:
            # Accommodate color_left/trajectory vs depth/trajectory by matching
            # trajectory suffix and frame token only when unambiguous.
            matches = [p for (traj, tok), p in depths.items() if tok == token and traj.split("/")[-1] == trajectory.split("/")[-1]]
            depth = matches[0] if len(matches) == 1 else None
        if depth is None:
            missing_depth.append(rgb.relative_to(root).as_posix())
            continue
        number = re.findall(r"\d+", rgb.stem)
        frame_index = int(number[-1]) if number else len(records)
        records.append(
            FrameRecord(
                trajectory_id=trajectory,
                frame_index=frame_index,
                rgb_path=rgb.relative_to(root).as_posix(),
                depth_path=depth.relative_to(root).as_posix(),
                metadata={"environment": rgb.relative_to(root).parts[0]},
            )
        )
    records.sort(key=lambda r: (r.trajectory_id, r.frame_index))
    records, navigation_sources = _attach_hdf5_navigation(root, records)
    report = {
        "dataset_root": str(root),
        "rgb_files": len(rgb_files),
        "rgb_selection": rgb_selection,
        "depth_files": len(depth_files),
        "paired_frames": len(records),
        "trajectory_count": len({r.trajectory_id for r in records}),
        "navigation_frames": sum(bool(r.navigation) for r in records),
        "navigation_sources": navigation_sources,
        "missing_depth_count": len(missing_depth),
        "missing_depth_examples": missing_depth[:20],
        "documented_depth_convention": "Euclidean range from camera focal point, metres; float16 bits stored in 16-bit PNG",
        "local_depth_verification": "pending real MidAir files",
        "depth_note": "Convention is established by the official FAQ and loader; actual files still require local metadata/value/visual confirmation before real training.",
    }
    if strict and not records:
        raise ValueError(f"No synchronized RGB/depth pairs discovered under {root}")
    return records, report


def dataset_fingerprint(records: Iterable[FrameRecord]) -> str:
    canonical = "\n".join(
        f"{r.trajectory_id}|{r.frame_index}|{r.rgb_path}|{r.depth_path}" for r in records
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def split_trajectories(records: list[FrameRecord], seed: int = 42) -> dict[str, list[str]]:
    trajectories = sorted({r.trajectory_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(trajectories)
    n = len(trajectories)
    if n < 3:
        raise ValueError("At least three complete trajectories are required for train/val/test splitting")
    n_train = max(1, round(0.70 * n))
    n_val = max(1, round(0.15 * n))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    return {
        "train": sorted(trajectories[:n_train]),
        "val": sorted(trajectories[n_train:n_train + n_val]),
        "test": sorted(trajectories[n_train + n_val:]),
    }


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    names = ("train", "val", "test")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise ValueError(f"Trajectory leakage between {left} and {right}: {sorted(overlap)}")


def write_manifest(records: list[FrameRecord], output: str | Path, seed: int = 42) -> dict[str, Any]:
    splits = split_trajectories(records, seed)
    assert_no_leakage(splits)
    counts = {name: sum(r.trajectory_id in tids for r in records) for name, tids in splits.items()}
    manifest = {
        "version": 1,
        "seed": seed,
        "generation_method": "sorted trajectory IDs; Python random.Random(seed).shuffle; 70/15/15 rounded with nonempty splits",
        "dataset_fingerprint": dataset_fingerprint(records),
        "trajectory_count": len({r.trajectory_id for r in records}),
        "frame_count": len(records),
        "splits": splits,
        "frame_counts": counts,
        "records": [asdict(r) for r in records],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: str | Path, split: str) -> tuple[list[FrameRecord], dict[str, Any]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_no_leakage(manifest["splits"])
    selected = set(manifest["splits"][split])
    records = [FrameRecord(**r) for r in manifest["records"] if r["trajectory_id"] in selected]
    return records, manifest


def load_depth(path: Path, scale: float = 1.0, encoding: str = "midair_float16_bits") -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    raw = np.asarray(Image.open(path))
    if encoding == "midair_float16_bits":
        # Official MidAir utility: load PNG as uint16 then reinterpret the
        # underlying bits as IEEE float16. Values are already metres.
        return raw.astype(np.uint16, copy=False).view(np.float16).astype(np.float32)
    if encoding == "linear_uint16":
        return raw.astype(np.float32) * scale
    if encoding == "metric":
        return raw.astype(np.float32)
    raise ValueError(f"Unsupported depth encoding: {encoding}")


class MidAirDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path, records: list[FrameRecord], config: dict[str, Any], training: bool = False):
        self.root = Path(root)
        self.records = records
        self.size = tuple(config.get("image_size", [192, 320]))
        self.min_depth = float(config.get("min_depth", 0.1))
        self.max_depth = float(config.get("max_depth", 1000.0))
        self.depth_scale = float(config.get("depth_scale", 1.0))
        self.depth_encoding = str(config.get("depth_encoding", "midair_float16_bits"))
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rgb = np.asarray(Image.open(self.root / record.rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = load_depth(self.root / record.depth_path, self.depth_scale, self.depth_encoding)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(f"RGB/depth alignment mismatch at {record.rgb_path}: {rgb.shape[:2]} vs {depth.shape[:2]}")
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        valid = torch.isfinite(depth_t) & (depth_t >= self.min_depth) & (depth_t <= self.max_depth)
        depth_t = torch.where(valid, depth_t, torch.zeros_like(depth_t))
        rgb_t = F.interpolate(rgb_t.unsqueeze(0), self.size, mode="bilinear", align_corners=False).squeeze(0)
        depth_t = F.interpolate(depth_t.unsqueeze(0), self.size, mode="nearest").squeeze(0)
        valid = F.interpolate(valid.float().unsqueeze(0), self.size, mode="nearest").squeeze(0).bool()
        if self.training and torch.rand(()) < 0.5:
            rgb_t, depth_t, valid = rgb_t.flip(-1), depth_t.flip(-1), valid.flip(-1)
        return {"rgb": rgb_t, "depth": depth_t, "valid": valid, "record": asdict(record)}


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rgb": torch.stack([x["rgb"] for x in batch]),
        "depth": torch.stack([x["depth"] for x in batch]),
        "valid": torch.stack([x["valid"] for x in batch]),
        "record": [x["record"] for x in batch],
    }
