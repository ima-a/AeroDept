from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(config: dict[str, Any], manifest: Path | None = None) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[2]
    bundle = project / "SOURCE_PROVENANCE.json"
    provenance = json.loads(bundle.read_text()) if bundle.exists() else None
    bundle_receipt = project.parent / "code_bundle.json"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(["git", "-C", str(project), "status", "--porcelain"], text=True)
    except Exception:
        commit = None
        status = None
    return {
        "seed": config.get("seed", 42),
        "config": config,
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "git_status_porcelain": status.splitlines() if status is not None else None,
        "source_file_sha256": {str(p.relative_to(project)): sha256_file(p)
                               for pattern in ("src/aerodepth_lite/*.py", "configs/*.yaml", "scripts/*.py", "pyproject.toml")
                               for p in sorted(project.glob(pattern))},
        "source_provenance": provenance,
        "code_bundle": json.loads(bundle_receipt.read_text()) if bundle_receipt.exists() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision") if importlib.util.find_spec("torchvision") else None,
        "numpy": np.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("Pillow", "PyYAML", "matplotlib")},
        "pretrained_weights": _weight_identity(config),
        "manifest_sha256": sha256_file(manifest) if manifest and manifest.exists() else None,
    }


def _weight_identity(config: dict[str, Any]) -> dict[str, Any] | None:
    if config["model"].get("name") != "mobilenet_v3_small_depth":
        return None
    from torchvision.models import MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    return {"identity": "MobileNet_V3_Small_Weights.IMAGENET1K_V1", "url": weights.url,
            "initialization_requested": config["model"].get("pretrained", True),
            "cached_path": str(path), "sha256": sha256_file(path) if path.exists() else None}
