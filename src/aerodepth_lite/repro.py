from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
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
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = None
    return {
        "seed": config.get("seed", 42),
        "config": config,
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "manifest_sha256": sha256_file(manifest) if manifest and manifest.exists() else None,
    }
