"""Build a small allowlisted source snapshot; Git commands are read-only."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def package(output: Path) -> dict:
    paths = sorted([*ROOT.glob("src/aerodepth_lite/*.py"), *ROOT.glob("configs/*.yaml"),
                    *ROOT.glob("tests/*.py"), *ROOT.glob("scripts/*.py"),
                    ROOT / "notebooks/kaggle_train.ipynb", ROOT / "docs/kaggle_training.md",
                    ROOT / "pyproject.toml"])
    files = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}
    files["README.md"] = b"# AeroDepth-Lite Kaggle source snapshot\n\nInstall with pip install --no-deps --no-build-isolation -e .\nSee docs/kaggle_training.md and notebooks/kaggle_train.ipynb.\nSynthetic checks are software verification only. Source provenance is in SOURCE_PROVENANCE.json.\n"
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    provenance = {"base_git_commit": commit, "dirty_worktree": bool(status),
                  "git_status_porcelain": status.splitlines(),
                  "source_claim": "working-tree snapshot with uncommitted edits" if status else "clean source snapshot",
                  "generated_files": ["README.md", "SOURCE_PROVENANCE.json"],
                  "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["SOURCE_PROVENANCE.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"zip": str(output), "bytes": output.stat().st_size, "sha256": digest,
            "file_count": len(files), "dirty_worktree": bool(status), "base_git_commit": commit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/aerodepth_kaggle_code.zip")
    print(json.dumps(package(parser.parse_args().output), indent=2))
