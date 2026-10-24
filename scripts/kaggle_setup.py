"""Standard-library bootstrap. Deliberately never installs torch/torchvision."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_project(bundle, destination, expected_sha256=None):
    bundle, destination = Path(bundle), Path(destination)
    digest = sha256(bundle)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Code bundle checksum mismatch")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate bundle members")
        if sum(info.file_size for info in archive.infolist()) > 20_000_000:
            raise ValueError("Unexpectedly large code bundle")
        for info in archive.infolist():
            if (Path(info.filename).is_absolute() or ".." in Path(info.filename).parts or
                    ((info.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe code bundle member")
        provenance = json.loads(archive.read("SOURCE_PROVENANCE.json"))
        if set(names) != set(provenance["file_sha256"]) | {"SOURCE_PROVENANCE.json"}:
            raise ValueError("Bundle inventory differs from provenance")
        for name, expected in provenance["file_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Bundled source hash differs: {name}")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
    (destination.parent / "code_bundle.json").write_text(json.dumps({
        "bundle": str(bundle), "sha256": digest, "provenance": provenance}, indent=2))
    return destination


def install_project(project):
    import torch
    import torchvision
    original = (torch.__version__, torchvision.__version__)
    # Install only missing utility packages. --no-deps prevents resolver changes
    # to the Kaggle CUDA stack. Existing packages are checked below.
    utilities = {"numpy": "numpy>=1.26", "PIL": "Pillow>=10", "yaml": "PyYAML>=6",
                 "matplotlib": "matplotlib>=3.8", "h5py": "h5py>=3.10", "packaging": "packaging",
                 "setuptools": "setuptools>=77", "wheel": "wheel"}
    import importlib.util
    for module, requirement in utilities.items():
        if importlib.util.find_spec(module) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", requirement])
    from packaging.requirements import Requirement
    for requirement in ("setuptools>=77", "wheel"):
        parsed = Requirement(requirement)
        if not parsed.specifier.contains(importlib.metadata.version(parsed.name)):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", requirement])
    for requirement in [*utilities.values(), "torch>=2.2", "torchvision>=0.17"]:
        parsed = Requirement(requirement)
        if not parsed.specifier.contains(importlib.metadata.version(parsed.name)):
            raise RuntimeError(f"Installed package does not satisfy {requirement}; resolve explicitly before training")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(project)])
    if original != (importlib.metadata.version("torch"), importlib.metadata.version("torchvision")):
        raise RuntimeError("PyTorch stack changed during setup")
    # Current notebook kernel does not process newly written editable .pth files.
    sys.path.insert(0, str(Path(project) / "src"))
    return {"torch": original[0], "torchvision": original[1]}


def stage_project(source, destination, expected_sha256=None):
    """Kaggle may expose an uploaded ZIP either intact or already extracted."""
    source, destination = Path(source), Path(destination)
    if source.is_file():
        return extract_project(source, destination, expected_sha256)
    provenance = json.loads((source / "SOURCE_PROVENANCE.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in provenance["file_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe project path")
        path = source / name
        if path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(source / "SOURCE_PROVENANCE.json", destination / "SOURCE_PROVENANCE.json")
    (destination.parent / "code_bundle.json").write_text(json.dumps({
        "source": str(source), "original_bundle_sha256": expected_sha256,
        "original_zip_checksum_verified": False, "extracted_file_hashes_verified": True,
        "provenance": provenance}, indent=2))
    return destination


def environment_report(working="/kaggle/working"):
    import torch
    import torchvision
    usage = shutil.disk_usage(working)
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
              "gpu_available": torch.cuda.is_available(),
              "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "storage_free_gb": usage.free / 1e9, "storage_total_gb": usage.total / 1e9}
    print(json.dumps(report, indent=2))
    return report
