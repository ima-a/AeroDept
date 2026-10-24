# Installation and tested versions

Use Python3.13 from the repository root. The exact CPU commands are in README. `requirements-tested.txt` constrains versions tested for this checkpoint. The original working CUDA environment was preserved, not rebuilt.

For a fresh compatible NVIDIA/CUDA13 environment, the recorded pair can be installed with:

```bash
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install '.[dev,midair]' -c requirements-tested.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use the [official installer](https://pytorch.org/get-started/locally/) to choose wheels compatible with your actual driver/GPU. Do not run the CPU-wheel install over an existing CUDA environment you intend to keep. CPU inference and unit/synthetic checks do not need a GPU. Dataset validation/replay additionally needs h5py and the separately acquired data.

Training was Python3.13.12, torch2.14.1+cu130, torchvision0.29.1+cu130, NumPy2.5.3, Pillow12.3.0, PyYAML6.0.3, matplotlib3.11.2 and h5py3.16.0. Clean-clone verification uses fresh downloaded CPU wheels; the clean-clone protocol is recorded in docs/publication.md and executed outcomes accompany the published commit. CPU/GPU rounding differences are expected; the public runner checks declared metric tolerances rather than promising identical floating-point bytes.
