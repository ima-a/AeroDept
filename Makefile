.PHONY: install install-cpu test smoke inspect split
PYTHON ?= python3
SMOKE_DIR ?= artifacts/synthetic_smoke

# Preserve any existing CUDA PyTorch. For a fresh CPU-only venv use install-cpu.
install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install '.[dev,midair]'

install-cpu:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/python -m pip install '.[dev,midair]'

test:
	.venv/bin/python -m pytest -q

smoke:
	.venv/bin/python scripts/smoke.py --output "$(SMOKE_DIR)"

inspect:
	.venv/bin/aerodepth inspect --dataset-root "$(DATASET_ROOT)" --output artifacts/inspection

split:
	.venv/bin/aerodepth split --dataset-root "$(DATASET_ROOT)" --output artifacts/midair_seed42.json
