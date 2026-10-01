.PHONY: install test smoke inspect split replay report

PYTHON ?= python3

install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/python -m pip install -e '.[dev,midair]'

test:
	.venv/bin/python -m pytest -q

smoke:
	.venv/bin/aerodepth make-fixture --output /tmp/aerodepth-fixture
	.venv/bin/aerodepth inspect --config configs/smoke.yaml --dataset-root /tmp/aerodepth-fixture --output /tmp/aerodepth-inspection
	.venv/bin/aerodepth split --dataset-root /tmp/aerodepth-fixture --output /tmp/aerodepth-fixture/splits.json
	.venv/bin/aerodepth train --config configs/smoke.yaml --dataset-root /tmp/aerodepth-fixture --manifest /tmp/aerodepth-fixture/splits.json --output /tmp/aerodepth-smoke
	.venv/bin/aerodepth replay --config configs/smoke.yaml --dataset-root /tmp/aerodepth-fixture --manifest /tmp/aerodepth-fixture/splits.json --checkpoint /tmp/aerodepth-smoke/best.pt --output /tmp/aerodepth-replay
	.venv/bin/aerodepth report --input /tmp/aerodepth-replay --output /tmp/aerodepth-report

inspect:
	.venv/bin/aerodepth inspect --dataset-root "$(DATASET_ROOT)" --output artifacts/inspection

split:
	.venv/bin/aerodepth split --dataset-root "$(DATASET_ROOT)" --output manifests/midair_seed42.json
