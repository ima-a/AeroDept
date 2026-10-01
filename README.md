# AeroDepth-Lite

A small supervised monocular depth model and a rule-based scheduler for reducing depth inference calls on UAV video. The scheduler may reuse the last prediction when motion, image change, near-field risk, and prediction age are low. Replay compares every-frame, every-2nd-frame, every-5th-frame, and adaptive inference on complete trajectories.

**Research status:** the software workflow is complete, but the real MidAir evaluation is a preliminary two-trajectory pilot: one trajectory for training, one for validation, and no test trajectory. Fewer inference calls are a compute proxy; energy use and battery life were not measured. See the [pilot evidence](evidence/real_midair_pilot/README.md).

## What a fresh clone can reproduce

| Item | From this repository alone? | Requirement |
|---|---|---|
| Unit tests and synthetic end-to-end smoke run | Yes | Python 3.10+ and installable dependencies |
| Real MidAir inspection, training, and replay | No | Download licensed MidAir RGB, depth, and HDF5 records |
| Exact saved pilot result | No | Original data, manifest, and checkpoint are intentionally excluded; the tracked evidence preserves results and fingerprints |

The smoke fixture is synthetic and verifies the pipeline, **not** depth accuracy on MidAir. Dependency ranges and hardware can change numerical results; the pilot's exact software versions and source commit are recorded in [metadata.json](evidence/real_midair_pilot/metadata.json). No bit-for-bit retraining guarantee is claimed.

## Repository layout

```text
configs/                    Smoke, preliminary, and frozen pilot settings
src/aerodepth_lite/         Data, model, training, scheduler, replay, CLI, plots
tests/                      Synthetic tests and opt-in real-data integration check
evidence/real_midair_pilot/ Small tracked pilot summary and fingerprints
```

The repository excludes downloaded data, virtual environments, generated manifests, checkpoints, runs, and plots. Put MidAir data in an external directory or ignored `datasets/`; outputs may go under ignored `artifacts/` or `runs/`. Do not commit downloaded MidAir data or model checkpoints by default.

## Install and verify

On Linux with a CPU, from a fresh clone:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev,midair]'
.venv/bin/python -m pytest -q
make smoke
```

For CUDA, install the matching official PyTorch build before installing the project. `make smoke` creates a 30-frame synthetic fixture and generated outputs in `/tmp/aerodepth-*`; it trains a tiny one-epoch model and runs all four replay policies.

## Run on MidAir

Download MidAir from the [official page](https://midair.ulg.ac.be/download.html) under its license. Obtain left RGB, depth/range PNGs, and the associated `sensor_records.hdf5` for **enough complete trajectories to form train, validation, and test sets**. Preserve the archive paths. The loader discovers trajectories recursively, so a single fixed top-level directory name is not required.

The [official FAQ](https://midair.ulg.ac.be/faq.html) defines MidAir depth as Euclidean distance from the camera focal point. The [official example loader](https://github.com/montefiore-institute/midair-dataset/blob/master/hdf5-depth_viewer.py) reads the 16-bit PNG values as float16 bit patterns in metres; `midair_float16_bits` uses that convention. Inspect the local extraction and saved synchronized visuals before training.

```bash
export MIDAIR_ROOT=/path/to/extracted/MidAir

.venv/bin/aerodepth inspect --config configs/preliminary.yaml \
  --dataset-root "$MIDAIR_ROOT" --output artifacts/inspection \
  --sample-count 10 --stats-sample-count 32 --seed 42

.venv/bin/aerodepth split --dataset-root "$MIDAIR_ROOT" \
  --seed 42 --output manifests/midair_seed42.json

.venv/bin/aerodepth train --config configs/preliminary.yaml \
  --dataset-root "$MIDAIR_ROOT" --manifest manifests/midair_seed42.json \
  --output runs/preliminary

.venv/bin/aerodepth replay --config configs/preliminary.yaml \
  --dataset-root "$MIDAIR_ROOT" --manifest manifests/midair_seed42.json \
  --checkpoint runs/preliminary/best.pt --split val \
  --output runs/preliminary/replay

.venv/bin/aerodepth report --input runs/preliminary/replay \
  --output runs/preliminary/report
```

Review `inspection.json`, the sample visuals, and the generated manifest before training. Splits use whole trajectories, seed 42, and a 70/15/15 train/validation/test allocation; the manifest records relative paths and a dataset fingerprint. Tune the model and scheduler only on training and validation. Run `replay --split test` once after freezing all choices.

To check a local real MidAir extraction:

```bash
MIDAIR_ROOT="$MIDAIR_ROOT" .venv/bin/python -m pytest -q -m integration
```

## Method and outputs

The depth network uses depthwise-separable encoder/decoder blocks with skip connections and positive metric-depth output. Losses combine masked L1, log-depth, gradient, and optional near-field terms. Replay evaluates reused predictions against each **current** frame's ground truth and writes per-frame CSV logs, per-trajectory metrics, critical-event accounting, and plots.

The scheduler combines normalized navigation/IMU motion, 64×36 grayscale change, near-field risk from the previous prediction, and prediction age. It always infers on the first frame and by `MAX_SKIP`, and can force inference for danger or extreme change. Missing navigation is logged explicitly. It has no learned weights. The provisional critical event is a near-zone ground-truth depth entry or sharp decrease; it needs validation on a larger held-out dataset.

Seed 42 is applied to Python, NumPy, PyTorch, and data-loader workers. Runs save configuration, code revision, library versions, manifest hash, checkpoints, metrics, and timing. CUDA operations may still be nondeterministic.

## Pilot result and limits

The tracked [pilot evidence](evidence/real_midair_pilot/README.md) covers 4,409 synchronized frames from two Kite/foggy trajectories. The frozen validation replay used 1,481 adaptive calls over 2,204 frames (32.80% fewer than every-frame) and updated all four provisional critical events on time. Its AbsRel was 0.374984. This is a preliminary validation result with thresholds selected on the sole validation trajectory. It is not a held-out benchmark, and no test result exists.

There is no pretrained encoder, FLOP measurement, quantization result, edge-device latency, power reading, or battery test in this repository.
