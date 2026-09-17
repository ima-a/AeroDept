# AeroDepth-Lite

Safety-constrained adaptive inference for energy-efficient UAV monocular depth estimation. The MVP trains a compact supervised metric-depth network, reuses its latest prediction on selected frames, and forces refreshes through a transparent non-neural scheduler. It compares every-frame, every-2nd-frame, every-5th-frame, and adaptive complete-trajectory replay.

Inference-call reduction and runtime are **compute proxies**, not measured energy savings or evidence of longer UAV battery life.

## What is novel and what is not

Frame skipping is not itself novel. AeroDepth-Lite combines normalized navigation/IMU motion, cheap 64×36 grayscale change, robust near-field risk from the previous prediction, prediction age, hard danger/extreme-change triggers, and a mandatory `MAX_SKIP`. The scheduler has no learned parameters and adds no FLOPs to an executed depth inference. Its intended contribution is this safety-constrained combination and UAV trajectory evaluation.

## Current status

The complete software path is implemented and covered by a deterministic synthetic fixture. **No real MidAir files were available on this machine**, so there is no legitimate MidAir checkpoint, validation metric, tuned scheduler result, or local depth-file verification yet. Synthetic smoke metrics are software diagnostics only and must not be cited as research results. See [MVP_STATUS.md](MVP_STATUS.md).

## Architecture

- `aerodepth_lite.data`: layout-tolerant RGB/depth discovery, float16-bit depth decoding, optional HDF5 navigation attachment, deterministic trajectory manifests, masks and aligned resize/augmentation.
- `aerodepth_lite.model`: a 14,753-parameter smoke configuration (base width 8) or wider preliminary configuration using depthwise-separable encoder/decoder blocks and skip connections. Softplus produces positive metric depth.
- `aerodepth_lite.losses`: masked L1, log-depth, gradient, and optional near-field losses.
- `aerodepth_lite.metrics`: pixel-weighted AbsRel, RMSE, δ1/δ2/δ3, near-field metrics, and per-trajectory aggregation.
- `aerodepth_lite.scheduler`: non-neural motion, visual-change, risk, age, first-frame and `MAX_SKIP` logic.
- `aerodepth_lite.replay`: honest evaluation of reused predictions against each current frame, per-frame CSV logs, critical-event accounting, and four policies.
- `aerodepth_lite.artifacts`: figures generated from saved histories/results.

## MidAir data and depth convention

Download MidAir through the [official download page](https://midair.ulg.ac.be/download.html), accepting its CC BY-NC-SA license, and select at minimum left RGB, depth/range maps, and the associated `sensor_records.hdf5`. Extract archives while preserving their paths. The loader does not require one exact root layout; it identifies trajectories and synchronized frame tokens recursively.

The official [data-organization documentation](https://midair.ulg.ac.be/data_organization.html) specifies 25 Hz visual streams and HDF5 positioning/IMU records. The [official FAQ](https://midair.ulg.ac.be/faq.html) states that depth is **Euclidean distance from the camera focal point**, not camera-axis/projective z-depth. The [official example loader](https://github.com/montefiore-institute/midair-dataset/blob/master/hdf5-depth_viewer.py) reads each PNG as `uint16`, reinterprets its bits as `float16`, and treats the result as metres. AeroDepth-Lite implements that exact decoding with `depth_encoding: midair_float16_bits`.

Before a real run, `inspect` still checks actual dimensions, finite/range statistics, corrupt files, synchronization, navigation presence, and produces a visual. This local inspection is required even though the documented convention is known.

Typical extracted layout (other nesting is accepted):

```text
MIDAir/
  <climate>/
    color_left/trajectory_XXXX/*.JPEG
    depth/trajectory_XXXX/*.PNG
    sensor_records.hdf5
```

Raw data is ignored by Git and must remain outside the repository or under ignored `data/`/`datasets/` directories.

## Installation

Python 3.10+ is required. On a CPU machine, avoid downloading a CUDA runtime unnecessarily:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev,midair]'
```

For a CUDA machine, install the appropriate official PyTorch build first, then install the project. All commands below use `.venv/bin/aerodepth` explicitly.

## Reproducible workflow

Set `DATASET_ROOT` to the extracted root:

```bash
export DATASET_ROOT=/path/to/MidAir

.venv/bin/aerodepth inspect \
  --config configs/preliminary.yaml \
  --dataset-root "$DATASET_ROOT" \
  --output artifacts/inspection

.venv/bin/aerodepth split \
  --dataset-root "$DATASET_ROOT" \
  --seed 42 \
  --output manifests/midair_seed42.json

.venv/bin/aerodepth visualize \
  --config configs/preliminary.yaml \
  --dataset-root "$DATASET_ROOT" \
  --manifest manifests/midair_seed42.json \
  --split val \
  --output artifacts/inspection/validation_sample.png

.venv/bin/aerodepth train \
  --config configs/preliminary.yaml \
  --dataset-root "$DATASET_ROOT" \
  --manifest manifests/midair_seed42.json \
  --output runs/preliminary

.venv/bin/aerodepth replay \
  --config configs/preliminary.yaml \
  --dataset-root "$DATASET_ROOT" \
  --manifest manifests/midair_seed42.json \
  --checkpoint runs/preliminary/best.pt \
  --split val \
  --output runs/preliminary/replay

.venv/bin/aerodepth report \
  --input runs/preliminary/replay \
  --output runs/preliminary/report
```

Use `--policies every`, `fixed2`, `fixed5`, or `adaptive` to run one policy. Test evaluation is deliberately opt-in with `--split test`; do it only after freezing model and scheduler choices on validation.

The complete software smoke test is:

```bash
make smoke
```

It creates a visibly labelled synthetic fixture in `/tmp`; it is not MidAir evidence.

## Configuration and reproducibility

`configs/smoke.yaml` is intentionally tiny; `configs/preliminary.yaml` is the real-data starting point. Configuration controls image/depth processing, architecture width, optimization, loss weights, scheduler thresholds, visual-score resolution, `MAX_SKIP`, and critical-event thresholds.

Seed 42 is applied to Python, NumPy, PyTorch CPU/CUDA, and DataLoader workers. Manifests contain relative paths, seed, trajectory/frame counts, split method, and dataset fingerprint. Runs record the full configuration, Git commit, Python/PyTorch/NumPy versions, device, manifest SHA-256, duration, checkpoint, parameter count, checkpoint size, losses, and validation metrics. CUDA deterministic algorithms are requested with warnings for operations that cannot guarantee determinism.

Splits are 70/15/15 by complete trajectory and automatically checked for overlap. Training augmentations currently consist only of synchronized horizontal RGB/depth/mask flips. RGB is resized bilinearly; metric depth and validity masks use nearest-neighbour interpolation, which preserves sampled metric values but does not create subpixel geometry.

## Scheduler and safety logging

The scheduler uses

```text
score = wm*motion + wv*visual_change + wr*near_risk + wa*normalized_age
```

and executes inference on the first frame; at `MAX_SKIP`; for configured critical risk/extreme motion/extreme visual change; or when the combined score reaches its threshold. Missing navigation is represented by `motion_available=false`, not silently presented as a measured zero. Every frame logs all raw/normalized signals, reason, decision threshold, ages, forced status, latencies, and prediction source.

The provisional critical event is a 10th-percentile valid ground-truth depth entering the configured near zone, or falling by the configured amount while within it. This definition and every scheduler threshold must be tuned only on validation trajectories.

## Testing

```bash
.venv/bin/python -m pytest -q
```

Fast tests do not require MidAir. They cover discovery/path pairing, float16 depth decoding, HDF5 navigation, deterministic splitting/leakage, masks/alignment, hand-calculated metrics, finite losses, model shape/size, checkpoints, scheduler signals/constraints/reset, logging schema, fixed/adaptive accounting, honest reused-frame evaluation, and smoke training.

## Limitations and next work

- Real MidAir acquisition, local depth inspection, training, validation replay, and validation-only threshold tuning remain blocked by dataset absence.
- HDF5 navigation support reads common MidAir vector fields and resamples using `sampling_frequency`; it must be checked against the downloaded release's exact paths and camera-path ordering.
- No pretrained encoder, FLOP measurement, quantization, edge-device latency, power, or battery test is claimed.
- The tiny smoke model is a pipeline baseline, not a converged accuracy result. The wider preliminary config also requires real validation before any performance claim.
- A future hardware study should record wall power/energy, device utilization, temperature, latency distribution, and joules per full trajectory for FP32/FP16/INT8 under controlled repeats.
