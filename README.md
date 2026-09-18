# AeroDepth-Lite

Safety-constrained adaptive inference for energy-efficient UAV monocular depth estimation. The MVP trains a compact supervised metric-depth network, reuses its latest prediction on selected frames, and forces refreshes through a transparent non-neural scheduler. It compares every-frame, every-2nd-frame, every-5th-frame, and adaptive complete-trajectory replay.

Inference-call reduction and runtime are **compute proxies**, not measured energy savings or evidence of longer UAV battery life.

## What is novel and what is not

Frame skipping is not itself novel. AeroDepth-Lite combines normalized navigation/IMU motion, cheap 64×36 grayscale change, robust near-field risk from the previous prediction, prediction age, hard danger/extreme-change triggers, and a mandatory `MAX_SKIP`. The scheduler has no learned parameters and adds no FLOPs to an executed depth inference. Its intended contribution is this safety-constrained combination and UAV trajectory evaluation.

## Current status

The complete software path is implemented and covered by a deterministic synthetic fixture. A two-trajectory real MidAir pilot (4,409 synchronized frames) passed discovery, decoding, synchronization, navigation-resampling, and visual-plausibility checks. A deliberately limited one-trajectory-train/one-trajectory-validation run also completed, including validation-only scheduler tuning and all four replay policies. These are **pilot preliminary validation results**, not a meaningful benchmark: there is no test trajectory and the final test split remains untouched. See [MVP_STATUS.md](MVP_STATUS.md) and the small tracked [pilot evidence bundle](evidence/real_midair_pilot/README.md).

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

Before a real run, `inspect` checks every pair's dimensions and decodability, reports finite/range statistics and invalid-pixel causes, cross-checks HDF5 camera paths and navigation resampling, and creates 10 deterministic synchronized visuals by default. This local inspection is required even though the documented convention is known.

The validated local pilot is under ignored `datasets/midair_pilot`. It contains Kite/foggy training trajectories `trajectory_2003` (2,205 frames) and `trajectory_2021` (2,204 frames), plus their `sensor_records.hdf5`. The full inspection evidence is under ignored `artifacts/real_midair_pilot/inspection`. The two-trajectory manifest may be used only for software/pilot validation; it cannot form the required train/validation/test research split.

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
  --output artifacts/inspection \
  --sample-count 10 \
  --stats-sample-count 32 \
  --seed 42

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

`configs/smoke.yaml` is intentionally tiny; `configs/preliminary.yaml` is the real-data starting point. `configs/preliminary_pilot_frozen.yaml` records the scheduler setting selected only on the single pilot validation trajectory and must not be presented as a final benchmark configuration. Configuration controls image/depth processing, architecture width, optimization, loss weights, scheduler thresholds, visual-score resolution, `MAX_SKIP`, and critical-event thresholds.

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

Fast tests do not require MidAir. They cover discovery/path pairing, float16 depth decoding, HDF5 navigation, HDF5 path/resampling validation, deterministic splitting/leakage, masks/alignment, hand-calculated metrics, finite losses, model shape/size, checkpoints, scheduler signals/constraints/reset, logging schema, fixed/adaptive accounting, honest reused-frame evaluation, and smoke training. The opt-in real-data integration check is:

```bash
MIDAIR_ROOT=datasets/midair_pilot .venv/bin/python -m pytest -q -m integration
```

## Limitations and next work

- The real-data inspection gate and a CPU pilot experiment passed on two trajectories, but this is only one training trajectory and one validation trajectory. It is not a meaningful trajectory-disjoint train/validation/test experiment, and no test result exists.
- The pilot best checkpoint used 109,153 parameters and achieved validation AbsRel 0.375082, RMSE 155.280 m, δ1/δ2/δ3 0.3882/0.6356/0.7348, near AbsRel 0.2207, and near RMSE 2.5316 m on `trajectory_2021`. Treat these as preliminary diagnostics because the split is too small.
- The validation-selected adaptive scheduler made 1,481 calls over 2,204 frames (32.80% fewer than every-frame), with AbsRel 0.374984 and all 4 provisional critical events updated on time. This is inference-call reduction, not measured energy or battery saving.
- Expansion remains blocked by the official download form requiring a fresh email/agreement/reCAPTCHA token. The largest currently storage-safe plan adds complete left RGB and depth for trajectories `2000`, `2001`, and `2022` (about 3.166 GiB compressed) to `datasets/midair_pilot`, yielding five trajectories. A 10–12 trajectory study also needs more free storage; only about 12 GiB is currently available.
- HDF5 navigation support reads common MidAir vector fields and resamples using `sampling_frequency`; the pilot matched all 4,409 camera-path entries and sampled resampled values exactly.
- No pretrained encoder, FLOP measurement, quantization, edge-device latency, power, or battery test is claimed.
- Both the tiny smoke model and the two-trajectory pilot are pipeline baselines, not converged or publication-grade accuracy results.
- A future hardware study should record wall power/energy, device utilization, temperature, latency distribution, and joules per full trajectory for FP32/FP16/INT8 under controlled repeats.
