# AeroDepth-Lite MVP status

Date: 2026-09-18. Branch: `feat/aerodepth-mvp`. Starting commit: `c056401`.

## Implemented and tested

- Recursive MidAir RGB/depth discovery, exact frame pairing, invalid/corrupt/alignment reporting, relative paths, and synchronized visualization.
- Official MidAir float16-bit PNG decoding. Local decoding and synchronized visual review confirm spatially plausible Euclidean focal-point ranges on the real pilot.
- HDF5 navigation attachment for velocity, acceleration, and angular velocity with sampling-frequency resampling and explicit missing-signal state.
- Deterministic seed-42 whole-trajectory 70/15/15 manifests with fingerprint and leakage checks.
- Aligned preprocessing, validity masks, train-only flips, lightweight positive-depth model, masked component losses, standard and near-field metrics.
- Reproducible CPU/CUDA training, best/last checkpointing, resume, run metadata, loss curve, and prediction comparison.
- Every-frame, fixed-2, fixed-5, and adaptive chronological replay with honest reused-prediction evaluation.
- First-frame and `MAX_SKIP` constraints, full frame CSV schema, critical-event accounting, per-trajectory metrics, and generated comparison/timeline/trade-off plots.

## Verification executed

Baseline before implementation:

```text
python -m pytest -q
/usr/bin/python: No module named pytest
```

Fast automated suite before the final documentation pass:

```text
.venv/bin/python -m pytest -q
21 passed, 1 skipped
```

Opt-in real MidAir integration check:

```text
MIDAIR_ROOT=datasets/midair_pilot .venv/bin/python -m pytest -q -m integration
1 passed, 21 deselected
```

The clean software workflow was executed with the commands in `make smoke` using a deterministic fixture of 6 trajectories × 5 frames (30 frames). This fixture is synthetic and exists solely for software verification.

Environment: Python 3.14.7, PyTorch 2.14.0+cpu, Intel i5-10300H (4 cores/8 threads), 15 GiB RAM, and approximately 12 GiB disk free at handoff. A GeForce GTX 1650 Mobile is present, but `nvidia-smi` cannot communicate with its driver and this PyTorch build has no CUDA support; all reported training and replay ran on CPU.

Smoke model: `AeroDepthTiny`, base width 8, 14,753 trainable parameters, approximate FP32 weights 0.059 MB, checkpoint approximately 0.24 MB. The smoke checkpoint and metrics are **not real MidAir results**.

## Real-data pipeline validation

The official MidAir download form supplied Kite/training/foggy left-camera RGB and depth for `trajectory_2003` and `trajectory_2021`, plus `sensor_records.hdf5`. The downloaded archives total approximately 1.687 GiB; the retained compressed and extracted tree is approximately 4.1 GiB under ignored `datasets/midair_pilot`.

The gate scanned all **4,409 frames across 2 trajectories**: 2,205 from `trajectory_2003` and 2,204 from `trajectory_2021`. RGB/depth counts and ordering matched the HDF5 camera lists exactly; there were no missing depth frames, corrupt pairs, dimension mismatches, unmatched trajectories, or sampled navigation-resampling errors. Velocity, accelerometer, and gyroscope values are present for every image record.

Float16-bit decoding across all 4,623,171,584 pixels produced a raw finite range of 5.87890625–65,504 m. With the configured valid interval of 0.1–1,000 m, 24.6001% of pixels are invalid, entirely because they exceed 1,000 m; no below-minimum or non-finite pixels were found. Across 6.4 million valid pixels sampled deterministically from 32 frames, p01/p05/p25/median/p75/p95/p99 were 8.890625/11.7265625/22.5/36.75/85.1875/516.5/873.5 m. Ten synchronized visuals were reviewed: terrain, water, vegetation, and horizon boundaries align, near surfaces remain nearer than distant slopes, and masked far/fog regions are consistent with the 1,000 m ceiling.

Evidence: `artifacts/real_midair_pilot/inspection/inspection.json`, its 10 `samples/*.png`, and `contact_sheet.png`. These paths are persistent and Git-ignored.

## Real-data pilot preliminary validation

Because expansion was blocked, a clearly labelled non-final manifest used `trajectory_2003` for training (2,205 frames), `trajectory_2021` for validation (2,204 frames), and an empty test split. The sets are trajectory-disjoint by construction, but two trajectories are insufficient for a meaningful train/validation/test study. The test split was not run.

The 20-epoch wider model completed on CPU in 5,018.95 seconds (83.65 minutes). It has 109,153 trainable parameters (0.437 MB FP32 weights); the best checkpoint is 1.385 MB at `artifacts/real_midair_pilot/preliminary_pilot_v2/best.pt`. Epoch-level history and scheduler state are now persisted in checkpoints so interrupted runs can resume without losing prior history. The prediction is spatially non-constant but visibly blurred and materially inaccurate, consistent with the one-training-trajectory limitation.

Every-frame evaluation of the best checkpoint on the full validation trajectory produced: AbsRel **0.375082**, RMSE **155.280 m**, δ1 **0.388164**, δ2 **0.635602**, δ3 **0.734800**, near-field AbsRel **0.220725**, and near-field RMSE **2.531563 m**, over 89,276,771 valid pixels. The near field is configured as depth below 10 m. These are genuine MidAir validation measurements but only pilot preliminary results.

Scheduler normalization and decision thresholds were tuned only on `trajectory_2021`. With `motion_scale=16`, decision threshold `0.34`, and frozen `MAX_SKIP=4`, the clean validation replay was:

| Policy | Calls | Call reduction | Max age | AbsRel | RMSE (m) | δ1 | Critical events on time |
|---|---:|---:|---:|---:|---:|---:|---:|
| every | 2,204 | 0.00% | 0 | 0.375082 | 155.280 | 0.388164 | 4/4 |
| fixed2 | 1,102 | 50.00% | 1 | 0.375457 | 155.238 | 0.387896 | 2/4 |
| fixed5 | 441 | 79.99% | 4 | 0.375226 | 155.279 | 0.388870 | 0/4 |
| adaptive | 1,481 | 32.80% | 4 | 0.374984 | 155.259 | 0.388437 | 4/4 |

Per-frame logs and raw metrics are under `artifacts/real_midair_pilot/preliminary_pilot_v2/replay_frozen`; plots are under `report_frozen`. A concise, machine-independent subset is retained in the tracked [pilot evidence bundle](evidence/real_midair_pilot/README.md). Call reduction and CPU inference runtime are compute proxies only. No energy or battery saving is claimed.

## Synthetic software diagnostics

Synthetic fixture replay is only evidence that accounting and artifacts execute. It must not appear in a paper table as MidAir performance.

Synthetic software diagnostic (5 validation frames):

| Policy | Inference calls | Skip rate | Max age | AbsRel |
|---|---:|---:|---:|---:|
| every | 5 | 0% | 0 | 0.901169 |
| fixed2 | 3 | 40% | 1 | 0.901169 |
| fixed5 | 1 | 80% | 4 | 0.901172 |
| adaptive | 5 | 0% | 0 | 0.901169 |

The one-epoch fixture model predicted persistent near-field risk, so the adaptive safety trigger conservatively inferred on every frame. This is correct rule execution, not a tuned efficiency result.

## Current blocker and exact next steps

Only two complete trajectories are local. The largest safe next download with current storage adds complete left RGB and depth for `trajectory_2000`, `trajectory_2001`, and `trajectory_2022`, about 3.166 GiB compressed, under `datasets/midair_pilot`, producing five total trajectories. The official source is `https://midair.ulg.ac.be/download.html`. No transfer was started because the official API requires a fresh email/agreement/reCAPTCHA token. The preferred 10-trajectory set (`2000, 2001, 2003, 2008, 2010, 2014, 2015, 2016, 2021, 2022`) needs another 9.137 GiB compressed before extraction and is unsafe with only about 12 GiB free.

1. Complete the official form token and add the three-trajectory storage-safe expansion, or attach/free enough storage for the preferred 10–12 trajectory set. The existing HDF5 covers the foggy trajectories.
2. Re-run the real inspection and integration gate over the expanded set.
3. Generate and review the seed-42 manifest; prove no overlap and preserve the test split untouched.
4. Retrain on the expanded training split and inspect validation predictions.
5. Retune scheduler thresholds using only the expanded validation split, then freeze preprocessing, model, scheduler, near-field definition, and `MAX_SKIP`.
6. Only then perform one deliberate final test evaluation. The current empty test split remains untouched.

Exact starting command after data acquisition:

```bash
.venv/bin/aerodepth inspect --config configs/preliminary.yaml --dataset-root datasets/midair_pilot --output artifacts/real_midair_pilot/inspection --sample-count 10 --stats-sample-count 32 --seed 42
```

## Git

Raw data, environments, checkpoints, runs, and generated artifacts are ignored. Commit hashes, if created during handoff, are reported in the final response rather than embedded here.
