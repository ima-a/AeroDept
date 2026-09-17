# AeroDepth-Lite MVP status

Date: 2026-09-17. Branch: `feat/aerodepth-mvp`. Base commit: `41db4e1`.

## Implemented and tested

- Recursive MidAir RGB/depth discovery, exact frame pairing, invalid/corrupt/alignment reporting, relative paths, and synchronized visualization.
- Official MidAir float16-bit PNG decoding. Documentation establishes Euclidean focal-point range in metres; local file confirmation remains pending.
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

Final automated suite:

```text
.venv/bin/python -m pytest -q
19 passed
```

The clean software workflow was executed with the commands in `make smoke` using a deterministic fixture of 6 trajectories × 5 frames (30 frames). This fixture is synthetic and exists solely for software verification.

Environment: Python 3.14.7, PyTorch 2.14.0+cpu, Intel i5-10300H, CUDA unavailable. The smoke train performs 2 train batches and 2 validation batches for one epoch. Its measured duration and exact diagnostics are in `/tmp/aerodepth-smoke/run.json` during this session; paths under `/tmp` are ephemeral.

Smoke model: `AeroDepthTiny`, base width 8, 14,753 trainable parameters, approximate FP32 weights 0.059 MB, checkpoint approximately 0.24 MB. The smoke checkpoint and metrics are **not real MidAir results**.

## Experimentally evaluated

No real-data experiment has been evaluated. Real dataset volume used: **0 frames, 0 trajectories**. No genuine MidAir checkpoint or validation metric exists. No scheduler threshold was tuned on MidAir. No energy result is claimed.

Synthetic fixture replay is only evidence that accounting and artifacts execute. It must not appear in a paper table as MidAir performance.

Synthetic software diagnostic (5 validation frames):

| Policy | Inference calls | Skip rate | Max age | AbsRel |
|---|---:|---:|---:|---:|
| every | 5 | 0% | 0 | 0.901169 |
| fixed2 | 3 | 40% | 1 | 0.901169 |
| fixed5 | 1 | 80% | 4 | 0.901172 |
| adaptive | 5 | 0% | 0 | 0.901169 |

The one-epoch fixture model predicted persistent near-field risk, so the adaptive safety trigger conservatively inferred on every frame. This is correct rule execution, not a tuned efficiency result.

## External blocker and exact next steps

MidAir is absent from all searched local locations. Obtain a licensed subset containing left RGB, depth, and `sensor_records.hdf5`, then:

1. Run `inspect`; compare decoded value ranges/visuals with official documentation and record that evidence.
2. Generate and review the manifest; preserve the test split untouched.
3. Train `configs/preliminary.yaml`; check finite/nontrivial predictions and validation curves.
4. Tune scheduler normalization, thresholds, `MAX_SKIP`, and provisional critical definition using validation only.
5. Freeze configuration, run all validation policies and plots, then deliberately run test once if a final evaluation is wanted.
6. Benchmark on target edge hardware before making latency, energy, or battery claims.

Exact starting command after data acquisition:

```bash
.venv/bin/aerodepth inspect --config configs/preliminary.yaml --dataset-root /path/to/MidAir --output artifacts/inspection
```

## Git

Raw data, environments, checkpoints, runs, and generated artifacts are ignored. Commit hashes, if created during handoff, are reported in the final response rather than embedded here.
