# Historical local laptop training workflow

The 50-epoch run is now complete. This file preserves the preparation/launch record; historical progress statements below describe the launch date. Use [the public reproduction guide](reproduction.md) and [completed results](../results/expanded_validation/REPORT.md) for the current checkpoint. `ARCHIVE_VERIFICATION_REPORT` refers to a locally retained prior archive-verification receipt; it is not required by the public reproduction runner.

This workflow reads the selected MidAir extraction directly from the external
disk. The software deliverable is MidAir replay/simulation and evaluation; no
drone or Raspberry Pi is required. The custom CNN baseline, pilot artifacts,
original downloads and Kaggle dataset package are preserved.

Run from the project root. Paths contain spaces and must remain quoted.

```bash
export MIDAIR_ROOT='/path/to/MidAir'
export TORCH_HOME="$PWD/artifacts/local/torch-cache"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

The existing `.venv` has Python 3.13.12, torch 2.14.1+cu130 and torchvision
0.29.1+cu130. `pip check` passes. The GTX 1650 has 4096 MiB reported by
NVIDIA-SMI, driver 615.78.08; PyTorch reports about 3730 MiB usable device memory.
A real CUDA sum succeeded. Sandboxed commands cannot see the NVIDIA devices;
run GPU commands in the host execution context. No driver or environment changes
were needed. Exact hardware and package records are in
`artifacts/local/hardware.json` and `environment-freeze.txt`.

## Dataset and split

```bash
.venv/bin/python -u scripts/local_training.py prepare \
  --dataset-root "$MIDAIR_ROOT" \
  --verification-report "$ARCHIVE_VERIFICATION_REPORT" \
  --output artifacts/local/preparation
```

Preparation checks the inventory against the prior verified 12-combination,
26,481-pair extraction, every RGB/depth image header, HDF5 camera ordering and
sampled navigation alignment. Prior archive CRC verification is reused. It saves
seven HDF5 hashes and a path inventory fingerprint, rather than claiming a fresh
full image-content audit. It never opens the Kaggle ZIP for writing.

The existing two-path pilot manifest cannot supply three independent splits.
The seed-42 grouped manifest is frozen at
`artifacts/local/preparation/manifest.json`. Each environment/path modulo 1000
stays together across weather/season realizations. The pilot's Kite paths 0003
and 0021 cannot enter test. Preparation compares Kite groundtruth position streams
between base sunny IDs and climate-prefixed HDF5 IDs in `path_id_evidence.json`.
The split is custom and is not the official MidAir benchmark protocol.

Numerical range review reads only first/middle/last frames from each train and
validation path, and saves `depth_range_review.json`. The 0.1–1000 m mask and
masked L1/log/gradient weights 1/0.2/0.1 are retained. Over-range and invalid pixels
are reported; no range adjustment is made to improve validation. RGB stays in
[0,1] in the loader, and ImageNet normalization occurs once inside the model.
The official [MidAir FAQ](https://midair.ulg.ac.be/faq.html) defines Euclidean
camera range, and [torchvision's model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
documents the selected `IMAGENET1K_V1` weights and normalization.

## Smoke, resume, benchmark and main run

`configs/mobilenet_local.yaml` uses 192×320, batch 4, two loader workers, CUDA
AMP, ImageNet initialization, AdamW 3e-4 and a fixed 50-epoch cosine horizon.
The model has 1,027,489 parameters. The original MobileNet and CNN configs are
unchanged. Weight URL, full cached-file SHA-256, source hashes/dirty state,
configuration, manifest hash and actual package versions are recorded per run.

```bash
.venv/bin/python -u scripts/local_training.py smoke \
  --dataset-root "$MIDAIR_ROOT" --manifest artifacts/local/preparation/manifest.json \
  --output runs/local_mobilenet/smoke
.venv/bin/python -u scripts/local_training.py smoke \
  --dataset-root "$MIDAIR_ROOT" --manifest artifacts/local/preparation/manifest.json \
  --output runs/local_mobilenet/smoke_resumed --resume runs/local_mobilenet/smoke/last.pt

.venv/bin/python -u scripts/local_training.py benchmark --detach \
    --dataset-root "$MIDAIR_ROOT" --manifest artifacts/local/preparation/manifest.json \
    --output runs/local_mobilenet/benchmark
```

Smoke runs three training and three validation batches per invocation, deliberately
stops at an epoch boundary, then resumes for a second bounded epoch. It verifies
epoch/global step, optimizer, scaler, RNG and unchanged schedule state. Automated
tests additionally compare resumed and uninterrupted final model weights.
Benchmark covers a complete train epoch and complete validation pass, including
external-disk reads, setup and checkpoint writing. Its total is multiplied by 50
and also reported with a 20% allowance. Resource telemetry records sampled process
tree RSS (including loader workers), GPU temperature/utilization/clocks/power and
reported thermal slowdowns. CUDA allocator peaks are recorded by the trainer.
RSS sums can count shared pages more than once; it is not exclusive memory use.

By default, main training is permitted only when CUDA works, smoke/resume and tests pass,
the range evidence has been reviewed, at least 2 GiB internal space remains,
and the benchmark estimate including allowance is at most 12 hours. A larger
duration requires explicit user authorization and `--max-hours`. For this run,
the user explicitly authorized the measured 19–23-hour duration after the full
benchmark. `--max-hours 24` records that approval in `launch_decision.json`;
it is an estimate gate, not a timed termination of training. The YAML review's
12-hour policy is the default; the explicit launch decision overrides it.

```bash
.venv/bin/python -u scripts/local_training.py main --max-hours 24 --detach \
    --dataset-root "$MIDAIR_ROOT" --manifest artifacts/local/preparation/manifest.json \
    --benchmark runs/local_mobilenet/benchmark/measurement.json \
    --output runs/local_mobilenet/main
```

Main starts a fresh seeded pretrained model. It never resumes smoke or benchmark
weights automatically. Its post-training hook evaluates best.pt on validation
only, writing AbsRel/RMSE/delta/near-field metrics and RGB/GT/prediction/error plots.
There is no final-test inference or adaptive-scheduler tuning in this script.

Resume an interrupted **main** run with the same shell exports and:

```bash
.venv/bin/python -u scripts/local_training.py main --max-hours 24 --detach \
    --dataset-root "$MIDAIR_ROOT" --manifest artifacts/local/preparation/manifest.json \
    --benchmark runs/local_mobilenet/benchmark/measurement.json \
    --output runs/local_mobilenet/main --resume runs/local_mobilenet/main/last.pt
```

`last.pt` and `best.pt` contain optimizer/scheduler/AMP/RNG/history state and global
step; best is selected only by minimum validation AbsRel. Epoch numbers inside
checkpoints/CSV are zero-based; `progress.json` reports human-readable epochs.
Checkpoints and CSV history use temporary-file replacement, preserving the previous
complete file during interrupted writes. An interrupted epoch repeats from the
last completed epoch. CUDA can remain nondeterministic. Power loss and external
disk disconnection are not recoverable guarantees.

`tmux` is absent on this laptop. Detached execution uses a new process session
around `systemd-inhibit`, with process IDs and exact command recorded in
`session.json`, stdout/stderr in `training.log`, progress in `progress.json`, and
telemetry in `resources.jsonl`. No system packages are installed by this workflow.

## Observed execution status

The 2026-10-19 dataset check passed: 26,481 matched RGB/depth pairs, all 12
trajectories matched to HDF5, no path mismatches, no navigation resampling error,
and no spatial-header mismatch. Seven HDF5 SHA-256 values are saved in the
verification artifact. Existing pilot evidence was read and retained.

| Split | Actual release IDs (environment/climate) | Frames |
|---|---|---|
| Train | Kite/cloudy 3002, Kite/foggy 2021, Kite/sunny 0000, Kite/sunset 1001; PLE/fall 4001 and 4004, PLE/winter 6002 and 6005 | 17,635 |
| Validation | Kite/foggy 2003; PLE/spring 5000 | 4,410 |
| Test | Kite/sunny 0004; PLE/spring 5003 | 4,436 |

Manifest SHA-256:
`e7b093be69f1942f281a09563bd63efa479d4352de97da722fb0504208c1fbf5`.
Inventory fingerprint:
`b68296e4052d4d4c76b7b1c765cb92319eeff9a6d74cc2a0cb311b973ce1563b`.
Actual Kite climate HDF5 position streams exactly equal their sunny base-path
streams, including 2003 ↔ 0003 and 2021 ↔ 0021. This supports the climate-digit
grouping with local numerical evidence.

The 30 train/validation range-review frames contain 31,457,280 pixels:
22,501,689 valid (71.531%), 8,955,591 above 1000 m (28.469%), none below 0.1 m
and none nonfinite. 4,039,431 pixels are valid and nearer than 10 m. These are
sample statistics, and over-range regions are excluded by the declared protocol.
The range and loss weights were retained; no test depth statistics were computed.

Pretrained-file SHA-256:
`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.
The completed automated test run passed **42 tests, 1 skipped** (the opt-in real
data integration test). Explicit dataset preparation supplies the real-data
structural check. A sandbox-only NVML warning is expected; host CUDA works.
`git diff --check` passed. Logs are in `artifacts/local/tests.log`.

Fresh CUDA smoke passed in `runs/local_mobilenet/smoke_verified`: 12 training
and 12 validation samples, finite losses, checkpoints saved, 217.71 MB peak CUDA
allocation (270.53 MB reserved). The resumed bounded run in `smoke_resumed`
reached epoch 2/global step 6 with optimizer, scaler and all RNG states restored,
and retained cosine `T_max=50`; see `resume_verified.json`. Its first epoch came
from the checkpointed initial smoke. The initial smoke's post-run JSON writer
failed on undefined near-field metrics; serialization was corrected to JSON null,
tested, and the fresh smoke above completed successfully. This is software
verification on 12 validation samples, not validation of a trained candidate.

Train/validation predictions were visually checked. They remain near their
initial sub-metre range after the bounded smoke. GT and prediction plots now share
one metre scale, avoiding misleading independent automatic color ranges.

The full-epoch benchmark completed on 2026-10-19. Its 17,635 training frames took
1197.94 seconds (14.72 frames/s) and its 4,410 validation frames took 171.82 seconds
(25.67 frames/s). Total wall time including setup and checkpoint writes was
1371.23 seconds, **22m 51s**. The 50-epoch estimate is **19.04 hours**, or
**22.85 hours with 20% allowance**. This exceeds the original 12-hour gate; the
user subsequently authorized proceeding. No batch/resolution/loss/range change
was made, and no larger-batch experiment was needed after that authorization.

Measured loader waits were 658.83 seconds during training (55.00%) and 141.63
seconds during validation (82.43%): loading stalls are a material bottleneck in
this external-disk/CPU decoding pipeline. These timers do not isolate physical
disk reads from decoding/preprocessing/IPC. No unsupported attribution to the
disk alone is claimed. Sampled GPU temperatures were 52–78°C, with no software
thermal slowdown flag in the 137 collected samples; hardware slowdown reasons
were not queried. Peak CUDA allocation/reservation was 217.71/270.53 MB.
Peak sampled process-tree RSS was 5.74 GB (5.35 GiB), including shared pages.

Benchmark validation, **one epoch only**, with the declared 0.1–1000 m label mask:

| Metric | Value |
|---|---:|
| AbsRel | 0.305814 |
| RMSE (m) | 127.146186 |
| Delta1 / Delta2 / Delta3 | 0.481514 / 0.786652 / 0.902863 |
| Near-field AbsRel (<10 m) | 0.357977 |
| Near-field RMSE (m, <10 m) | 3.978276 |
| Validation loss | 42.214163 |
| Valid resized pixels | 212,903,814 |

Full metrics, timing, resource records, loss curve and the four inspected
train/validation prediction plots are in `runs/local_mobilenet/benchmark`.
`best.pt` and `last.pt` are both readable, 12,661,787 bytes each, epoch 0 (human
epoch 1), global step 4409, cosine horizon 50. They are benchmark checkpoints;
the final run did not resume them. The large far-range RMSE and visibly smooth
predictions remain limitations of this early result. No final test prediction
or performance metric has been generated, and the adaptive scheduler is untuned.

The **main run started at 23:01:49 IST on 2026-10-19**, from seed 42 and official
pretrained encoder weights, in `runs/local_mobilenet/main`. Wrapper/session PID:
**80373**; Python training PID: **80376**. The active systemd sleep inhibitor and
CUDA process were checked. The run is in **epoch 1 of 50**, not complete.
`training.log`, `progress.json`, `resources.jsonl` and `session.json` contain live
status. `metadata.json` records `resume_checkpoint: null`, parameter counts,
source hashes, versions and manifest identity. The first main checkpoints will
exist only after epoch 1 completes; until then, there is no main checkpoint to
resume. Checkpoint/CSV history are updated at each completed epoch. Final selected
validation metrics and plots are generated automatically after training finishes.

Monitor progress without launching another run:

```bash
tail -f runs/local_mobilenet/main/training.log
cat runs/local_mobilenet/main/progress.json
```

Use the resume command above only after the current process has stopped and a
main `last.pt` exists. The detached launcher refuses an existing live wrapper PID.
The exact configured resume command is also saved as:

```bash
bash runs/local_mobilenet/main/resume.sh
```

The working-tree source snapshot, file hashes, base revision, dirty status and
read-only Git diff are saved under `artifacts/local/source_snapshot*`. The source
snapshot's runtime hashes were checked against the launched main run's metadata.
No commit was created. Original archives and the Kaggle dataset package retain
their recorded sizes and modification times.
