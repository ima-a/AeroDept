# Reproduce the expanded teacher/student workflow

This is a separate, ongoing experiment. At the [27 October snapshot](../results/expanded_status_20261027T150226Z/REPORT.md), supervised MobileNet finished 10 epochs, ResNet-50 was training, and the teacher gate and KD had not started. There is no completed distillation comparison or new test result. Snapshot files are dated observations, not live telemetry.

The historical final study is complete and remains quarantined. Its source-identical checkout and dataset-free saved-result verification are documented in [final study](final_study.md). The original validation/V1/V2 reproduction commands remain in [reproduction](reproduction.md).

## Install and choose the correct inputs

Clone `main`, follow [installation](installation.md), and record `git rev-parse HEAD`. Use the pinned versions in `requirements-tested.txt`. The recorded laptop uses Python 3.13 and CUDA PyTorch; this publication reuses that environment for focused CPU checks, rather than claiming a fresh dependency installation. Runtime/hardware changes can change numerical results. Do not modify code, dependencies or configurations during a run or between arms.

Run expanded commands from the repository root with `PYTHONPATH="$PWD/src"`, as shown below. This runner locates the frozen manifest, reference metrics and source identities relative to its source checkout; a non-editable site-packages installation alone does not provide that layout. The explicit path also prevents accidentally importing another installed revision.

The public package contains the original epoch-4 inference weights, source, tests, expanded manifest/reservations, approved configurations, protocols and compact verification evidence. Datasets, downloaded ImageNet initialization, environments, full training states, active logs and the laptop's user service stay private. The small failed-launch log and startup observation published with their receipts are historical evidence, not active logs. These commands reproduce new runs; they do not download or resume the laptop's private checkpoints.

Obtain left RGB and depth/range for the exact **20 training and two validation realizations** in [the verified trajectory table](../results/expanded_preparation_20261026/REPORT.md#verified-realizations-and-page-ids). That table maps official form IDs to internal climate-prefixed IDs and counts. The [original subset documentation](dataset.md) and [expansion checklist](expanded_experiment_20261025/PLAN.md) explain acquisition; their original-test/new-test selections are not needed for this TRAIN/VAL workflow. Corresponding climate HDF5 records support independent preparation checks; the training runner uses the frozen manifest's metadata and does not open sensor archives/HDF5. Respect [MidAir attribution and terms](../NOTICE.md).

Use a real directory, with no dataset symlinks. Set your own common root while retaining these relative locations:

```text
<MIDAIR_EXPANDED_ROOT>/
  Kite_training/<climate>/{color_left,depth}/trajectory_<ID>/<frame>.<JPEG|PNG>
  PLE_training/<season>/{color_left,depth}/trajectory_<ID>/<frame>.<JPEG|PNG>
  MidAir-expansion/training_staging_20261026T000911/MidAir/
    Kite_training/<climate>/{color_left,depth}/trajectory_<ID>/<frame>.<JPEG|PNG>
    PLE_training/<season>/{color_left,depth}/trajectory_<ID>/<frame>.<JPEG|PNG>
```

The first two locations contain original TRAIN/VAL inputs; the nested staging location contains the 12 added realizations. The frozen manifest specifies which location each pair uses. Inspect its relative records, not a recursive dataset discovery. Independently acquired inputs must be arranged at those exact relative paths without overwriting an existing dataset. Changing the staging prefix changes manifest identity and is rejected. Keep the original manifest unchanged. The full local archive/extraction audit script and per-member extraction index are private provenance; this publication supplies the resulting manifest and integrity/validation receipts, not a portable acquisition/extraction program.

| Assignment | Independent path groups | Realizations | Frames |
|---|---:|---:|---:|
| Training | 12 | 20 | 44,098 |
| Unchanged validation | 2 | 2 | 4,410 |
| New test, reserved and unacquired at snapshot | 4 | 4 reserved | 8,820 expected |
| Original completed test, quarantined | 2 | 2 historical | 4,436 historical |

All weather equivalents of validation and both test sets are excluded from training. Pilot-exposed paths cannot become new test. There are no test image records in the expanded manifest; do not acquire/preview/evaluate test inputs for this workflow. Both models predict Euclidean camera range in metres with the same masks, preprocessing and metrics described in [the training contract](expanded_training.md).

## Frozen identities and approved settings

The current [recovery protocol](../results/expanded_relaunch_20261026T114024/protocol.json) retains the original approved scientific settings. Its [freeze receipt](../results/expanded_relaunch_20261026T114024/freeze.json) records all seven core source hashes; this published workflow matches those hashes. Absolute HDD paths, UUIDs and service identities in receipts describe the original laptop and must not be treated as portable launch settings.

Use the approved configuration copies in `results/expanded_launch_20261026T104420/configs/`. They are byte-identical to the recovery launch's active configurations. `configs/expanded_*.yaml` preserve the original proposals: their numerical settings are identical, but their `protocol_status` differs, which matters to strict checkpoint identity. Do not switch between those copies on resume.

| Setting | Supervised MobileNet | ResNet-50 teacher | KD MobileNet |
|---|---|---|---|
| Initialization | Original epoch 4, including BN | ImageNet1K V2 encoder; new range decoder | Independently from identical original epoch 4 |
| Epochs / successful updates | 10 / 220,490 | 20 / 440,980 | 10 / 220,490, conditional |
| Physical/effective batch | 2 / 2 | 2 / 2 | 2 / 2 |
| AdamW LR | 1e-4 | Encoder 1e-5; decoder 1e-4 | 1e-4 |
| Fixed cosine horizon / minimum LR | 10 / 1e-6 | 20 / 1e-6 | 10 / 1e-6 |
| KD coefficient | 0 | 0 | 0.2 |

All arms retain seed 42, weight decay 1e-4, clip norm 1, no accumulation/early stopping/search, and FP32 validation every epoch with minimum pooled AbsRel checkpoint selection. Student arms have fresh matched optimizer/scheduler/scaler/history/RNG state, training order and paired flips. KD never initializes from the newly trained supervised checkpoint. Teacher inference during KD is frozen FP32 eval/no_grad on the same augmented RGB. [Numerical recovery](expanded_numerical_recovery.md) retries the same batch after scaler backoff and BN/RNG restoration; each accepted batch advances exactly once, with at most eight retries before stopping.

Manifest uncompressed SHA-256: `67c6cad39767d847a2087a81d1ecc4de2ad8c9ecc0fa01e491f3950b25f8dd3a`.

Epoch-4 initialization SHA-256: `b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6`.

Teacher ImageNet V2 initialization SHA-256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. TorchVision acquires it on first teacher initialization if absent; full-checkpoint resume does not redownload it. It is not a pretrained depth teacher until trained on this experiment's TRAIN inputs.

## Check setup without image reads

From the repository root, after installing this checkout:

```bash
export MIDAIR_EXPANDED_ROOT='/path/to/prepared/MidAir'
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$PWD/src"
python -B - <<'PY'
import hashlib, os
from pathlib import Path
from aerodepth_lite.expanded import INITIAL_SHA, PREP, check_record, load_inputs
root = Path(os.environ['MIDAIR_EXPANDED_ROOT']).absolute()
records, manifest = load_inputs(PREP/'manifest_train_val.json.gz', PREP/'reservations.json', root)
assert hashlib.sha256(Path('models/mobilenet_v3_small_epoch4.state_dict.pt').read_bytes()).hexdigest() == INITIAL_SHA
for split, rows in records.items():
    for row in rows:
        check_record(row, split, root)
        for name in (row.rgb_path, row.depth_path):
            path = root/name
            assert path.is_file() and path.stat().st_size > 0, name
    print(split, len(rows), 'pairs present; metadata/path checks only')
PY
```

Presence does not prove decoding/integrity. The recorded preparation receipts cover the original acquisition; independently acquired archives need their own integrity and explicit TRAIN/VAL validation. Do not use a generic discovery/split command. The implemented runner installs its allowlist/read guards before image access and rejects changed manifest/reservation identities.

Dataset-free focused software checks are:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B -m pytest -q tests/test_expanded.py tests/test_final_study.py
```

These tests use synthetic inputs, published weights and manifest metadata, not MidAir test inputs. Existing bounded TRAIN-only numerical/CUDA checks and their limitations are recorded in [workflow verification](../results/expanded_workflow_20261026/REPORT.md) and [recovery verification](../results/expanded_numerics_20261026T110550/REPORT.md). They are distinct from full training accuracy evidence.

## Launch a new independent run

The following Bash block runs the supported commands sequentially on a separate prepared machine. It stops on any command error and explicitly stops if the saved teacher gate fails. Do not run it alongside the laptop's active experiment. Choose a fresh output parent on a reliable disk, outside the dataset; existing directories are refused. Check CUDA, free output space, AC power and dataset availability first.

`systemd-inhibit` below is a Linux sleep inhibitor while each command runs; this shell alone does not survive terminal disconnection. For unattended use, run it in a persistent session or user service with logging and an inhibitor covering the full sequence. The laptop's [recorded service and limitations](../results/expanded_relaunch_20261026T114024/HANDOFF.md) are evidence, not an installable service for another host. Lid behavior must be verified locally; the original machine's physical lid behavior was not tested and full logout may stop its user service.

```bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$PWD/src"
export TORCH_HOME="$PWD/artifacts/local/torch-cache"
export MIDAIR_EXPANDED_ROOT='/path/to/prepared/MidAir'
EXPANDED_OUT='/path/to/outputs/new-expanded-run'
EXPANDED_PY="$PWD/.venv/bin/python"
CONFIG_DIR='results/expanded_launch_20261026T104420/configs'
MANIFEST='results/expanded_preparation_20261026/manifest_train_val.json.gz'
RESERVATIONS='results/expanded_preparation_20261026/reservations.json'
mkdir "$EXPANDED_OUT"

systemd-inhibit --what=sleep --mode=block --why='Expanded supervised run' -- \
  "$EXPANDED_PY" -u -B scripts/expanded_training.py train --device cuda \
  --config "$CONFIG_DIR/expanded_supervised.yaml" --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --manifest "$MANIFEST" --reservations "$RESERVATIONS" \
  --initialize models/mobilenet_v3_small_epoch4.state_dict.pt \
  --output "$EXPANDED_OUT/A_supervised" --confirm-full-training

systemd-inhibit --what=sleep --mode=block --why='Expanded teacher run' -- \
  "$EXPANDED_PY" -u -B scripts/expanded_training.py train --device cuda \
  --config "$CONFIG_DIR/expanded_teacher.yaml" --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --manifest "$MANIFEST" --reservations "$RESERVATIONS" \
  --output "$EXPANDED_OUT/B_teacher" --confirm-full-training

"$EXPANDED_PY" -B scripts/expanded_training.py gate \
  --baseline "$EXPANDED_OUT/A_supervised/best.pt" \
  --teacher "$EXPANDED_OUT/B_teacher/best.pt" --output "$EXPANDED_OUT/teacher_gate.json"
"$EXPANDED_PY" -B -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else "Teacher gate failed; stop before KD")' \
  "$EXPANDED_OUT/teacher_gate.json"
read -r TEACHER_SHA _ < <(sha256sum "$EXPANDED_OUT/B_teacher/best.pt")

systemd-inhibit --what=sleep --mode=block --why='Expanded KD run' -- \
  "$EXPANDED_PY" -u -B scripts/expanded_training.py train --device cuda \
  --config "$CONFIG_DIR/expanded_kd.yaml" --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --manifest "$MANIFEST" --reservations "$RESERVATIONS" \
  --initialize models/mobilenet_v3_small_epoch4.state_dict.pt \
  --teacher "$EXPANDED_OUT/B_teacher/best.pt" --teacher-sha256 "$TEACHER_SHA" \
  --gate "$EXPANDED_OUT/teacher_gate.json" \
  --output "$EXPANDED_OUT/D_kd" --confirm-full-training
```

The gate requires strictly lower teacher pooled validation AbsRel and, on **each** unchanged validation path, finite overall/near AbsRel and RMSE no worse than the supervised baseline, with equal valid/near counts and no tolerance. A better pooled score alone does not pass. KD recomputes and validates the gate and its source/checkpoint identities. Failure means stop, not tune/retry until acceptance.

`scripts/expanded_sequence.py` is the guarded supervisor used by the laptop's frozen bundle: it verifies source/input hashes, disk identity, full completion budgets, required checkpoints and the gate between stages. It requires a separately prepared bundle with `frozen/`, `freeze.json`, `freeze.sha256`, logs/completion directories and host-specific disk metadata; cloning this repository is not sufficient to invoke it. The portable per-arm sequence above uses the same verified training/gate implementation without pretending the private host bundle is portable.

## Stop and resume

Preserve the complete output directory. Resume only from that arm's own `last.pt`, using unchanged source, approved configuration and compatible runtime. Keep adjacent `best.pt` and `previous_best.pt` when present; for KD retain the baseline best, teacher best, gate and their hashes as well. The checkpoint contains optimizer/scheduler/scaler/history/RNG and exact identity. An incomplete epoch replays from the previous completed boundary. No checkpoint exists before the first epoch completes; an interrupted attempt without `last.pt` needs a fresh output directory.

For the supervised arm, replace its launch command with:

```bash
systemd-inhibit --what=sleep --mode=block --why='Resume expanded supervised run' -- \
  "$EXPANDED_PY" -u -B scripts/expanded_training.py train --device cuda \
  --config "$CONFIG_DIR/expanded_supervised.yaml" --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --manifest "$MANIFEST" --reservations "$RESERVATIONS" \
  --resume "$EXPANDED_OUT/A_supervised/last.pt" \
  --output "$EXPANDED_OUT/A_supervised" --confirm-full-training
```

Teacher resume uses its own approved config/output/last checkpoint. KD resume uses its own last checkpoint instead of `--initialize`, while retaining its original teacher/hash/gate arguments. Do not rerun `mkdir`, retrain completed arms, change budgets or replace a teacher to bypass resume identity. For the active laptop only, the frozen `stop.sh`/`resume.sh` in its recorded handoff control the entire sequence; do not substitute these portable example paths.

Historical reports, hashes and exposure ledgers remain unchanged. The prepublication protocol wording is retained at commit `21a73a2`; the initial failed launch and its numerical diagnosis have separate receipts. Public progress snapshots do not contain or substitute for private resumable checkpoints. Expanded final-test acquisition/evaluation requires a separate future protocol.
