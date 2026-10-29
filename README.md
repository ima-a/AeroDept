# AeroDepth-Lite

A reproducible research checkpoint for monocular drone-camera range prediction and causal inference scheduling. The repository includes the actual trained **epoch-4 MobileNetV3-Small depth weights**, a CPU inference path, frozen relative split manifest, validation results, original negative scheduler study, and bounded scheduler-v2 results. A [frozen final-test study](results/final_test/REPORT.md) now reports all five policies on 4,436 held-out frames, after repeated validation timing and split-isolation checks. This is simulation research, not a flight-safety system.

The original final result is mixed: against the fixed validation-budget rule, V2 improves pooled near RMSE by 1.89%, but uses 14.86% more model calls and 26.53% more CPU compute; maximum event-update delay rises from 1 to 4 frames. PLE near RMSE regresses. Model depth errors remain substantial. [Protocol, guarded runner and reproduction](docs/final_study.md). Published training/validation/V1/V2/pilot evidence remains unchanged; that final study performed no retraining or additional policy search.

## Local presentation dashboard

The expanded supervised A and teacher B runs, plus the separately authorized exploratory KD run, are complete. Teacher B failed its original gate; that rejection is preserved. The validation presentation defaults to **supervised A epoch9**, offers **original epoch4** as the near-field reference and **exploratory KD epoch9** as a negative-result comparison. A improves pooled AbsRel11.84% but worsens pooled near RMSE16.18% versus original; it is not a near-field safety upgrade. Published original weights and normal CLI defaults remain unchanged.

```bash
python -m pip install -e '.[dashboard]'
python -m streamlit run scripts/dashboard.py
```

Localhost only, optional Python3.11+ UI; no cloud deployment or automatic training/tuning/test evaluation. [Dashboard, artifacts, guarded validation reproduction and limits](docs/presentation_dashboard.md). Earlier progress snapshots remain dated historical records; the presentation results give the completed status.

The [consolidated final report](docs/FINAL_REPORT.md) brings together the model comparison, negative teacher/KD findings, scheduler ablations and actual timing. Independent reserved final testing is **NOT RUN** pending its eight RGB/depth archives and a verified manifest. The improved dashboard keeps this status explicit. A relocated checkout uses `configs/presentation.json`; local datasets/caches/private KD weights are opt-in environment locations. The two compact inference exports are included in source delivery; private training states remain excluded.

## Expanded teacher/student experiment — completed training

The expanded supervised A and teacher B runs and the separately authorized exploratory KD run are complete. Teacher B failed the original acceptance gate, which remains unchanged. Supervised A epoch9 is the general-depth presentation default; original epoch4 remains the near-field reference; exploratory KD epoch9 retains its negative validation result. [Final student comparison](results/student_validation_comparison_20261029T152648Z/REPORT.md) and [completed presentation validation](results/presentation_validation_20261029T155521Z/REPORT.md). The [27 October progress snapshot](results/expanded_status_20261027T150226Z/REPORT.md) records the earlier teacher-running observation and remains historical.

Start with the [expanded reproduction guide](docs/expanded_reproduction.md): exact dataset layout, approved configurations, numerical-recovery policy, guarded commands, checkpoint/resume requirements and historical source versions. Code, manifests, protocols and compact evidence are included. MidAir inputs and private full-run checkpoints must be acquired or generated separately.

## Install

Run from a fresh clone with Python 3.13. CPU installation:

```bash
git clone --branch main https://github.com/ima-a/AeroDept.git
cd AeroDept
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[dev,midair]' -c requirements-tested.txt
```

For an existing compatible CUDA environment, skip the CPU-wheel command. For a fresh NVIDIA environment, install the matching torch/torchvision pair using the [official selector](https://pytorch.org/get-started/locally/), then install the project. The recorded laptop used torch2.14.1+cu130/torchvision0.29.1+cu130; CPU verification uses the corresponding +cpu builds. `make install` does not explicitly replace CUDA PyTorch with CPU wheels; `make install-cpu` is for a fresh CPU environment. [Installation details](docs/installation.md).

## Dataset-free inference

No MidAir download or ImageNet cache is needed. Supply an RGB image:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 aerodepth infer \
  --weights models/mobilenet_v3_small_epoch4.state_dict.pt \
  --metadata models/mobilenet_v3_small_epoch4.json \
  --image /path/to/image.jpg --output artifacts/prediction.npy --device cpu
```

The FP32 `.npy` output is 192×320 **Euclidean camera range in metres**. RGB is resized over the full frame, without a crop; ImageNet normalization occurs inside the model exactly once. Predictions have a 0.1m floor and no upper clamp. Out-of-domain images can produce poor depth estimates. [Model metadata and notices](models/README.md).

A fast, dataset-free software verification creates clearly labelled **synthetic** images, trains a tiny synthetic model, checks replay/reporting, and runs the published MobileNet on one synthetic RGB:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/smoke.py --output artifacts/synthetic_smoke
# Dataset-free example image created by the smoke workflow:
aerodepth infer --image artifacts/synthetic_smoke/fixture/environment_0/trajectory_0000/rgb/frame_000000.JPEG \
  --output artifacts/example_prediction.npy --device cpu
```

Synthetic outputs verify software, not research accuracy. Choose a new output directory when repeating a workflow; existing evidence is preserved.

## Research reproduction

Acquire **left RGB, depth/range and sensor HDF5** through the [official MidAir download form](https://midair.ulg.ac.be/download.html), accept its terms and extract in the original layout. [Exact subset, IDs and directory structure](docs/dataset.md). The full research manifest is compressed under `results/expanded_validation/manifest.json.gz`; decompression preserves its SHA-256 identity.

```bash
export MIDAIR_ROOT='/path/to/MidAir'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/reproduce.py \
  --dataset-root "$MIDAIR_ROOT" --study validation --device cpu --output artifacts/validation_repeat
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/reproduce.py \
  --dataset-root "$MIDAIR_ROOT" --study scheduler-v1 --device cuda --output artifacts/scheduler_v1_repeat
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/reproduce.py \
  --dataset-root "$MIDAIR_ROOT" --study scheduler-v2 --device cuda --output artifacts/scheduler_v2_repeat
```

These **public, versioned** runners use the supplied inference export, restrict image/depth reads to validation, preserve existing outputs, check agreement with the audited FP32 reference, and separate cached search from actual policy timing. They need the same dataset/manifest; reproducing research metrics is different from standalone inference. Plan roughly 3GB output space per run plus dependencies. GPU/CPU versions, precision and hardware affect exact numerical reproduction and timing. [Protocol, bounded plans and commands](docs/reproduction.md).

## Recorded findings

All 50 training epochs completed in **16h28m on GTX1650**. Best epoch4 was selected by recorded AMP validation AbsRel; the audited FP32 validation AbsRel is **0.301540**, RMSE **127.335m**, near-field RMSE **3.362m**. This model overfits after epoch4, misses far-depth geometry and has local floor saturation, particularly problematic alongside foggy-scene errors.

Original scheduler: all 4,410 calls because its mixed-unit motion score saturates. Its tuned choice exactly equals fixed-2 on both trajectories and adds overhead. V2 removes privileged velocity, separately scales held simulated GPS and IMU proxies, and intentionally compares RGB with the last inferred frame. Its eight-point validation search selects 2,455 calls (44.33% skipped). Compared with exact call-matched fixed scheduling, near RMSE improves 2.25% on Kite and 8.77% on PLE, but Kite overall errors worsen slightly and PLE maximum event-update delay increases 1→2 frames. This is a **mixed, exploratory validation result**, not uniform adaptive superiority or independent generalization. No energy/battery improvement was measured.

[Training/validation results](results/expanded_validation/REPORT.md) · [Original scheduler](results/scheduler_v1/REPORT.md) · [V2](results/scheduler_v2/REPORT.md) · [Historical custom-CNN pilot](evidence/real_midair_pilot/README.md). The pilot used different data/protocol and is not a fair expanded-data architecture comparison. Only the compatible expanded-data MobileNet weights are published.

## Training and resume

For a new run matching the recorded ImageNet-initialized training protocol:

```bash
python -c "import gzip,pathlib; p=pathlib.Path('artifacts/frozen_manifest.json'); p.parent.mkdir(exist_ok=True); p.write_bytes(gzip.decompress(pathlib.Path('results/expanded_validation/manifest.json.gz').read_bytes()))"
aerodepth train --config models/training_config.yaml --dataset-root "$MIDAIR_ROOT" \
  --manifest artifacts/frozen_manifest.json --output runs/new_mobilenet
# Resume requires a trusted FULL training checkpoint, not the published state_dict:
aerodepth train --config models/training_config.yaml --dataset-root "$MIDAIR_ROOT" \
  --manifest artifacts/frozen_manifest.json --output runs/resumed_mobilenet --resume /path/to/private/last.pt
```

A fresh training run initializes the encoder from ImageNet and downloads those weights. For literal random initialization, create a new config with `model.pretrained: false`; this is a different, unmeasured experiment. Resume of the original run needs its private full `last.pt` and adjacent prior `best.pt`, including optimizer/scheduler/AMP/RNG state. Allow about 16.5 h on the recorded 4 GB GTX1650, plus setup/I/O; timings are hardware dependent. CPU is supported for inference/synthetic checks, not recommended for this 50-epoch workload. [Historical laptop workflow](docs/local_training.md) and [optional Kaggle notebook](notebooks/kaggle_train.ipynb) remain available.

```text
src/aerodepth_lite/  implementation, inference and public research runner
configs/            portable candidate and synthetic configurations
scripts/            export, reproduction and synthetic workflow entry points
tests/              automated checks
notebooks/          optional maintained Kaggle runner
models/             trained inference weights, frozen config and metadata
results/            historical studies, expanded manifests/protocols and dated evidence
docs/               installation, dataset, methodology and publication record
evidence/           distinct historical pilot
# datasets/, runs/, artifacts/, environments and private tooling stay local/ignored
```

See [NOTICE.md](NOTICE.md) for dataset attribution and upstream source/weight terms. Broader independent trajectories and actual deployment/energy investigation remain future work. The completed final study supports no measured energy, battery or flight-safety improvement.

The [bounded HDD/USB acquisition inventory](results/acquisition_publication_20261029T180214Z/REPORT.md) now checks the formerly unmounted HDD read-only as well. All8 reserved RGB/depth streams are absent in the documented locations and original Kaggle package; the HDD hardware failure remains unresolved. No new-test evaluation was run.
