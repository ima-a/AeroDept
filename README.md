# AeroDepth-Lite

Compact monocular drone-camera **range prediction** with causal inference scheduling. The current demonstration uses supervised **MobileNetV3-Small epoch 9 (A)**; the original epoch-4 model is retained as the near-field reference. Both inference exports are included. Python code, frozen manifests, saved results, tests and presentation files are organized below.

A improves pooled validation AbsRel by **11.84%**, but near-field RMSE worsens **16.18%**. Exploratory distillation with the rejected ResNet-50 teacher underperforms A on validation. The eight reserved test archives are verified; the single frozen **8,820-frame independent study is completed and verified**. See [its protocol and reproduction guide](docs/independent_final_test.md). This is simulation research, with no measured flight-safety, energy or battery benefit.

[Project report](docs/FINAL_REPORT.md) · [Documentation index](docs/README.md) · [Viva code guide](docs/VIVA_GUIDE.md) · [PowerPoint](presentations/aerodepth_lite/AeroDepth_Lite_Project_Presentation.pptx)

## Run the project

On the prepared laptop, use the existing environment:

```bash
. .venv/bin/activate
python -m streamlit run scripts/dashboard.py
```

The dashboard uses `configs/presentation_local.json` when present, otherwise the portable `configs/presentation.json`. Saved comparison tables and curves are available without a dataset. Real validation images, playback caches and the exploratory KD checkpoint require the explicitly configured external inputs. Missing inputs are shown as unavailable. [Dashboard instructions](docs/presentation_dashboard.md).

For a fresh source checkout, use Python 3.13 and the recorded CPU dependency pair:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,midair,dashboard]' -c requirements-tested.txt
```

Keep an existing compatible CUDA environment when you need GPU execution. See [installation](docs/installation.md) for the recorded CUDA versions and hardware caveats.

## Predict range from your own image

This command uses the A inference export, without MidAir, private checkpoints or an ImageNet download:

```bash
aerodepth infer \
  --weights models/presentation_A_epoch9_20261029T155521Z/supervised_A_epoch9.state_dict.pt \
  --metadata models/presentation_A_epoch9_20261029T155521Z/supervised_A_epoch9.json \
  --image /path/to/image.jpg --output artifacts/prediction_A.npy --device cpu
```

Output: a 192 × 320 FP32 `.npy` map of **Euclidean camera range in metres**. RGB is resized over the whole frame; ImageNet normalization occurs once inside the model. Predictions have a 0.1 m floor and no upper clamp. Generalization outside the simulated data has not been established. The CLI still defaults to the original epoch-4 export if weight arguments are omitted.

## Directory map

```text
src/aerodepth_lite/  current implementation; one-word code section markers
scripts/            dashboard, inference/reproduction and training entry points
configs/            portable settings; presentation_local.json stays private
models/             two compact inference exports, metadata and hashes
results/            canonical saved research evidence and frozen protocols
  student_validation_comparison_20261029T152648Z/  original / A / KD comparison
  presentation_validation_20261029T155521Z/        A scheduler, timing and curves
  expanded_preparation_20261026/                  expanded split and reservations
  final_test/                                    historical original-model test
  ...                                            dated supporting evidence

evidence/           historical diagnostic source snapshots and pilot evidence
docs/               current guides, report, viva map and maintenance receipts
presentations/      editable 13-slide PowerPoint and preview
notebooks/          optional Kaggle workflow
tests/              dataset-free software checks; optional data-dependent checks

# Local only / ignored:
datasets/           retained extracted pilot RGB/range and sensor files
runs/local_mobilenet/main/  original full training checkpoints and history
artifacts/
  local/torch-cache/                           shared initialization weights
  student_validation_comparison_20261029T152648Z/  dashboard example cache
  archive/2026-10-30/                          retained historical local records
.venv/              working Python environment
```

Result paths stay stable because code, protocols and figure references depend on them. A PNG in `results/` may be scientific evidence; file extension alone does not make it disposable. Raw data, private training states, environments and local archives are excluded from source delivery. [Storage and cleanup details](docs/archive_locations.md).

## Dataset and recorded results

The custom expanded MidAir subset has **44,098 training frames** across 12 independent paths / 20 climate realizations and **4,410 validation frames** across 2 paths. Weather variants of a path stay grouped. The new independent test contains 4 reserved paths / 8,820 verified frames, isolated by underlying path from training, validation, pilot and historical test exposure. The older completed 4,436-frame test belongs to the original model and remains quarantined.

| Student | Validation AbsRel ↓ | RMSE (m) ↓ | Near RMSE (m) ↓ |
|---|---:|---:|---:|
| Original epoch 4 | 0.301540 | 127.335 | 3.362 |
| Supervised A epoch 9 | 0.265831 | 119.944 | 3.906 |
| Exploratory KD epoch 9 | 0.270049 | 121.207 | 3.961 |

These are familiar validation paths used for development and checkpoint selection. The original model used a different training subset/budget. Matched A/KD comparisons cover one seed. Near pixels are heavily concentrated in PLE. [Full student comparison](results/student_validation_comparison_20261029T152648Z/REPORT.md).

A's frozen V2 scheduler uses 2,456 calls on validation, the same pooled count as the fixed-budget rule. Pooled near RMSE improves 9.14%, while overall error slightly worsens and maximum event-update delay rises from 1 to 2 frames. Risk caching reduces scheduler work without a robust measured end-to-end speedup. [Scheduler and timing evidence](results/presentation_validation_20261029T155521Z/REPORT.md).

## Reproduce or verify

- **Software checks, no MidAir needed:**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -B -m pytest -q -p no:cacheprovider -m 'not integration'
  ```

- **Dataset-free end-to-end synthetic check:** `python scripts/smoke.py --output artifacts/new_synthetic_smoke`. Generated fixtures test software, not scientific accuracy.
- **Original validation and scheduler reproduction:** [reproduction guide](docs/reproduction.md); requires separately acquired validation data and a fresh output directory.
- **Expanded training and historical resume:** [expanded reproduction](docs/expanded_reproduction.md). Source hashes are strict: use the frozen revision for old checkpoints.
- **A validation replay and timing:** [dashboard reproduction](docs/presentation_dashboard.md); use a new config/protocol/output namespace.
- **Historical final-study accounting:** [frozen study guide](docs/final_study.md), using its exact historical source checkout. Saved-result checking does not rerun test inputs.

On 30 October 2026, one-word navigation comments were added to the current implementation. Python syntax trees were checked against the preceding sources; calculation logic is unchanged, but file hashes differ. Historical source seals and results are retained, not rewritten. [Marker receipt](docs/maintenance/2026-10-30_code_markers.json).

Dataset acquisition, attribution and formats: [dataset guide](docs/dataset.md) and [NOTICE.md](NOTICE.md). The current [final report](docs/FINAL_REPORT.md) explains the negative findings, exposure controls and independent-test protocol and outcome.

Read-only independent-study audit after completion: `python scripts/verify_independent_test.py`. Classroom code locations: [CODE_GUIDE.md](CODE_GUIDE.md).

On the independent test, A versus Original changes pooled AbsRel by -1.82% and near RMSE by +1.41%. KD versus A changes pooled AbsRel by -0.13% and near RMSE by +1.80%. Negative changes mean lower error. These new outcomes do not erase the negative validation findings or the failed teacher gate; A remains the preselected demonstration model.
