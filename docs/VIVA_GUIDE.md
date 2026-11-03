# Code to show during the viva

Checked on **3 November 2026** against runtime-code revision `c24aa6a`, including the published KD/ResNet exports and the validation-only teacher dashboard changes. Search for the comment markers or function names in the tables. `model.py` defines the architectures; the shared training loop is in `expanded.py`. ResNet training does not happen inside the model class.

## Current models and their roles

| Model | Architecture / parameters | Selected epoch | What to explain |
|---|---|---:|---|
| Original | MobileNetV3-Small / 1,027,489 | 4 | Reference model; trained for 50 epochs on the earlier 17,635-frame subset. |
| Supervised A | MobileNetV3-Small / 1,027,489 | 9 | Default demonstration model; 10 expanded-training epochs. |
| Exploratory KD | MobileNetV3-Small / 1,027,489 | 9 | Same student architecture, trained for 10 epochs with the rejected ResNet teacher. KD is not a ResNet model. |
| ResNet-50 teacher | ResNet-50 V1.5 plus range decoder / 23,999,633 | 3 | Trained for 20 epochs; failed the teacher gate. Its dashboard metrics are validation-only. |

All four have [published inference weights, metadata and checksums](../models/README.md). The KD and teacher exports match their exact selected checkpoints and were checked with synthetic CPU FP32 parity inputs. KD inference needs no teacher. These exports contain parameters and buffers, not full training/resume state. Raw datasets and playback caches remain separate inputs.

## Models and learning

| If asked to show… | File | Search for | What to point out |
|---|---|---|---|
| Student architecture | [model.py](../src/aerodepth_lite/model.py) | `# MobileNet` | `MobileNetV3SmallDepth`: feature encoder, projection and four skip-decoder blocks. |
| Teacher architecture | [model.py](../src/aerodepth_lite/model.py) | `# ResNet` | `ResNet50Depth`: ResNet-50 V1.5 encoder and projected skips. |
| Small pilot model | [model.py](../src/aerodepth_lite/model.py) | `# Tiny` | Historical/synthetic `AeroDepthTiny`; not the current A model. |
| Decoder / skip connections | [model.py](../src/aerodepth_lite/model.py) | `# Decoder`, `# Skips`, `# Upsampling` | Bilinear resizing and concatenation with encoder features. |
| Input normalization | [model.py](../src/aerodepth_lite/model.py) | `# Normalization` | ImageNet mean/std inside the model, exactly once. |
| Positive range output | [model.py](../src/aerodepth_lite/model.py) | `softplus` | One-channel output plus 0.1 m floor; no upper output clamp. |
| Parameter count | [model.py](../src/aerodepth_lite/model.py) | `# Parameters` | Count tensor elements. Each student has 1,027,489; the teacher has 23,999,633. |
| ResNet training setup | [expanded.py](../src/aerodepth_lite/expanded.py) | `# ResNet` | Teacher arm uses ImageNet V2 initialization; all arms use the shared `Experiment`. |
| Training step | [expanded.py](../src/aerodepth_lite/expanded.py) | `# Training` | `step()` performs prediction, loss, backward pass and update; `full_training()` controls epochs. |
| Optimizer | [expanded.py](../src/aerodepth_lite/expanded.py) | `# AdamW` | Weight decay and separate teacher encoder/decoder learning rates. |
| Learning-rate schedule | [expanded.py](../src/aerodepth_lite/expanded.py) | `# Cosine` | Fixed cosine horizon/minimum; not an inference scheduler. |
| Gradient update | [expanded.py](../src/aerodepth_lite/expanded.py) | `# Backpropagation`, `# Clipping` | AMP scaler, finite-gradient checks and norm clipping. |
| Validation / selection | [expanded.py](../src/aerodepth_lite/expanded.py) | `# Validation`, `# Training` | FP32 evaluation; `full_training()` selects minimum pooled validation AbsRel. |
| Checkpoints | [expanded.py](../src/aerodepth_lite/expanded.py) | `# Checkpoint` | Weights plus optimizer, schedule, scaler, RNG, history and strict identities. |
| Original training | [train.py](../src/aerodepth_lite/train.py) | `# Training`, `# Selection` | Original epoch loop and validation-based checkpoint selection. |

## Losses, metrics and data

| If asked to show… | File | Search for | What to point out |
|---|---|---|---|
| Supervised objective | [losses.py](../src/aerodepth_lite/losses.py) | `# L1`, `# Log`, `# Gradient`, `# Weights` | Range L1 + 0.2 absolute log-range + 0.1 gradient L1. Near-loss weight is zero in the recorded expanded run. |
| Distillation objective | [losses.py](../src/aerodepth_lite/losses.py) | `# Distillation` | Masked absolute log-range consistency; teacher is detached. This is regression KD, not classification logits. |
| KD in training | [expanded.py](../src/aerodepth_lite/expanded.py) | `# KD` | Add `kd_weight × consistency` to supervised loss. Recorded coefficient: 0.2. |
| Teacher quality gate | [expanded.py](../src/aerodepth_lite/expanded.py) | `# Gate` | Better pooled AbsRel plus no worse overall/near error on either validation path. Teacher B failed. |
| Rejected-teacher experiment | [exploratory_kd.py](../src/aerodepth_lite/exploratory_kd.py) | `# Exploration`, `# Lineage` | Separate explicit experimental lineage; original failed gate stays failed. |
| AbsRel | [metrics.py](../src/aerodepth_lite/metrics.py) | `# AbsRel` | Sum `abs(pred−GT)/GT`, then divide by total valid pixels. |
| RMSE | [metrics.py](../src/aerodepth_lite/metrics.py) | `# RMSE`, `# Pooling` | Sum squared error, divide by pooled valid count, take square root. |
| Delta accuracy | [metrics.py](../src/aerodepth_lite/metrics.py) | `# Delta` | Fraction with `max(pred/GT, GT/pred) < 1.25**k`, for k=1,2,3. |
| Near-field evaluation | [metrics.py](../src/aerodepth_lite/metrics.py) | `# Near` | Valid GT below 10 m; pooled independently. |
| Depth format | [data.py](../src/aerodepth_lite/data.py) | `# Decoding` | uint16 PNG bits reinterpreted as float16 Euclidean range. |
| Invalid data / resize | [data.py](../src/aerodepth_lite/data.py) | `# Mask`, `# Resizing` | Finite GT in [0.1,1000] m; RGB bilinear, GT/mask nearest. |
| Augmentation | [data.py](../src/aerodepth_lite/data.py) | `# Augmentation` | Paired horizontal flips only during training. |
| Actual expanded split | [expanded.py](../src/aerodepth_lite/expanded.py) | `# Splits`, `# Leakage` | Loads the frozen allowlist/reservations and guards reads. The generic splitter in `data.py` is not used to redraw this study. |

Accuracy/precision/recall/F1 and confusion matrices from classification are not interchangeable with these depth-regression metrics. Delta is a thresholded ratio metric, not object-detection accuracy.

## Scheduler and demonstration

| If asked to show… | File | Search for | What to point out |
|---|---|---|---|
| Inference scheduling | [presentation_scheduler.py](../src/aerodepth_lite/presentation_scheduler.py) | `# Scheduler`, `# Decision` | First frame, max age, risk/motion/visual triggers, weighted score, then infer or reuse. |
| Motion signals | [presentation_scheduler.py](../src/aerodepth_lite/presentation_scheduler.py) | `# Motion` | Simulated IMU and held GPS velocity; no privileged ground-truth velocity. |
| Visual change | [presentation_scheduler.py](../src/aerodepth_lite/presentation_scheduler.py) | `# Visual` | Current thumbnail versus last inferred thumbnail. |
| Risk definition | [scheduler.py](../src/aerodepth_lite/scheduler.py) | `# Risk` | Risk from the previous predicted range, not current GT. |
| Risk cache | [presentation_scheduler.py](../src/aerodepth_lite/presentation_scheduler.py) | `# Cache` | Refresh only for a fresh prediction token; age remains separate. |
| Baselines / ablations | [presentation_scheduler.py](../src/aerodepth_lite/presentation_scheduler.py) | `# Baselines`, `# Ablation` | Fixed schedules and remove-one-signal variants. |
| GPS alignment | [presentation.py](../src/aerodepth_lite/presentation.py) | `# Sensors` | Causal held simulated GPS aligned to camera frames. |
| Replay versus timing | [presentation.py](../src/aerodepth_lite/presentation.py) | `# Replay`, `# Timing` | Cached replay scores quality; actual timing executes model calls. |
| Standalone inference | [inference.py](../src/aerodepth_lite/inference.py) | `# Weights`, `# Preprocessing`, `# Inference` | Verify weights, resize RGB, run eval-mode prediction and save metres. |
| Dashboard pages | [dashboard_views.py](../src/aerodepth_lite/dashboard_views.py) | `# Results`, `# Comparison`, `# Viewer`, `# Scheduler`, `# Artifacts` | Five research pages backed by saved evidence; entry point is `scripts/dashboard.py`. |

## Recent dashboard changes and exact code locations

| If asked to show… | File | Search for | What to point out |
|---|---|---|---|
| Which models appear in each study | [dashboard_data.py](../src/aerodepth_lite/dashboard_data.py) | `study_bundle`, `models` | Validation contains Original/A/KD/Teacher; independent test contains Original/A/KD; historical test contains Original only. |
| ResNet validation metrics | [dashboard_data.py](../src/aerodepth_lite/dashboard_data.py) | `teacher_context_reused_no_inference` | Reuses pooled and per-path metrics from the existing audit; no new teacher evaluation. Unknown teacher call/timing values stay unavailable. |
| Failed-gate explanation | [dashboard_views.py](../src/aerodepth_lite/dashboard_views.py) | `teacher_gate_panel` | Uses the saved `teacher_gate_passed` verdict and shows Kite near-field regressions against A. The verification record's `passed` field means the audit passed, not that the teacher was accepted. |
| Model table and comparisons | [dashboard_views.py](../src/aerodepth_lite/dashboard_views.py) and [dashboard_data.py](../src/aerodepth_lite/dashboard_data.py) | `comparison_page`, `relative_change` | Overall and near error, per-trajectory scope, and differences against a selected reference. Lower error is better. |
| Viewer model choices and frame persistence | [dashboard_views.py](../src/aerodepth_lite/dashboard_views.py) | `viewer_page`, `viewer_position`, `seek` | Viewer supports the three students; selected frames survive navigation. Recorded playback is distinct from inference. |
| Explicit forward pass | [presentation_ui.py](../src/aerodepth_lite/presentation_ui.py) | `actual_inference` | Requires the button and allowed validation inputs; model/BatchNorm state is checked. No automatic training or test evaluation. |
| Portable model paths | [dashboard.json](../configs/dashboard.json) and [dashboard.py](../scripts/dashboard.py) | `models`, `default_model`, `dashboard_local.json` | Current dashboard uses four exported models and keeps A as default. Local data/cache locations are ignored configuration. |
| Config resolution | [presentation_ui.py](../src/aerodepth_lite/presentation_ui.py) | `load_dashboard_config` | Checkout-relative model paths plus explicit external data/cache locations. Missing inputs are unavailable, not discovered automatically. |
| Offline model loading | [inference.py](../src/aerodepth_lite/inference.py) | `load_inference_model` | Checks export SHA-256, loads a tensor-only state dictionary strictly, includes BN buffers, and disables pretrained downloads. |
| Published model lineage | [Models guide](../models/README.md) | `original_checkpoint_sha256`, `strict_export_parity` in each export's JSON | Four inference exports are public; full optimizer/RNG/training checkpoints remain private. Teacher CLI inference is available even though the viewer is student-only. |

For a demonstration:

1. Open **Model comparison → Validation** to show all four models, then inspect Kite and the teacher-gate details. Teacher trained for 20 epochs versus A's 10; this is not a matched-budget architecture ranking.
2. Open **Results → Validation → ResNet-50 teacher** for the teacher's selected-scope metrics.
3. Switch to **Independent test**: teacher disappears because it was not evaluated there. Do not describe its validation scores as test results.
4. Use **Depth viewer** for Original/A/KD validation imagery and the explicit inference button. Its playback rate is a display setting.
5. Open **Artifacts → Validation** for teacher metadata and gate verification; use the [model guides](../models/README.md) for inference on your own images.

The current [dashboard config](../configs/dashboard.json) is separate from the frozen `configs/presentation.json` bound by the completed study. Publishing exports and updating the UI did not rewrite frozen protocols, exposure ledgers or scientific results.

## Independent study: code and results to explain

| If asked to show… | File | Search for | What to point out |
|---|---|---|---|
| Archive integrity and path isolation | [independent_data.py](../src/aerodepth_lite/independent_data.py) | `safe_members`, `sensor_records`, `verify_isolation` | ZIP members, sensor correspondence and underlying-path/weather-equivalent isolation. |
| Freeze and exposure controls | [independent_test.py](../src/aerodepth_lite/independent_test.py) | `freeze`, `validate_protocol`, `require_unused`, `# Exposure` | Fixed inputs/models/policies before exposure; existing exposure/output prevents automatic retry. |
| Current-frame scoring and actual calls | [independent_test.py](../src/aerodepth_lite/independent_test.py) | `Scores`, `forward`, `evaluate_stream` | All frames are scored, including reused maps; timing and call accounting are separate from shared I/O/decode. |
| Dataset-free audit | [verify_independent_test.py](../scripts/verify_independent_test.py) | `# Verification`, `verify` | Verifies source/artifact hashes, schedules, ages, events, pixel pooling and the 35-row table without reading images or running models. |

The dataset has **44,098 expanded training frames**, **4,410 validation frames**, and **8,820 independent-test frames on four reserved paths**. The historical 4,436-frame Original-model test is a separate exposed study. Validation informed selection and development; its scores are not untouched generalization evidence.

The independent study completed seven comparisons: five A policies plus Original/KD every-frame. A versus Original improves pooled AbsRel by **1.82%** but worsens near RMSE by **1.41%**. KD versus A improves pooled AbsRel by only **0.13%** while worsening near RMSE by **1.80%**. A remains the preselected demonstration model.

For V2 versus budget-fixed, report **3,718 versus 4,912 calls**, **17.08% lower measured inference-plus-scheduler time**, **0.43% worse AbsRel**, **0.25% better near RMSE**, and maximum event-update delay **4 versus 1 frames**. This was one CPU pass; shared I/O/decode and offline scoring are excluded from that compute time. These numbers do not establish battery savings, end-to-end speedup or flight safety.

The teacher gate is a per-path check, not just a pooled ranking: Kite teacher near AbsRel **0.742757** and near RMSE **6.636090 m** exceed A's **0.504336** and **4.695176 m**. Its better pooled score therefore did not pass the gate. The separate KD run remains exploratory with the rejected teacher.

## Hyperparameters and evidence to keep open

- Approved training configs: `results/expanded_launch_20261026T104420/configs/`. The similarly named top-level configs retain proposal headers.
- [Student comparison](../results/student_validation_comparison_20261029T152648Z/comparison.csv): exact original/A/KD metrics.
- [Teacher metrics](../results/student_validation_comparison_20261029T152648Z/metrics.json): `teacher_context_reused_no_inference.scopes`, including pooled/Kite/PLE values.
- [Teacher gate verification](../results/expanded_validation_diagnostic_20261028T172239Z/gate_verification.json): failed teacher verdict and the original gate identity.
- [Training curves](../results/presentation_validation_20261029T155521Z/training_curves.json): actual recorded values.
- [Validation scheduler table](../results/presentation_validation_20261029T155521Z/scheduler_comparison.csv): validation calls, errors and delays; separate from the independent study.
- [Independent comparison](../results/independent_test_20261030/comparison.csv) and [protocol/results guide](independent_final_test.md): completed test scores, calls and timing boundaries.
- [Published inference models](../models/README.md) and [dashboard configuration](../configs/dashboard.json): current portable model availability.
- [Final report](FINAL_REPORT.md): teacher rejection, timing uncertainty, dataset counts and limitations.

Quick locator from the project root:

```bash
rg -n '^\s*# (MobileNet|ResNet|Training|AbsRel|RMSE|Delta|Distillation|Decision)$' src scripts
```

For a presentation, read the code and saved results. Audit the completed study with `python scripts/verify_independent_test.py`; there is no need to retrain or repeat test exposure. The 30 October core-code annotations changed source bytes without changing Python ASTs. Later dashboard updates added real UI/display functionality; historical checkpoint resume still requires the matching frozen source revision described in [expanded reproduction](expanded_reproduction.md).
