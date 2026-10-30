# Code to show during the viva

Open the file and search for the exact one-word comment below. `model.py` defines the architectures; the shared training loop is in `expanded.py`. ResNet training does not happen inside the model class.

## Models and learning

| If asked to show… | File | Search for | What to point out |
|---|---|---|---|
| Student architecture | [model.py](../src/aerodepth_lite/model.py) | `# MobileNet` | `MobileNetV3SmallDepth`: feature encoder, projection and four skip-decoder blocks. |
| Teacher architecture | [model.py](../src/aerodepth_lite/model.py) | `# ResNet` | `ResNet50Depth`: ResNet-50 V1.5 encoder and projected skips. |
| Small pilot model | [model.py](../src/aerodepth_lite/model.py) | `# Tiny` | Historical/synthetic `AeroDepthTiny`; not the current A model. |
| Decoder / skip connections | [model.py](../src/aerodepth_lite/model.py) | `# Decoder`, `# Skips`, `# Upsampling` | Bilinear resizing and concatenation with encoder features. |
| Input normalization | [model.py](../src/aerodepth_lite/model.py) | `# Normalization` | ImageNet mean/std inside the model, exactly once. |
| Positive range output | [model.py](../src/aerodepth_lite/model.py) | `softplus` | One-channel output plus 0.1 m floor; no upper output clamp. |
| Parameter count | [model.py](../src/aerodepth_lite/model.py) | `# Parameters` | Count tensor elements. Student has 1,027,489 parameters. |
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
| Dashboard tabs | [dashboard.py](../scripts/dashboard.py) | `# Overview`, `# Comparison`, `# Inference`, `# Scheduler`, `# Evidence` | Five UI sections backed by saved evidence. |

## Hyperparameters and evidence to keep open

- Approved training configs: `results/expanded_launch_20261026T104420/configs/`. The similarly named top-level configs retain proposal headers.
- [Student comparison](../results/student_validation_comparison_20261029T152648Z/comparison.csv): exact original/A/KD metrics.
- [Training curves](../results/presentation_validation_20261029T155521Z/training_curves.json): actual recorded values.
- [Scheduler table](../results/presentation_validation_20261029T155521Z/scheduler_comparison.csv): calls, errors and delays.
- [Final report](FINAL_REPORT.md): teacher rejection, timing uncertainty, dataset counts and limitations.

Quick locator from the project root:

```bash
rg -n '^\s*# (MobileNet|ResNet|Training|AbsRel|RMSE|Delta|Distillation|Decision)$' src scripts
```

For a presentation, read the code and saved results. There is no need to retrain or run the reserved test. The 30 October annotations change source bytes but not Python ASTs; historical checkpoint resume uses the matching frozen revision described in [expanded reproduction](expanded_reproduction.md).
