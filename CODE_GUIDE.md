# Classroom code guide

Search the one-word labels at the actual operation. Architecture definitions are in `model.py`; training happens in `expanded.py`. The longer [viva guide](docs/VIVA_GUIDE.md) includes more questions and exact labels.

| Topic | Definition to show | Label / evidence |
|---|---|---|
| Student | [model.py](src/aerodepth_lite/model.py), `MobileNetV3SmallDepth` | `# MobileNet`, `# Decoder`, `# Normalization`; A epoch 9 is the demonstration default |
| Teacher | [model.py](src/aerodepth_lite/model.py), `ResNet50Depth` | `# ResNet`; architecture only |
| Training and optimizer | [expanded.py](src/aerodepth_lite/expanded.py), `Experiment.step`, `full_training` | `# Training`, `# AdamW`, `# Cosine`; no training is needed for a demo |
| Loss and KD | [losses.py](src/aerodepth_lite/losses.py), `depth_loss`, `masked_log_range_consistency` | `# L1`, `# Log`, `# Gradient`, `# Distillation`; regression consistency, not classification logits |
| Teacher rejection | [expanded.py](src/aerodepth_lite/expanded.py), `teacher_gate` | `# Gate`; retain the failed near-field checks and exploratory KD designation |
| Data and masks | [data.py](src/aerodepth_lite/data.py), `load_depth`, `MidAirDataset.__getitem__` | `# Decoding`, `# Mask`, `# Resizing`; range metres, finite GT [0.1,1000], near <10 |
| AbsRel / RMSE / delta | [metrics.py](src/aerodepth_lite/metrics.py), `DepthMetricAccumulator` | `# AbsRel`, `# RMSE`, `# Delta`, `# Pooling`; strict delta and valid-pixel pooling |
| Causal scheduling | [presentation_scheduler.py](src/aerodepth_lite/presentation_scheduler.py), `PresentationScheduler` | `# Decision`, `# Motion`, `# Visual`, `# Cache`; last prediction risk, current sensors/RGB, bounded age |
| Archive identity / leakage | [independent_data.py](src/aerodepth_lite/independent_data.py), `safe_members`, `sensor_records`, `verify_isolation` | Eight CRC-verified archives, HDF correspondence and grouped path hashes |
| Freeze / exposure | [independent_test.py](src/aerodepth_lite/independent_test.py), `freeze`, `run` | `# Compatibility`, `# Exposure`; exclusive ledger before decoding, no automatic retry |
| Final scoring / timing | [independent_test.py](src/aerodepth_lite/independent_test.py), `Scores`, `forward`, `evaluate_stream` | `# Metrics`, `# Inference`, `# Evaluation`; every current frame scored, including reused predictions |
| Verify saved results | [verify_independent_test.py](scripts/verify_independent_test.py), `verify` | `# Verification`; hashes, calls, source ages, events and pooling, with no image reads or model calls |
| UI / actual inference | [dashboard.py](scripts/dashboard.py), [presentation_ui.py](src/aerodepth_lite/presentation_ui.py), `actual_inference` | Five tabs; validation-only viewer, explicit forward-pass button, clearly labelled recorded replay |

Keep [the final report](docs/FINAL_REPORT.md), [validation comparison](results/student_validation_comparison_20261029T152648Z/comparison.csv), and the new study's `results/independent_test_20261030/comparison.csv` open. The original exposed 4,436-frame study stays under `results/final_test/` and is historical.

The existing comment-only changes preserve Python ASTs; their byte identities and checks are recorded in [the marker receipt](docs/maintenance/2026-10-30_code_markers.json). The independent runner is new functionality with its own frozen source identities. Do not modify frozen numerical sources to update slides or the dashboard.
