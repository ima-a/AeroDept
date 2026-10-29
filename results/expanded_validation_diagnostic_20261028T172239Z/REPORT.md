# Expanded teacher diagnostic audit — 28 October 2026

**Observed result:** the selected teacher genuinely overestimates range during a short foggy close-approach segment. The saved gate rejection reproduces. No implementation defect was found in the audited preprocessing, units, masking, spatial alignment, normalization, selection or evaluation mode. The underlying cause of the learned bias remains uncertain.

This is a validation-only audit. No optimizer, backward training, KD, retraining, protocol amendment, alternate checkpoint promotion, reserved-test image/depth access or Git publication was performed. The failing HDD was not mounted or read.

## Frozen completion and identity

A completed 10 epochs / 220,490 updates; B completed 20 / 440,980. The original service was stopped (`MainPID=0`, exit status 2) at `teacher_rejected`; no KD output exists. A remains its pooled-AbsRel-selected epoch 9; B remains epoch 3. Best/last hashes matched completion receipts, all seven core source hashes matched the frozen identities, and the gate was recomputed from saved metrics without rewriting its historical absolute paths.

| Model | Selected checkpoint SHA-256 |
|---|---|
| Original epoch 4 | `b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6` |
| Supervised A (epoch 9) | `84b393a61f73ab95666a907931bff93fd21f65c2b5c1b2e170ce37b3c8f3f8b8` |
| Teacher B (epoch 3) | `6f9b1d27703ef0feba2f24441948a458650b239746632f9fc32d59ff0630b660` |

[Identity and immutable-state checks](identities.json) · [Gate verification](gate_verification.json) · [Runtime](runtime.json). The full original protocols/receipts stay in the recovered bundle; the original manifest is unchanged.

## Identical FP32 comparison

All 4,410 unchanged validation frames were evaluated with batch 2, identical unaugmented RGB/GT tensors and the existing metric accumulator. All models used eval/inference mode, no autocast, full-frame 192×320 resize, one in-model ImageNet normalization, metric Euclidean range and the same finite inclusive [0.1,1000] GT mask / GT<10 near mask. No fitted scale, upper output clamp, prediction filtering or crop was introduced.

The audit reproduced the saved overall/per-path metrics within a maximum absolute difference of **2.84e-14** (including counts). Counts were exactly **212,903,814 valid pixels** and **32,267,840 near pixels**. Guard logs cover exactly 8,820 unique permitted image paths; no unexpected dataset open occurred. The separate six illustrations used only their selected VAL frames and original batch partners.

| Scope / model | AbsRel | RMSE (m) | Near AbsRel | Near RMSE (m) |
|---|---:|---:|---:|---:|
| Pooled / Original epoch 4 | 0.301540 | 127.335 | 0.196167 | 3.362 |
| Pooled / Supervised A (epoch 9) | 0.265831 | 119.944 | 0.201899 | 3.906 |
| Pooled / Teacher B (epoch 3) | 0.240330 | 99.828 | 0.197856 | 3.873 |
| Kite_training foggy / Original epoch 4 | 0.357499 | 165.193 | 0.427121 | 4.379 |
| Kite_training foggy / Supervised A (epoch 9) | 0.290636 | 155.964 | 0.504336 | 4.695 |
| Kite_training foggy / Teacher B (epoch 3) | 0.258787 | 128.240 | 0.742757 | 6.636 |
| PLE_training spring / Original epoch 4 | 0.236536 | 57.876 | 0.189583 | 3.328 |
| PLE_training spring / Supervised A (epoch 9) | 0.237017 | 53.311 | 0.193278 | 3.881 |
| PLE_training spring / Teacher B (epoch 3) | 0.218889 | 49.378 | 0.182324 | 3.764 |

Teacher pooled AbsRel improves over A, and PLE passes all four trajectory checks. **Kite fails near AbsRel (0.742757 > 0.504336) and near RMSE (6.636 > 4.695 m)**. Kite overall errors improve. Correct/equal pixel counts do not excuse a per-path near regression.

All original GT bands are retained: [0.1,10), [10,50), [50,100), [100,300), [300,1000] m. [Per-path/pooled depth-band CSV](depth_band_metrics.csv), [raw metrics](metrics.json), [pooled supplementary metrics](pooled_supplementary.json) and [depth-band figure](depth_bands.png) retain every model and band; original epoch-4 band counts/errors reproduce the prior audit.

## Foggy near-field failure

| Model | Mean prediction (m) | Mean GT (m) | Bias P−GT (m) | Near MAE (m) | GT-near pixels predicted ≥10m |
|---|---:|---:|---:|---:|---:|
| Original epoch 4 | 12.203 | 8.614 | +3.589 | 3.614 | 79.26% |
| Supervised A (epoch 9) | 12.934 | 8.614 | +4.320 | 4.321 | 96.21% |
| Teacher B (epoch 3) | 14.937 | 8.614 | +6.323 | 6.324 | 99.60% |

Teacher median near signed error is +6.255 m; the 95th percentile is +9.497 m and 99th +12.929 m. Its mean signed bias (+6.323 m) nearly equals its near MAE (6.324 m): errors are predominantly overestimation. This is a pixel-threshold diagnostic, not an obstacle detector or flight-safety metric.

**Floor saturation is not the cause of this near-field rejection:** all three models have zero foggy GT-near pixels with prediction ≤0.101 m. They do have floor predictions elsewhere on valid Kite pixels:

| Model | Kite valid-pixel floor rate | Kite near-pixel floor rate | PLE valid-pixel floor rate |
|---|---:|---:|---:|
| Original epoch 4 | 4.4611% | 0% | 0.004397% |
| Supervised A (epoch 9) | 3.3423% | 0% | 0.004141% |
| Teacher B (epoch 3) | 2.2483% | 0% | 0.000391% |

The teacher reduces valid-pixel floor saturation while worsening foggy near range. [All counts/rates, including pooled and near-only rates](floor_rates.csv), [exact all-near-pixel quantiles and histograms](near_distributions.json), [quantile CSV](near_quantiles.csv) and [distribution plot](near_error_distributions.png) separate these phenomena. Quantiles use all valid near pixels, not a sample; figure axes trim tails for readability, but CSV/JSON retain their full range.

Only **83/2,205 Kite frames** contain valid GT<10m pixels: one consecutive interval, **1526–1608**. Their GT ranges from **6.715 to 9.992 m**; this path supplies no validation evidence below 6.715 m. These are correlated frames from one approach, not 83 independent flights. They account for **2.77%** of pooled validation near pixels, while PLE supplies the remaining 97.23%. Therefore better pooled near scores can conceal this subgroup failure.

Teacher near SSE is worse than A in **78/83** active Kite frames. Its worst single frame contributes only 2.85% of Kite near SSE; top 10 contribute 24.99%, top 25 53.84%, and top 50 84.18%. This is concentrated in one short sequence but is not a single-frame numerical anomaly.

The lower image third contains **95.54%** of Kite near observations and **89.85%** of teacher near SSE. Exact fixed 3×3 regional counts, bias/RMSE and error energy are in [spatial regions](spatial_regions.csv); [spatial bias maps](near_spatial_bias.png) retain the per-position support.

PLE has 2,084 near-containing frames, and teacher near SSE is worse than A in 976 despite its better aggregate PLE result. Rare large PLE errors remain: the worst one contributes 6.23% of teacher PLE near SSE. No claim of uniform teacher superiority is supported.

## Deterministic visual checks

Illustrations were selected by the preregistered rule: per path, eligible frames have ≥64 near pixels; choose nearest median teacher near AbsRel, maximum teacher near AbsRel, and maximum teacher-minus-A near SSE, with earliest-frame ties and duplicate removal. This eligibility threshold affects illustrations only; all pixels/frames remain in metrics. Original FP32 batch partners were retained, and per-frame metrics reproduced.

- [Kite_training/foggy/trajectory_2003 frame 1534: poor](examples/Kite_training__foggy__trajectory_2003__001534__poor.png).
- [Kite_training/foggy/trajectory_2003 frame 1567: typical](examples/Kite_training__foggy__trajectory_2003__001567__typical.png).
- [Kite_training/foggy/trajectory_2003 frame 1577: regression](examples/Kite_training__foggy__trajectory_2003__001577__regression.png).
- [PLE_training/spring/trajectory_5000 frame 445: typical](examples/PLE_training__spring__trajectory_5000__000445__typical.png).
- [PLE_training/spring/trajectory_5000 frame 1691: poor](examples/PLE_training__spring__trajectory_5000__001691__poor.png).
- [PLE_training/spring/trajectory_5000 frame 1928: regression](examples/PLE_training__spring__trajectory_5000__001928__regression.png).

The inspected Kite examples show foggy water/rock/shoreline foregrounds with a broad positive range bias across the near patch. The regression frame 1577 has teacher near bias +7.512 m versus A +3.619 m. This is a visual scene description, not a semantic dataset label or proof that fog/water caused the error. The PLE regression example shows large errors around thin trunks/foliage despite reasonable surrounding ground predictions; it illustrates why mean bias and RMSE can tell different stories.

[Example selection](example_selection.json), [example verification](example_verification.json), [per-frame metrics](per_frame.csv), [top regression frames](top_regression_frames.csv) and [concentration accounting](concentration.json) make selection reproducible. All example RGB/depth derives only from the existing validation split; attribution/CC BY-NC-SA 4.0 is printed on figures.

## Contract and implementation checks

- First-batch encoder input equalled `(RGB−mean)/std` exactly for every model (maximum difference 0): normalization occurs once. Shared loader RGB remained [0,1].
- Native RGB/depth dimensions were checked by the unchanged loader on every pair. Model output and resized GT/mask aligned at 192×320. Teacher projected features followed /2, /4, /8, /16, /32 shapes; full-frame upsampling targets actual skip/input dimensions.
- All predictions were finite, positive FP32 metres with the unchanged 0.1 m floor. GT used uint16 PNG reinterpreted as float16 metre values; invalid/out-of-range GT is masked before nearest resizing. Metric counts exactly match the historical reference.
- All model/BatchNorm modules stayed in eval mode. Complete parameter/buffer digests were identical before/after inference. Source inspection confirms training calls `model.train()` at every successful/retried batch, so evaluation does not leave the following epoch permanently in eval mode.
- Nonzero pretrained BN counters were verified against ImageNet initialization: MobileNet starts at 699,033; ResNet-50 at 673,830. Original/student-A/teacher decoder counters reflect 17,636 / 216,077 / 66,147 updates; encoder counters include the appropriate pretrained offsets. Finite small running variances alone do not prove a BN defect, and correct eval mode does not establish that learned population statistics are ideal.

[Contract checks](contract_checks.json), [independent buffer/validation-record/floor checks](independent_contract_checks.json), and [BN initialization provenance](batchnorm_initialization_provenance.json) support these findings. No audited contract/evaluation/selection implementation defect was found. This is not a proof against every possible training bug or a causal diagnosis of optimization.

## Saved epoch histories

B’s frozen pooled selection remains epoch **3**; no alternate checkpoint was promoted or evaluated as a replacement. Compared with selected A, B’s foggy near RMSE regresses in **17/20 epochs**, and near AbsRel in **18/20**. Near RMSE ranges **3.553–8.248 m**, so the problem is recurrent, not literally present at every epoch. The minimum near error in history is not the authorized selection rule. A itself worsens foggy near RMSE from original epoch 4’s 4.379 m to selected A’s 4.695 m, despite better pooled overall error.

[All saved A/B epochs](saved_history.csv), [history summary](history_summary.json) and [history plot](training_history.png) retain negative results and unchanged selections.

## Supported explanation, hypotheses and uncertainty

The supported explanation of rejection is a reproducible, predominantly positive near-range bias in one short foggy approach segment. Pooled valid-pixel selection and aggregation do not guarantee performance on this small subgroup. Bigger architecture and more training did not guarantee better close-range predictions.

| Candidate explanation | Evidence and limit |
|---|---|
| Loss/selection trade-off | Plausible: far/overall metrics improve, near bias worsens, the frozen supervised objective has no near-MAE weight, and pooled selection does not constrain this subgroup. Gradient contribution/causation was not measured here. |
| Missing fog/near training data | A simple absence claim is contradicted by cached preparation: new foggy train paths 2005 and 2006 have 10.26% and 32.02% near pixels among valid pixels. Semantic scene diversity or support for this shoreline approach remains unknown; no TRAIN images were opened. |
| BatchNorm | Eval mode, unchanged buffers and counter provenance pass. Quality of learned statistics with physical batch 2 remains an untested hypothesis; no VAL calibration/adaptation was performed. |
| Architecture/decoder | Correct shapes/normalization/units do not isolate architectural capacity or representation. Appearance, architecture, initialization and optimization differ between models; a causal architecture claim needs controls. |
| Recent HDD corruption | Selected checkpoint hashes match pre-incident completion receipts, and fresh USB validation reproduces pre-incident metrics for all three models. This provides no support for corruption as the explanation of this recorded rejection. It does not certify the entire HDD. |

Only two familiar validation paths are available; foggy near evidence is one temporally correlated segment with no GT below 6.715 m. No test generalization, weather-only cause, energy or flight-safety claim follows.

## Next experiment — proposal only

**Recommend a small, non-updating TRAIN-only loss-gradient contribution probe first**, using the existing deterministic 64 TRAIN frames and the same frozen teacher. Compare the current objective with one fixed proposed near-MAE term; measure component norms/alignment while parameters and BN/RNG stay unchanged. This tests the cheapest plausible objective mechanism before another long teacher run. No probe or proposed training was launched.

[Concrete controls, estimated cost and acceptance criteria](NEXT_EXPERIMENT_PROPOSAL.md). The probe is expected to take minutes, but FP32 multi-gradient runtime/memory and USB I/O are not measured. It cannot pass the teacher gate. A conditional five-epoch teacher-only near-loss pilot would cost roughly the historical first-five-epoch **8.98 h** (uncertain on USB); the completed 20-epoch teacher cost **40.48 h**. Such a pilot requires a new protocol, new outputs and unchanged original rejection; it is not a resume or authorization to run KD.

## Preservation and artifacts

New compact evidence/figures live in this directory. Executed audit scripts, raw exact-near prediction/GT arrays and image-open logs live under `artifacts/expanded_validation_diagnostic_20261028T172239Z/`. Raw arrays contain validation near pixels only; they are not a training/test cache. Existing reports, manifests, configs, weights, gates and exposure ledgers are preserved; preservation/checksum receipts accompany the audit. No commit or push was made.

A second verified copy of this new results directory and its private audit artifacts is saved under the same relative `project/results/` and `project/artifacts/` paths on the AERODEPTH USB. The backup receipt is stored separately under the USB `recovery/` directory.
