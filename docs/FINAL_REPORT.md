# AeroDepth-Lite — consolidated final research report

Status: training and validation are complete. The separately authorized independent reserved study is **COMPLETED and verified**, with 8,820 frames and seven frozen comparisons. No new training, tuning, teacher repair, KD rerun or presentation generation is part of this delivery. Existing presentation/evidence and one-word code markers are preserved. See [storage records](archive_locations.md), [the code guide](../CODE_GUIDE.md) and [the viva guide](VIVA_GUIDE.md).

## Problem and objectives

Estimate dense Euclidean camera range from one drone RGB image with a compact student, then decide causally when to compute a fresh range map or reuse the previous map. Evaluate overall and near-field errors together, investigate teacher-to-student distillation, and measure actual execution cost. A prediction update is not obstacle detection; lower averaged error is not flight safety.

## Related work and what was actually implemented

Eigen, Puhrsch and Fergus motivate learned monocular depth and scale-aware evaluation ([2014 paper](https://arxiv.org/abs/1406.2283)). Our output is metric Euclidean range, not their architecture or a reproduced benchmark result. Howard et al. introduce efficient MobileNetV3 encoders ([2019 paper](https://arxiv.org/abs/1905.02244)); this project uses MobileNetV3-Small features with its own skip decoder. He et al. introduce residual learning ([paper](https://arxiv.org/abs/1512.03385)); the teacher uses torchvision ResNet-50 V1.5, not an accuracy guarantee. Hinton, Vinyals and Dean describe knowledge transfer to a smaller model ([paper](https://arxiv.org/abs/1503.02531)); the implemented depth KD is masked absolute log-range consistency, not classification-logit temperature distillation. These references support methods, not measured superiority here.

Fonder and Van Droogenbroeck provide the synthetic multi-modal MidAir dataset ([CVPRW2019 paper](https://orbi.uliege.be/handle/2268/234665)). Its [FAQ](https://midair.ulg.ac.be/faq.html) defines depth as Euclidean distance to the focal point. MidAir terms require attribution/noncommercial/share-alike use; see [NOTICE](../NOTICE.md).

## Architecture, preprocessing and simulated sensors

The student has1,027,489 parameters: ImageNet MobileNetV3-Small feature encoder, captured skips at feature indices0/1/3/8, 576→128 projection, decoder output widths64/32/24/16 and3×3 one-channel head. Classification pooling/head are discarded. Bilinear upsampling targets skip/input sizes; final softplus+0.1 produces positive range in metres with no upper output clamp.

Teacher: ImageNet1K_V2-initialized ResNet-50 V1.5 encoder, /2,/4,/8,/16 skips projected to16/32/64/128 channels and /32 bottleneck128; decoder widths128/64/32/16 and the same positive metric-range head. Teacher parameters are frozen, eval-mode, FP32/no-grad during KD and receive the same augmented RGB as the student.

All comparisons use full-frame192×320 RGB, bilinear RGB resize, one in-model ImageNet normalization, float16-bit-decoded range and nearest GT/mask resize. Valid GT is finite inclusive[0.1,1000]m; near is valid GT<10m. FP32 validation uses no AMP/TF32. Pixel-weighted additive metric sums give AbsRel, RMSE and strict ratio<1.25^k δ1/2/3; pooled RMSE is from pooled SSE, never a mean of per-frame RMSE. Supplementary bias is signed P−GT; floor rate is prediction≤0.101m on valid pixels. Undefined quantities stay null. Colour clipping changes illustrations only.

Scheduler inputs are causal simulated IMU accelerometer/gyroscope and held simulated GPS velocity, plus current RGB relative to last inferred RGB and risk from the last prediction. Ground-truth velocity, future navigation and current GT depth never enter decisions. The frozen manifest contains historical privileged velocity fields, but scheduler navigation explicitly whitelists sensor fields and reads recorded GPS. HDF5 hashes and hold alignment are retained in the presentation provenance.

## Grouped splits and exposure controls

| Assignment | Independent paths | Realizations | Frames | Exposure status |
|---|---:|---:|---:|---|
| Expanded TRAIN |12|20|44,098|Training completed|
| Unchanged VAL |2|2|4,410|Development and selection; familiar paths|
| Reserved independent test |4|4|8,820 verified|Completed once; independent frozen evaluation|
| Legacy completed test |2|2|4,436|Previously exposed; historical and quarantined|

Different climates of the same underlying flight count as one path. Training has Kite0000/0001/0002/0005/0006/0021 and PLE0001/0002/0004/0005/0006/0007; validation is Kite0003 and PLE0000; reserved groups are Kite0007/0008 and PLE0008/0009. Legacy test Kite0004/PLE0003 and every equivalent climate are excluded from training. Pilot-exposed Kite0003/0021 cannot be new test: Kite0003 stays validation and0021 training. HDF5 position-stream hashes establish weather equivalence. Different paths may still share scenery; this is not geographic isolation.

[Authoritative reservations](../results/expanded_preparation_20261026/reservations.json) and compressed TRAIN/VAL manifest remain unchanged. Uncompressed manifest SHA256: `67c6cad39767d847a2087a81d1ecc4de2ad8c9ecc0fa01e491f3950b25f8dd3a`. Metadata-only rechecks found disjoint groups and zero test records. The original exposed test is not reused as untouched evidence.

## Training methods, selection and numerical recovery

Original student completed50 epochs on17,635 training frames; epoch4 was selected using its recorded AMP validation. The later FP32 audit uses that fixed checkpoint. Expanded A completed10 epochs and teacher B20, both batch2, seed42, AdamW/weight decay1e-4 and clipping1. Student LR1e-4; teacher encoder/decoder1e-5/1e-4. Fixed cosine minima1e-6 and approved budgets were not tuned here. A selected epoch9 and B epoch3 by minimum pooled FP32 validation AbsRel. Current proposal-template headers are not retroactive approval records; frozen launch copies and receipts are authoritative ([reproduction guide](expanded_reproduction.md)).

Supervised objective is valid-pixel range L1 +0.2 absolute log-range +0.1 spatial gradient L1; near weight remains0. AMP recovery retries an overflowed batch without an optimizer step, clears gradients and restores BN/RNG before retry, checkpoints scaler state and stops at its retry bound. No skipped successful-update accounting or silent shortened budget.

The original teacher gate requires strictly better pooled FP32 AbsRel plus no worse per-path overall/near AbsRel and RMSE, with identical pixel counts. B improves pooled scores but fails Kite foggy near-field checks: near AbsRel0.742757 versus A0.504336; near RMSE6.636090m versus A4.695176m. The failed gate remains unchanged (SHA256 `10e749c38271d360c2c0dc50cde6ec5416f410532b8f6817166f6ac6d3e36b1f`). The original sequence stopped; original gated KD never ran.

A separate explicitly exploratory KD arm then completed10 epochs using the rejected selected B teacher. Its objective adds0.2×masked absolute log-range consistency on valid training GT pixels. Both A and KD independently started from identical original epoch4 weights and BN buffers with fresh matched optimizer/scheduler/scaler/RNG/counters, not from completed A. KD also selected epoch9 using pooled FP32 validation. This is not an accepted-teacher experiment. The fixed64-frame TRAIN gradient probe, after its separate numerical amendment, did not meet the preserved teacher-repair criteria; no repair was launched ([probe report](../results/expanded_gradient_resolution_20261028T185216Z/REPORT.md)).

## Selected-student validation results

| Scope | Model | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | Near AbsRel ↓ | Near RMSE m ↓ |
|---|---|---:|---:|---:|---:|---:|
|pooled|Original epoch4|0.301540|127.335|0.498036|0.196167|3.362|
|pooled|Supervised A epoch9|0.265831|119.944|0.626189|0.201899|3.906|
|pooled|Exploratory KD epoch9|0.270049|121.207|0.618070|0.209946|3.961|
|Kite_training/foggy/trajectory_2003|Original epoch4|0.357499|165.193|0.399073|0.427121|4.379|
|Kite_training/foggy/trajectory_2003|Supervised A epoch9|0.290636|155.964|0.612216|0.504336|4.695|
|Kite_training/foggy/trajectory_2003|Exploratory KD epoch9|0.295285|157.806|0.604495|0.542571|4.986|
|PLE_training/spring/trajectory_5000|Original epoch4|0.236536|57.876|0.612996|0.189583|3.328|
|PLE_training/spring/trajectory_5000|Supervised A epoch9|0.237017|53.311|0.642421|0.193278|3.881|
|PLE_training/spring/trajectory_5000|Exploratory KD epoch9|0.240734|53.187|0.633840|0.200465|3.928|

A versus original: pooled AbsRel−11.84% and overall RMSE−5.80%, but near RMSE+16.18%. A is the general-depth demonstration, never a near-field safety upgrade. Original remains the near-field reference. KD versus A: pooled AbsRel+1.59%, near RMSE+1.42%; PLE overall RMSE improves slightly while near errors worsen. Preserve this negative outcome.

Counts match across students:212,903,814 valid/32,267,840 near pixels; Kite114,412,073/894,287; PLE98,491,741/31,373,553. Only83 consecutive Kite frames contain near GT;97.23% of pooled near pixels come from PLE. Pixel counts are not independent sample counts. Complete δ2/δ3, bias/floor/counts, depth bands and absolute/relative changes remain in the [student comparison](../results/student_validation_comparison_20261029T152648Z/REPORT.md). Teacher audited metrics are reused, not reevaluated.

## Validation scheduler ablations

Frozen V2 parameters remain unchanged from v2_02. Comparator: per-path K=ceil(N×2455/4410), calls at ceil(kN/K), hence1228/path and2456 pooled on2205-frame paths. It was not fitted to A calls. All4410 current frames are scored, using unwarped reused predictions. First-frame inference, path reset, causal navigation and hard MAX_SKIP4 remain. Remove-one ablations disable both weights and forced triggers; surviving weights are unchanged. Age-only equals fixed5 and is not duplicated.

| Policy | Calls | AbsRel ↓ | Near RMSE m ↓ | Maximum age | Event max delay |
|---|---:|---:|---:|---:|---:|
|every|4410|0.265831|3.906|0|0|
|fixed2|2206|0.267708|4.163|1|1|
|fixed5|882|0.282528|6.586|4|4|
|budget_fixed|2456|0.267810|4.161|1|1|
|v2|2456|0.268088|3.780|4|2|
|v2_no_motion|1900|0.268943|3.874|4|2|
|v2_no_visual|1789|0.274759|5.529|4|4|
|v2_no_risk|2446|0.268167|3.770|4|2|
|v2_no_age|2454|0.268108|3.779|4|2|

V2 versus budget-fixed uses equal2456 pooled calls but redistributes them: Kite905/PLE1551 versus1228/1228. Near RMSE improves9.14% pooled, overall AbsRel/RMSE slightly regress and maximum event-update delay rises1→2. Kite near RMSE is slightly worse; PLE improves. Removing visual change gives near RMSE5.529m/delay4. Ablations change call budgets and state trajectories, so they are not equal-compute causal estimates. Stale predictions can incidentally lower averaged error on correlated frames.

Critical events use the frozen GT p10<8m entry or ≥2m drop rule, only offline for scoring. There are1 Kite and10 PLE events, no observed censoring here. Zero-event denominators remain null and right-censoring is retained. Full per-path metrics, reasons, ages, sources and logs: [presentation report](../results/presentation_validation_20261029T155521Z/REPORT.md).

## Live risk cache and actual execution timing

The separate versioned scheduler caches predicted-risk value and availability until a fresh source token/reset; age stays independent. Five adaptive variants×4410 frames match uncached full-prediction computation for decisions/reasons/source indices/ages. Threshold-boundary and reset tests and six actual live timed traces agree. The mechanism command wrote its complete receipt before its shell returned143; this caveat is preserved, not relabelled as clean exit. No completed experiment was rerun here.

Actual timing: FP32/eval batch1, CPU2 threads and GTX1650 CUDA; ten excluded warmups and three alternating repetitions per policy/device. Two fixed128-frame stress windows reset locally, so budgets differ from full-path replay. I/O, decode/resize, transfers, synchronized model, scheduler including cache updates and end-to-end are separate. GT loading serves the validation-display pipeline, never decisions. Model/manifest/sensor setup, warmup, offline scoring, final writes, browser rendering/WebSocket are outside the per-frame boundary. OS/file cache and desktop load are uncontrolled.

| Device | Policy | Calls/run | End-to-end median [min,max] s | Model-call median ms |
|---|---|---:|---:|
|cpu|every|[256, 256, 256]|10.4203 [10.2725,14.3289]|14.475|
|cpu|budget_fixed|[144, 144, 144]|8.6725 [8.3859,9.1672]|13.744|
|cpu|v2_cached|[222, 222, 222]|10.4290 [10.1152,10.6138]|13.731|
|cpu|v2_uncached|[222, 222, 222]|10.1185 [9.7162,10.3029]|14.293|
|cpu|fixed5|[52, 52, 52]|7.2661 [7.0481,7.3484]|14.256|
|cuda|every|[256, 256, 256]|9.0176 [8.8663,9.4765]|9.664|
|cuda|budget_fixed|[144, 144, 144]|7.8625 [7.7724,8.3073]|9.660|
|cuda|v2_cached|[222, 222, 222]|8.7650 [8.7047,9.3071]|9.629|
|cuda|v2_uncached|[222, 222, 222]|8.7976 [8.7502,9.3273]|9.609|
|cuda|fixed5|[52, 52, 52]|6.9060 [6.8450,7.2114]|9.652|

Risk caching reduces median scheduler time13.43% CPU/15.81% CUDA. End-to-end median is3.07% worse CPU and0.37% better CUDA, with overlapping repetition ranges: no robust end-to-end speedup. Every-frame model-only medians are14.48ms CPU/9.66ms CUDA, not playback FPS or full-pipeline throughput. No energy or battery reduction was measured.

## Independent final study · 30 October 2026

**COMPLETED:** one authorized frozen study; 8,820 current frames in each of seven comparisons; 35 pooled/per-path comparison rows. All counts, saved identities, fixed schedules, source ages, events/censoring and pooled tables passed read-only verification. No training, tuning or automatic repeat occurred.

Eight verified ZIPs supplied 17,640 images. Kite foggy 2007 has 2,204 pairs; Kite sunny 0008 has 2,205; PLE spring 5008 has 2,206; PLE winter 6009 has 2,205. Archives occupy 5,866,144,441 bytes; extracted images 6,668,899,690 bytes; four reused HDF5 copies 310,171,768 bytes. Exact filenames/counts/CRC/HDF correspondence and path/weather-equivalent leakage checks passed before freeze. All downloads are retained, with staging outside the repository on healthy local storage.

Protocol SHA-256: `6a1bb71c98ba3a5b513ff46e7babb4c33dca168886f9561072e8095cff71473f`. Manifest SHA-256: `4220cb10bd3d7671a4e08498b2f9ce34860ccdc95debce6eb93cb25cfb7da9f2`. [Ready protocol](../results/independent_test_20261030/PROTOCOL.json), [exclusive exposure ledger](../results/independent_test_20261030/EXPOSURE.json), [completion](../results/independent_test_20261030/COMPLETION.json), [saved-evidence verification](../results/independent_test_20261030/VERIFICATION.json). The ledger preceded the first test decoder access; existing output/exposure prevents retry.

Primary: five A policies, with full V2 versus validation-frozen budget-fixed as the predeclared contrast. Secondary: Original and exploratory KD every-frame versus A every-frame. The fixed comparator uses K=ceil(N×2455/4410), positions ceil(kN/K), independently of observed adaptive test calls. Numerical preprocessing, strict deltas, GT validity/near thresholds and pixel pooling are unchanged.

On the independent test, A versus Original changes pooled AbsRel by -1.82% and near RMSE by +1.41%. KD versus A changes pooled AbsRel by -0.13% and near RMSE by +1.80%. Negative changes mean lower error. These new outcomes do not erase the negative validation findings or the failed teacher gate; A remains the preselected demonstration model.

The frozen primary V2-versus-budget-fixed comparison uses 3,718 versus 4,912 actual calls. AbsRel changes +0.43%, near RMSE -0.25%, and measured inference-plus-scheduler time -17.08%. Maximum observed event-update delay is 4 versus 1 frames; right-censored events are 0 versus 0. A fresh update is not a detection guarantee.

### pooled

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ | Calls |
|---|---|---|---|---|---|---|---|---|
| A_every | 0.383690 | 81.813343 | 0.406799 | 0.661703 | 0.792140 | 0.439396 | 4.144905 | 8820 |
| A_fixed2 | 0.387019 | 81.975304 | 0.405759 | 0.660284 | 0.790959 | 0.442566 | 4.369061 | 4411 |
| A_fixed5 | 0.412312 | 83.462253 | 0.399407 | 0.653437 | 0.785177 | 0.464057 | 7.756830 | 1765 |
| A_budget_fixed | 0.386539 | 81.962581 | 0.406137 | 0.660468 | 0.791392 | 0.442450 | 4.416516 | 4912 |
| A_v2 | 0.388208 | 82.298352 | 0.406031 | 0.659257 | 0.790126 | 0.439805 | 4.405438 | 3718 |
| Original_every | 0.390815 | 81.798271 | 0.381188 | 0.644062 | 0.791480 | 0.487086 | 4.087260 | 8820 |
| KD_every | 0.383202 | 82.173502 | 0.411403 | 0.661888 | 0.787492 | 0.436249 | 4.219446 | 8820 |

### Kite_training/foggy/trajectory_2007

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ | Calls |
|---|---|---|---|---|---|---|---|---|
| A_every | 0.233710 | 78.364065 | 0.608260 | 0.886045 | 0.960080 | 0.182073 | 3.102119 | 2204 |
| A_fixed2 | 0.235758 | 78.566267 | 0.608514 | 0.884996 | 0.958970 | 0.183720 | 3.382324 | 1102 |
| A_fixed5 | 0.249707 | 80.321071 | 0.604302 | 0.880481 | 0.954849 | 0.192619 | 5.271800 | 441 |
| A_budget_fixed | 0.235239 | 78.531337 | 0.608936 | 0.885844 | 0.959255 | 0.184402 | 3.549451 | 1227 |
| A_v2 | 0.241673 | 79.524608 | 0.608158 | 0.882499 | 0.957243 | 0.183913 | 4.016049 | 621 |
| Original_every | 0.282769 | 77.610310 | 0.490006 | 0.778229 | 0.904023 | 0.272358 | 2.756589 | 2204 |
| KD_every | 0.230382 | 79.045396 | 0.621987 | 0.895550 | 0.961209 | 0.186724 | 3.085750 | 2204 |

### Kite_training/sunny/trajectory_0008

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ | Calls |
|---|---|---|---|---|---|---|---|---|
| A_every | 0.544881 | 126.223733 | 0.065705 | 0.199668 | 0.400885 | 0.404106 | 4.044662 | 2205 |
| A_fixed2 | 0.544626 | 126.220357 | 0.065890 | 0.200499 | 0.401088 | 0.405437 | 4.408612 | 1103 |
| A_fixed5 | 0.545245 | 126.408625 | 0.065506 | 0.198532 | 0.402090 | 0.409206 | 4.813383 | 441 |
| A_budget_fixed | 0.544577 | 126.245126 | 0.065748 | 0.199383 | 0.402016 | 0.407058 | 4.346573 | 1228 |
| A_v2 | 0.545369 | 126.372778 | 0.066001 | 0.198130 | 0.398508 | 0.409256 | 4.981709 | 695 |
| Original_every | 0.491769 | 126.794208 | 0.117185 | 0.289704 | 0.491811 | 0.204875 | 2.255965 | 2205 |
| KD_every | 0.551861 | 126.564154 | 0.062087 | 0.190963 | 0.381596 | 0.418402 | 4.164355 | 2205 |

### PLE_training/spring/trajectory_5008

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ | Calls |
|---|---|---|---|---|---|---|---|---|
| A_every | 0.334740 | 49.392397 | 0.493041 | 0.776523 | 0.888847 | 0.379440 | 3.365328 | 2206 |
| A_fixed2 | 0.336127 | 49.549236 | 0.491859 | 0.775349 | 0.888354 | 0.379725 | 3.366251 | 1103 |
| A_fixed5 | 0.350586 | 51.117740 | 0.489056 | 0.770847 | 0.884622 | 0.380790 | 3.690481 | 442 |
| A_budget_fixed | 0.335786 | 49.514489 | 0.492624 | 0.775737 | 0.888929 | 0.379598 | 3.360605 | 1229 |
| A_v2 | 0.340827 | 50.083864 | 0.491641 | 0.773423 | 0.887035 | 0.378321 | 3.619280 | 738 |
| Original_every | 0.348892 | 51.077478 | 0.483851 | 0.756080 | 0.875094 | 0.443289 | 4.031056 | 2206 |
| KD_every | 0.339704 | 50.012651 | 0.483985 | 0.769286 | 0.884139 | 0.396170 | 3.506637 | 2206 |

### PLE_training/winter/trajectory_6009

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ | Calls |
|---|---|---|---|---|---|---|---|---|
| A_every | 0.440121 | 55.320550 | 0.422947 | 0.735320 | 0.877592 | 0.606827 | 5.014265 | 2205 |
| A_fixed2 | 0.449838 | 55.826714 | 0.419619 | 0.731271 | 0.874442 | 0.612906 | 5.337441 | 1103 |
| A_fixed5 | 0.519251 | 59.925677 | 0.402322 | 0.715413 | 0.858910 | 0.655943 | 10.533072 | 441 |
| A_budget_fixed | 0.448878 | 55.784023 | 0.420046 | 0.731742 | 0.874446 | 0.612386 | 5.376952 | 1228 |
| A_v2 | 0.443212 | 55.511515 | 0.421146 | 0.733813 | 0.876556 | 0.607600 | 5.055119 | 1664 |
| Original_every | 0.451437 | 53.665238 | 0.407627 | 0.716455 | 0.864294 | 0.627238 | 4.647456 | 2205 |
| KD_every | 0.430639 | 55.166191 | 0.438641 | 0.741015 | 0.880190 | 0.584823 | 5.083697 | 2205 |

Subgroup caution: A regresses on Kite sunny, with AbsRel +10.80% and near RMSE +79.29% versus Original. V2 near RMSE is worse than budget-fixed on Kite foggy, Kite sunny and PLE spring; its PLE winter improvement produces the small pooled near-RMSE benefit. The sunny trajectory has zero defined critical events, so event rates/delays there remain undefined rather than zero-error claims.

### Counts and execution boundaries

| Scope | Valid pixels / comparison | Near pixels / comparison |
|---|---|---|
| pooled | 377372106 | 105333081 |
| Kite_training/foggy/trajectory_2007 | 99812690 | 22311420 |
| Kite_training/sunny/trajectory_0008 | 87117266 | 945433 |
| PLE_training/spring/trajectory_5008 | 93182086 | 35039499 |
| PLE_training/winter/trajectory_6009 | 97260064 | 47036729 |

All seven traces score every current frame, including reused predictions. Per-path/pooled signed bias, floor diagnostics, mean/max ages, reasons, event delays and censoring, actual inference/scheduler/transfer timing, and complete traces are in the [full independent study report](../results/independent_test_20261030/REPORT.md), [comparison CSV](../results/independent_test_20261030/comparison.csv) and [results JSON](../results/independent_test_20261030/results.json).

Actual timing was one CPU FP32 pass, batch 1, two threads, rotated seven-policy order and 10 excluded synthetic warmups per model. Shared I/O was 33.710 s, shared decode/resize 402.683 s, and whole-matrix wall 1669.140 s. Inference includes transfer and forward; scheduler includes decision and fresh risk-cache update. Offline scoring/logging are excluded from per-policy compute. Matrix wall includes all seven comparisons, so it is not single-policy end-to-end throughput. CUDA was unavailable in this runtime; historical validation CUDA timings are separate. One pass and uncontrolled OS/cache/load do not establish reproducible speedups or energy savings.

No samples were silently discarded and no nonfinite predictions were hidden. Per-path errors and ranking reversals remain visible; pixel totals do not represent independent samples. Four reserved simulated paths and possible shared scenery limit generalization. Use [the study guide](independent_final_test.md) for full-data requirements and the safe `python scripts/verify_independent_test.py` audit; do not remove the exposure ledger to rerun a completed study.

## Demonstration, reproduction and delivery

Five existing tabs are retained with improved navigation/context, explicit statuses, paired overall/near charts, preselected-frame buttons, fixed colour legends, compact evidence groups and fresh/reused playback indicators. A remains default; all actual inference requires the explicit button. Replay is labelled precomputed with its requested display rate separate from measured performance. No UI path trains, tunes, benchmarks or opens reserved inputs. The same localhost8501 service/installation is refreshed; no second server or cloud deployment. Missing external inputs show unavailable.

```bash
python -m pip install -e '.[dashboard]'
python -m streamlit run scripts/dashboard.py
```

Relocated source checkouts use portable `configs/presentation.json` with explicit `MIDAIR_ROOT`, `AERODEPTH_PRESENTATION_CACHE`, `AERODEPTH_COMPARISON_CACHE` and private `AERODEPTH_KD_CHECKPOINT` environment locations. Unset locations resolve to unavailable, never broad discovery/current directory. The laptop keeps its ignored local config. Both small inference exports are explicitly included; private full checkpoints, datasets/caches/environments/credentials remain excluded. [Dashboard guide](presentation_dashboard.md) includes validation reproduction into fresh frozen outputs. Do not rerun the completed local study to make slides or refresh UI.

Dataset-free A inference from the source delivery:

```bash
aerodepth infer \
  --weights models/presentation_A_epoch9_20261029T155521Z/supervised_A_epoch9.state_dict.pt \
  --metadata models/presentation_A_epoch9_20261029T155521Z/supervised_A_epoch9.json \
  --image /path/to/your_RGB.jpg --output /path/to/new_prediction.npy --device cpu
```

Classroom sequence: Overview trade-off cards and test status → Model comparison pooled then Kite foggy → Depth demonstration typical/poor presets, original/A/KD selectors and one explicit inference → A recorded V2 replay with age/source → Scheduler V2 versus budget quality and actual timing → Evidence downloads for the independent protocol, exposure, completion and result table. Fresh is not necessarily correct; finish with limitations rather than a safety claim.

## Limitations and conclusion

Only two familiar validation paths informed selection and analysis, including pilot exposure. Conditions and neighbouring frames are correlated; pooled pixels heavily favour PLE near field. Synthetic scenes/simulated sensors and shared scenery limit real-world transfer. Matched A/KD controls permit a narrow run comparison, not statistical proof across seeds; original training differs in data/budget. Teacher near regression mechanisms (loss weighting, BN, coverage, architecture) remain hypotheses. The gradient probe did not justify repair; rejected KD does not establish that distillation always fails. Timing windows, I/O/cache state and desktop activity limit extrapolation. The independent study adds four reserved simulation paths; it does not measure energy/battery gains, flight safety or guaranteed obstacle detection.

Validation conclusion: expanded supervised A improves overall range prediction with a near-field regression; exploratory KD underperforms A on that recorded comparison. Live risk caching reduced scheduler work without a robust measured end-to-end improvement.

On the independent test, A versus Original changes pooled AbsRel by -1.82% and near RMSE by +1.41%. KD versus A changes pooled AbsRel by -0.13% and near RMSE by +1.80%. Negative changes mean lower error. These new outcomes do not erase the negative validation findings or the failed teacher gate; A remains the preselected demonstration model.

The frozen primary V2-versus-budget-fixed comparison uses 3,718 versus 4,912 actual calls. AbsRel changes +0.43%, near RMSE -0.25%, and measured inference-plus-scheduler time -17.08%. Maximum observed event-update delay is 4 versus 1 frames; right-censored events are 0 versus 0. A fresh update is not a detection guarantee.

Verification and exact changed-source identities are recorded separately in this delivery results directory; completed historical protocols/results remain unchanged. The owner subsequently authorized a selective commit/push after final verification. No automatic deployment promotion or history rewrite is part of this delivery.

The [29 October acquisition inventory](../results/acquisition_publication_20261029T180214Z/REPORT.md) is historical. The 30 October Downloads inventory and new ready protocol supersede its acquisition blocker; frozen historical records remain unchanged.
