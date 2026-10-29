# AeroDepth-Lite — consolidated final research report

Status: completed training and validation; independent reserved final study **NOT RUN**. No slides, new training, tuning, test-pixel access or Git publication in this delivery. This report consolidates saved evidence; it does not repeat the completed experiments.

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
| Reserved independent test |4|4|8,820 expected|NOT RUN; image acquisition missing|
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

## Independent final study — NOT RUN

Current healthy rescue indices contain no reserved image records, and explicit extracted targets are absent. The only surviving official configuration contains24 completed training image links plus6 sensor links and zero reserved image links. No archives were found in the inspected healthy non-dataset storage locations. Historical acquisition metadata also records reserved archives absent. A later bounded local inventory mounted the kernel-write-protected HDD read-only and checked14 documented HDD/USB roots plus the original Kaggle ZIP central-directory filenames. None of the four reserved realizations was present. The HDD was safely unmounted; its recorded hardware failure was not repaired. The detailed inventory is linked below. No reserved RGB/depth/header or archive member was opened.

All four corresponding climate HDF5 files are present. Request exactly8 archives (left RGB + depth each) on the [official form](https://midair.ulg.ac.be/download.html):

| Environment | Condition | Official page ID | Internal realization | Expected HDF5 frames |
|---|---|---|---|---:|
|Kite_training|foggy|0007|trajectory_2007|2204|
|Kite_training|sunny|0008|trajectory_0008|2205|
|PLE_training|spring|4008|trajectory_5008|2206|
|PLE_training|winter|4009|trajectory_6009|2205|

Human action: select only these conditions/IDs and left RGB/depth; enter your own email and accept the terms to receive a fresh authorized configuration. Complete CAPTCHA if shown (not confirmed by text inspection). Do not view sample overviews or request all paths. Provide access privately, never paste signed URLs publicly. Do not redownload identical HDF5 records. Current USB free space is about9.4GiB; requested compressed/extracted sizes are UNKNOWN. Measure official lengths and member sizes before promising download+extraction capacity; use healthy external storage and preserve backups.

The user has authorized ONE study without another confirmation, but actual prerequisites are not met. The old blocked proposal is preserved. This delivery adds a separate blocked V2 specification: A five frozen policies primary, plus original/KD every-frame predeclared secondary analyses. No ready protocol, exposure ledger or completion was fabricated. The existing opt-in runner supports the original A matrix; the newly requested supplementary/secondary/timing scope needs a separately versioned extension and focused guards before a genuine ready freeze. This is a remaining implementation prerequisite, not a completed final evaluation. Exact counts/root/archive/HDF5/manifest/source/model identities and measurement/aggregation definitions must be frozen before the first test decoder access. No split redraw or automatic retry after exposure.

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

Classroom sequence: Overview trade-off cards and test status → Model comparison pooled then Kite foggy → Depth demonstration typical/poor presets, original/A/KD selectors and one explicit inference → A recorded V2 replay with age/source → Scheduler V2 versus budget quality and actual timing → Evidence downloads and blocked final-study checklist. Fresh is not necessarily correct; finish with limitations rather than a safety claim.

## Limitations and conclusion

Only two familiar validation paths informed selection and analysis, including pilot exposure. Conditions and neighbouring frames are correlated; pooled pixels heavily favour PLE near field. Synthetic scenes/simulated sensors and shared scenery limit real-world transfer. Matched A/KD controls permit a narrow run comparison, not statistical proof across seeds; original training differs in data/budget. Teacher near regression mechanisms (loss weighting, BN, coverage, architecture) remain hypotheses. The gradient probe did not justify repair; rejected KD does not establish that distillation always fails. Timing windows, I/O/cache state and desktop activity limit extrapolation. There is no independent new-test result, measured energy/battery gain, flight safety or guaranteed obstacle detection.

Supported conclusion: expanded supervised A improves overall validation range prediction with a near-field regression; this rejected teacher does not yield a superior exploratory KD student. V2 offers a mixed quality/call trade-off, and caching reduces scheduler work without a robust measured end-to-end improvement. Preserve every negative result and complete the reserved independent study only after genuine acquisition/integrity/manifest/protocol prerequisites.

Verification and exact changed-source identities are recorded separately in this delivery results directory; completed historical protocols/results remain unchanged. No Git publication or automatic deployment promotion is performed.

Current bounded HDD/USB evidence: [acquisition inventory](../results/acquisition_publication_20261029T180214Z/REPORT.md). The exact8 missing streams remain separate from already-present HDF5 records. No duplicate download or final evaluation was started.
