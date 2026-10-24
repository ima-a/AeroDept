# Frozen MobileNet validation-only scheduler experiments

All tuned results below are exploratory validation selection results on two trajectories, not an independent estimate of generalization. Final test images/depth were never loaded. No retraining, preprocessing/depth-limit changes or Git operations. Original run, audit, source and config files were hashed and preserved.

## Protocol and reference

Epoch-4 checkpoint, SHA-256 `c1d4279cdaf594d1c15bce1f4f1b6ae322c7ec3ef28349df1dd47d975818ebd8`. Full chronological 4410-frame replay; 2205 each Kite/foggy and PLE/spring. FP32 eval-mode batch1; frozen 192×320 preprocessing and mask; near GT<10m; strict delta ratios <1.25^k; valid-pixel pooled metrics. No prediction warp/interpolation, just last prediction reuse scored against EVERY current GT. State resets at each trajectory. Audited batch4 comparison passed all predeclared tolerances: errors/deltas 1e-5, RMSE 1e-3m, exact pixel counts. See reference_agreement.json for numerical differences.

## Causality and motion provenance

The original AdaptiveScheduler is reused unchanged. Actual policy decisions consume current [0,1] RGB, preceding RGB, current navigation, previous inferred depth and age. Predicted ROI validity means finite positive PREDICTIONS only. No GT depth, GT mask or future navigation enters decisions. Offline scoring, including GT event labels and mask-based predicted hazard scoring, occurs after the decision. Cached search accepts only decision_features.json and selects risk from the last inferred source frame. Actual selected decisions match all 4410 cached decisions exactly.

Navigation is privileged simulated groundtruth/velocity (100Hz), plus simulated imu/accelerometer and imu/gyroscope (100Hz). HDF5 gps/velocity is visited first but overwritten by groundtruth/velocity; effective decisions use no GPS velocity estimate. The motion score takes a maximum across velocity, acceleration and angular-velocity norms with different units; acceleration includes gravity, so it is an empirical score rather than a calibrated physical-motion estimate. Sample indices floor(ordinal*100/25), never future indices. All 4410 manifest vectors checked against both selected validation HDF5 groups. Assumes common time origin and ideal availability; real sensor latency, noise and online velocity estimation are not established.

## Quality comparison

| Policy | Scope | AbsRel | RMSE m | d1 | d2 | d3 | Near AbsRel | Near RMSE m | Valid pixels | Near pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| every | Overall | 0.301540 | 127.335344 | 0.498036 | 0.767152 | 0.894330 | 0.196167 | 3.362006 | 212903814 | 32267840 |
| every | Kite_training/foggy/trajectory_2003 | 0.357499 | 165.193258 | 0.399073 | 0.684972 | 0.857466 | 0.427121 | 4.379248 | 114412073 | 894287 |
| every | PLE_training/spring/trajectory_5000 | 0.236536 | 57.876310 | 0.612996 | 0.862615 | 0.937152 | 0.189583 | 3.328456 | 98491741 | 31373553 |
| fixed2 | Overall | 0.302756 | 127.337361 | 0.497085 | 0.766642 | 0.894053 | 0.197344 | 3.633127 | 212903814 | 32267840 |
| fixed2 | Kite_training/foggy/trajectory_2003 | 0.357646 | 165.083112 | 0.398691 | 0.685203 | 0.857895 | 0.434435 | 4.449627 | 114412073 | 894287 |
| fixed2 | PLE_training/spring/trajectory_5000 | 0.238995 | 58.249776 | 0.611384 | 0.861244 | 0.936057 | 0.190586 | 3.607145 | 98491741 | 31373553 |
| fixed5 | Overall | 0.312564 | 127.946823 | 0.493304 | 0.762815 | 0.890481 | 0.204631 | 5.610869 | 212903814 | 32267840 |
| fixed5 | Kite_training/foggy/trajectory_2003 | 0.360177 | 165.292124 | 0.399604 | 0.684575 | 0.856774 | 0.455130 | 4.641937 | 114412073 | 894287 |
| fixed5 | PLE_training/spring/trajectory_5000 | 0.257254 | 60.408138 | 0.602151 | 0.853702 | 0.929636 | 0.197490 | 5.636046 | 98491741 | 31373553 |
| adaptive_existing | Overall | 0.301540 | 127.335344 | 0.498036 | 0.767152 | 0.894330 | 0.196167 | 3.362006 | 212903814 | 32267840 |
| adaptive_existing | Kite_training/foggy/trajectory_2003 | 0.357499 | 165.193258 | 0.399073 | 0.684972 | 0.857466 | 0.427121 | 4.379248 | 114412073 | 894287 |
| adaptive_existing | PLE_training/spring/trajectory_5000 | 0.236536 | 57.876310 | 0.612996 | 0.862615 | 0.937152 | 0.189583 | 3.328456 | 98491741 | 31373553 |
| adaptive_selected | Overall | 0.302756 | 127.337361 | 0.497085 | 0.766642 | 0.894053 | 0.197344 | 3.633127 | 212903814 | 32267840 |
| adaptive_selected | Kite_training/foggy/trajectory_2003 | 0.357646 | 165.083112 | 0.398691 | 0.685203 | 0.857895 | 0.434435 | 4.449627 | 114412073 | 894287 |
| adaptive_selected | PLE_training/spring/trajectory_5000 | 0.238995 | 58.249776 | 0.611384 | 0.861244 | 0.936057 | 0.190586 | 3.607145 | 98491741 | 31373553 |
| fixed_matched | Overall | 0.302756 | 127.337361 | 0.497085 | 0.766642 | 0.894053 | 0.197344 | 3.633127 | 212903814 | 32267840 |
| fixed_matched | Kite_training/foggy/trajectory_2003 | 0.357646 | 165.083112 | 0.398691 | 0.685203 | 0.857895 | 0.434435 | 4.449627 | 114412073 | 894287 |
| fixed_matched | PLE_training/spring/trajectory_5000 | 0.238995 | 58.249776 | 0.611384 | 0.861244 | 0.936057 | 0.190586 | 3.607145 | 98491741 | 31373553 |

## Calls, freshness and critical events

| Policy | Scope | Calls | Skip rate | Mean age | Max age | Events | Fresh on event | Missed-update rate | Mean delay | Max delay | Censored | Near estimated on event |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| every | Overall | 4410 | 0.000000 | 0.000000 | 0 | 11 | 11 | 0.000000 | 0.000000 | 0 | 0 | 7 |
| every | Kite_training/foggy/trajectory_2003 | 2205 | 0.000000 | 0.000000 | 0 | 1 | 1 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| every | PLE_training/spring/trajectory_5000 | 2205 | 0.000000 | 0.000000 | 0 | 10 | 10 | 0.000000 | 0.000000 | 0 | 0 | 7 |
| fixed2 | Overall | 2206 | 0.499773 | 0.499773 | 1 | 11 | 6 | 0.454545 | 0.454545 | 1 | 0 | 8 |
| fixed2 | Kite_training/foggy/trajectory_2003 | 1103 | 0.499773 | 0.499773 | 1 | 1 | 1 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| fixed2 | PLE_training/spring/trajectory_5000 | 1103 | 0.499773 | 0.499773 | 1 | 10 | 5 | 0.500000 | 0.500000 | 1 | 0 | 8 |
| fixed5 | Overall | 882 | 0.800000 | 2.000000 | 4 | 11 | 4 | 0.636364 | 1.181818 | 4 | 0 | 8 |
| fixed5 | Kite_training/foggy/trajectory_2003 | 441 | 0.800000 | 2.000000 | 4 | 1 | 0 | 1.000000 | 1.000000 | 1 | 0 | 0 |
| fixed5 | PLE_training/spring/trajectory_5000 | 441 | 0.800000 | 2.000000 | 4 | 10 | 4 | 0.600000 | 1.200000 | 4 | 0 | 8 |
| adaptive_existing | Overall | 4410 | 0.000000 | 0.000000 | 0 | 11 | 11 | 0.000000 | 0.000000 | 0 | 0 | 7 |
| adaptive_existing | Kite_training/foggy/trajectory_2003 | 2205 | 0.000000 | 0.000000 | 0 | 1 | 1 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| adaptive_existing | PLE_training/spring/trajectory_5000 | 2205 | 0.000000 | 0.000000 | 0 | 10 | 10 | 0.000000 | 0.000000 | 0 | 0 | 7 |
| adaptive_selected | Overall | 2206 | 0.499773 | 0.499773 | 1 | 11 | 6 | 0.454545 | 0.454545 | 1 | 0 | 8 |
| adaptive_selected | Kite_training/foggy/trajectory_2003 | 1103 | 0.499773 | 0.499773 | 1 | 1 | 1 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| adaptive_selected | PLE_training/spring/trajectory_5000 | 1103 | 0.499773 | 0.499773 | 1 | 10 | 5 | 0.500000 | 0.500000 | 1 | 0 | 8 |
| fixed_matched | Overall | 2206 | 0.499773 | 0.499773 | 1 | 11 | 6 | 0.454545 | 0.454545 | 1 | 0 | 8 |
| fixed_matched | Kite_training/foggy/trajectory_2003 | 1103 | 0.499773 | 0.499773 | 1 | 1 | 1 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| fixed_matched | PLE_training/spring/trajectory_5000 | 1103 | 0.499773 | 0.499773 | 1 | 10 | 5 | 0.500000 | 0.500000 | 1 | 0 | 8 |

Critical event: valid GT full-frame p10 enters <8m or decreases >=2m while <8m; reset per trajectory, including first-frame near entry. Missed-update rate means no NEW inference on the event frame. Delays count frames to next inference within the same trajectory; terminal unresolved events are right-censored. Zero events produce undefined rates/delays and no safety evidence. “Near estimated on event” independently checks used-prediction p10<8m on the current GT-valid pixels, offline only; it is a coarse depth-estimation check, not obstacle detection or collision safety. A fresh inference can still estimate depth incorrectly.

## Bounded search and selection

The first cached-search attempt failed while writing its first CSV because the output subdirectory was missing; search_attempt1_error.log is preserved. Directory creation was fixed, tests rerun, and the unchanged plan restarted before any selection. 12 explicit configurations and a deterministic selection rule were saved in search_plan.json only AFTER all four baseline replays and BEFORE search. Varied max_skip1/2/4, motion scale20/40, motion threshold .95/.99, visual scale .05/.15/.30 and threshold .90/.99, risk threshold .80/.90/.98, score threshold .60/.70/.80/.85 and four score-weight settings. MAX_SKIP counts skipped frames, so max_skip=1 allows a refresh every second frame. Risk distance remains10m; all scoring limits remain frozen. Each configuration and every frame decision/result remains in search/. Search uses cached counterfactual FP32 predictions for scoring and never labels cache speed as inference performance.

Eligibility on EACH trajectory: AbsRel/RMSE <=5% worse, near AbsRel/RMSE <=10% worse than every-frame; no unresolved critical updates and maximum observed update delay <=2frames. Among eligible minimize worst per-trajectory call fraction, then worst relative quality, mean observed event delay, config id. No-event trajectories do not imply safety. Selected `candidate_03`, eligible=True. See selection.json for full configuration.

Compute-matched fixed schedule uses exact same number K of calls on each validation trajectory, at ceil(k*N/K), k=0..K-1. Uniform deterministic cadence uses preselected validation budgets and sequence lengths, with no event/GT alignment. Neither this nor tuning is an independent held-out evaluation.

## Runtime measurement

| Policy | Calls | Inference seconds | Scheduler seconds | Policy compute seconds | Wall incl IO/scoring/instrumentation |
|---|---|---|---|---|---|
| every | 4410 | 56.240649 | 0.013249 | 56.253898 | 149.997791 |
| fixed2 | 2206 | 31.233973 | 0.014786 | 31.248759 | 145.601180 |
| fixed5 | 882 | 12.014766 | 0.015099 | 12.029865 | 108.691655 |
| adaptive_existing | 4410 | 61.298120 | 8.458052 | 69.756172 | 110.536218 |
| adaptive_selected | 2206 | 27.749681 | 7.623687 | 35.373369 | 153.075338 |
| fixed_matched | 2206 | 27.676940 | 0.015324 | 27.692264 | 141.752735 |

All table runtimes execute actual model calls on GTX1650, same FP32 batch1 model, 10 excluded warm-up calls per policy, CUDA synchronized before/after each model call; inference timer includes CPU→GPU input transfer, model execution and GPU→CPU output transfer. CPU scheduler overhead includes RGB features and previous-prediction quantile risk. Offline dataset replay permits slower-than-camera processing; age is frame age, not queue/latency age. Warm-ups use an already GPU-resident tensor and time is excluded. Wall includes I/O, scoring and diagnostics, so wall is not live inference performance. First every-frame pass additionally writes the cache and performs floor diagnostics. Single pass per policy; no randomized repeated hardware benchmark, confidence interval, energy or battery measurement. Per-trajectory timings and full per-frame latency logs in comparison.csv and *_frames.csv.

## Floor/risk evidence

| Trajectory | Floor all pixels | Floor valid GT pixels | ROI floor fraction | Risk p50 | Risk p90 | Fresh risk>=.9 | Motion=1 frames | Existing risk reason | Selected risk reason |
|---|---|---|---|---|---|---|---|---|---|
| Kite_training/foggy/trajectory_2003 | 20148494 | 5104001 | 0.000009 | 0.000000 | 0.009486 | 0 | 2205 | 0 | 0 |
| PLE_training/spring/trajectory_5000 | 13694321 | 4331 | 0.001153 | 0.349471 | 0.618063 | 0 | 2205 | 0 | 0 |

Risk distributions, floor fractions and trigger counts cover BOTH complete trajectories, not the four older ROIs. Fresh-prediction risk threshold exceedances are counterfactual full-split diagnostics. Actual decision risk uses the last selected prediction, which can differ; those exceedances and critical_risk reasons are separately saved. Priority rules (first frame/max_skip before risk, risk before motion) mean threshold exceedances need not equal inference reason counts. Whole-image floor prevalence is substantial, but the mean central lower-ROI floor fraction is only about 0.0009% for Kite and 0.1153% for PLE. No fresh risk reaches .9 on either trajectory; frozen adaptive has zero critical_risk reasons. Therefore the whole-image floor count does not explain triggering by the frozen risk gate. Floor predictions could still affect lower risk thresholds, and the scheduler never receives GT masks to remove invalid-GT regions.

## Evidence

baseline_comparison.csv, tuned_comparison.csv, comparison.csv, search_comparison.csv; quality_vs_inference.png; scheduler_timelines.png; reason_counts.csv/png; floor_risk_diagnostics.json/png; *_frames.csv and search/*_frames.csv; search_plan.json, selection.json, matched_fixed_positions.json; protocol.json, motion_provenance.json, motion_diagnostics.json, reference_agreement.json, cached_actual_agreement.json, verification_checks.json, actual_latency_distributions.csv, critical_event_details.csv, preservation.json, hashes.json and the public reproduction guide. Selected audit prediction examples and training curves are published in ../expanded_validation/.

## Conclusions

Kite_training/foggy/trajectory_2003: selected adaptive skips 50.0%; relative to every-frame changes: absrel +0.04%, rmse -0.07%, near_absrel +1.71%, near_rmse +1.61%. Versus exact call-matched fixed: absrel +0.00%, rmse +0.00%, near_absrel +0.00%, near_rmse +0.00%. Adaptive/fixed missed-update rates 0.000000/0.000000; max event delays 0/0 frames.

PLE_training/spring/trajectory_5000: selected adaptive skips 50.0%; relative to every-frame changes: absrel +1.04%, rmse +0.65%, near_absrel +0.53%, near_rmse +8.37%. Versus exact call-matched fixed: absrel +0.00%, rmse +0.00%, near_absrel +0.00%, near_rmse +0.00%. Adaptive/fixed missed-update rates 0.500000/0.500000; max event delays 1/1 frames.

The selected adaptive policy is EXACTLY fixed-2 on all 4410 frames, including inference decisions, source predictions and ages, and equals the matched fixed schedule. Its useful compute/quality trade-off comes entirely from fixed-rate reuse. It provides no adaptive quality or freshness improvement on either trajectory and adds scheduler overhead. Within this bounded validation study, adaptive scheduling does not help relative to compute-matched fixed scheduling. The original adaptive policy also provides no savings. More aggressive searched configurations fail the predeclared per-trajectory quality/freshness limits; this does not prove every possible adaptive scheduler is ineffective.

The original configuration’s full-split behavior and tuned quality/call trade-offs must be interpreted per trajectory. A compute saving is a reduction in model calls; actual inference-plus-scheduler time is reported separately. Compute-matched comparisons above determine whether quality or freshness improves; mixed quality changes do not establish uniform superiority. Model quality remains limited: foggy Kite near-field error is high, floor saturation remains a model issue, far-depth geometry is poor and training overfits after epoch4. Two selected validation trajectories and privileged navigation prevent stronger generalization or safety conclusions. No measured energy/battery improvement is claimed.


Public reproduction: see ../../docs/reproduction.md and scripts/reproduce.py in the repository. Full local frame logs/cache and original immutable artifacts remain excluded from Git. This public copy retains the original scientific measurements; references to private evidence filenames describe archived provenance, not required runtime dependencies.
