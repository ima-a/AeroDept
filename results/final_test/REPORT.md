# Frozen independent final test

Protocol SHA-256: `f167c49078bde5bc09eed2a552000d8f9fe5812df13d78a60eaf40085cfa1f8f`. One study; no tuning or retraining.

All current frames are scored, including reused predictions. Overall metrics pool valid pixels. See comparison.csv for deltas/counts/timing and per-policy JSON/CSV for event delays and frame evidence.

| Policy | Scope | Calls | AbsRel | RMSE m | Near AbsRel | Near RMSE m | Events | Missed update | Max delay | Compute s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| every | Overall | 4436 | 0.473621 | 108.588 | 0.806198 | 5.93942 | 24 | 0 | 0 | 66.1203 |
| every | Kite_training/sunny/trajectory_0004 | 2232 | 0.497952 | 157.356 | 1.39895 | 9.71225 | 7 | 0 | 0 | 33.4645 |
| every | PLE_training/spring/trajectory_5003 | 2204 | 0.452954 | 27.8297 | 0.708577 | 5.05485 | 17 | 0 | 0 | 32.6558 |
| fixed2 | Overall | 2218 | 0.474396 | 108.632 | 0.806075 | 5.94853 | 24 | 0.583333 | 1 | 33.0091 |
| fixed2 | Kite_training/sunny/trajectory_0004 | 1116 | 0.496569 | 157.408 | 1.38145 | 9.91029 | 7 | 0.714286 | 1 | 16.6996 |
| fixed2 | PLE_training/spring/trajectory_5003 | 1102 | 0.455562 | 27.8992 | 0.711316 | 5.00376 | 17 | 0.529412 | 1 | 16.3095 |
| fixed5 | Overall | 888 | 0.493873 | 108.819 | 0.855973 | 7.20454 | 24 | 0.791667 | 4 | 13.1962 |
| fixed5 | Kite_training/sunny/trajectory_0004 | 447 | 0.522871 | 157.487 | 1.67926 | 13.1027 | 7 | 0.571429 | 3 | 6.63545 |
| fixed5 | PLE_training/spring/trajectory_5003 | 441 | 0.46924 | 28.8492 | 0.720384 | 5.6727 | 17 | 0.882353 | 4 | 6.56071 |
| v2_selected | Overall | 2837 | 0.474483 | 108.478 | 0.805441 | 5.93092 | 24 | 0.208333 | 4 | 46.5923 |
| v2_selected | Kite_training/sunny/trajectory_0004 | 1158 | 0.498595 | 157.184 | 1.39763 | 9.66865 | 7 | 0.285714 | 3 | 19.6104 |
| v2_selected | PLE_training/spring/trajectory_5003 | 1679 | 0.454002 | 27.8653 | 0.707913 | 5.05699 | 17 | 0.176471 | 4 | 26.9819 |
| fixed_validation_budget | Overall | 2470 | 0.473906 | 108.581 | 0.804753 | 6.04513 | 24 | 0.458333 | 1 | 36.8222 |
| fixed_validation_budget | Kite_training/sunny/trajectory_0004 | 1243 | 0.497305 | 157.334 | 1.39123 | 10.2064 | 7 | 0.428571 | 1 | 18.6278 |
| fixed_validation_budget | PLE_training/spring/trajectory_5003 | 1227 | 0.45403 | 27.8822 | 0.708165 | 5.04044 | 17 | 0.470588 | 1 | 18.1943 |

Primary comparison: V2 versus fixed_validation_budget (same pooled validation fraction for each test trajectory, frozen before exposure). Positive error changes are regressions; all remain in v2_comparisons.csv. No retrospective test-call-matched policy was evaluated.

Fresh inference is an update, not evidence of correct obstacle detection. Zero-event rates are undefined; terminal delays are right censored. Two simulation paths and validation-selected model/policy limit generalization. Compute excludes I/O, preprocessing and scoring; this is CPU timing, not measured energy, battery or flight safety.

## Interpretation and retained regressions

The selected V2 does not show a uniform advantage on the held-out paths. Against the fixed validation-budget comparator, pooled near RMSE falls 1.889% (6.04513→5.93092 m), while near AbsRel rises 0.0855% and overall AbsRel rises 0.1219%. V2 makes 14.858% more model calls (2837 versus 2470) and uses 26.533% more measured CPU compute (46.5923 versus 36.8222 s). These are different realized test budgets; the comparator was independently frozen from validation, without test-call matching.

Kite: V2 near RMSE improves 5.269%, but near AbsRel regresses 0.4597% and overall AbsRel regresses 0.2594%. Its 1158 calls are fewer than the comparator's 1243, yet compute is 5.275% higher. PLE: V2 near RMSE regresses 0.3284%, calls increase 36.838% (1679 versus 1227) and compute increases 48.298%. Other PLE error changes versus the primary comparator are small improvements. These path differences remain visible rather than being hidden by pooling.

V2 has fewer missed fresh updates (5/24 events versus 11/24) and slightly lower mean delay (0.4167 versus 0.4583 frames), but its maximum delays are 3 frames on Kite and 4 on PLE, versus 1 for fixed budget on each. Both paths exceed the 2-frame delay ceiling used during validation selection. There are no right-censored events. These failures are retained; no new policy is selected. A 4-frame camera delay is 0.16 s under ideal 25 Hz timing, without queue or sensor latency.

Depth quality is weak on these paths even with every-frame inference: pooled AbsRel 0.473621, RMSE 108.587793 m, near AbsRel 0.806198 and near RMSE 5.939422 m. The corresponding published validation AbsRel is 0.301540 and near AbsRel 0.196167. The path distributions differ, so this is a descriptive generalization limitation, not an estimate of a single causal effect. V2 cannot correct the model's underlying range errors.

| Scope | Frames | Valid pixels | Near pixels | Near share of pooled near pixels | Events |
|---|---:|---:|---:|---:|---:|
| Overall | 4436 | 187182008 | 48016063 | 100.000% | 24 |
| Kite_training/sunny/trajectory_0004 | 2232 | 85971621 | 6789662 | 14.140% | 7 |
| PLE_training/spring/trajectory_5003 | 2204 | 101210387 | 41226401 | 85.860% | 17 |

The offline current-GT-mask predicted-p10<8 m check is positive for only 9 of 24 V2 event frames. This coarse range-estimation check is separate from the 19 fresh updates and is not an obstacle-detection score. All event positions, delays, used-prediction p10 values and censoring are retained in critical_event_details.csv and per-policy JSON/CSV.

Validation timing was completed first: V2 compute median 40.926128 s, range 40.490593–41.757841; matched fixed median 36.343187 s, range 36.099892–36.685819, with 2455 actual calls per run. V2 is 12.61% slower at the medians. All six individual measurements, per-trajectory results, scheduler durations and min/max/spread are retained under validation_timing/. CPU timing is specific to this runtime.

The manifest and original pilot exclusions passed; 192 available weather-variant position streams showed no cross-split identity. This does not establish independent geographic scenery. The frozen model/V2 were selected on two validation trajectories, with prior V1 work and joint motion/RGB changes; no isolated causal gain is established. No energy, battery, flight safety or correct obstacle-detection improvement is claimed. Zero-event rates remain undefined by contract.

The study made 12849 measured actual model calls plus 50 excluded warmup calls. Offline test wall time was 470.717 s, including I/O, preprocessing, scoring and instrumentation. It is not live inference throughput. The saved-result accounting audit passed without reading data again or invoking a model.

See [the frozen protocol](protocol.json), [validation timing](validation_timing/REPORT.md), [reproduction commands](reproduction_commands.sh), [accounting verification](verification.json), [preservation](preservation.json), [quality plot](quality_vs_calls.png), [freshness/event plot](freshness_events.png), and [hash manifest](SHA256SUMS). The 58-test suite passed; the optional generic real-data discovery test was skipped to avoid broad data discovery.
