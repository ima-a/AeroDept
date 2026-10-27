# Validation timing verification

Protocol SHA-256: `f167c49078bde5bc09eed2a552000d8f9fe5812df13d78a60eaf40085cfa1f8f`.

Three measured repetitions per policy; predeclared alternating order. FP32 batch1 CPU, two torch threads, ten identical excluded warmup calls before each run. Actual calls, including all scheduler decision overhead, are measured. Cache generation and I/O are excluded from compute; offline replay wall is separate.

| Rep | Order in pair | Policy | Calls | Model s | Scheduler s | Compute s |
|---|---|---|---:|---:|---:|---:|
| 1 | 1 | v2_selected | 2455 | 35.746649 | 4.743944 | 40.490593 |
| 1 | 2 | fixed_validation_matched | 2455 | 36.090598 | 0.009294 | 36.099892 |
| 2 | 1 | fixed_validation_matched | 2455 | 36.333889 | 0.009298 | 36.343187 |
| 2 | 2 | v2_selected | 2455 | 36.167056 | 4.759072 | 40.926128 |
| 3 | 1 | v2_selected | 2455 | 36.906174 | 4.851666 | 41.757841 |
| 3 | 2 | fixed_validation_matched | 2455 | 36.676656 | 0.009163 | 36.685819 |

| Policy | Median compute s | Min s | Max s | Spread s |
|---|---:|---:|---:|---:|
| v2_selected | 40.926128 | 40.490593 | 41.757841 | 1.267248 |
| fixed_validation_matched | 36.343187 | 36.099892 | 36.685819 | 0.585927 |

Each policy executes 2,455 measured calls per repetition (905 Kite; 1,550 PLE); 14,730 measured calls plus 60 excluded warmups overall. This matched comparator uses the original per-trajectory validation budgets. The final-test primary comparator instead uses the separately frozen pooled validation fraction, identically applied to each unseen path. No timing result changes any policy.

Individual per-trajectory results and median/min/max/spread are in CSV; every actual call and scheduler duration is retained in frame logs. Three repetitions quantify this run's spread, not a population confidence interval. CPU results cannot establish GTX1650 speedup, energy, battery or flight-safety improvement.
