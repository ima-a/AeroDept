# Bounded scheduler v2: validation selection, not final test

Eight declared configurations only. Frozen epoch4 checkpoint and scoring contract; no retraining or test image/depth access. Existing v1 negative results remain preserved.

Motion change: causal 1Hz simulated GPS velocity held from floor(frame/25), normalized at20m/s; simulated IMU accelerometer differences and gyroscope norms normalized separately. These are proxies, not calibrated dynamics or measured physical sensors. No privileged groundtruth velocity enters v2 decisions. Sensor latency is idealized.

Separate RGB design change: compare current 36x64 thumbnail with last inferred frame. Original scheduler uses immediately preceding frame even when skipped. Both changes are evaluated jointly; this is not an ablation proving either change caused a gain. See timelines.png for original/v2 inputs.

Same deterministic selection rule as v1: each trajectory AbsRel/RMSE <=5% degradation, near errors <=10%, max event-update delay <=2, no terminal censoring; then minimize worst trajectory call fraction, worst relative error, mean event delay and id. Selected v2_02, eligible=True. Search was not expanded.

Complete metrics, delta values, near valid-pixel counts, call/freshness statistics and measured runtime are in comparison.csv; all eight configs/results/frame logs are retained.

| Policy | Scope | Calls | AbsRel | RMSE m | Near AbsRel | Near RMSE m | Missed-update rate | Max delay |

|---|---|---:|---:|---:|---:|---:|---:|---:|

| every | Overall | 4410 | 0.3015399580446375 | 127.33534435672304 | 0.19616659020141977 | 3.3620062389556646 | 0.0 | 0 |

| every | Kite_training/foggy/trajectory_2003 | 2205 | 0.35749885801497044 | 165.19325793792444 | 0.42712065017288436 | 4.379248064552112 | 0.0 | 0 |

| every | PLE_training/spring/trajectory_5000 | 2205 | 0.23653578933565478 | 57.87631012899888 | 0.1895833634489477 | 3.3284562317991266 | 0.0 | 0 |

| original_adaptive | Overall | 4410 | 0.3015399580446375 | 127.33534435672304 | 0.19616659020141977 | 3.3620062389556646 | 0.0 | 0 |

| original_adaptive | Kite_training/foggy/trajectory_2003 | 2205 | 0.35749885801497044 | 165.19325793792444 | 0.42712065017288436 | 4.379248064552112 | 0.0 | 0 |

| original_adaptive | PLE_training/spring/trajectory_5000 | 2205 | 0.23653578933565478 | 57.87631012899888 | 0.1895833634489477 | 3.3284562317991266 | 0.0 | 0 |

| v1_selected | Overall | 2206 | 0.3027562801832172 | 127.33736126525586 | 0.19734374649530972 | 3.6331267042070126 | 0.45454545454545453 | 1 |

| v1_selected | Kite_training/foggy/trajectory_2003 | 1103 | 0.35764558126513796 | 165.08311222755026 | 0.43443533176277577 | 4.449627095706972 | 0.0 | 0 |

| v1_selected | PLE_training/spring/trajectory_5000 | 1103 | 0.2389946017059965 | 58.24977597995772 | 0.190585572739405 | 3.607144611991745 | 0.5 | 1 |

| v2_selected | Overall | 2455 | 0.3025490565931531 | 127.6556844881935 | 0.19438840294929866 | 3.2565133200459253 | 0.09090909090909091 | 2 |

| v2_selected | Kite_training/foggy/trajectory_2003 | 905 | 0.3588411862154233 | 165.5975943786333 | 0.42712065017288436 | 4.379248064552112 | 0.0 | 0 |

| v2_selected | PLE_training/spring/trajectory_5000 | 1550 | 0.23715779456166314 | 58.059176975082174 | 0.18775448988332114 | 3.218774908216256 | 0.1 | 2 |

| v2_matched_fixed | Overall | 2455 | 0.30249158573995255 | 127.34454899165782 | 0.1967687307211628 | 3.557915633047273 | 0.2727272727272727 | 1 |

| v2_matched_fixed | Kite_training/foggy/trajectory_2003 | 905 | 0.3579664495676108 | 165.14071720112184 | 0.43552226158594154 | 4.480013333029176 | 1.0 | 1 |

| v2_matched_fixed | PLE_training/spring/trajectory_5000 | 1550 | 0.23804969339981108 | 58.09385600381288 | 0.18996318406036636 | 3.5281008152588926 | 0.2 | 1 |

Actual selected/matched timing: 10 identical excluded warmups; synchronized CUDA FP32 actual policy inference, including H2D/D2H and CPU scheduler overhead. Cache-search seconds are offline only. No energy/battery claims. One hardware pass each; GPU timing variability remains.

Selected configuration: `{"max_skip": 4, "accel_change_scale": 2.0, "gyro_scale": 0.5, "visual_scale": 0.08, "critical_risk": 0.9, "extreme_motion": 0.9, "extreme_visual": 0.9, "decision_threshold": 0.65, "motion_weight": 0.25, "visual_weight": 0.3, "risk_weight": 0.3, "age_weight": 0.15, "gps_speed_scale": 20.0}`.

## Conclusion

Kite_training/foggy/trajectory_2003: skips 58.96%; compared to call-matched fixed errors: absrel +0.244%, rmse +0.277%, near_absrel -1.929%, near_rmse -2.249%; missed-update rate 0.0/1.0 and max delay 0/1.

PLE_training/spring/trajectory_5000: skips 29.71%; compared to call-matched fixed errors: absrel -0.375%, rmse -0.060%, near_absrel -1.163%, near_rmse -8.767%; missed-update rate 0.1/0.2 and max delay 2/1.

Critical events are offline GT p10<8m entry or >=2m drop; fresh updates are distinct from correct predicted hazard/depth. Zero-event rates are undefined. Nearfield model errors, floor saturation, poor far-depth geometry, only two selected trajectories and simulated navigation limit conclusions. These exploratory validation results cannot establish generalization or flight safety.

Preservation and cached/actual agreement passed. Initial thumbnail generation failed because sandbox worker IPC was blocked; the error log is preserved and thumbnail generation succeeded with host IPC access.


Public reproduction: see ../../docs/reproduction.md and scripts/reproduce.py in the repository. Full local frame logs/cache and original immutable artifacts remain excluded from Git. This public copy retains the original scientific measurements; references to private evidence filenames describe archived provenance, not required runtime dependencies.
