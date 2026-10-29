# Final selected-student validation comparison

Completed 2026-10-29T21:12:39.477597+05:30 (IST). Validation only; no training, teacher repair, scheduler tuning, test access, new checkpoint selection or automatic promotion.

The official selections remain original epoch4, supervised A epoch9 and exploratory KD epoch9. Checkpoint byte hashes, saved selected-epoch records, parent sources/model contracts and KD completion/authorization were verified before any pixel inference. Actual paths come from the prior audit and recovered exploratory authorization, not the unmounted failed HDD. Handoff is the dated startup record; completion.json and10-epoch history provide the newer completion truth.

Original/A/teacher complete metrics and supplementary statistics were reused from the checksum-verified prior validation audit. Only KD needed a full4410-frame inference pass for bias, floor and depth bands. Its recomputed standard scores were compared to the existing selected checkpoint record. The accepted original/A contact inputs use six original two-frame batches (12 frames/model), plus one earlier aborted two-frame direct replay (14 actual forward frames/model total); teacher was not evaluated. No other checkpoint was loaded for selection.

All inference uses FP32 eval/inference mode, batch2, no autocast/TF32,192×320 full frame, unchanged normalization, GT decoding and finite inclusive[0.1,1000]m mask before nearest resize, near GT<10m. Delta rates use strict max(P/G,G/P)<1.25^k; metrics pool pixels and squared errors, rather than averaging frame/path RMSE. Floor rate is prediction<=0.101m divided by valid GT pixels. Near bias is mean(P−GT) on valid GT<10m, positive meaning overestimated range. Undefined JSON values are null; corresponding CSV cells are blank. Colour limits never clip the underlying metric predictions.

| Scope | Model | AbsRel | RMSE m | δ1 | δ2 | δ3 | Near AbsRel | Near RMSE m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pooled | Original epoch4 | 0.301540 | 127.335344 | 0.498036 | 0.767152 | 0.894330 | 0.196167 | 3.362006 |
| Pooled | Supervised A epoch9 | 0.265831 | 119.944144 | 0.626189 | 0.863269 | 0.922667 | 0.201899 | 3.906072 |
| Pooled | Exploratory KD epoch9 | 0.270049 | 121.206946 | 0.618070 | 0.860105 | 0.922285 | 0.209946 | 3.961365 |
| Kite foggy | Original epoch4 | 0.357499 | 165.193258 | 0.399073 | 0.684972 | 0.857466 | 0.427121 | 4.379248 |
| Kite foggy | Supervised A epoch9 | 0.290636 | 155.963703 | 0.612216 | 0.848630 | 0.901462 | 0.504336 | 4.695176 |
| Kite foggy | Exploratory KD epoch9 | 0.295285 | 157.806185 | 0.604495 | 0.844948 | 0.900903 | 0.542571 | 4.985648 |
| PLE spring | Original epoch4 | 0.236536 | 57.876310 | 0.612996 | 0.862615 | 0.937152 | 0.189583 | 3.328456 |
| PLE spring | Supervised A epoch9 | 0.237017 | 53.311489 | 0.642421 | 0.880274 | 0.947300 | 0.193278 | 3.881228 |
| PLE spring | Exploratory KD epoch9 | 0.240734 | 53.186731 | 0.633840 | 0.877711 | 0.947123 | 0.200465 | 3.928255 |

| Scope | Model | Valid pixels | Near pixels | Near bias m | Valid floor count | Valid floor % |
|---|---|---:|---:|---:|---:|---:|
| Pooled | Original epoch4 | 212,903,814 | 32,267,840 | 0.560990 | 5,108,332 | 2.399361 |
| Pooled | Supervised A epoch9 | 212,903,814 | 32,267,840 | 0.536461 | 3,828,105 | 1.798044 |
| Pooled | Exploratory KD epoch9 | 212,903,814 | 32,267,840 | 0.554255 | 4,191,844 | 1.968891 |
| Kite foggy | Original epoch4 | 114,412,073 | 894,287 | 3.589278 | 5,104,001 | 4.461069 |
| Kite foggy | Supervised A epoch9 | 114,412,073 | 894,287 | 4.320245 | 3,824,026 | 3.342327 |
| Kite foggy | Exploratory KD epoch9 | 114,412,073 | 894,287 | 4.651245 | 4,187,885 | 3.660352 |
| PLE spring | Original epoch4 | 98,491,741 | 31,373,553 | 0.474670 | 4,331 | 0.004397 |
| PLE spring | Supervised A epoch9 | 98,491,741 | 31,373,553 | 0.428606 | 4,079 | 0.004141 |
| PLE spring | Exploratory KD epoch9 | 98,491,741 | 31,373,553 | 0.437472 | 3,959 | 0.004020 |

Pixel counts match exactly across all models, cached audit, KD saved record and frozen validation reference: pooled212,903,814/32,267,840 valid/near; Kite114,412,073/894,287; PLE98,491,741/31,373,553. Band counts partition each valid count exactly. Only83 consecutive Kite frames1526–1608 contain near GT; their GT range is6.715–9.992m. They supply2.77% of pooled near pixels;97.23% come from PLE. These correlated frames and two familiar trajectories are not independent generalization evidence. No count discrepancy was omitted.

| Scope | Change | Δ AbsRel (relative %) | Δ RMSE m (relative %) | Δ Near AbsRel (relative %) | Δ Near RMSE m (relative %) | Δ Near bias m | Δ Floor percentage points |
|---|---|---:|---:|---:|---:|---:|---:|
| Pooled | KD − A | +0.004218 (+1.587%) | +1.262802 (+1.053%) | +0.008048 (+3.986%) | +0.055293 (+1.416%) | +0.017794 | +0.170847 |
| Pooled | A − original | -0.035709 (-11.842%) | -7.391200 (-5.805%) | +0.005732 (+2.922%) | +0.544066 (+16.183%) | -0.024529 | -0.601317 |
| Kite foggy | KD − A | +0.004649 (+1.600%) | +1.842482 (+1.181%) | +0.038235 (+7.581%) | +0.290471 (+6.187%) | +0.331000 | +0.318025 |
| Kite foggy | A − original | -0.066863 (-18.703%) | -9.229555 (-5.587%) | +0.077215 (+18.078%) | +0.315928 (+7.214%) | +0.730967 | -1.118741 |
| PLE spring | KD − A | +0.003717 (+1.568%) | -0.124759 (-0.234%) | +0.007187 (+3.719%) | +0.047027 (+1.212%) | +0.008866 | -0.000122 |
| PLE spring | A − original | +0.000481 (+0.203%) | -4.564821 (-7.887%) | +0.003695 (+1.949%) | +0.552772 (+16.607%) | -0.046064 | -0.000256 |

All requested absolute and relative changes, including delta rates, counts, bias, floor rates and every band, are in changes.csv/changes.json. Relative changes divide by absolute baseline and stay null for zero/null baselines; a signed bias change is not an error-improvement score. Rate differences in CSV are fractions; display tables convert floor differences to percentage points.

| Scope | GT band m | Valid pixels | Original AbsRel / RMSE m | A AbsRel / RMSE m | Exploratory KD AbsRel / RMSE m |
|---|---|---:|---:|---:|---:|
| Pooled | 0.1-10m | 32,267,840 | 0.196167 / 3.362006 | 0.201899 / 3.906072 | 0.209946 / 3.961365 |
| Pooled | 10-50m | 109,936,062 | 0.236179 / 10.787866 | 0.205351 / 10.355340 | 0.211460 / 10.479110 |
| Pooled | 50-100m | 27,393,245 | 0.271053 / 22.983747 | 0.185398 / 18.329250 | 0.184217 / 18.242772 |
| Pooled | 100-300m | 25,160,613 | 0.512176 / 165.804212 | 0.459585 / 171.438747 | 0.454187 / 171.408210 |
| Pooled | 300-1000m | 18,146,054 | 0.638864 / 388.069862 | 0.598699 / 356.171579 | 0.606131 / 361.156514 |
| Kite foggy | 0.1-10m | 894,287 | 0.427121 / 4.379248 | 0.504336 / 4.695176 | 0.542571 / 4.985648 |
| Kite foggy | 10-50m | 59,186,434 | 0.246454 / 10.608176 | 0.174365 / 8.412997 | 0.183052 / 8.696115 |
| Kite foggy | 50-100m | 18,654,690 | 0.251297 / 20.494309 | 0.156016 / 14.928776 | 0.154703 / 14.857228 |
| Kite foggy | 100-300m | 20,117,893 | 0.534278 / 179.957955 | 0.481748 / 187.340103 | 0.475341 / 187.320924 |
| Kite foggy | 300-1000m | 15,558,769 | 0.674670 / 397.318759 | 0.634949 / 364.629092 | 0.643748 / 370.403602 |
| PLE spring | 0.1-10m | 31,373,553 | 0.189583 / 3.328456 | 0.193278 / 3.881228 | 0.200465 / 3.928255 |
| PLE spring | 10-50m | 50,749,628 | 0.224196 / 10.993719 | 0.241489 / 12.237155 | 0.244591 / 12.234575 |
| PLE spring | 50-100m | 8,738,555 | 0.313227 / 27.555579 | 0.248121 / 24.028891 | 0.247224 / 23.916998 |
| PLE spring | 100-300m | 5,042,720 | 0.423999 / 89.258866 | 0.371165 / 81.430400 | 0.369792 / 81.285545 |
| PLE spring | 300-1000m | 2,587,285 | 0.423540 / 326.979989 | 0.380706 / 300.330649 | 0.379916 / 299.588977 |

Full band δ1/2/3, bias and floor data are in depth_band_metrics.csv and machine-readable metrics.json. The first band[0.1,10) is exactly the existing near subset; final band includes1000m.

Teacher context (reused; no teacher inference): selected B epoch3 remains rejected under the original gate.

| Scope | Teacher AbsRel | Teacher RMSE m | Teacher near AbsRel | Teacher near RMSE m | Teacher near bias m |
|---|---:|---:|---:|---:|---:|
| Pooled | 0.240330 | 99.827872 | 0.197856 | 3.872763 | 0.676583 |
| Kite foggy | 0.258787 | 128.240364 | 0.742757 | 6.636090 | 6.323239 |
| PLE spring | 0.218889 | 49.377967 | 0.182324 | 3.764378 | 0.515628 |

Known rejection: B Kite near AbsRel0.742757 and RMSE6.636090m exceed A0.504336 and4.695176m despite better pooled/overall metrics. The original failed gate/hash and protocol remain unchanged. Exploratory KD is a separately authorized rejected-teacher run and does not retrospectively become the original accepted-teacher KD arm.

**Figures and interpretation:** [contact sheet](figures/student_contact_sheet.png), [overall/near trade-offs](figures/overall_near_tradeoffs.png), [depth bands](figures/depth_bands.png). The six exact teacher-audit frames were fixed before KD existed: Kite1567 typical,1534 poor,1577 regression; PLE445 typical,1691 poor,1928 regression. These are selected stress examples, not a representative validation sample. Frame-selection records include original batch partners and input/cache hashes. Shared log range colour[0.1,1000]m, signed near-error colours±10m, invalid/outside-near grey, near boundary yellow. MidAir attribution is retained.

Measured KD-versus-A changes are predominantly adverse: pooled AbsRel+0.00421789 (+1.5867%), RMSE+1.262802m (+1.0528%), near AbsRel+0.00804768 (+3.9860%) and near RMSE+0.055293m (+1.4156%). Kite foggy near AbsRel rises7.5812%, near RMSE6.1866%, and near bias+0.331000m. PLE overall RMSE improves0.124759m (0.2340%) and its already tiny valid floor rate decreases, while its AbsRel and both near errors worsen. Pooled floor rate rises from1.798044% to1.968891% (+0.170847 percentage points). Thus KD is not uniformly worse on every number, but has no pooled or near-error advantage over A here.

A-versus-original is also mixed: pooled AbsRel improves11.8422% and RMSE5.8045%, while pooled near AbsRel worsens2.9221% and near RMSE16.1828%. Foggy near AbsRel worsens18.0781% and near RMSE7.2142%; PLE near RMSE worsens16.6075%. A's lower pooled signed near bias does not contradict larger near RMSE because mean signed bias can hide cancellation/dispersion. Original retains the best aggregate near AbsRel/RMSE among these three on both paths. A's PLE overall AbsRel is also0.2034% worse than original, despite lower PLE RMSE. None dominates all metrics.

Depth-band exceptions are retained: KD modestly improves pooled50–100m AbsRel/RMSE and100–300m AbsRel/RMSE versus A, while near,10–50m and300–1000m errors worsen. A's100–300m RMSE is worse than the original even though its pooled overall errors improve. See all per-path/band values and counts, not just pooled rankings.

In the visualizations, all three students overestimate the illustrated foggy near patches. A can outperform original on an individual illustrated frame (for example1567) even though A's aggregate foggy near error is worse over all83 near-active frames. The prior example selection is explicitly not a representative sample. The PLE stress examples show large local errors around thin foreground/foliage regions; these are scene observations, not semantic labels or demonstrated causes.

**Measured findings:** See the tables for trade-offs. Pooled overall selection rewards A relative to the original but does not preserve its near-field accuracy. KD must be judged separately on each trajectory and near group, not inferred from the rejected teacher’s pooled advantage. A lower signed bias alone can coexist with larger error variance/RMSE. The gradient probe did not support a teacher-repair run; this comparison does not overturn it. No causal claim that distillation, fog, BN or a specific loss term alone produced any subgroup change follows from one trained seed and these familiar two validation paths.

**Demonstration proposal:** Use supervised A epoch9 as the primary full-depth visualization candidate because it has the lowest pooled validation AbsRel among the three students and improves overall RMSE over the original. Keep the original epoch4 as the explicit near-field reference/toggle: A sacrifices near accuracy, especially in fog, so do not label it a proximity-safety upgrade. KD remains an exploratory comparison rather than a preferred default on pooled AbsRel. This is a recommendation only; no export, published-weight change, config promotion or deployment was made. The demo should visibly show model identity, metric range units, stale/fresh state if scheduling is used, and validation-only/fog/near limitations; it is not a navigation or collision-avoidance system.

**Exact remaining work (none executed or newly authorized here):**

1. Scheduler ablations: retain completed original V1/V2 and negative results unchanged. For the chosen new student, preregister a separate validation-only causal replay using unchanged event/mask definitions, no privileged future/GT scheduler inputs, per-sequence reset and the frozen policy settings. Compare every-frame, fixed2, fixed5, original validation-budget fixed rule, and V2; include RGB-only, motion-only and combined controls to isolate the previously joint changes. Report calls, stale-depth pooled/per-path/near errors and event-update delays. Freeze any new validation selection before test; no scheduler search or retraining is authorized now. Historical epoch4 policy results do not establish A/KD scheduler behavior.
2. Timing: measure the selected exact export on the intended laptop/runtime, FP32 eval batch1, fixed threads/device, synchronized CUDA when used, identical excluded warmups and at least3 alternating measured repetitions for adaptive versus fixed comparisons. Separate model-only inference, scheduler overhead, preprocessing/I/O and end-to-end latency; report distribution, memory, power/thermal/storage conditions and call counts. Historical CPU scheduler timing and short training benchmarks do not substitute for this matched GPU/CPU deployment timing. No energy/battery claims without direct measurements.
3. Final testing: preserve original exposed test study/ledger and do not reuse it as untouched selection data. Verify healthy storage/acquisition for reserved independent new groups Kite0007/0008, PLE0008/0009 without inspecting their RGB/depth during planning; current expanded TRAIN/VAL manifest has zero test records. Freeze final model/checkpoint, preprocessing, policies, comparisons, analysis and one-use exposure protocol before separately authorized test access. Acquisition availability and expected8820 frames must be verified, not assumed or forced. A separate guarded test manifest/receipt and exposure ledger are prerequisites. No new test result exists from this task.
4. Demonstration: after a human accepts the explicit trade-off, create a separate pinned inference export/config and model card, verify byte/state identity and input/output parity, wire it to the visualization with honest stale/fresh indicators and caveats, then perform the bounded inference smoke/timing above on permitted inputs. Keep published original weights untouched unless a later explicit promotion is authorized. Teacher online inference is not needed to deploy any student. There is no remaining authorized training step.

Provenance: PLAN/FREEZE, source_inputs, identities, configurations, runtime, completion, saved-reference differences, example verification, visualization record, execution record and preservation verification accompany this report. All original checkpoints/protocols/gates/audits remain separate. The checks verify full model/BN state unchanged and persistent bytes before/after; training and optimizer instances=0. Exact commands and any runtime differences are recorded.

Execution transparency and runtime compatibility: the first full KD pass reached4410 but a report lookup used `overall` where the historical file stores pooled scores under `metrics`, so its aggregates were not accepted/saved. That failed script/log/status is preserved. A second full KD pass persisted results and reproduced selected saved FP32 metrics with maximum absolute difference1.42e-14; every required pixel count matches exactly. No third full pass was run. The later direct main-process contact attempt failed a bitwise RGB parity check, so its predictions were not mixed with cached worker predictions. A bounded comparison on the same first pair measured worker/main RGB max absolute difference1.1920928955078125e-7 with identical GT and masks; workers used1 torch thread and main used2. Different execution contexts are observed, but thread count alone was not causally isolated. Final figures instead run all three students on the identical worker-produced tensors and retain all original per-frame numerical checks unchanged. The final visual script completed all six caches, state/reference checks and completion.json, then hit a missing datetime import only while writing the status timestamp. That metadata failure is retained in its log/exit record; status was finalized separately from verified outputs, with no new inference. All commands and exit statuses remain in execution_record.json. This is an audit-script recording issue, not evidence of a trained-model defect.

The completed KD validation pass is an accuracy/diagnostic pass, not a latency benchmark: USB decoding, CPU FP64 statistics and recording are included. Runtime versions/hardware/model/input precision match the reusable audit (runtime.json), and the accepted figure replay uses its same worker context. Total unique evaluated validation frames remains4410; repeated processing adds no statistical sample size. No test RGB/depth or teacher predictions were read/recomputed. All107 protected existing files and all selected checkpoint byte hashes were reverified, and all three loaded model/BN before/after digests match.

Scheduler implementation prerequisite: the public historical runner pins original export/config/manifest metadata and includes historical policy search. A later expanded-student scheduler study needs a separate new export/metadata lineage, updated validation-only input/reference binding and an explicit fixed-policy/ablation mode. Do not substitute new weights under the old metadata or invoke a historical search as though it were a fixed new ablation. Preserve the historical runner and its completed outcomes. No such implementation was made here.
