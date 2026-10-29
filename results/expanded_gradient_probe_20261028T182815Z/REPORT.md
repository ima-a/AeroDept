# Fixed TRAIN gradient probe — incomplete after frozen validity failure

Report date: 2026-10-28T18:40:00.172899+00:00 (UTC). The service stopped after 14.39 s. **One batch / two frames passed; the second batch failed. The 64-frame probe did not complete and cannot support a teacher-repair training experiment under its registered criteria.**

## Frozen design and scope

The exact 64-frame order, checkpoint identities, reductions, coefficients, decision rules and code were sealed before image/gradient results. Freeze SHA-256: `f187c97fac75fe0a7eb73a32734aaf18d0bb961f6c72f756444943c3a2af13d9`. [Plan](PLAN.json), [sampling](sampling.json), [freeze index](FREEZE.json), and [analysis-code seal](ANALYSIS_FREEZE.json) remain unchanged. The new descriptive support cutoffs were explicitly registered before execution; they were not represented as numeric cutoffs from the older qualitative proposal.

The selected, rejected teacher B is unchanged: SHA-256 `6f9b1d27703ef0feba2f24441948a458650b239746632f9fc32d59ff0630b660`, original selected epoch 3. FP32 forward/loss/gradients, autocast and TF32 off, `model.eval()` with parameter gradients enabled. No optimizer, scaler, clipping, steps, training-mode BN, calibration, validation/test evaluation, coefficient search, checkpoint selection or gate amendment was performed. These are **eval-mode gradients**, not train-mode optimization evidence.

## Sampling and temporal coverage

Planned: 64 sparse frames, 32 fixed pairs, 20 weather/season realizations, 12 underlying path groups. The unmodified sampler iterates fractions 0.5, 0, 1 and 0.25 over sorted realizations and takes the first 64 unique records. Each realization therefore contributes start/middle/end frames; only the first four sorted Kite realizations receive an additional quarter-position frame. Weather equivalents and repeated temporal positions remain correlated. No significance test treats these as 64 independent flights or 32 independent trials.

[Exact trajectory/condition/index coverage](planned_trajectory_coverage.csv) and [actual completed scope](coverage_summary.json) distinguish planned, raw-read, prefetched and completed observations. All 128 image paths passed raw SHA checks against the verified USB rescue index. The image guard allowed only those frozen TRAIN paths. Loader prefetch opened extra TRAIN inputs, but that does not make them completed gradient observations. There were no successful validation/test or failing-HDD reads.

The completed batch paired **Kite cloudy trajectory 3002 frame 1100** and **Kite foggy trajectory 2005 frame 1102**, both midpoint samples from different path groups. It contains **88,934 valid pixels and 19,445 near pixels (21.8645%)**. These are combined batch counts; the aborted program did not persist per-frame near counts. Full 64-frame/per-trajectory near-count coverage is unavailable, rather than guessed or filled from unrelated preparation averages.

| Planned trajectory | Condition | Path group | Frame indices |
|---|---|---|---|
| Kite_training/cloudy/trajectory_3002 | cloudy | Kite/0002 | 1100,0,2201,550 |
| Kite_training/foggy/trajectory_2005 | foggy | Kite/0005 | 1102,0,2205,551 |
| Kite_training/foggy/trajectory_2006 | foggy | Kite/0006 | 1102,0,2205,551 |
| Kite_training/foggy/trajectory_2021 | foggy | Kite/0021 | 1101,0,2203,550 |
| Kite_training/sunny/trajectory_0000 | sunny | Kite/0000 | 1102,0,2204 |
| Kite_training/sunny/trajectory_0005 | sunny | Kite/0005 | 1102,0,2205 |
| Kite_training/sunset/trajectory_1001 | sunset | Kite/0001 | 1102,0,2204 |
| Kite_training/sunset/trajectory_1006 | sunset | Kite/0006 | 1102,0,2205 |
| PLE_training/fall/trajectory_4001 | fall | PLE/0001 | 1101,0,2203 |
| PLE_training/fall/trajectory_4004 | fall | PLE/0004 | 1101,0,2203 |
| PLE_training/fall/trajectory_4006 | fall | PLE/0006 | 1102,0,2204 |
| PLE_training/spring/trajectory_5001 | spring | PLE/0001 | 1101,0,2203 |
| PLE_training/spring/trajectory_5002 | spring | PLE/0002 | 1102,0,2204 |
| PLE_training/spring/trajectory_5004 | spring | PLE/0004 | 1101,0,2203 |
| PLE_training/spring/trajectory_5005 | spring | PLE/0005 | 1102,0,2205 |
| PLE_training/spring/trajectory_5006 | spring | PLE/0006 | 1102,0,2204 |
| PLE_training/spring/trajectory_5007 | spring | PLE/0007 | 1102,0,2204 |
| PLE_training/winter/trajectory_6002 | winter | PLE/0002 | 1102,0,2204 |
| PLE_training/winter/trajectory_6005 | winter | PLE/0005 | 1102,0,2205 |
| PLE_training/winter/trajectory_6007 | winter | PLE/0007 | 1102,0,2204 |

## Exact loss reductions

Current objective is `L1_V + 0.2 Log_V + 0.1 Gradient_valid_edges`. L1/log use all valid pixels as the denominator. The spatial term is a sum of separate horizontal/vertical means over all valid adjacent pairs.

The current near partition restricts L1/log numerators to GT<10m but retains the **all-valid denominator**. For spatial differences, a valid edge is near-incident if at least one endpoint is near; the denominator still includes all valid edges of that direction. Near/far boundary edges are explicitly assigned to the near partition, so this is an edge-attribution convention, not a unique pixel attribution. The far partition is its exact complement.

The separate near-MAE objective is `sum_N |P−GT| / |N|`, with differentiable zero for empty N. The candidate is exactly current + **0.2 near-MAE**. These have different normalizations; no scalar loss magnitude alone is interpreted as gradient dominance.

For the completed batch:

| Scalar quantity | Value |
|---|---:|
| L1 | 16.030344009 |
| log | 0.239831105 |
| gradient | 25.952804565 |
| near_MAE | 2.863368034 |
| current | 18.673589706 |
| current_near | 1.736043930 |
| current_far | 16.937547684 |
| near_L1_global | 0.626061916 |
| candidate | 19.246263504 |

Here the current near L1 contribution is **0.626062**, equal to the near fraction × separately normalized near-MAE (**2.863368**). The complete current near partition is **1.736044**, while the whole current objective is **18.673590**. That scalar split is not an additive split of parameter-gradient norms, because vectors can align or cancel.

## Partial gradient evidence — completed batch only

Gradients were obtained with `autograd.grad` from one shared forward graph and identical tensors. Parameters/BN and input/prediction hashes were unchanged after the passed batch; every reverse pass preserved Python/NumPy/CPU/CUDA/loader RNG identities and left parameter `.grad` fields empty. Block summaries use FP64 norm/dot accumulation on the calculated FP32 gradients.

Encoder has 23,508,032 parameters; decoder has 491,601 **including** the 145-parameter output head. The head row is a subset, not a third disjoint block. Raw L2 and per-parameter RMS values, per-component norms and cosines are retained in [the saved passed batch](batches.json).

| Block | ‖current‖ | ‖near partition under current normalization‖ | ‖0.2 × separate near-MAE‖ | cos(current, near-MAE) | cos(candidate, near-MAE) |
|---|---:|---:|---:|---:|---:|
| encoder | 60.596121 | 24.764835 | 22.174197 | -0.078993 | 0.276610 |
| decoder | 57.065855 | 26.132738 | 23.440989 | -0.064602 | 0.327734 |
| output_head | 0.982789 | 0.568883 | 0.510240 | -0.149576 | 0.350142 |

Observed in this single mixed-condition batch: base/near-MAE cosines are negative in all blocks and become positive with the fixed candidate. Current near/far-partition cosines are −0.440 encoder, −0.464 decoder, −0.593 head, indicating cancellation. Near-partition/current norm ratios are 0.409, 0.458 and 0.579, so this batch does **not** show the preregistered ≤0.10 “weak near contribution” signal. The added term is only about 3.1% of the scalar objective, yet its norm is about 36.6%, 41.1% and 51.9% of the current block gradient: this is precisely why scalar magnitudes cannot establish gradient dominance. Norm ratios are not additive percentages.

This is evidence consistent with local objective conflict, not evidence that all foggy TRAIN data conflicts, that AdamW would take the raw gradient direction, or that validation accuracy would improve. The paired cloudy/foggy frames cannot localize the conflict to one condition. One completed batch cannot satisfy the frozen count/breadth rules.

## Frozen numerical failure

All 16 synthetic reduction/reconstruction checks passed across mixed, all-near, no-near and all-invalid FP32 examples. Batch 0 passed all four parameter-gradient reconstructions across all 195 parameter tensors: current components, candidate components, near/far partition, and near-L1 normalization. [Passed checks](gradient_reconstruction.json).

**Batch 1 failed the first reconstruction:** `g_current ≈ g_L1 + 0.2 g_log + 0.1 g_gradient`. The per-element allowance was frozen at `1e-6 + 1e-4 |g_current|`. The largest observed ratio `|difference|/allowance` was **2.767938**. Eight early encoder weight tensors contained at least one out-of-bound entry. This is not a 2.77× error in the whole gradient norm, and the failed-batch absolute errors/vectors were not persisted; no stronger numerical claim is made.

Predictions, objectives and all nine gradient vectors passed finite checks before that assertion. The failed batch was the next fixed pair: foggy trajectory 2006 frame 1102 and foggy trajectory 2021 frame 1101. No fallback, rerun, tolerance relaxation, mode/device switch or modified candidate was attempted.

FP32 summation/cancellation through deep backpropagation, numerical differences between reverse passes, or a probe implementation issue are possible explanations. The available log does not distinguish them. The passing output-level synthetic algebra does not prove full-network parameter-level agreement. This failure does not establish a defect in the trained teacher or invalidate the separate saved FP32 validation audit.

## State and evidence preservation

The passed first batch has an exact post-batch model/BN digest match to the initial in-memory model. The failed second batch aborted before its post-batch digest was captured; a full-run final in-memory-state acceptance receipt therefore does not exist. No optimizer or parameter update was possible in the executed script and no checkpoint was saved. Persistent checkpoint/source/protocol/gate identities are rechecked separately; this distinction is retained rather than claiming all 64-frame invariants passed.

## Decision and next action

**The predefined criteria do not support a teacher-repair training experiment from this attempt.** Full validity and coverage failed. The minimum informative-batch and path/fraction-breadth requirements cannot be met by one passed mixed batch, even though its local direction conflict is suggestive. [Explicit acceptance result](acceptance.json). This is an invalid/incomplete probe, not proof that near weighting is ineffective.

The justified next proposal is a small **numerical-consistency diagnostic**, not another teacher run: freeze the original first two pairs/checkpoint/mode/inputs again in a separate record; collect three repetitions of direct and component FP32 gradients and compare output-space versus parameter-space reconstruction, FP32 versus FP64 accumulation of the same computed vectors, and per-parameter absolute/relative error scales. No parameter/BN/RNG updates, optimizer, VAL/test inputs, sample expansion or coefficient search. Retain the original tolerances for assessment and never rewrite this failed result to pass. Expected budget is four TRAIN frames, six batch-repetitions and minutes of compute (proposal, not measured). Success means an explained/reproducible numerical error budget and a reviewed validation method before any new full 64-frame attempt; unexplained failures remain inconclusive. **This diagnostic was not launched.**

The prior conditional teacher pilot remains unapproved and unsupported by this incomplete probe: its one intervention would be near_weight 0→0.2, 5 epochs / 110,245 updates, unchanged 20-epoch cosine horizon, fresh original initialization/seed and all other controls; historical reference cost 8.98 h, with the unchanged original teacher gate as acceptance. No pilot, retraining, gate bypass or KD was launched.

## Limitations

Only one pair has complete valid gradient evidence; failed/prefetched inputs do not become additional samples. The intended 64 sparse frames are 20 correlated weather realizations over 12 path groups, not independent statistical replicates. Eval-mode gradients at the already trained, rejected checkpoint differ from train-mode BN, AMP, momentum/preconditioning, clipping and the trajectory of a freshly initialized repair run. No validation improvement or causal loss/BN/architecture conclusion follows.

All new evidence is saved separately. The frozen plan, sampling and code remain unchanged; existing rejected teacher, gate, protocol, checkpoints and prior audit reports are preserved. No source training/model code, configuration, Git history or publication was changed.

[Preservation verification](preservation_verification.json) confirms existing files and the freeze are unchanged. [Execution record](execution_record.json) records the stopped service. A separate verified USB backup of this probe’s results and artifacts is retained under the same relative project paths; its receipt is in the USB recovery directory.
