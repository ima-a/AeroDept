# Bounded gradient numerical diagnosis and fixed TRAIN probe

Completed 2026-10-28T19:02:08.322040+00:00. Two-batch diagnosis plus **one** 64-frame rerun took 316.43 s (5.27 min), including their preflight/reporting overhead; code inspection excluded. No optimizer instances or steps, training, VAL/test evaluation, teacher repair, gate changes, checkpoint promotion or KD. Prior failed probe remains failed and unchanged.

The failure was reproducible FP32 arithmetic non-distributivity through the network, rather than an observed loss, mask, graph, parameter-order, accumulation or state-mutation defect. All four repeated component/direct gradients were bitwise identical on both original batches. All nine objectives shared a single prediction tensor per batch. Original source/config/checkpoint identity checks stayed active. The genuinely FP64 reference used independent copies of the same checkpoint values with FP64 RGB, GT, normalization buffers, every floating module input/output, loss terms, predictions, parameters and gradients. The source depth_loss preserves dtype; the strict_supervised FP32 wrapper was deliberately not used for this reference.

The original check compared separately backpropagated and combined FP32 objectives pointwise. The second batch reproduced 35 failing encoder elements for current loss, 13 for candidate, and 8 for near/far partition. These are counts per comparison, not necessarily distinct elements across comparisons. Decoder/head had zero original failures on these two batches.

| Original batch | Comparison | Block | FP32 max absolute error | FP32 relative L2 error | Original failed elements | FP64 max absolute error | FP64 relative L2 error |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | current_components | encoder | 7.09295e-06 | 9.989e-07 | 0 | 3.10862e-15 | 1.08489e-15 |
| 0 | current_components | decoder | 2.86102e-06 | 3.28994e-07 | 0 | 2.30926e-14 | 1.73652e-15 |
| 0 | current_components | output_head | 5.21541e-08 | 1.76286e-07 | 0 | 1.38778e-16 | 3.89513e-16 |
| 0 | candidate_components | encoder | 3.39746e-06 | 7.0791e-07 | 0 | 2.66454e-15 | 8.0176e-16 |
| 0 | candidate_components | decoder | 4.29153e-06 | 2.33943e-07 | 0 | 1.86517e-14 | 1.29933e-15 |
| 0 | candidate_components | output_head | 4.47035e-08 | 1.86494e-07 | 0 | 1.11022e-16 | 3.91399e-16 |
| 1 | current_components | encoder | 3.29018e-05 | 1.44828e-06 | 35 | 3.73035e-14 | 1.74314e-15 |
| 1 | current_components | decoder | 1.3113e-05 | 4.54106e-07 | 0 | 4.61853e-14 | 1.6788e-15 |
| 1 | current_components | output_head | 1.19209e-07 | 2.1422e-07 | 0 | 3.05311e-16 | 4.18641e-16 |
| 1 | candidate_components | encoder | 2.95639e-05 | 1.10355e-06 | 13 | 2.39808e-14 | 1.29746e-15 |
| 1 | candidate_components | decoder | 3.8147e-06 | 9.33779e-08 | 0 | 7.10543e-15 | 2.44767e-16 |
| 1 | candidate_components | output_head | 4.76837e-07 | 3.03733e-07 | 0 | 4.44089e-16 | 4.36458e-16 |

At predicted-range output, current reconstruction max errors were 3.64e-12 and 1.82e-12 (FP32); the same-prediction FP64 reference reduced them to 6.78e-21 and 3.39e-21. Replacing only CPU summation of the already computed FP32 parameter gradients with FP64 did not resolve the encoder discrepancy (batch 1 relative L2 1.4473e-6, still 35 failures). A VJP using the weighted output-space derivatives instead of independent parameter reverse passes reduced batch-1 encoder relative discrepancy to 6.55e-7, consistent with sensitivity to arithmetic ordering along reverse computation. The full FP64 forward/backward reconstructed at approximately 1e-15. A particular kernel or single operation is not claimed as the sole source.

Failures concentrate in smaller/cancelled coordinates, but are not exclusively near zero. In batch 1 current loss, 27/35 failures had |actual|<=0.01, 5/35 <=0.001, and 1/35 met the prespecified |actual|<=0.001*block RMS definition (cutoff 3.57e-5). Failed |actual| ranged 4.97e-6 to 0.03403; the maximum overall absolute discrepancy 3.29e-5 need not be the largest normalized failing coordinate. Reporting only relative element error would overstate the vector-direction error.

Direct versus component-reconstructed cosine effects on the first two pairs were at most 1.03e-7 across all recorded relations/blocks. However, doing an entirely new full FP64 forward/backward changed encoder cosines by up to 2.30e-4 versus FP32, larger than reconstruction error because the full piecewise network is precision-sensitive. No first-two conflict/effect classification changed. This is empirical four-frame reference evidence, not a universal numerical bound or train-mode guarantee.

A separate [numerical amendment V2](../expanded_gradient_resolution_20261028T185216Z_probe_v2/NUMERICAL_AMENDMENT_V2.json) was frozen before any remaining batch. It retains the old pointwise failures as diagnostics and requires relative vector L2<=1e-5, an additional maximum-element safeguard, cosine reconstruction change<=1e-5, and unchanged scientific classifications. Exact-zero vectors require exact-zero reconstruction. The active prelaunch implementation seal is FREEZE_V2_1.json SHA-256 `d5db00778f92d8001c53d85ca308b713b091f07e3af0aa43cb6a00297a65bc79`; the earlier prelaunch source/seal are retained. This reference-supported metric amendment does not revise the original failed result or scientific thresholds. Full PLAN, sampling and analysis script are byte-identical copies of the originals.

All **32 batches / 64 frames** completed and passed the amended numerical and unchanged state/coverage checks. Across 384 comparison/block rows, maximum relative L2 error was 1.59035e-06, maximum absolute error 0.000217438, maximum cosine reconstruction effect 2.13458e-07. The original pointwise check failed in 74 of 128 comparisons, with 1830 failing element occurrences over comparisons and overlapping blocks; these are retained in gradient_reconstruction.json. All direct/reconstructed scientific classifications agreed.

Sampling stayed frozen: 64 sparse frames from **20 realizations / 12 underlying path groups**, seed42, batch2, no augmentation/shuffle, passes midpoint/start/end then quarter for only first four sorted Kite realizations. Temporal/weather observations are correlated; neither 64 frames nor 32 pairs is an independent sample count. Both model and all modules stayed **eval** with gradients enabled; findings are eval-mode gradient evidence. Decoder includes the output head (145 parameters), which is shown separately as a subset.

The 64 frames contained **2,804,432 valid pixel occurrences and 672,221 near-valid occurrences**. 6 batches had no near GT: [2, 3, 10, 12, 22, 30]. These were retained, not filtered. Batch1 has only one near-valid pixel. Its separately normalized near-MAE gradient therefore depends on one pixel, while the original near partition retains all-valid denominators; this is very limited near-field support, and sampling variance is not estimated. Frame-level residuals/counts and complete coverage are in sample_coverage.csv; per-realization totals below.

| Trajectory | Condition | Path group | Frames | Near / valid occurrences |
|---|---|---|---|---:|
| Kite_training/cloudy/trajectory_3002 | cloudy | Kite/0002 | 1100,0,2201,550 | 32,248 / 171,215 |
| Kite_training/foggy/trajectory_2005 | foggy | Kite/0005 | 1102,0,2205,551 | 0 / 180,321 |
| Kite_training/foggy/trajectory_2006 | foggy | Kite/0006 | 1102,0,2205,551 | 32,785 / 203,324 |
| Kite_training/foggy/trajectory_2021 | foggy | Kite/0021 | 1101,0,2203,550 | 0 / 155,945 |
| Kite_training/sunny/trajectory_0000 | sunny | Kite/0000 | 1102,0,2204 | 0 / 119,948 |
| Kite_training/sunny/trajectory_0005 | sunny | Kite/0005 | 1102,0,2205 | 0 / 134,386 |
| Kite_training/sunset/trajectory_1001 | sunset | Kite/0001 | 1102,0,2204 | 0 / 136,281 |
| Kite_training/sunset/trajectory_1006 | sunset | Kite/0006 | 1102,0,2205 | 17,699 / 146,117 |
| PLE_training/fall/trajectory_4001 | fall | PLE/0001 | 1101,0,2203 | 36,188 / 118,209 |
| PLE_training/fall/trajectory_4004 | fall | PLE/0004 | 1101,0,2203 | 56,151 / 150,718 |
| PLE_training/fall/trajectory_4006 | fall | PLE/0006 | 1102,0,2204 | 49,790 / 132,730 |
| PLE_training/spring/trajectory_5001 | spring | PLE/0001 | 1101,0,2203 | 36,043 / 118,206 |
| PLE_training/spring/trajectory_5002 | spring | PLE/0002 | 1102,0,2204 | 62,589 / 127,217 |
| PLE_training/spring/trajectory_5004 | spring | PLE/0004 | 1101,0,2203 | 55,652 / 146,143 |
| PLE_training/spring/trajectory_5005 | spring | PLE/0005 | 1102,0,2205 | 32,828 / 129,734 |
| PLE_training/spring/trajectory_5006 | spring | PLE/0006 | 1102,0,2204 | 49,443 / 132,750 |
| PLE_training/spring/trajectory_5007 | spring | PLE/0007 | 1102,0,2204 | 60,232 / 122,920 |
| PLE_training/winter/trajectory_6002 | winter | PLE/0002 | 1102,0,2204 | 60,347 / 127,005 |
| PLE_training/winter/trajectory_6005 | winter | PLE/0005 | 1102,0,2205 | 31,823 / 129,435 |
| PLE_training/winter/trajectory_6007 | winter | PLE/0007 | 1102,0,2204 | 58,403 / 121,828 |

Only one of the three sampled foggy TRAIN realizations contains near GT in this fixed subset: trajectory2006 has32,785 near occurrences; trajectory2005 and trajectory2021 have zero across their four sampled frames each. In the conflicting first mixed cloudy/foggy batch, all19,445 near pixels belong to the cloudy frame. Fog presence in a batch is not evidence that its near-gradient conflict originates in fog. This sparse subset cannot establish the loss mechanism of the earlier validation foggy regression.

Loss definitions remain current = L1_V + 0.2 log_V + 0.1 spatial_valid_edges, and candidate = current + **0.2 separately normalized GT<10m MAE**. Current near L1/log partitions restrict numerators but retain |V| denominator; near-incident edge numerators retain all valid-edge denominators by direction. Far partition is complementary, assigning boundary edges to near. Separately normalized near MAE divides by |N|. No scalar-loss size was treated as gradient dominance. Example original batch0: all-valid-normalized near L1=0.626062 = (19445/88934)*2.863368 separately normalized near MAE.

| Block | Informative batches | Conflict count / required | Candidate effect among conflict | Conflict breadth passed | Weak count / required | Weak signal passed | Supports pilot review |
|---|---:|---:|---:|---|---:|---|---|
| encoder | 26 | 1/7 | 1/1 | False | 5/13 | False | False |
| decoder | 26 | 4/7 | 4/4 | True | 4/13 | False | False |
| output_head | 26 | 5/7 | 5/5 | True | 2/13 | False | False |

| Block | Current/near cosine min; median; max | Candidate/near cosine min; median; max | Near-partition/current norm ratio median | 0.2 near-MAE/current norm ratio median |
|---|---|---|---:|---:|
| encoder | -0.0789932; 0.512572; 0.8975 | 0.0100429; 0.611796; 0.931872 | 0.188643 | 0.132818 |
| decoder | -0.429906; 0.755369; 0.956102 | -0.342196; 0.821939; 0.976504 | 0.376696 | 0.237294 |
| output_head | -0.854741; 0.895883; 0.999419 | -0.742718; 0.958571; 0.999691 | 0.457351 | 0.31294 |

**Observed scientific decision:** The unchanged preregistered descriptive rules do not support a teacher-repair training experiment from this fixed probe.
This decision is based on full validity and the original count/effect/breadth rules, not tuning cutoffs after seeing results. Exact triggering batches/path/fraction breadth and all norm/cosine distributions are in summary.json. Mixed pairs cannot attribute a gradient direction to a condition or frame. A conflict is local first-order objective evidence, not proof that new optimization would improve validation, nor evidence of AdamW update direction. BN, data coverage and architecture remain hypotheses; no trained-model defect was established.

No repair experiment was launched. Selected rejected teacher B, failed gate and frozen protocol remain unchanged. Models/checkpoints, source/configuration and all prior audit/probe evidence in the protected index were rehashed successfully (175 files). The exact original first-two input/prediction hashes, losses and gradient summaries agree between diagnosis and rerun. Raw input identities were checked only against the original fixed TRAIN allowlist; all128 TRAIN RGB/depth paths verified, no reserved-test or validation images opened, no failing-HDD reads. Final model state digest stayed `dbdc6b27563761a82a9d9a06679e565407ad4862423b810f682b9eb141883a1e`, with no BN-buffer changes, empty parameter .grad fields and unchanged forward/reverse RNG states. See completion.json, diagnostic_rerun_parity.json, preservation_verification.json and execution_record.json.

Limitations: only four TRAIN frames received genuine full FP64 references; full64 scientific measurements remain FP32 eval-mode evidence. Nonsmooth forward precision effects are larger than same-graph reconstruction differences, and reference consistency does not identify one GPU kernel or guarantee train-mode/AMP behavior. Sparse correlated frame/pixel occurrences are not independent flights. No causal explanation of the validation foggy regression or accuracy gain is established. All training recommendations remain proposals.

Full retained evidence: [per-component gradient norms and cosines](../expanded_gradient_resolution_20261028T185216Z_probe_v2/block_gradients.csv), [exact frame counts and residuals](../expanded_gradient_resolution_20261028T185216Z_probe_v2/sample_coverage.csv), [unchanged scientific-rule decisions and distributions](../expanded_gradient_resolution_20261028T185216Z_probe_v2/summary.json), [all old and amended numerical checks](../expanded_gradient_resolution_20261028T185216Z_probe_v2/gradient_reconstruction.json), [full dtype-verified FP64 reference](full_FP64_reference.json), and [first-two detailed diagnostics](diagnostic_batches.json).
