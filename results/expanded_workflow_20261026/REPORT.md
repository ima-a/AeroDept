# Expanded workflow implementation and bounded verification — 26 October 2026

Implemented the separate ResNet-50 teacher / matched epoch-4 MobileNet supervised and KD workflow. **Full training has not started. All budgets/rates below are proposals pending review.** No test RGB/depth was accessed; no final-test evaluation, scheduler search, cleanup or Git write was performed.

## Changed files

- Extended `src/aerodepth_lite/model.py` with ResNet50Depth registration, normalization contract and encoder/decoder counts. Original MobileNet implementation/state keys are unchanged.
- Added masked absolute log-range KD to `src/aerodepth_lite/losses.py`; historical `depth_loss` is unchanged.
- Added isolated `src/aerodepth_lite/expanded.py`, `scripts/expanded_training.py`, three `configs/expanded_{supervised,teacher,kd}.yaml` configs, and `tests/test_expanded.py`.
- Added [protocol and exact future launch/resume commands](../../docs/expanded_training.md); corrected the expansion plan student initialization while preserving its previous text locally.
- Saved these new receipts separately from all historical evidence. Pre-existing modified/untracked work, including README and final-study files, was retained.

## Guarded inputs and initialization

Common root: `/run/media/Ima/Low storage/AeroDepth/MidAir`. Frozen manifest: `results/expanded_preparation_20261026/manifest_train_val.json.gz`, uncompressed SHA-256 `67c6cad39767d847a2087a81d1ecc4de2ad8c9ecc0fa01e491f3950b25f8dd3a`. Train: 44,098 frames / 20 realizations / 12 independent paths. Unchanged validation: 4,410 / 2 / 2. Reserved new test: four paths, 8,820 expected frames, unacquired; original completed test quarantined. Manifest contains no test records.

The runner pins manifest/reservation hashes and exact group assignments before image opens, rejects unsafe/symlink paths and discovery, and uses only explicit allowlisted records. Bounded modes restrict all image reads to a deterministic 64-frame TRAIN subset; no validation images were used during this task. Pilot exclusions and all test/validation climate-equivalent group exclusions are retained.

Both students independently initialize from the published epoch-4 state dict (including BN), SHA-256 `b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6`, with fresh optimizer/scheduler/scaler/history/counters and identically reset RNG after construction. Teacher uses ImageNet1K V2 only for its encoder; initialization SHA is retained in checkpoint identities. Offline resume builds models without downloading weights.

## Correctness evidence

Final regression: **79 passed, 1 integration test deselected**. The single warning was inability to initialize NVML inside the CPU-test sandbox. Command and final log identity are recorded in [executed commands](executed_commands.json) and [source identity](source_identity.json). Tests cover odd/full-size alignment, single normalization, offline weight round-trip, loss masks/nonfinite failures, epoch-4 buffers, arm parity, identity mismatch, read guards, near-field gate failure and interrupted best/last recovery.

Each arm ran bounded CUDA optimization on real TRAIN data: two batches → save → two continuation batches → reconstruct/resume the save → repeat the continuation. Intended encoder/head updates and finite losses/unscaled gradients passed. Resume state comparisons had maximum absolute difference **0.0** in all arms. A/C initial state digests, frame order and augmented RGB hashes matched exactly. KD teacher parameters, buffers and gradients remained frozen. See [smoke checks](smoke_checks.json). The smoke teacher is not accuracy evidence and cannot pass the full-run scope guard.

## Measured bounded GTX 1650 benchmark

GTX 1650, torch 2.14.1+cu130 / torchvision 0.29.1+cu130, 192×320 full-frame inputs, physical/effective batch 2, no accumulation, two loader workers, external HDD, synchronized CUDA timing. Two warm-up batches excluded; eight measured batches = 16 examples per arm. Same fixed 64 TRAIN frames; OS cache uncontrolled. Teacher inference in KD is FP32; trained model forward uses AMP.

| Arm | Frames/s | Wall seconds (16 frames) | Peak allocated MiB | Peak reserved MiB | Encoder / decoder parameters |
|---|---:|---:|---:|---:|---:|
| supervised | 7.414 | 2.158 | 108.3 | 150.0 | 927,008 / 100,481 |
| teacher | 3.518 | 4.548 | 514.5 | 554.0 | 23,508,032 / 491,601 |
| kd | 6.717 | 2.382 | 202.5 | 228.0 | 927,008 / 100,481 |

These are allocator peaks during measured training, not total VRAM usage. The sample lasts seconds; it does not establish sustained thermal behavior, full-dataset I/O or multi-day feasibility. Frozen smoke-teacher outputs establish runtime only. Raw batch timings, loader wait and per-arm source identities: [benchmarks](benchmarks.json).

KD teacher forward alone consumed 0.851 seconds over 16 measured examples (0.0532 s/frame). A/C wall differences also contain uncontrolled loader/cache effects; they should not be interpreted as a controlled compute-only difference.

## Duration extrapolations, not measured full epochs

Estimate train time as 44,098 / measured frames/s. Estimate validation overhead from four FP32 TRAIN inference batches (eight examples), scaled to 4,410 frames; this is **not a validation evaluation**. Use the larger A/C validation-overhead proxy for both students to avoid attributing startup/cache differences to KD. Teacher uses its own proxy. Add two measured full checkpoint writes and one measured recovery-copy/hash sequence every epoch, conservatively assuming every epoch improves. The 20% allowance below is a planning margin, not a statistical interval.

| Arm | Train h/epoch | Validation proxy h/epoch | Checkpoints h/epoch | Total h/epoch | Proposed epochs / updates | Budget h | With 20% allowance h |
|---|---:|---:|---:|---:|---:|---:|---:|
| supervised | 1.65 | 0.24 | 0.00038 | 1.89 | 10 / 220,490 | 18.9 | 22.7 |
| teacher | 3.48 | 0.16 | 0.00471 | 3.65 | 20 / 440,980 | 73.0 | 87.6 |
| kd | 1.82 | 0.24 | 0.00014 | 2.06 | 10 / 220,490 | 20.6 | 24.8 |

Sequential A+B+C: approximately **112.6 hours**, or **135.1 hours with the allowance**, conditional on a passing teacher gate. Exact arithmetic and limitations: [duration estimates](duration_estimates.json).

| Arm | FP32 TRAIN proxy (8 examples), seconds | One full checkpoint write, seconds | Recovery copy + hashes, seconds |
|---|---:|---:|---:|
| supervised | 1.573 | 0.298 | 0.770 |
| teacher | 1.061 | 5.456 | 6.026 |
| kd | 0.612 | 0.160 | 0.190 |

Checkpoint writes/recovery copies were measured on the internal repository filesystem; dataset reads were on the external HDD. Relocating run outputs can change overhead. Checkpoints remain ignored/private. Retain best/last/previous-best and gate/baseline identities for reliable resume.

## Proposed protocol and remaining blockers

- Full order A → B → numerical validation gate → C. Students: 10 epochs each, AdamW LR 1e-4, weight decay 1e-4; teacher: 20 epochs, encoder LR 1e-5 / decoder 1e-4. Clip norm 1; seed 42; fixed cosine horizons 10/20 with minimum LR 1e-6; no early stopping or scheduler search.
- Both student arms use identical batch 2, all 44,098 TRAIN frames, shuffles/flips, optimizer initialization and 220,490-update budget. Full FP32 unchanged-validation selection runs every epoch and selects minimum pooled AbsRel.
- KD is supervised L1/log/gradient weights 1/0.2/0.1 plus fixed proposed `0.2 mean_valid |log(S)-log(stop_gradient(T))|`; teacher eval/no_grad FP32 on the same augmented RGB. Empty masks safe; bad predictions fatal. Range/masks/preprocessing/metrics unchanged.
- Teacher gate: pooled validation AbsRel strictly below A; on each validation path, finite overall/near AbsRel and RMSE no worse than A; exact valid/near pixel counts. Failed gate blocks full KD. No teacher-quality decision has been made.
- Review/freeze rates, budgets, batch/BN policy, stopping rule, coefficient and gate before full training. The short benchmark supports memory feasibility for these batches, not sustained laptop reliability. Accept the rough multi-day runtime and arrange stable power/mount/storage.
- New test acquisition and a separate frozen final-evaluation/exposure protocol remain future work; original test remains quarantined. No new test performance is available or used.

## Preservation and evidence locations

All **155 protected original files** retained exact size/SHA-256, including original models/configs/results, prepared manifest/reservations and original main best/last checkpoints. See [preservation verification](preservation_verification.json). No historical reports, hashes or exposure ledgers were rewritten.

The evolved model/loss source naturally differs from historical final-study execution hashes. The original saved-result verifier was run from the exact pre-change source snapshot with explicit `results/final_test` inputs: saved accounting passed, without images or model calls. Original verifier and historical hashes remain unchanged; reproduction of that source identity requires its snapshot/matching checkout.

Local raw logs, initial source snapshot, pre-edit plan, earlier checks and ignored software checkpoints: `artifacts/expanded_workflow_20261026T095357`. Definitive source checks: `artifacts/expanded_workflow_20261026T095357/verification`. Other verification folders are retained as earlier source revisions. [Checkpoint SHA identities](checkpoints.json), [executed commands](executed_commands.json), [source and historical verification](source_identity.json).

No full training command was executed. Exact launch and per-arm resume commands are in [the protocol](../../docs/expanded_training.md#commands-for-a-later-reviewed-full-run--not-executed).
