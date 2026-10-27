# Expanded workflow numerical diagnosis and fix — 26 October 2026

**The AMP loss-scaling failure was reproduced and fixed. Verification was bounded to TRAIN inputs. Full training remains stopped.** No scheduler, LR, batch, epoch budget, loss coefficient, range/preprocessing, metric, initialization or teacher-gate settings were changed. No validation/test images were read.

## Confirmed cause

The original frozen run was replayed with its exact source, initial epoch-4 weights, configuration, sampler seed and training pipeline, with an explicit 250-batch bound. It stopped at **batch 39 after 38 successful updates**. The original launch did not persist its own exact step count; this count comes from the deterministic reproduction. The preceding 38 batches, augmented RGB hashes, losses and pre-failure model tensors were later matched exactly by the fixed runner.

Failure gradients contained NaNs in `encoder.1.block.1.fc2.weight` (8 entries) and `.bias` (1 entry), inside MobileNet squeeze/excitation. Failing TRAIN frames: Kite foggy internal 2005 frame 1686; PLE fall internal 4006 frame 597.

| Captured-state control | Loss | Gradients |
|---|---:|---|
| Original AMP / loss scale 1024 | 33.769989 | Nonfinite |
| Same AMP forward / unscaled backward | 33.769989 | All finite |
| FP32 / unscaled backward | 33.626579 | All finite |

These fixed mechanistic controls made no optimizer updates and were not a hyperparameter search. They isolate loss-scaling overflow on this captured batch. Predicted ranges and supervised loss had already passed finite checks. See [reproduction](reproduction.json) and [failing records/gradients](failure.json).

## Implementation

- `src/aerodepth_lite/expanded.py`: added bounded same-batch AMP loss-scale backoff, buffer/RNG rollback, explicit retry logging/counts, first-batch loss/scale progress, and checkpoint identity format 2 with numerical policy and restored retry totals.
- `tests/test_expanded.py`: verifies scaler unscale-state/growth reset without optimizer updates, scale-underflow rejection and numerical-policy resume mismatch.
- `docs/expanded_numerical_recovery.md`: records [the numerical-policy amendment](../../docs/expanded_numerical_recovery.md). Current `docs/expanded_training.md` links the amendment while its earlier text is preserved in local provenance and the unchanged failed freeze.

On an AMP gradient overflow, no optimizer update occurs. The scaler backs off by its configured factor (0.5), growth tracking/unscale bookkeeping reset, all model buffers and training/data RNG restore, and the same augmented batch is retried. The frozen FP32 teacher output is computed once and reused. Exactly one finite update advances global step. BatchNorm and dropout RNG therefore reflect one successful forward rather than multiple trials. The initial loss scale remains 1024.

At most eight retries/nine backward attempts are permitted. Persistent bad gradients hard-stop with no optimizer update and restored buffers/RNG. Nonfinite/nonpositive predictions and nonfinite loss remain immediately fatal, as do non-AMP gradient failures and unsafe scale reduction. There is no invalid-pixel masking workaround, batch dropping or silently skipped optimizer step. Schedules and successful-update budgets are unchanged.

The gradient policy is `same_batch_amp_backoff_v1`. New checkpoints explicitly record it and strict source/policy identity; old source identities cannot resume as new ones. The failed bundle was not patched in place.

## Verification

- **43 focused unit/regression tests passed** (`test_expanded`, model/loss/metrics, MobileNet and local-training checks). The entire suite and archive audits were not repeated.
- The captured failing batch recovered by reducing scale **1024 → 512**, with the same AMP forward loss 33.769989, all finite gradients and **exactly one optimizer update**.
- Recovery model parameters, optimizer state, BatchNorm buffers, scaler state and RNG matched a single successful finite-scale pass exactly. Every BN counter advanced once.
- Save/reconstruct/resume after a recovered batch produced exactly the same subsequent model/optimizer/scaler state as uninterrupted continuation; overflow counter restored.
- A controlled persistent NaN gradient on the cached real TRAIN batch exhausted the bounded retries and stopped with **zero optimizer updates**, unchanged parameters/optimizer and restored buffers/RNG.
- A new **100-batch / 200-example** full-manifest training-prefix replay completed with 100 successful updates and finite parameters. No full epoch, validation or scheduler epoch step was run. Overflow events occurred at batch 39 (one retry) and batch 68 (two retries); final scale 128. These are bounded software checks, not accuracy evidence.
- Supervised student, teacher and KD smoke/resume checks passed with **zero maximum model-state difference**. Student initial state/BN digests and augmented input/order traces matched.
- KD fault injection verified exactly one teacher forward, identical augmented teacher/student RGB on every retry, a frozen teacher, one student optimizer update, and one BN update. Bad predictions still failed immediately without backoff/update.

Receipts: [fixed-run checks](fix_verification.json), [KD protection checks](kd_overflow_verification.json), [three arm checks](arm_smoke_checks.json), [smoke commands](smoke_commands.json), [source identities and selected log hashes](source_identity.json). Raw bounded diagnostic scripts, captured TRAIN tensors/state, logs and software checkpoints remain ignored/private under `artifacts/expanded_numerics_20261026T110550`. The earlier `fix_verification.log` is a superseded helper run that stopped at a JSON-array/tuple comparison; the normalized final helper run is `fix_verification_v2.log`.

## Preservation and next launch boundary

All **155 protected original files** retained exact size/SHA-256: old configurations, manifests, published weights, original best/last checkpoints and study evidence. All indexed files in the failed frozen bundle still match freeze SHA-256 `2b53a27cd4491843e92a7218a883158cbff6a6842ede83f966620497e7131de9`. Existing uncommitted work was retained. See [preservation](preservation.json).

The original user service remains stopped/failed, MainPID=0; there is no active Python CUDA workload. No new full run, teacher gate or test evaluation was started. This authorization was for bounded diagnosis and a verified fix only.

Before any future full launch, create a **fresh frozen bundle and fresh output directories** with this amended source/policy, preserving the failed attempt and old proposal. The old attempt has no checkpoint and cannot resume. Keep A/C initialization and budgets matched and retain the unchanged numerical teacher acceptance gate. No accuracy benefit or sustained stability is claimed from a 100-batch check. Earlier laptop timing estimates did not include this recovery policy; long-run retry frequency/runtime are unmeasured.
