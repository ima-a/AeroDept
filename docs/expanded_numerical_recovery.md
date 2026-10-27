# Expanded workflow numerical recovery — 26 October 2026

Status: implemented and verified with bounded existing TRAIN inputs. **Full training remains stopped.** This amendment changes numerical handling and checkpoint identity; it does not change epochs, batch size, learning rates, schedules, loss coefficients, input groups, preprocessing, metrics or validation selection. The failed frozen bundle and its original stop policy remain unchanged.

## Confirmed failure

An exact prefix replay of the failed supervised configuration stopped at batch 39, after 38 successful optimizer updates. The failing batch contains TRAIN frames `Kite_training/foggy/trajectory_2005:1686` and `PLE_training/fall/trajectory_4006:597`. Two parameter gradients in MobileNet's squeeze/excitation block (`encoder.1.block.1.fc2.weight` and `.bias`) contained NaNs.

The captured pre-step model/optimizer/RNG state and augmented TRAIN tensors were replayed under three fixed mechanistic controls, with no optimizer updates during those controls:

| Control | Loss | Parameter gradients |
|---|---:|---|
| Original AMP, scale 1024 | 33.769989 | Nonfinite |
| Same AMP forward, unscaled backward | 33.769989 | All finite |
| FP32 reference, unscaled backward | 33.626579 | All finite |

This isolates loss-scaling overflow in this captured batch. The original runner stopped after unscale and before the scaler could back off. These checks measure software/numerical behavior, not model accuracy. The original run's exact update count was not logged; 38 is the count reproduced deterministically by the bounded replay.

## Recovery policy

`same_batch_amp_backoff_v1` in `src/aerodepth_lite/expanded.py`:

1. Snapshot all model buffers and Python/NumPy/CPU/CUDA/data-loader RNG before the student's forward. Compute the frozen FP32 teacher output once when KD is active.
2. Run the existing AMP forward and FP32 supervised/KD reductions. Nonfinite/nonpositive predictions or a nonfinite loss remain fatal.
3. If unscaled gradients are nonfinite under AMP, do **not** call the optimizer or advance the batch/global step. Reduce the loss scale by the scaler's backoff factor, reset its growth tracking and per-optimizer unscale bookkeeping using public scaler APIs, restore model buffers/RNG, clear gradients, and retry the **same augmented batch**. Reuse the same detached FP32 teacher prediction.
4. Clip and apply exactly one optimizer update only after all gradients are finite. A retry does not fetch another batch, change augmentation, consume another dropout RNG stream, or count another BatchNorm update. The scheduler still advances only at an epoch boundary.
5. Allow at most eight overflow retries (nine backward attempts total) per batch. Persistent nonfinite gradients, an unsafe loss scale, or nonfinite predictions/loss stop the run. A rejected batch has no optimizer update; buffers and RNG are restored before propagating the error. Non-AMP gradient failures remain immediately fatal.

There are no skipped batches or silently skipped optimizer updates. This differs deliberately from simply allowing default GradScaler step-skipping, which would break the fixed successful-update budget and retain unwanted buffer updates unless handled explicitly. It is not an LR/scale search: initial scale remains 1024, and ordinary overflow backoff is recorded at runtime.

Each retry logs its frame IDs, offending parameters, old/new loss scale and current successful-update count. Batch rows and epoch history record retry counts. The first batch and every 250th batch now log loss, global step, scale and total retries for startup monitoring. Teacher inference time remains separate.

## Identity, history and future launches

New checkpoints use identity format 2 and include the complete gradient policy. They save/restore the scaler, growth tracker, total overflow retries and all previously required optimizer/scheduler/history/RNG state. Source/policy mismatches reject resume. Earlier checkpoint identities are not weakened or migrated.

Both matched student arms and the trained teacher use this same numerical policy. Students still initialize independently from exact published epoch-4 weights and BatchNorm buffers. The teacher remains frozen/eval/FP32 during KD, with no parameter or BatchNorm changes. Teacher acceptance criteria remain unchanged.

The failed bundle `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420` retains its old source, hashes and logs. It has no resumable checkpoint. Do not overwrite its frozen code or rerun its old start script. A separately authorized full launch must create a fresh bundle/output destination with this amendment and current source hashes, then follow A → B → validation gate → KD. No full training was launched as part of this fix. Earlier runtime estimates did not include this recovery implementation; sustained retry frequency and runtime are unmeasured.

## Verification

See [the numerical diagnosis and fix report](../results/expanded_numerics_20261026T110550/REPORT.md) for exact receipts and local artifacts. Verification includes the captured failing batch, a 100-batch real-TRAIN prefix, a one-pass finite-scale reference, BatchNorm/RNG/optimizer invariants, recovered-checkpoint continuation, persistent-gradient fault rejection, all three arm smoke/resume checks, matched student initialization/augmentations, and KD teacher/input protection. No validation or test images were read. The before-amendment current protocol text is preserved in `artifacts/expanded_numerics_20261026T110550/expanded_training.before.md`; the historical frozen proposal also remains in the failed bundle.
