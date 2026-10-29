# Next experiment proposal — NOT LAUNCHED

The current selected teacher failed its original gate. No audit result changes that outcome, selects another checkpoint or authorizes KD.

## Smallest justified next experiment

A **non-updating, TRAIN-only loss-gradient contribution probe** on the existing deterministic 64-frame TRAIN subset (`fixed_training_subset`, 32 batches of two). This addresses a plausible objective trade-off before committing another roughly 40.5 hours to teacher training. It is a proposed diagnostic experiment, not a teacher-quality improvement claim.

Controls:

- Use a fresh in-memory copy of selected teacher B, SHA-256 `6f9b1d27703ef0feba2f24441948a458650b239746632f9fc32d59ff0630b660`; preserve every original checkpoint and gate.
- Use only the explicit frozen TRAIN allowlist. Use identical FP32 unaugmented RGB/GT tensors for both objective calculations, seed 42, batch 2, no resplitting or new imagery. Validation/test are never calibration/gradient inputs.
- Keep the model in eval mode during this mechanism probe to avoid BN-buffer changes. Compute parameter gradients without an optimizer or parameter updates. Reset temporary gradients and check parameter/BN/RNG identities between branches.
- Compare the existing loss `L1 + 0.2 log + 0.1 gradient` against exactly the same loss plus **one fixed proposed** `0.2 mean_near |prediction−GT|`, using the existing GT<10m mask. The existing `near` component is MAE in metres, not relative error. Do not search coefficients or change architecture/initialization/normalization.
- Record per-component loss, encoder/decoder gradient norms, signed gradient alignment/conflict with the near MAE component, near/far residuals and actual near-pixel coverage. Frames with no near pixels retain zero near contribution; do not filter them out to make a result look better.

Estimated cost (proposal, not measured): 32 teacher batches with several sequential FP32 gradient calculations, likely **a few minutes**, plus USB I/O. The historical eight-batch AMP timing is not a measured FP32 multi-gradient runtime or memory result. Stop on nonfinite values or memory failure; do not silently adapt the batch or settings.

Validity/acceptance criteria: exact input parity; finite losses/gradients; component gradients reconstruct the frozen weighted loss gradient within fixed FP32 tolerances (atol=1e-6, rtol=1e-4 per parameter); no parameter/BN/RNG mutation or optimizer/update/checkpoint-selection changes; full per-batch coverage and near-count accounting. Report measured norm ratios and signed conflicts without selecting the teacher, promoting a model, or treating a probe as a gate pass. Evidence of an objective conflict/weak near contribution would justify reviewing a near-weighted teacher pilot; its absence would not justify that expensive change. Learned BN statistics and scene/domain coverage remain alternative hypotheses.

## Conditional later training experiment, requiring a new review/freeze

Only if that probe supports loss rebalancing: a separate teacher pilot with one changed scalar, teacher-only `near_weight: 0 → 0.2`, all other data, initial ImageNet V2/decoder seed, training order, batch/BN policy, optimizer, rates, fixed **20-epoch cosine horizon**, numerical recovery and FP32 selection unchanged. A predeclared first-five-epoch pilot would cover 110,245 successful updates and include the original selected teacher's epoch-3 point; use the saved original first-five-epoch history as an explicitly historical control, not a fresh randomized comparison. If exact data/source/RNG parity cannot be established, add a matched zero-near-weight pilot instead of claiming a clean ablation.

Historical cost: original B's first five epochs took **8.98 h**; all 20 took **40.48 h**. USB input performance, thermal conditions and a new loss can change runtime. These are reference observations, not a new benchmark or a guaranteed estimate. A paired fresh five-epoch control would roughly double the pilot budget. Do not shorten a cosine horizon or change a checkpoint identity to resume the original completed run.

The current full trainer intentionally rejects a nonzero near weight and requires exact source identities across A/B/gate. A future teacher-only loss/config extension would create a new source identity; it therefore also needs reviewed compatibility/lineage or newly matched controls before any future KD gate. Do not relax identity checks or rewrite the existing failed gate as a shortcut. The small gradient probe above needs no production training-code change.

Predeclare pilot outputs and scope separately; select only by the original pooled FP32 AbsRel rule. Model acceptance must still require the **unchanged original gate against selected A**: strictly better pooled AbsRel, finite per-path overall/near AbsRel and RMSE no worse than A, identical valid/near counts. Failure means stop, with all negative results retained. A pilot cannot be promoted silently into full scope or authorize KD. Any later full training/continuation needs a separately frozen identity/budget/resume design; do not rewrite the completed A/B experiment.

Neither this probe nor the pilot was executed in the validation audit. No test input or performance can be used to choose the change.
