# Real MidAir pilot evidence

This directory retains a small, reviewable record of the completed real-data pilot while the raw MidAir data, checkpoint, full manifest, per-frame logs, and generated artifact tree remain Git-ignored.

## Contents and provenance

- `results.json` is a concise transcription of the four policies in the saved frozen replay result at `artifacts/real_midair_pilot/preliminary_pilot_v2/replay_frozen/results.json`.
- `metadata.json` retains the dataset/run facts and SHA-256 fingerprints from the saved inspection, manifest, run metadata, frozen replay, checkpoint, and frozen configuration.
- The frozen scheduler and training settings are tracked at [`configs/preliminary_pilot_frozen.yaml`](../../configs/preliminary_pilot_frozen.yaml).

The ignored source artifacts remain the authority for the retained values. Their hashes make accidental substitution detectable. No value in this bundle was produced by retraining or by a new replay.

## Explicit limitations

- This is a two-trajectory software pilot: one trajectory was used for training and one for validation. The manifest has an empty test split, and no test result exists.
- Scheduler thresholds were selected on the sole validation trajectory, so the replay is preliminary validation evidence, not an unbiased benchmark.
- Inference-call reduction and CPU runtime are compute proxies. They are not measured energy savings, battery-life improvements, or edge-device performance.
- The model is a small pipeline baseline trained on insufficient trajectory diversity; its depth accuracy is not publication-grade.
- Critical-event results use the provisional definition recorded in the replay output and require validation on a meaningful held-out dataset.

## Verify the local pilot, when its ignored data is available

```bash
MIDAIR_ROOT=datasets/midair_pilot .venv/bin/python -m pytest -q -m integration
sha256sum configs/preliminary_pilot_frozen.yaml \
  artifacts/real_midair_pilot/pilot_train_val_manifest.json \
  artifacts/real_midair_pilot/inspection/inspection.json \
  artifacts/real_midair_pilot/preliminary_pilot_v2/run.json \
  artifacts/real_midair_pilot/preliminary_pilot_v2/replay_frozen/results.json \
  artifacts/real_midair_pilot/preliminary_pilot_v2/best.pt
```

Compare the hashes with `metadata.json`. This does not retrain or evaluate a test split.
