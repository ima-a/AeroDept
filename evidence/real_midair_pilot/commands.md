# Verification commands

These commands verify the committed software and the existing local real-data pilot without training, replaying, downloading data, or touching the test split.

```bash
.venv/bin/python -m pytest -q
MIDAIR_ROOT=datasets/midair_pilot .venv/bin/python -m pytest -q -m integration
git diff --check
```

The source fingerprints were recorded with:

```bash
sha256sum configs/preliminary_pilot_frozen.yaml \
  artifacts/real_midair_pilot/pilot_train_val_manifest.json \
  artifacts/real_midair_pilot/inspection/inspection.json \
  artifacts/real_midair_pilot/preliminary_pilot_v2/run.json \
  artifacts/real_midair_pilot/preliminary_pilot_v2/replay_frozen/results.json \
  artifacts/real_midair_pilot/preliminary_pilot_v2/best.pt
```
