# Expanded training progress snapshot — 27 October 2026

Observed at `2026-10-27T15:02:26.079648+00:00`. This is a saved observation, not live status or a completed experiment. Source bundle: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery`. Freeze SHA-256: `d49ab655ff5b1a94e7809e24bad60eac1cdba26bd7c9ce44853e924918e89d90`.

| Arm | Completed epochs | Status at observation | Best FP32 validation AbsRel |
|---|---:|---|---:|
| Supervised MobileNet | 10/10 | Complete; completion receipt verifies full budget and best/last identities | 0.265831085 (epoch 9) |
| ResNet-50 teacher | 13/20 | Running epoch 14, 5,000/22,049 batches | 0.240329789 (epoch 3) |
| Teacher acceptance gate | — | Pending full teacher completion | — |
| KD MobileNet | 0/10 | Not started; conditional on passing gate | — |

Teacher successful updates: 291,637/440,980. Latest logged loss: 8.197812080, finite; cumulative recovered AMP overflow retries: 118. No fatal error is recorded. Latest completed teacher-epoch validation AbsRel: 0.256210836. Best selection is preserved even when subsequent validation regresses; a better pooled score alone does not establish gate acceptance.

Both student arms independently start from the original epoch-4 parameters and BatchNorm buffers with fresh matched training state. The supervised checkpoint is a comparison baseline, not KD initialization. Teacher training still has its fixed 20-epoch budget; the unchanged multi-metric, per-path validation gate runs afterward. There is no test performance or completed KD comparison in this snapshot.

Files:

- [Observation provenance](observation.json) and [copied live status](status.json).
- [Supervised epoch history](supervised_history.json) and [supervisor completion receipt](supervised_completion.json).
- [Teacher completed-epoch history](teacher_history.json). Current incomplete-epoch progress is only in the copied status.
- `SHA256SUMS` identifies the copied files; private full checkpoints are not published.

Readiness, protocol and independent reproduction are documented in [the reproduction guide](../../docs/expanded_reproduction.md). All metrics here were read from existing TRAIN/VAL run records; this publication performed no image reads, training or model evaluation. Original test exposure records and original/expanded manifests remain unchanged.
