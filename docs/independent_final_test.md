# Reserved independent test · 30 October 2026

This completed and verified study is the separately authorized seven-comparison study under `results/independent_test_20261030/`. It preserves the old blocked specification and the original exposed test under `results/final_test/`. Check `COMPLETION.json` for completion; `EXPOSURE.json` alone means exposure started, not success. A failure is recorded as `INCOMPLETE` and is never automatically retried.

## Inputs and scope

| Reserved realization | Website ID | Paired frames |
|---|---|---:|
| Kite_training/foggy/trajectory_2007 | 0007 | 2,204 |
| Kite_training/sunny/trajectory_0008 | 0008 | 2,205 |
| PLE_training/spring/trajectory_5008 | 4008 | 2,206 |
| PLE_training/winter/trajectory_6009 | 4009 | 2,205 |

Eight left-RGB/range ZIPs contain 17,640 images, or 8,820 synchronized pairs. Compressed size is 5,866,144,441 bytes; image extraction is 6,668,899,690 bytes. Four existing climate HDF5 files add 310,171,768 bytes. Official acquisition paths, exact member lists, CRC checks, per-file hashes, HDF5 camera correspondence and position-hash leakage checks are saved in `INVENTORY.json` and `prepared/`. Metadata-only preparation preceded the ready freeze; no test scene/depth distribution inspection guided settings.

Staging is outside the repository on healthy storage. Downloads originals are retained. No writes, mounts or repairs used the failed HDD. The ignored `configs/independent_test_local.json` supplies only explicit `dataset_root` and `kd_checkpoint` locations; the ready protocol pins its hash. Repository manifests and artifact links use relative paths. Keep datasets, signed download configs and private checkpoints outside source control.

Primary: A epoch 9 with every-frame, fixed-2, fixed-5, validation-frozen budget-fixed and V2. Primary comparison is V2 versus budget-fixed. Secondary: Original epoch 4 and exploratory KD epoch 9 every-frame versus A every-frame. No outcomes alter the selection or policy settings. The rejected teacher remains rejected.

## Frozen contract

The [ready protocol](../results/independent_test_20261030/PROTOCOL.json) binds input/model/source identities, the five policies, secondary comparisons, metrics, pooling, events/censoring and timing before exposure. Its SHA-256 is `6a1bb71c98ba3a5b513ff46e7babb4c33dca168886f9561072e8095cff71473f`.

Full-frame 192×320, ImageNet normalization once inside the model, Euclidean range metres, finite valid GT [0.1,1000], near GT <10, FP32, strict delta thresholds and additive pixel pooling are unchanged. There is no prediction upper clamp, crop or scale correction. Nonfinite predictions anywhere stop the study. Every current frame is scored, including unwarped reused maps. Ground truth, validity masks and event labels do not enter the scheduler.

Budget-fixed uses K=ceil(N×2455/4410), at positions ceil(kN/K), independently of observed V2 calls. First-frame inference and per-path reset are mandatory. The test traverses all frames once, rotates seven-policy execution order and executes each policy's actual model calls. It retains only each policy's latest prediction and compact accounting, without a full prediction cache.

Timing is one CPU pass with two Torch threads; CUDA was unavailable in the execution runtime. Ten synthetic zero-image warmups per model precede exposure and are excluded. Per-policy compute is model/transfer plus scheduler including fresh risk-cache updates. Shared RGB/range I/O and decode/resize are measured once per frame. Offline scoring/logging are excluded from policy compute; whole-matrix wall time includes all seven comparisons and is not single-policy end-to-end throughput. OS cache/load are uncontrolled. No energy, battery or flight-safety measurement follows from these timings.

## Read-only verification and reproduction

From the source checkout or extracted source distribution:

```bash
python scripts/verify_independent_test.py
```

This standard-library verifier checks frozen sources, exports, preparation and output identities; per-frame order, counts, actual fixed schedules, ages and reasons; trajectory-local events/censoring; pooled metrics and the 35-row table. It opens no test images and makes no inference calls. The dashboard also verifies the completed protocol and displayed result hashes before presenting test data.

The original execution used the following command **once**. It now refuses to run because exposure/output already exist:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/independent_test.py run \
  --locations configs/independent_test_local.json \
  --protocol-sha256 6a1bb71c98ba3a5b513ff46e7babb4c33dca168886f9561072e8095cff71473f \
  --confirm-new-test-exposure
```

Do not remove the ledger, edit the protocol or rerun this namespace to refresh a report. A genuinely separate, explicitly authorized replication needs a new versioned study namespace, verified data and private KD checkpoint, the historical source revision for compatibility, focused checks, and a new freeze before decoding. It cannot restore the current test's unexposed status. `scripts/prepare_independent_test.py` exposes metadata-only `inventory` and guarded `extract` stages; `scripts/independent_test.py freeze` requires current source-bound passing checks. Historical byte identities are never rewritten to accommodate new source.

The inference-only compatibility record verifies AST equality to checkpoint source at `fb1b4aedd15280abdb1126293e022d78cdcbf4e5`; it does not bypass teacher acceptance or allow training resume. The portable public A/Original exports load without that private checkpoint. Full-data reproduction also needs the exact external staged images, HDF5s and private KD checkpoint recorded by the protocol.

Use [the final report](FINAL_REPORT.md) for conclusions, [the code guide](../CODE_GUIDE.md) for viva locations, and [the dashboard guide](presentation_dashboard.md) for classroom launch and validation-only inference/replay.

[Complete results and interpretation](../results/independent_test_20261030/REPORT.md) · [Saved-output verification](../results/independent_test_20261030/VERIFICATION.json).
