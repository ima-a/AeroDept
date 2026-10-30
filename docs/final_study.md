# Frozen final study

This study continues published commit `e27ed7e289b9af6ea30a35ff234dfbcd8dfe168a`. Model training, scheduler development and parameter search have ended. The original checkpoint, validation, V1, V2 and pilot evidence are preserved. The historical reports describe their original scope and are not rewritten.

## Source version and saved-result verification

Commit `8f32f71` publishes the exact historical source identities recorded by this study, including the original model/loss code and byte-identical evidence. Current `main` also contains the separate expanded experiment; its evolved source intentionally fails this study's strict historical source check. Verify the recorded results in a separate checkout without accessing MidAir or running model inference:

```bash
git clone https://github.com/ima-a/AeroDept.git AeroDept-final-study
cd AeroDept-final-study
git checkout --detach 8f32f71
mkdir -p artifacts
python3 scripts/verify_final_study.py --study-dir results/final_test \
  --protocol results/final_test/protocol.json \
  --output artifacts/saved_final_verification.json
```

This verifier uses only the standard library and saved metrics/accounting/source/weight hashes. It does not reevaluate test inputs. The execution and exposure records remain unchanged. The training-completion statements below refer to the original study, not the later completed expanded training and separate [reserved test study](independent_final_test.md). Use this historical checkout for any independently authorized reproduction of the historical runner as well.

## Recorded protocol and independent reproduction

The machine-readable [protocol](../results/final_test/protocol.json) was saved before test image/depth access, with SHA-256 `f167c49078bde5bc09eed2a552000d8f9fe5812df13d78a60eaf40085cfa1f8f`. It freezes the exact published model, `v2_02`, preprocessing/masks, metrics, sensor provenance, event/freshness definitions, comparisons, outcomes, aggregation and selection limitations. No test performance can alter these choices.

The independent fixed-budget primary comparator uses the pooled validation fraction **2455/4410 for both test trajectories**. For length N, choose K=ceil(N×2455/4410), with calls at ceil(kN/K) for k=0…K−1. This rule needs sequence length but no observed test V2 calls, predictions or event labels. Fixed-2, fixed-5 and every-frame are additional frozen comparisons. Retrospective exact test-call matching is omitted. The validation timing comparator retains the original validation budgets, 905 and 1550 calls, for an equal-call timing check.

Timing uses the current Intel Core i5-10300H CPU because this runtime exposes no CUDA device nodes. All policies use FP32, batch1, two torch threads and identical excluded model warmups. Validation executes three measured repetitions of V2 and matched fixed, in the protocol's alternating order. Scheduler time includes feature/risk calculation and the decision; inference includes transfers and the actual forward call. Compute excludes disk I/O, preprocessing, offline scoring and logs; wall time is reported separately. Validation preprocessed RGB preparation is excluded from timing. Its regenerable local cache was removed on 30 October; recorded timing results remain unchanged. These measurements do not establish speed on the historical GTX1650.

The leakage audit recomputes environment/path groups for every weather variant, checks the original pilot exclusions, and compares exact position-stream hashes across all locally available climates. It reads metadata and navigation, without reading test pixels. Shared geographic scenery can remain despite disjoint paths; this is a custom two-path simulation test, not the official MidAir benchmark.

Install the published dependencies and set the dataset root as in [reproduction](reproduction.md). Run from the repository root:

```bash
export MIDAIR_ROOT='/path/to/MidAir'
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$PWD/src"
python scripts/final_study.py --stage validation-timing \
  --dataset-root "$MIDAIR_ROOT" --output artifacts/final_study/validation_timing
python scripts/final_study.py --stage audit \
  --dataset-root "$MIDAIR_ROOT" --output artifacts/final_study/isolation.json
python scripts/final_study.py --stage test --allow-final-test \
  --dataset-root "$MIDAIR_ROOT" \
  --timing-dir artifacts/final_study/validation_timing \
  --output artifacts/final_study/test
env -u MIDAIR_ROOT OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python scripts/verify_final_study.py --study-dir artifacts/final_study/test \
  --output artifacts/final_study/verification.json
```

The test command requires completed, hash-checked validation timing and a passing isolation audit. Immediately before the first test pixel read, it creates an exclusive `results/final_test/EXPOSURE.json` ledger. Existing output directories and a second exposure are refused. Retain partial outputs if execution fails; no automatic retry is permitted. The recorded checkout already contains the completed study's ledger. Independent reproduction belongs in a separate checkout; it is a new reproduction, not another untouched test selection. If that checkout includes the published ledger, archive it **in that separate copy only** as `results/final_test/EXPOSURE.recorded.json` before executing the commands above. Keep the recorded study checkout and its ledger intact. The runner intentionally has no bypass or resume flag.

The test makes a single chronological traversal, maintaining independent state and executing actual model calls for all five policies. Policy execution order cycles across frames. Ten identical first-frame warmup calls per policy are counted separately. Every current frame is scored against its current GT, including unwarped reused predictions. Metrics pool additive valid-pixel sums; event statistics reset per trajectory. Model calls, age, reasons, events, delays, censoring, near counts and compute durations are retained in per-frame CSV and per-policy JSON. Coarse offline predicted p10 checks are separate from fresh-update statistics.

Fresh inference does not establish correct obstacle detection. Zero-event rates stay undefined. No energy, battery or flight-safety improvement is measured or claimed. The model and V2 were selected on validation, with known overfit, floor saturation, poor far-depth geometry, only two validation trajectories and jointly changed RGB/motion design. All test regressions remain part of the study.

See `results/final_test/REPORT.md`, `validation_timing/REPORT.md`, the CSV tables, plots, preservation receipts and `SHA256SUMS` for recorded outcomes. Full generated validation RGB cache stays local under `artifacts/final_study/`; raw MidAir data is not redistributed.

Recorded outcome: all 4,436 frames were scored once for each policy. V2 used 2,837 calls versus the primary comparator's 2,470 and 46.59 versus 36.82 CPU compute seconds. Pooled near RMSE improved 1.89%, while PLE near RMSE regressed 0.33% and maximum event-update delay rose from 1 to 4 frames. Both paths exceeded the 2-frame validation selection delay ceiling. No new selection followed these outcomes. The full suite passed 58 tests with one optional integration test skipped.
