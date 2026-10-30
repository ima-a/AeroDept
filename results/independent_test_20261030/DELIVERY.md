# AeroDepth-Lite completion report

The one authorized independent study completed and passed saved-evidence verification. No additional training, tuning, teacher repair, KD rerun or new presentation was performed. Existing local changes and historical evidence were preserved. The owner subsequently authorized commit/push after completion; the final handoff supplies its commit identity.

## Done / blocked / pending

| Work | Status | Evidence |
|---|---|---|
| Acquisition, CRC, HDF5 correspondence and grouped leakage checks | Done | INVENTORY.json and prepared/PREPARED.json |
| Source/model/input freeze and exclusive exposure | Done once | PROTOCOL.json, EXPOSURE.json |
| Seven comparisons, all 8,820 frames | Done | COMPLETION.json, results.json, comparison.csv |
| Artifact and accounting audit without inference | Passed | VERIFICATION.json |
| Documentation and one-word code navigation | Done | ../../docs/FINAL_REPORT.md, ../../CODE_GUIDE.md |
| Unit/guard/UI tests | 127 passed, 4 skipped, 1 integration test deselected | TEST_VERIFICATION.json |
| Isolated sdist/wheel, strict exports, CLI and portable UI | Passed | PACKAGE_VERIFICATION.json |
| Existing-service browser flows | Passed | BROWSER_VERIFICATION.json |
| Safe local cleanup and historical preservation | Done | CLEANUP.json, PRESERVATION_VERIFICATION.json |
| Git publication | Authorized; verified delivery prepared | Commit/push result is in the final handoff; no force push |
| Required user actions / blocked research dependencies | None | No remaining acquisition or study blocker |

## Verified data and exposure

Eight exact RGB/range archives, 17,640 image files and 8,820 pairs: Kite foggy 2007 (2,204), Kite sunny 0008 (2,205), PLE spring 5008 (2,206), PLE winter 6009 (2,205). Zero partial study downloads; five unrelated/redundant ZIPs were excluded. The eight archive files total 5,866,144,441 compressed bytes and 6,668,899,690 extracted bytes; reused HDF5 files add 310,171,768 bytes. Downloads originals and staged data are retained. Leakage checks covered 192 weather/path identities and found no reserved/exposed-group overlap. No failed-HDD operation occurred.

Ready protocol SHA-256: `6a1bb71c98ba3a5b513ff46e7babb4c33dca168886f9561072e8095cff71473f`. Manifest SHA-256: `4220cb10bd3d7671a4e08498b2f9ce34860ccdc95debce6eb93cb25cfb7da9f2`. Completion SHA-256: `a641728f92e8fa9cbdc6ae8db853b790c05618e1b00e4c62be8ff85022bdf9fc`. The exposure ledger preceded decoding; existing exposure/output prevent a second run. All seven comparisons scored exactly **377,372,106 valid pixels and 105,333,081 near pixels**.

## Pooled model/policy results

| Model / policy | AbsRel ↓ | RMSE m ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | Near AbsRel ↓ | Near RMSE m ↓ | Calls |
|---|---|---|---|---|---|---|---|---|
| A_every | 0.383690 | 81.813343 | 0.406799 | 0.661703 | 0.792140 | 0.439396 | 4.144905 | 8820 |
| A_fixed2 | 0.387019 | 81.975304 | 0.405759 | 0.660284 | 0.790959 | 0.442566 | 4.369061 | 4411 |
| A_fixed5 | 0.412312 | 83.462253 | 0.399407 | 0.653437 | 0.785177 | 0.464057 | 7.756830 | 1765 |
| A_budget_fixed | 0.386539 | 81.962581 | 0.406137 | 0.660468 | 0.791392 | 0.442450 | 4.416516 | 4912 |
| A_v2 | 0.388208 | 82.298352 | 0.406031 | 0.659257 | 0.790126 | 0.439805 | 4.405438 | 3718 |
| Original_every | 0.390815 | 81.798271 | 0.381188 | 0.644062 | 0.791480 | 0.487086 | 4.087260 | 8820 |
| KD_every | 0.383202 | 82.173502 | 0.411403 | 0.661888 | 0.787492 | 0.436249 | 4.219446 | 8820 |

[Full study report](REPORT.md) contains all 35 pooled/per-path metric rows, diagnostics/counts, reasons, ages, event delays/censoring, actual timing and measurement boundaries. [The consolidated final report](../../docs/FINAL_REPORT.md) includes the complete quality tables and previous validation/teacher/KD findings.

A improves pooled AbsRel 1.82% versus Original but worsens near RMSE 1.41%; on Kite sunny its near RMSE is 79.29% worse. KD slightly improves pooled AbsRel 0.13% versus A (a validation ranking reversal), but worsens near RMSE 1.80%. This does not accept the rejected teacher or replace A as the preselected demonstration model.

V2 uses 3,718 calls versus budget-fixed's 4,912. Measured inference-plus-scheduler time is 17.08% lower, AbsRel 0.43% worse, pooled near RMSE 0.25% lower and maximum event-update delay 4 versus 1 frames. Near RMSE worsens on three of four paths; winter accounts for the pooled benefit. There are 20 events and no right-censored events in this run. Kite sunny has zero defined events, so its rates/delays are undefined. Timing is one CPU pass, not energy, battery, flight safety or guaranteed detection evidence.

## Reproducibility, organization and cleanup

[CODE_GUIDE.md](../../CODE_GUIDE.md) maps viva topics to real definitions and one-word labels. The earlier 13-file comment-only AST receipt is preserved; 12 numerical files still match those annotated bytes, and the dashboard's later functional changes have separate tests. Source directories remain src/, configs/, scripts/, tests/, models/, results/ and docs/. Private data/checkpoints/configs/caches stay ignored.

Run the dataset-free saved-evidence audit from the checkout:

```bash
python scripts/verify_independent_test.py
```

The isolated package verification loaded both exports strictly, ran synthetic CLI inference, rendered five dataset-free dashboard tabs and new-test selectors, and verified the saved study from the extracted sdist. Third-party dependencies were reused read-only; fresh network dependency resolution was not claimed. The working PyTorch environment was unchanged. Package hashes identify the checked build before later report/verification receipts.

This delivery removed 683,763,922 bytes of disposable isolated build/test copies and exact duplicate browser downloads; [CLEANUP.json](CLEANUP.json) lists each item. The earlier 12.66 GiB cleanup is recorded separately and not counted again. All 484 protected historical files are unchanged. No sole-copy historical private material, dataset, original download or working environment was deleted.

## Dashboard and classroom sequence

The original `aerodepth-dashboard-20261029t155521z.service` is active at **http://127.0.0.1:8501**. Its archived runtime log directory was restored; no competing server was launched. One-command start while this existing transient unit is present:

```bash
systemctl --user start aerodepth-dashboard-20261029t155521z.service
```

For another checkout/machine with no active server, use `python -m streamlit run scripts/dashboard.py` after installing the optional dashboard dependencies. Saved metrics/assets are dataset-free; actual validation viewing/replay require the explicit healthy external inputs. Full-data requirements are in [the independent guide](../../docs/independent_final_test.md).

1. Overview: distinguish validation and independent-test cards; explain A's role and ranking reversals.
2. Model comparison: choose New independent test, then pooled and Kite sunny; compare overall and near errors.
3. Depth demonstration: choose the validation Poor/Typical preset, switch A/Original/KD and explicitly run one model call.
4. Recorded A V2 replay: show source frame, fresh/reused status and age; playback FPS is not inference throughput.
5. Scheduler: choose New independent test · A; compare V2/budget-fixed and all four paths, calls and delays. Historical exposed test stays separate.
6. Evidence: download the protocol, completion and tables; use the code guide to locate model, training, loss, metrics and exposure controls.

Browser verification exercised all three splits, four test scopes, five A policies, timelines, historical/validation labels, three actual CPU validation model calls, advancing recorded replay, exact-byte downloads and mobile rendering. A first selector attempt timed out during rerender; the subsequent check added waits and completed. No test inputs were reopened.

## Changed files and identities

`DELIVERY_MANIFEST.json` lists exact hashes of the changed/new repository-facing files at sealing time; `SHA256SUMS` covers this study's saved evidence. The frozen numerical source hashes remain in PROTOCOL.json. Preserved records were not rewritten to fit new source. Local configs, environments, downloads, datasets, private checkpoints and caches are excluded from the Git delivery. The final user instruction authorizes commit and push; no merge, force push or history rewrite is needed.
