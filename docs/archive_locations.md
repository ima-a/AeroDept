# Local storage and cleanup

The 30 October 2026 cleanup removed **12.66 GiB of file contents** (715 files or file symlinks) and organized 55 retained historical paths under a local archive. The repository working directory fell from approximately 24 GiB to 11 GiB on disk. Canonical research evidence was retained.

The old 25 October plan to archive onto the HDD is superseded. The current archive is on the project filesystem:

```text
artifacts/archive/2026-10-30/
```

The exact removed paths, reasons, byte counts, ZIP verification and old→new mappings are in [the cleanup receipt](maintenance/2026-10-30_cleanup.json). Archived paths were renamed on the same filesystem and their inode and byte totals checked. This archive is Git-ignored and is not a second independent backup.

## What remains needed

| Location | Purpose |
|---|---|
| `src/`, `scripts/`, `configs/`, `tests/` | Current implementation, entry points, settings and checks. |
| `models/` | Original epoch-4 and A epoch-9 inference exports with metadata/hashes. |
| `results/` | Canonical protocols, split manifests, reservations, metrics, histories and figures. Stable paths support code and citations. |
| `evidence/` | Historical diagnostic source identities and pilot evidence. |
| `presentations/` | Editable PowerPoint, preview and generation sources. |
| `datasets/midair_pilot/` | Extracted RGB/range and HDF5; no extracted frame was removed. This pilot subset is not the complete expanded dataset. |
| `runs/local_mobilenet/main/` | Original resumable training states, split/configuration and history. |
| `artifacts/local/torch-cache/` | Shared pretrained initialization weights for new training. |
| `artifacts/student_validation_comparison_20261029T152648Z/` | Explicitly configured local dashboard comparison examples. |
| `.venv/` | Working Python/PyTorch environment. |

The local dashboard config still points at the same local comparison cache and external rescue-USB data/prediction cache/private KD checkpoint. No external storage was moved or cleaned. The portable config continues to take external locations from environment variables.

## What was removed

- Old validation prediction/target/mask arrays, the historical validation RGB timing cache and diagnostic near arrays. Saved quality, timing and comparison results remain. These derived arrays can be regenerated from the corresponding inputs into fresh output directories; removal does not authorize rerunning the reserved test.
- Bounded smoke/benchmark checkpoints and generated figures, plus redundant smoke manifests. These were not selected full-run research checkpoints; compact receipts remain archived or canonical under `results/`.
- Build, isolated package-install verification and interpreter/test/font caches. The main environment was retained.
- Four local `frames.zip` copies. **All 8,818 non-directory members** were SHA-256 compared with their extracted counterparts before deletion. Archive hashes and a digest of member identities are recorded. Extracted files and sensor HDF5 remain in place. This was byte comparison, not model evaluation.

## Where historical local files moved

| Old prefix | New prefix |
|---|---|
| `artifacts/<historical-run>/` | `artifacts/archive/2026-10-30/<historical-run>/` |
| `artifacts/local/<historical-item>/` | `artifacts/archive/2026-10-30/local/<historical-item>/` |
| `runs/local_mobilenet/{smoke,smoke_verified,smoke_resumed,benchmark}/` | `artifacts/archive/2026-10-30/runs/local_mobilenet/<name>/` |

The active initialization cache and dashboard comparison directory are exceptions and remain at their original paths. Use the receipt's `moves` list for exact mappings. The earlier cleanup receipt is now at `artifacts/archive/2026-10-30/cleanup_20261025T131018Z/`.

Historical reports retain their original recorded paths and hashes. Some archived scripts also contain old absolute paths. The archive is provenance, not a relocated runnable experiment bundle. Use the public reproduction guides, or prepare a separate source/input layout matching a historical run; resolve paths before using an archived shell script.

## Source versions and checks

One-word comments were added to current Python files for the viva. The [marker receipt](maintenance/2026-10-30_code_markers.json) records AST equality and before/after hashes. Old full checkpoints still require their original source identities. Use the pre-annotation expanded revision `fb1b4aedd15280abdb1126293e022d78cdcbf4e5`; see [expanded reproduction](expanded_reproduction.md).

The historical final-test verifier uses a different, earlier source revision (`8f32f71`). Its [guide](final_study.md) gives the separate-checkout procedure. Do not verify its source seal against current annotated code or weaken the seal. The existing exposure ledger remains unchanged.

See [maintenance verification](maintenance/2026-10-30_verification.json) for retained-file checks and the dataset-free test result. No new training, numerical research evaluation or test exposure was performed during cleanup.

## Independent study delivery · 31 October follow-through

The completed reserved-test inputs are staged outside the project at the explicit private location in ignored `configs/independent_test_local.json`. Downloads originals and staged data are retained. No failed-HDD mount or repair was used. Compact canonical evidence is in `results/independent_test_20261030/`; it has no full prediction cache.

The existing dashboard unit still writes to `artifacts/presentation_validation_20261029T155521Z/dashboard.log`. That small runtime log directory was restored after the restart exposed its old path. Historical contents remain separately archived under `artifacts/archive/2026-10-30/`; no old log was overwritten. Keep the active log directory while using this service.

The new delivery cleanup receipt is `results/independent_test_20261030/CLEANUP.json`. It supplements the earlier 12.66 GiB cleanup, without recounting those earlier deletions. Historical private material without a verified independent healthy backup remains retained.
