# Local research archive locations

The selective cleanup on 25 October 2026 removed only the approved bytecode/test
caches, four reverified duplicate files and an empty isolation log. Its receipts
are in `artifacts/cleanup_20261025T131018Z/` (local and Git-ignored).

**Archival is pending.** MidAir expansion downloads were active during cleanup.
No archive was created and none of the 16 approved archive sources was moved or
removed. Do not archive to the HDD while its downloads remain active.

The approved future archive root is:

```text
/run/media/Ima/Low storage/AeroDepth/repository-archive/2026-10-25/
```

Append each source's original repository-relative path to that root. The exact
pending list is recorded in the cleanup receipt's `archive_pending.json`.
Copy first; record file sizes, SHA-256 values and symlink targets; verify every
copy before removing a local source. Never overwrite a different destination.
Preserve the sibling V1/V2 directories and the relative V2 prediction symlink
without dereferencing it. Preserve historical receipts and reports unchanged.

Canonical public evidence remains under `results/`, published inference weights
and metadata under `models/`, and expansion reservations under
`docs/expanded_experiment_20261025/`. Original main-run checkpoints remain at
`runs/local_mobilenet/main/`; preparation records, initialization cache, datasets
and environments remain in place. Private artifacts and full training states
remain excluded from the publication package.

To verify saved final-study accounting, explicitly select the canonical results
and choose a new output filename:

```bash
python3 -B scripts/verify_final_study.py \
  --study-dir results/final_test \
  --output artifacts/new_saved_result_verification.json
```

This verifies recorded results without images, depth decoding or model calls.
Keep `results/final_test/EXPOSURE.json` intact; cleanup does not authorize another
final-test evaluation. Historical commands still refer to existing local paths
while archival is pending. After a verified archive transfer, document its path
mapping before using those historical local commands.
