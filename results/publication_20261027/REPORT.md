# Expanded workflow publication — 27 October 2026

This publication makes the existing completed final study and ongoing expanded teacher/student implementation available from `main`. It does not complete or relaunch the experiment. Current source still matches all seven core source hashes in the active recovery freeze; model/training code and all three approved configurations were not changed by packaging.

The [reproduction guide](../../docs/expanded_reproduction.md) documents acquisition/layout requirements, exact approved configurations, fixed settings and teacher gate, numerical handling, source import paths, independent commands and strict checkpoint resume. The [dated training snapshot](../expanded_status_20261027T150226Z/REPORT.md) records supervised completion and teacher progress, without claiming gate acceptance or a KD result.

## Verification actually performed

- Published historical source commit `8f32f71` matches all 17 execution-source hashes in the completed final study. Its 50 evidence files retain original bytes, including CRLF CSVs. Git attributes prevent future normalization of hashed research receipts.
- [Saved-result verification](historical_saved_verification.json) passed from a separate committed historical source copy, explicitly using `results/final_test`. This checks historical accounting and hashes; it performs no image reads or model calls.
- [Focused tests](focused_tests.txt): **30 passed**, from a separate committed expanded-source copy, using CPU only, explicit checkout PYTHONPATH, synthetic inputs/manifest metadata and the existing environment. No fresh dependency installation, full suite repeat, dataset-backed integration, CUDA benchmark or training was performed for publication.
- All [111 indexed research files](checksum_verification.json) matched their existing checksums, including the new dated progress snapshot. Original checksum lists, results, exposure ledgers, manifests and published weight bytes were not rewritten.
- [Preservation and documentation checks](preservation_and_docs.json) cover all 239 baseline files. Only the listed current documentation/Git attributes changed; prepublication documentation is preserved in Git history at `21a73a2`. Approved config and epoch-4 weight identities match the live freeze. Local Markdown destinations and Bash syntax were checked; reproduction commands were not executed.
- Candidate text was scanned for common credential, private-key and signed-download markers, reporting only filenames/types on matches. No markers were found. Datasets, archives, environments, initialization caches, private checkpoints and active logs remain excluded. Two small, explicitly selected historical launch/startup logs are included with their checksum receipts.

The final documentation pass adds explicit `PYTHONPATH="$PWD/src"` so source-relative manifests and identity files resolve to this checkout after a normal package installation. No source change was needed. Historical reports and private artifact references remain as provenance; private raw diagnosis/extraction files and complete training bundles are not distributed. The public per-arm commands operate on independently prepared inputs and fresh outputs; the host-specific frozen supervisor bundle is not recreated automatically.

## Scope and limitations

The original completed test remains quarantined; no test RGB/depth or new test archive was accessed. The new reserved test remains unacquired at the training snapshot. No additional evaluation, optimizer update, scheduler change, cleanup or package/environment modification was made by publication. The already authorized service continued independently from its frozen external source.

Absolute paths, disk UUIDs, PID/service identifiers and startup/progress receipts describe the recorded laptop and observation times, not another user's machine. Full optimizer/scaler/RNG checkpoints remain private; friends can reproduce new runs, but the public package cannot resume the laptop's private run. Hardware/library differences can change numerical results and timings. A separate frozen final-evaluation protocol is still required for the expanded experiment.
