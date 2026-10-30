# Publication handoff · final delivery

Current navigation: [documentation index](README.md), [viva guide](VIVA_GUIDE.md), and [local archive mapping](archive_locations.md). This document retains dated workflow/history; it is not live training or service status. The 30 October cleanup archived local historical records and removed disposable outputs; current completed results are summarized in [the final report](FINAL_REPORT.md).

The owner authorized a final commit/push on 30 October after all available study, dashboard, documentation and verification work finishes. Preserve existing modified and untracked research evidence and selectively include the completed new study. See its completion and verification records rather than inferring success from exposure alone.

- Review `.gitignore`, `MANIFEST.in`, the optional dashboard extras, dashboard/UI helper, portable config, tests, current docs and all intended compact result/provenance directories. Do not use blanket cleanup or discard untracked files.
- Include both exact inference-only exports: original epoch4 and the separately labelled supervised A epoch9. Only those two weight paths are allowlisted. Verify their SHA256SUMS; do not add full A/B/KD training checkpoints.
- Include the canonical frozen protocols, compressed relative manifests, reservations, receipts, metrics, negative results, exposure ledgers and essential figures. Preserve their bytes. Current final-delivery evidence is separate.
- Keep `configs/presentation_local.json` and `configs/independent_test_local.json` private/ignored. Publish `configs/presentation.json`, whose public paths are checkout-relative and external inputs require explicit environment variables.
- Keep datasets, raw archives, large validation memmaps, private fixed-example arrays, environments, signed download configurations, credentials, `.ai`/tool state and private training states excluded. Untracked/ignored does not mean disposable.
- Source distributions include the dashboard scripts/configs/reports/compact exports through MANIFEST.in. A wheel provides the library/CLI; use the source checkout/distribution for the dashboard assets. Build/install receipts record a fresh non-editable package venv with existing third-party dependencies reused read-only; full fresh dependency resolution was not retested.
- Review the final report's synthetic-data/simulated-sensor attribution and MidAir CC BY-NC-SA notices for figures. Do not claim energy savings or flight safety, and do not turn 1−AbsRel into accuracy.
- Keep the original failed teacher gate and exploratory KD designation explicit. Original gated KD never ran; the later separate exploratory run is not an accepted-teacher experiment.
- Include the new verified independent-test protocol, exposure, completion, manifest, compact frame logs, full metrics, timing and saved-artifact verifier. Preserve the older blocked proposal/specification. Do not repeat an exposed study to refresh a document.
- Reuse the delivery's focused tests, isolated installation, real browser/inference/export checks, preservation hashes and `git diff --check`. Do not repeat full training/validation/test experiments as a publication check.
- Review `git diff` and explicitly select intended source/evidence files before any later commit. Commit and push are now explicitly authorized after verification. Do not force-push, merge unrelated work or rewrite history. Existing CI remains manual-only.

Classroom launch:

```bash
python -m pip install -e '.[dashboard]'
python -m streamlit run scripts/dashboard.py
```

See [final report](FINAL_REPORT.md) and [dashboard/storage/reproduction guide](presentation_dashboard.md). Saved evidence works without a dataset; actual validation viewing/replay requires the verified external inputs and private KD checkpoint.


Publication authorization update,29 October2026: the owner now explicitly requested a selective commit/push of the existing work to main. The earlier no-push instruction describes the prior delivery, not this newly authorized action. Check the dated acquisition/publication manifest for included files and local-only exclusions. Historical evidence and source identities remain unchanged.
