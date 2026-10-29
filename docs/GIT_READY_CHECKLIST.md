# Publication handoff — no commit or push performed

The current work is ready for review as a source delivery, not a claim of completed independent final testing. Preserve current modified and untracked research evidence.

- Review `.gitignore`, `MANIFEST.in`, the optional dashboard extras, dashboard/UI helper, portable config, tests, current docs and all intended compact result/provenance directories. Do not use blanket cleanup or discard untracked files.
- Include both exact inference-only exports: original epoch4 and the separately labelled supervised A epoch9. Only those two weight paths are allowlisted. Verify their SHA256SUMS; do not add full A/B/KD training checkpoints.
- Include the canonical frozen protocols, compressed relative manifests, reservations, receipts, metrics, negative results, exposure ledgers and essential figures. Preserve their bytes. Current final-delivery evidence is separate.
- Keep `configs/presentation_local.json` private/ignored. Publish `configs/presentation.json`, whose public paths are checkout-relative and external inputs require explicit environment variables.
- Keep datasets, raw archives, large validation memmaps, private fixed-example arrays, environments, signed download configurations, credentials, `.ai`/tool state and private training states excluded. Untracked/ignored does not mean disposable.
- Source distributions include the dashboard scripts/configs/reports/compact exports through MANIFEST.in. A wheel provides the library/CLI; use the source checkout/distribution for the dashboard assets. Build/install receipts record a fresh non-editable package venv with existing third-party dependencies reused read-only; full fresh dependency resolution was not retested.
- Review the final report's synthetic-data/simulated-sensor attribution and MidAir CC BY-NC-SA notices for figures. Do not claim energy savings, flight safety or independent new-test accuracy.
- Keep the original failed teacher gate and exploratory KD designation explicit. Original gated KD never ran; the later separate exploratory run is not an accepted-teacher experiment.
- Mark independent new testing NOT RUN. Keep the old blocked proposal and new specification separate; do not publish a fake ready protocol, completion or exposure. One study is authorized but blocked on acquisition/manifest and the added secondary-analysis runner scope.
- Reuse the delivery's focused tests, isolated installation, real browser/inference/export checks, preservation hashes and `git diff --check`. Do not repeat full training/validation/test experiments as a publication check.
- Review `git diff` and explicitly select intended source/evidence files before any later commit. No commit, push, merge, history rewrite or automatic deployment is authorized by this delivery. Existing CI remains manual-only.

Classroom launch:

```bash
python -m pip install -e '.[dashboard]'
python -m streamlit run scripts/dashboard.py
```

See [final report](FINAL_REPORT.md) and [dashboard/storage/reproduction guide](presentation_dashboard.md). Saved evidence works without a dataset; actual validation viewing/replay requires the verified external inputs and private KD checkpoint.


Publication authorization update,29 October2026: the owner now explicitly requested a selective commit/push of the existing work to main. The earlier no-push instruction describes the prior delivery, not this newly authorized action. Check the dated acquisition/publication manifest for included files and local-only exclusions. Historical evidence and source identities remain unchanged.
