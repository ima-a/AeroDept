# Local presentation dashboard and A validation workflow

This is a separate, versioned validation workflow. The original published epoch-4 weights, CLI defaults, frozen V1/V2 implementations, final-test ledger and all historical results remain unchanged. Supervised A epoch9 is the dashboard's general-depth default; original epoch4 is the near-field reference; KD epoch9 is an exploratory comparison using the rejected teacher. The final student comparison records A's11.84% pooled AbsRel improvement and16.18% near-RMSE regression versus original. Displayed headline numbers are calculated from that verified artifact rather than embedded as new measurements.

Install and launch locally (Python3.11+ for the optional dashboard; the normal CLI's Python requirement is unchanged):

```bash
python -m pip install -e '.[dashboard]'
python -m streamlit run scripts/dashboard.py
```

The app binds only to `127.0.0.1:8501`, with usage statistics disabled. There is no cloud deployment, automatic training, scheduler search, benchmark or test-evaluation action. Normal CLI installation does not import or require Streamlit. For a different prepared local artifact config:

```bash
AERODEPTH_PRESENTATION_CONFIG=/absolute/path/to/presentation.json \
  python -m streamlit run scripts/dashboard.py
```

`configs/presentation_local.json` records the actual local locations: the verified rescue USB dataset root, explicit frozen TRAIN/VAL manifest/reservations, separate presentation cache, new results directory, pinned A inference export, original export and private KD selected checkpoint. It does not discover datasets or follow dataset symlinks. Missing files/results show unavailable/pending. A/KD training checkpoints stay private; only the separate A inference-only export is packaged for demonstration. No published default or deployment model is changed.

The five tabs show the project and path assignments; all students' pooled/per-path/near/floor metrics and recorded training histories; depth illustrations or explicit real inference; new A versus historical epoch4 scheduler evidence; and downloadable identities/protocols/tables/figures. Original recorded training validation used AMP, A/KD use FP32, and their data/budgets/objectives differ; the curve caption preserves this distinction.

Depth viewing is validation-only. Range colours are log0.1–1000m; near signed-error colours are±10m. Colour clipping never changes numeric predictions or metrics. The **Run actual model inference** button loads only the selected pinned model and performs one FP32/eval call. Its observed latency is not a repeated benchmark. Recorded A replay uses the full causal policy timeline and cached predictions, with fresh/reused status, source indices and ages. Requested playback frames/s is a display setting, not measured inference throughput. Original/KD recorded predictions are available for six fixed comparison examples; other allowed validation frames require the explicit button. No missing result is fabricated. Actual inference/playback pauses while the controlled timing workflow is active.

## Reproduction without overwriting evidence

Before a new reproduction, copy the config to a new file, use fresh `results`, `cache` and (if re-exporting) `export_directory` paths, and retain exact source/input hashes. Copy the frozen `PROTOCOL.json` into the new results directory and verify its model/source identity, or prepare a separately versioned equivalent before scoring if paths/source deliberately change. Do not change parameters or substitute a teacher. All protocol sources, selected model identities and measurement boundaries must be frozen before numerical evaluation.

```bash
# Only when creating a separate new export; preserves selected checkpoint bytes.
python scripts/presentation.py export --config /path/to/new_config.json
# The protocol must already bind the resulting export and current implementation.
python scripts/presentation.py cache --config /path/to/new_config.json
python scripts/presentation.py quality --config /path/to/new_config.json
python scripts/presentation.py cache-check --config /path/to/new_config.json
python scripts/presentation.py timing --config /path/to/new_config.json
```

Completed output guards refuse to overwrite cache/quality/cache-equivalence/timing evidence. Do not run those commands against the completed local config to repeat a study. `verify_presentation_cache.py` is an alternative one-thread mechanism checker for a fresh incomplete verification namespace; it does not provide timing measurements or overwrite existing checks. Actual timing always uses the frozen two-thread policy and records all repetitions/calls.

Quality evaluates every current validation frame for every-frame, fixed2, fixed5, the frozen `ceil(N*2455/4410)` budget rule, full V2 and four remove-one-signal ablations. Removed signals lose both their weighted contribution and associated forced triggers; surviving weights are unchanged. First-frame inference, per-trajectory state resets, causal held GPS/current IMU and hard MAX_SKIP4 remain. Age-only MAX_SKIP4 equals fixed5 and is not duplicated. Ablations also change call budgets; they are not equal-compute causal effects.

The recorded prediction cache is for complete offline quality scoring and playback. A separate live mechanism check recomputes risk from the full previous prediction for an uncached reference, versus cache updates only on fresh source tokens. Real timed runs execute the model and risk-cache update; they do not substitute prediction-cache lookup speed. Boundary tests and actual timed traces compare decisions/reasons/source indices/ages.

Timing uses three alternating repetitions on CPU and CUDA, batch1 FP32, ten identical excluded warmups, and fixed128-frame windows starting at Kite1500 and PLE400. These near-field/scene windows reset locally and have different call budgets from full-path quality. Individual measurements and median/min/max retain model-only time, scheduler including live cache updates, transfers, raw read-stage I/O, decode/resize and end-to-end processing. The end-to-end demonstration pipeline includes GT PNG loading for display/scoring context; GT never enters scheduler decisions. Model loading, manifest/sensor setup and dashboard rendering/WebSocket delivery are outside the timed per-frame boundary. File/OS cache and desktop conditions are not controlled. No energy, battery or flight-safety improvement is measured or inferred.

## Independent final test remains blocked

The new results directory contains a metadata-only acquisition-readiness record, a separate `FINAL_TEST_PROPOSAL.json` and the explicit opt-in command template. The healthy rescue index currently has no image records for the reserved foggy Kite2007, sunny Kite0008, spring PLE5008 or winter PLE6009. The failing HDD remains unmounted, so existence of archives there is unknown. There is no new reserved test manifest, and expected8820 frames must be verified from acquisition metadata rather than forced.

`scripts/presentation_final_test.py` refuses a proposed/blocked protocol, a missing opt-in flag, changed hashes, wrong realizations, unsafe/symlink paths, or an existing exposure/output. A future ready protocol must pin the actual independent manifest, sensor hashes, root, exact A export and unchanged policies before separately authorized exposure. The exclusive ledger precedes the first test decoder read; no automatic retry/resume or post-test selection exists. The original exposed final test remains historical, never an untouched selection source. No test images or headers are opened during this task.

[Streamlit testing reference](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) and [Playwright browser reference](https://playwright.dev/python/docs/browsers) document the optional verification tools. UI tests check missing inputs and no automatic inference; a real local browser exercises selectors, explicit inference, playback and downloads, with screenshots retained in the new results directory.


## Current delivery and portable source installation

The existing five-tab UI now has a study-context sidebar, explicit completed/blocked statuses, separate overall and near-RMSE charts, preset buttons for the previously fixed comparison illustrations, common colour legends, fresh/reused playback indicators and grouped evidence downloads. These are presentation changes; the frozen numerical backend, scheduler parameters and completed results are unchanged. [Consolidated final report](FINAL_REPORT.md); no PPT is produced.

The current independent test is **NOT RUN**. Metadata rechecks establish that the surviving official config has24 training image archives and6 sensor archives, with no reserved image links. Four climate HDF5 files are present, but all eight reserved RGB/depth targets are absent in healthy inventories. Request Kite foggy official0007/internal2007, Kite sunny0008/internal0008, PLE spring official4008/internal5008, PLE winter official4009/internal6009, only left RGB and depth. Use your email/terms on the official form for fresh access; CAPTCHA may be presented but was not confirmed. Do not preview trajectory samples or download duplicate HDF5. Sizes are unknown and current rescue free capacity is not a guarantee for download plus extraction. The failed HDD stays unmounted.

The new blocked study specification predeclares original/KD every-frame secondary analyses in addition to A's five-policy matrix. The preserved primary-only runner does not yet implement that added secondary/supplementary/timing scope. Acquisition, explicit integrity/leakage-verified manifest and a versioned runner extension are prerequisites for the genuine ready freeze. The user has already authorized one study; another approval is not the blocker. No exposure ledger or test completion was created.

### Portable configuration

`configs/presentation.json` uses repository-relative public artifact/export paths. It does not contain this laptop's mount paths. A relocated checkout automatically selects it; on this laptop the prepared `configs/presentation_local.json` remains in use and is excluded from publication. Override explicitly when needed:

```bash
AERODEPTH_PRESENTATION_CONFIG=configs/presentation.json \
  python -m streamlit run scripts/dashboard.py
```

Saved tables, figures and reports work without a dataset. For the optional validation viewer, set the following to verified local artifacts (no automatic acquisition/discovery):

```bash
export MIDAIR_ROOT=/your/healthy/verified/MidAir
export AERODEPTH_PRESENTATION_CACHE=/your/private/validation-cache
export AERODEPTH_COMPARISON_CACHE=/your/private/fixed-example-arrays
export AERODEPTH_KD_CHECKPOINT=/your/private/verified/KD-best.pt
```

Unresolved external variables map to an unavailable location, never the current directory. The original and A inference exports load without a private checkpoint or pretrained-weight download. KD still requires its exact trusted private selected checkpoint; absence is displayed honestly. Do not insert credentials or signed links into public configs. The normal dataset-free CLI continues to default to the original published model.

Source deliveries include the scripts, configs, reports and the two explicitly allowlisted compact inference exports through `MANIFEST.in`; wheel installation provides the library/CLI, while the source checkout/distribution supplies the dashboard assets. Only the known A state_dict is added to the Git allowlist, not `*.pt` generally. Dataset trees, private states, environments and bulky caches remain excluded.

### Classroom sequence

1. Overview: explain A's overall improvement and near-field regression; show the NOT RUN independent-test status.
2. Model comparison: switch pooled→Kite foggy and compare original/A/KD near errors.
3. Depth demonstration: use Typical/Poor presets; switch models; press **Run actual model inference** once. Read the colour legend and latency scope.
4. A recorded V2 replay: point out fresh/reused status, source frame and age. Playback rate is not measured throughput.
5. Scheduler: compare V2 and budget-fixed, then the visual ablation; show actual timing and its end-to-end regression/uncertainty.
6. Evidence: download the report/table and finish with the exact acquisition blocker.

Existing service: `aerodepth-dashboard-20261029t155521z.service`, localhost8501. Refresh with `systemctl --user restart aerodepth-dashboard-20261029t155521z.service` only after UI changes; do not launch another server on the same port. Stop with the corresponding `systemctl --user stop` command. The mounted rescue USB is needed for live validation viewing and replay; full logout may stop the user service.

The [bounded HDD/USB acquisition inventory](../results/acquisition_publication_20261029T180214Z/REPORT.md) now checks the formerly unmounted HDD read-only as well. All8 reserved RGB/depth streams are absent in the documented locations and original Kaggle package; the HDD hardware failure remains unresolved. No new-test evaluation was run.
