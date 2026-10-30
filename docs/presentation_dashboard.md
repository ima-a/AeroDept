# Local presentation dashboard and A validation workflow

This is a separate, versioned validation workflow. The original published epoch-4 weights, CLI defaults, final-test ledger and historical results remain unchanged. On 30 October, current source files received one-word navigation comments; Python ASTs are unchanged, while byte-level source hashes differ. Old sealed protocols use their matching source revision, as described in [expanded reproduction](expanded_reproduction.md). Supervised A epoch9 is the dashboard's general-depth default; original epoch4 is the near-field reference; KD epoch9 is an exploratory comparison using the rejected teacher. The final student comparison records A's11.84% pooled AbsRel improvement and16.18% near-RMSE regression versus original. Displayed headline numbers are calculated from that verified artifact rather than embedded as new measurements.

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

## Independent final test and current delivery

The new eight-archive acquisition has passed integrity, exact HDF5 pairing and grouped leakage checks. The separately versioned seven-comparison runner uses A for the five primary policies and Original/KD every-frame for secondary comparisons. [The independent study guide](independent_final_test.md) records exact input sizes, ready protocol, exposure controls, read-only verification and full-data requirements. The old blocked proposal and specification remain preserved as historical records.

The five existing tabs now distinguish validation, the new independent test and the historical exposed test. Overview labels validation headline metrics and separately shows the independent models. Model comparison has an Evaluation split selector and per-path scope. Scheduler has a separate independent A policy view with actual calls, errors, ages, events, timing and saved timelines. Evidence downloads contain the new protocol, exposure, completion, full tables and timing as well as older evidence. Completion is displayed only when saved artifact hashes verify; missing, in-progress, failed and changed evidence have distinct statuses. There is no UI path to evaluate test images.

A stays the preselected demonstration model even if test rankings reverse. The validation-only depth viewer, explicit actual inference button and recorded replay are unchanged. During the final timing run, actual inference/playback are paused to avoid competing model work. Existing teacher rejection, negative KD validation, scheduler regressions and timing limitations stay visible.

Saved metrics/assets work without the dataset. Full validation image viewing/replay needs the existing private cache; actual KD inference needs the pinned private checkpoint. The separate final-test staging tree is never a dashboard input. The [final report](FINAL_REPORT.md) and [code guide](../CODE_GUIDE.md) link the relevant evidence and implementation.

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

1. Overview: explain A's overall improvement and near-field regression; show the independently verified final-study status.
2. Model comparison: switch pooled→Kite foggy and compare original/A/KD near errors.
3. Depth demonstration: use Typical/Poor presets; switch models; press **Run actual model inference** once. Read the colour legend and latency scope.
4. A recorded V2 replay: point out fresh/reused status, source frame and age. Playback rate is not measured throughput.
5. Scheduler: compare V2 and budget-fixed, then the visual ablation; show actual timing and its end-to-end regression/uncertainty.
6. Evidence: download the report/table and finish with the independent study, subgroup reversals and limitations.

Historically recorded service: `aerodepth-dashboard-20261029t155521z.service`, localhost8501. Check its current status before choosing the service or direct launch command; a saved service name does not establish that it is running. Refresh with `systemctl --user restart aerodepth-dashboard-20261029t155521z.service` only after UI changes; do not launch another server on the same port. Stop with the corresponding `systemctl --user stop` command. The mounted rescue USB is needed for live validation viewing and replay; full logout may stop the user service.

The [29 October HDD/USB inventory](../results/acquisition_publication_20261029T180214Z/REPORT.md) remains a historical acquisition record. The final study uses newly verified Downloads and healthy local staging; the failed HDD stays untouched.

The 31 October restart check found the transient unit still referenced its original log directory. `artifacts/presentation_validation_20261029T155521Z/` was restored for new runtime logs; archived historical contents remain separate. The same unit and port are reused. While the unit exists, `systemctl --user start aerodepth-dashboard-20261029t155521z.service` starts it without creating a second server.
