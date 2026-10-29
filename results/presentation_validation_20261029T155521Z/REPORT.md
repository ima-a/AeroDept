# AeroDepth-Lite validation presentation — completed implementation and measurements

Report time 2026-10-29T22:22:24.271451+05:30 (IST). No training, search, scheduler retuning, reserved test pixels/headers, Git publication or cloud deployment. Original published export/defaults and frozen research remain unchanged.

Separate A export: `/home/Ima/work/Projects/DL-projects/AeroDept/models/presentation_A_epoch9_20261029T155521Z/supervised_A_epoch9.state_dict.pt`. Weights SHA256 `6820e7050dd5b6278ecdaf7055d9d69dbbe246a9644ebbdff77056da4278ae20`; metadata SHA256 `e9fed71377da6baa94b0a44ecdbf0ae6b1df44cc1ef1b4718a3d53a65e7faa12`. Architecture MobileNetV3-Small range decoder, 1,027,489 parameters. Original selected A checkpoint SHA256 `84b393a61f73ab95666a907931bff93fd21f65c2b5c1b2e170ce37b3c8f3f8b8`. Strict loading and complete model/BN state matched; CPU FP32 outputs were byte-equal for black, white, seed42 random and the first permitted validation RGB. No checkpoint was overwritten. Full192×320 RGB, one in-model ImageNet normalization, metric Euclidean range with0.1m floor/no upper clamp; finite GT[0.1,1000], near GT<10. Batch1 every-frame results reproduce the prior A FP32 reference within its existing1e-5/1e-3m tolerances and exact counts (cache_reference.json).

The presentation decision is fixed: A epoch9 general-depth default, original epoch4 near-field reference, exploratory KD epoch9 negative comparison. Source comparison metrics yield A pooled AbsRel11.84% better but near RMSE16.18% worse than original. Teacher B rejection remains unchanged; no accepted-teacher claim or automatic deployment promotion follows.

The full validation protocol was frozen before scoring, SHA256 `771bc3e9201238d11b170bfad3177bd437bbee74583cd41f2cb00d891822e813`. V2 `v2_02` parameters come byte-for-parameter from the historical final-study protocol. Independent budget rule is K=ceil(N*2455/4410), positions ceil(kN/K). For2205 frames/path it yields1228/path,2456 pooled; it is not retrospectively fitted to A's calls. All4410 current GT frames are scored per policy, with reused predictions unwarped. Full-split quality uses a causal counterfactual replay over the verified per-frame A cache; the following live timing runs execute the actual model separately.

| Policy | Calls | Skip % | AbsRel | RMSE m | Near AbsRel | Near RMSE m | Mean / max age | Events / max delay / censored |
|---|---:|---:|---:|---:|---:|---:|---|
| every | 4410 | 0.00 | 0.265831 | 119.944 | 0.201899 | 3.906 | 0.000 / 0 | 11 / 0 / 0 |
| fixed2 | 2206 | 49.98 | 0.267708 | 119.966 | 0.203273 | 4.163 | 0.500 / 1 | 11 / 1 / 0 |
| fixed5 | 882 | 80.00 | 0.282528 | 120.930 | 0.211366 | 6.586 | 2.000 / 4 | 11 / 4 / 0 |
| budget_fixed | 2456 | 44.31 | 0.267810 | 120.134 | 0.202683 | 4.161 | 0.443 / 1 | 11 / 1 / 0 |
| v2 | 2456 | 44.31 | 0.268088 | 120.283 | 0.200658 | 3.780 | 0.951 / 4 | 11 / 2 / 0 |
| v2_no_motion | 1900 | 56.92 | 0.268943 | 120.070 | 0.201134 | 3.874 | 1.241 / 4 | 11 / 2 / 0 |
| v2_no_visual | 1789 | 59.43 | 0.274759 | 120.657 | 0.205770 | 5.529 | 1.465 / 4 | 11 / 4 / 0 |
| v2_no_risk | 2446 | 44.54 | 0.268167 | 120.290 | 0.200724 | 3.770 | 0.956 / 4 | 11 / 2 / 0 |
| v2_no_age | 2454 | 44.35 | 0.268108 | 120.285 | 0.200653 | 3.779 | 0.953 / 4 | 11 / 2 / 0 |

| Path | Policy | Calls | AbsRel | RMSE m | Near AbsRel | Near RMSE m | Valid / near pixels | Event max delay |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Kite foggy | every | 2205 | 0.290636 | 155.964 | 0.504336 | 4.695 | 114,412,073 / 894,287 | 0 |
| Kite foggy | fixed2 | 1103 | 0.290545 | 155.778 | 0.508717 | 4.723 | 114,412,073 / 894,287 | 0 |
| Kite foggy | fixed5 | 441 | 0.294152 | 155.982 | 0.507954 | 4.728 | 114,412,073 / 894,287 | 1 |
| Kite foggy | budget_fixed | 1228 | 0.291007 | 156.039 | 0.503558 | 4.689 | 114,412,073 / 894,287 | 0 |
| Kite foggy | v2 | 905 | 0.293204 | 156.325 | 0.504336 | 4.695 | 114,412,073 / 894,287 | 0 |
| Kite foggy | v2_no_motion | 441 | 0.294152 | 155.982 | 0.507954 | 4.728 | 114,412,073 / 894,287 | 1 |
| Kite foggy | v2_no_visual | 905 | 0.293204 | 156.325 | 0.504336 | 4.695 | 114,412,073 / 894,287 | 0 |
| Kite foggy | v2_no_risk | 905 | 0.293204 | 156.325 | 0.504336 | 4.695 | 114,412,073 / 894,287 | 0 |
| Kite foggy | v2_no_age | 905 | 0.293204 | 156.325 | 0.504336 | 4.695 | 114,412,073 / 894,287 | 0 |
| PLE spring | every | 2205 | 0.237017 | 53.311 | 0.193278 | 3.881 | 98,491,741 / 31,373,553 | 0 |
| PLE spring | fixed2 | 1103 | 0.241180 | 54.047 | 0.194567 | 4.146 | 98,491,741 / 31,373,553 | 1 |
| PLE spring | fixed5 | 441 | 0.269024 | 57.869 | 0.202912 | 6.631 | 98,491,741 / 31,373,553 | 4 |
| PLE spring | budget_fixed | 1228 | 0.240863 | 53.975 | 0.194106 | 4.145 | 98,491,741 / 31,373,553 | 1 |
| PLE spring | v2 | 1551 | 0.238911 | 53.729 | 0.192002 | 3.751 | 98,491,741 / 31,373,553 | 2 |
| PLE spring | v2_no_motion | 1459 | 0.239658 | 53.860 | 0.192388 | 3.847 | 98,491,741 / 31,373,553 | 2 |
| PLE spring | v2_no_visual | 884 | 0.253331 | 55.514 | 0.197260 | 5.551 | 98,491,741 / 31,373,553 | 4 |
| PLE spring | v2_no_risk | 1541 | 0.239082 | 53.763 | 0.192070 | 3.740 | 98,491,741 / 31,373,553 | 2 |
| PLE spring | v2_no_age | 1549 | 0.238955 | 53.740 | 0.191997 | 3.750 | 98,491,741 / 31,373,553 | 2 |

**Measured scheduler conclusion:** full V2 and budget-fixed both use2456 pooled calls, but redistribute them (V2 Kite905/PLE1551 versus1228/1228). Pooled near RMSE improves9.14% versus budget-fixed; pooled AbsRel and RMSE regress, and maximum event-update delay rises1→2. Kite near RMSE is slightly worse than budget-fixed while PLE improves. Stale predictions may incidentally reduce aggregate error on correlated frames; this is not improved detection/generalization. There are only1 Kite and10 PLE events. Zero-event rates remain null by implementation; all observed final event waits are uncensored here. Pixel counts match exactly across all policies: pooled212,903,814 valid/32,267,840 near, with unchanged per-path counts. Full δ1/2/3, reasons and all ages/source indices are in quality_full_table.csv and policies/*_frames.json.

Removing motion reduces calls to1900 with a small pooled near regression. Removing visual change reduces calls to1789 but worsens near RMSE to5.529m and maximum delay to4 frames. Removing predicted risk or the soft-age score changes only10 or2 pooled calls; tiny near-score improvements do not prove those signals universally unnecessary. These ablations change inference budgets and state trajectories: no equal-compute causal claim is made. Both weights and associated forced triggers are removed; survivors are never renormalized. First-frame inference, resets, causal held GPS/IMU and hard MAX_SKIP4 stay active. Age-only equals fixed5 and was not duplicated. [Quality figure](figures/A_scheduler_quality.png).

**Live risk cache:** new PresentationScheduler stores both risk value and availability on an explicit fresh source token, retains them on reuse, and resets at trajectory boundaries. Age remains independent. Actual full-resolution cached-versus-uncached mechanism checks match all Decision fields, reasons, source indices and ages for4410 frames in each of five adaptive variants. Historical V2 feature-level equality passes4410 full-V2 decisions. Unit tests exercise empty/invalid predictions, repeated tokens, reset, threshold neighbours, removed forced triggers, hard age and no privileged GT velocity. Actual live timed traces also match cached/uncached decisions/reasons/source indices/ages across3 repetitions on each device (timed_live_cache_equivalence.json). This combines full-resolution mechanism evidence with real model execution; the earlier offline feature cache alone is not presented as live caching.

The original mechanism command wrote its complete five-policy passing receipt before its shell session returned143; no assertion failure was recorded. A defensive follow-up checker refused to overwrite that existing receipt. Both execution records/logs are retained; the completed receipt is independently validated alongside clean unit-test and live-timing trace checks. No whole experiment was automatically restarted.

**Actual timing boundaries:** FP32/eval batch1, CPU2 threads and GTX1650 CUDA, ten excluded warmups per run,3 alternating repetitions. Windows are Kite1500–1627 and PLE400–527 (128 each), resetting at each window. Actual calls per256-window run: every256, budget-fixed144, V2 cached/uncached222, fixed552. These stress-window budgets differ from full-path quality; no whole-path extrapolation or equal-call live-cost claim is made. Raw file read-stage I/O, RGB+GT PNG decode/resize, transfers, synchronized model-only, scheduler including risk-cache updates and end-to-end are recorded separately. GT is a display/validation pipeline input only, never a scheduler input. Offline quality scoring and final file writes are excluded. OS/file cache is uncontrolled and warm from prior validation; ordinary desktop/server load remains a limitation.

| Device | Policy | Actual calls ×3 | End-to-end median [min,max] s | Model-call median ms | Scheduler median s |
|---|---|---|---:|---:|
| cpu | every | [256, 256, 256] | 10.4203 [10.2725,14.3289] | 14.4753 | 0.004164 |
| cpu | budget_fixed | [144, 144, 144] | 8.6725 [8.3859,9.1672] | 13.7441 | 0.005005 |
| cpu | v2_cached | [222, 222, 222] | 10.4290 [10.1152,10.6138] | 13.7307 | 0.236049 |
| cpu | v2_uncached | [222, 222, 222] | 10.1185 [9.7162,10.3029] | 14.2932 | 0.272678 |
| cpu | fixed5 | [52, 52, 52] | 7.2661 [7.0481,7.3484] | 14.2555 | 0.003675 |
| cuda | every | [256, 256, 256] | 9.0176 [8.8663,9.4765] | 9.6642 | 0.003982 |
| cuda | budget_fixed | [144, 144, 144] | 7.8625 [7.7724,8.3073] | 9.6604 | 0.005051 |
| cuda | v2_cached | [222, 222, 222] | 8.7650 [8.7047,9.3071] | 9.6291 | 0.219143 |
| cuda | v2_uncached | [222, 222, 222] | 8.7976 [8.7502,9.3273] | 9.6089 | 0.260307 |
| cuda | fixed5 | [52, 52, 52] | 6.9060 [6.8450,7.2114] | 9.6523 | 0.003647 |

Risk caching changes median scheduler time by-13.43% on CPU and-15.81% on CUDA. But median end-to-end changes are+3.07% CPU (regression) and-0.37% CUDA; repetition ranges overlap. No robust end-to-end speedup is established. Preprocessing dominates this validation-display pipeline, and full V2 executes more window calls than budget-fixed. Batch1 every-frame model-call medians are about14.48ms CPU and9.66ms CUDA, excluding transfers/I/O/preprocessing/scheduler. These are measured windows/hardware, not dashboard playback FPS, training speed, energy or battery savings. [Timing figure](figures/A_live_timing.png). Individual30 runs and frame measurements are in timing_individual.csv/timing_frames.csv.

**Local dashboard:** http://127.0.0.1:8501. Five tabs: Overview, Model comparison, Depth demonstration, Scheduler, Evidence. Real inference requires its button; all three pinned models were exercised. A full recorded sequence uses cached causal source indices/ages and explicit playback labelling; original/KD only have fixed example recordings, otherwise unavailable until explicit inference. Model/trajectory/frame/policy/history selectors, missing configuration/results/cache handling, plots and byte-exact CSV export are verified. Historical epoch4 studies are separate. No UI control can train, benchmark, tune or evaluate test data. The optional Streamlit extra keeps normal CLI imports independent. Browser screenshots and verification receipt accompany this report; see docs/presentation_dashboard.md for installation/locations and reproduction.

```bash
python -m pip install -e '.[dashboard]'
python -m streamlit run scripts/dashboard.py
```

**Next final-test preparation:** reserved availability was checked through the rescue index/manifest metadata only: no reserved image records in the healthy rescue inventory; new independent test manifest is missing. Archives on the unmounted failed HDD remain unknown. FINAL_TEST_PROPOSAL.json identifies the exact A export and fixed primary V2-vs-budget comparison, secondary every/fixed2/fixed5, unchanged metric/event definitions, four reserved realizations and selection limitations. The future opt-in command refuses this blocked proposal; first verify acquisition/counts/sensors and freeze a ready protocol with exact manifest/root identities. The exclusive exposure ledger precedes test pixels; existing output/ledger is never reused or resumed. The original exposed final test stays historical. No reserved RGB/depth/header was opened here. See FINAL_TEST_COMMAND.txt; no further training is authorized.

Remaining blockers are reserved data/manifest readiness and separate final-test authorization/freeze; actual ablation timing beyond the predeclared selected live policies is unavailable, not fabricated. Published model promotion remains unperformed. All current model/checkpoint/protocol identities, historical source/results, exposure ledgers and prior dirty/untracked work are preserved. No commit, push, merge or cloud deployment.

End-to-end measurement ends after per-frame processing/cache update. It excludes model loading, manifest/sensor setup, dashboard rendering and WebSocket delivery, as well as warmups/offline scoring/final writes. It is not a measured browser-video frame rate.

**Final verification and delivery:** 15 focused tests pass (external Streamlit/Pandas deprecation warnings only), syntax/CLI help and `git diff --check` pass. Real rendered browser verification exercises all five tabs, three explicit CPU model inferences, an explicit CUDA A inference, model/trajectory/frame/policy/history selectors, playback, missing-data states, charts, byte-exact CSV and PNG downloads. Final overview/scheduler/depth screenshots were inspected. Verification receipts distinguish successful checks from earlier automation locator failures and the cache-check shell-exit caveat. `preservation_verification.json` verifies the156 protected original files, selected checkpoint/export identities, unchanged rejected gate, frozen sources and all new validation-cache hashes.

Changed current files: README/optional pyproject extras; `.streamlit/config.toml`; `configs/presentation_local.json`; `docs/presentation_dashboard.md`; new presentation backend/scheduler/UI modules, dashboard/stage/future-test/mechanism-check scripts and two focused test files. No original training/model/scheduler implementation was edited. New pinned export, measured results and source snapshots are separate. See `source_identity.json`, `execution_record.json`, `SHA256SUMS`, artifact test/browser logs and final Git status. The local dashboard service is `aerodepth-dashboard-20261029t155521z.service` (observed PID350385); stop with `systemctl --user stop aerodepth-dashboard-20261029t155521z.service`. It needs the mounted rescue USB and may stop at full logout.

A separate delivery backup preserves relative project paths under `/run/media/Ima/AERODEPTH/AeroDepth-rescue/project/presentation-delivery_20261029T155521Z/`. The3.29GB private validation cache already resides on that USB and is not duplicated. `artifacts/presentation_validation_20261029T155521Z/backup_verified.json` records copy verification; mutable server logs are represented by a fixed `dashboard.log.snapshot`.
