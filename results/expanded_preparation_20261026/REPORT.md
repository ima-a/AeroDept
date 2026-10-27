# Expanded MidAir training preparation — 26 October 2026

Training additions and the explicit train/validation manifest are prepared. No training, distillation, scheduler search, test evaluation or archival cleanup was run.

**Locations.**

- Download archives: `/run/media/Ima/Low storage/AeroDepth/MidAir/MidAir-expansion/MidAir`
- New training staging root: `/run/media/Ima/Low storage/AeroDepth/MidAir/MidAir-expansion/training_staging_20261026T000911/MidAir`
- Common dataset root for manifest paths: `/run/media/Ima/Low storage/AeroDepth/MidAir`
- Full local preparation script, extraction-member SHA index and receipts: `/home/Ima/work/Projects/DL-projects/AeroDept/artifacts/expanded_preparation_20261026T000911`

**Acquisition and extraction.** All 30 configured archives (24 left-RGB/depth ZIPs and six sensor ZIPs) passed full member CRC checks and expected-content/path checks. No symlinks, unsafe paths or unexpected members were accepted. All six HDF5 members are byte-identical to the existing climate records. Originals and archives were preserved; extraction used new exclusive files only.

Compressed archive bytes: 21,653,242,902. Extracted bytes: 24,355,581,588. Preserved incomplete backups: 1; these were inventoried without opening and are not counted as completed archives.

**Frozen assignments.**

| Assignment | Independent paths | Realizations | Frames | Verification scope |
|---|---:|---:|---:|---|
| Expanded train | 12 | 20 | 44,098 | All paths/pairs/HDF5/navigation; new 26,463 pairs fully decoded |
| Original validation | 2 | 2 | 4,410 | Records unchanged; fully decoded without augmentation |
| New final test | 4 | 4 reserved | 8,820 expected | HDF5 metadata only; image/depth archives absent in checked locations |
| Original completed test | 2 | 2 quarantined | 4,436 historical | Metadata only; no image/depth reads |

The new manifest is deliberately train/validation-only: `splits.test=[]`, `frame_counts.test=0`. Reserved and quarantined test assignments are frozen separately in `reservations.json`; their expected counts are not passed off as acquired data.

**Schema and read scope.** All image paths are relative to the common dataset root. Original train/validation paths remain unchanged; new paths include the explicit staging prefix. Existing `data.load_manifest` and `MidAirDataset` passed compatibility/contract checks without changing source. The generic Kaggle/preparation validator expects a nonempty three-way inventory and broad discovery; it must not be used for this prepared dataset. No random split was rerun.

A preparation-time audit hook rejected held-out RGB, depth and archive opens before I/O and rejected broad discovery at the original root. The allowlist permits only new training image directories and the two original validation image directories. All train/validation records were checked for relative nonescaping paths and disjoint groups. Sensor-only position hashes match the planned metadata and climate equivalents, and no exact position hash crosses assignments. Pilot groups remain excluded from new test; every equivalent of validation and both test sets remains excluded from training. These preparation guards are not a substitute for enforcing the same scope in future training orchestration.

17635 existing training pairs checked for filenames, HDF5 correspondence and all 9 navigation values per frame; prior original decode/header evidence retained, not re-decoded here.

**Dimensions, decoding and range.** Every new training and permitted validation pair decoded successfully, with paired dimensions and uint16-to-float16 metre decoding checked. Native and 192×320 mask counts, nonfinite/low/high/near counts and finite extrema are in `data_validation.json`. The 0.1–1000 m inclusive finite mask, <10 m near definition, preprocessing and losses remain unchanged.

**Verified realizations and page IDs.**

| Logical trajectory | Official page ID | Path group | Frames |
|---|---|---|---:|
| Kite_training/cloudy/trajectory_3002 | 0002 | Kite/0002 | 2,202 |
| Kite_training/foggy/trajectory_2005 | 0005 | Kite/0005 | 2,206 |
| Kite_training/foggy/trajectory_2006 | 0006 | Kite/0006 | 2,206 |
| Kite_training/foggy/trajectory_2021 | 0021 | Kite/0021 | 2,204 |
| Kite_training/sunny/trajectory_0000 | 0000 | Kite/0000 | 2,205 |
| Kite_training/sunny/trajectory_0005 | 0005 | Kite/0005 | 2,206 |
| Kite_training/sunset/trajectory_1001 | 0001 | Kite/0001 | 2,205 |
| Kite_training/sunset/trajectory_1006 | 0006 | Kite/0006 | 2,206 |
| PLE_training/fall/trajectory_4001 | 4001 | PLE/0001 | 2,204 |
| PLE_training/fall/trajectory_4004 | 4004 | PLE/0004 | 2,204 |
| PLE_training/fall/trajectory_4006 | 4006 | PLE/0006 | 2,205 |
| PLE_training/spring/trajectory_5001 | 4001 | PLE/0001 | 2,204 |
| PLE_training/spring/trajectory_5002 | 4002 | PLE/0002 | 2,205 |
| PLE_training/spring/trajectory_5004 | 4004 | PLE/0004 | 2,204 |
| PLE_training/spring/trajectory_5005 | 4005 | PLE/0005 | 2,206 |
| PLE_training/spring/trajectory_5006 | 4006 | PLE/0006 | 2,205 |
| PLE_training/spring/trajectory_5007 | 4007 | PLE/0007 | 2,205 |
| PLE_training/winter/trajectory_6002 | 4002 | PLE/0002 | 2,205 |
| PLE_training/winter/trajectory_6005 | 4005 | PLE/0005 | 2,206 |
| PLE_training/winter/trajectory_6007 | 4007 | PLE/0007 | 2,205 |
| Kite_training/foggy/trajectory_2003 | 0003 | Kite/0003 | 2,205 |
| PLE_training/spring/trajectory_5000 | 4000 | PLE/0000 | 2,205 |

**Training-only visualizations.** [Contact sheet](training_contact_sheet.png) uses the middle frame from each of the 12 new training realizations. RGB and metric-range maps are clearly labelled; invalid/far masks and metre scales are explicit. No validation/test images were selected for this figure. MidAir attribution/terms remain as documented in `NOTICE.md`.

**Manifest identities.**

- New uncompressed manifest SHA-256: `67c6cad39767d847a2087a81d1ecc4de2ad8c9ecc0fa01e491f3950b25f8dd3a`
- Original uncompressed manifest SHA-256, unchanged: `e7b093be69f1942f281a09563bd63efa479d4352de97da722fb0504208c1fbf5`

**Remaining work before training.** Acquire reserved test inputs separately without previews or decoding; implement the teacher/KD and matched epoch4-initialized student workflow; enforce the same grouped/read scope; freeze budgets, acceptance gate and resume identities; then measure laptop feasibility in a separately authorized bounded check. ImageNet student initialization in the earlier planning record is superseded by the user-specified epoch4 warm-start instruction; no configuration or historical plan was rewritten here.

Historical source, configurations, manifests, checkpoints, reports, hashes and exposure records passed the preservation comparison. See `readiness.json`, `manifest_verification.json`, `preservation_verification.json` and `SHA256SUMS` for exact evidence.
