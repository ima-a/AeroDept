# Bounded reserved-data inventory and selective publication

The four authoritative reserved realizations were checked using local metadata only. No test image/header/depth decoding, evaluation, downloads, Gmail or other accounts were used.

The HDD was identified by UUID90C20E17C20E01E8. Its recorded SMART FAILED condition remains unresolved; filesystem repair cannot resolve that hardware failure. After an unsupported `norecover` option was rejected, temporary kernel partition write protection was set/confirmed and NTFS was mounted `ro,nosuid,nodev,noexec`. No filesystem repair, dirty-flag clearing, journal writes, self-test or surface scan was performed. The volume was safely unmounted after inspection.

Checked14 explicit original/download/reserved/staging roots across HDD/USB. Also listed the central directory of the documented21,338,566,906-byte original Kaggle ZIP (52,970 members), with zero reserved realization matches. No archive member payload was read. No symlinks were followed. This is a bounded project inventory, not a search of unrelated personal directories.

| Realization | Completed original config included it? | RGB archive / extracted | Depth archive / extracted | Previously verified complete? | Finding |
|---|---|---|---|---|---|
|Kite_training/foggy/trajectory_2007|No|Absent in checked roots / absent|Absent in checked roots / absent|No|Both streams missing in bounded inventory|
|Kite_training/sunny/trajectory_0008|No|Absent in checked roots / absent|Absent in checked roots / absent|No|Both streams missing in bounded inventory|
|PLE_training/spring/trajectory_5008|No|Absent in checked roots / absent|Absent in checked roots / absent|No|Both streams missing in bounded inventory|
|PLE_training/winter/trajectory_6009|No|Absent in checked roots / absent|Absent in checked roots / absent|No|Both streams missing in bounded inventory|

The surviving completed configuration is `/home/Ima/Downloads/download_config(1).txt`, SHA256 `2330a5e127a6981ce61aa6d3d09da2ba07ff13dd327a711ccef884943ea88e4d`:24 training image archives plus6 sensor archives, no reserved image targets. The earlier `download_config.txt` referenced by historical launch code is no longer in the documented download folder; no assertion is made about an unavailable file beyond saved acquisition records. HDF5 climate records already exist on the rescue USB; they are not missing streams.

No reserved archive locations/sizes or extracted locations can be reported because none was found in the checked locations. Missing sizes are unknown, not zero. Completeness is not merely inferred from file existence: no matching prior reserved-data integrity receipt exists. No file found in this scope is being needlessly requested again.

Exact missing streams only:

```text
Kite_training/foggy/color_left/trajectory_2007/frames.zip
Kite_training/foggy/depth/trajectory_2007/frames.zip
Kite_training/sunny/color_left/trajectory_0008/frames.zip
Kite_training/sunny/depth/trajectory_0008/frames.zip
PLE_training/spring/color_left/trajectory_5008/frames.zip
PLE_training/spring/depth/trajectory_5008/frames.zip
PLE_training/winter/color_left/trajectory_6009/frames.zip
PLE_training/winter/depth/trajectory_6009/frames.zip
```

Detailed explicit paths and metadata: reserved_acquisition_inventory.json. Original reservations, manifests, teacher rejection, completed studies and checkpoint hashes are preserved. The new independent final study remains NOT RUN.

Publication scope: source/configs/tests/docs, two compact inference exports, scientific protocols/metrics/figures and the exact historical diagnostic source snapshots. Datasets, full private checkpoints, credentials, environments, caches, private account/backup operation records and redundant UI-debug screenshots stay local. Historical full-bundle SHA inventories remain unchanged; PUBLICATION_SHA256SUMS and publication_manifest.json identify the selected public subset. An omitted operational/debug artifact is not deleted and does not imply a missing scientific result.
