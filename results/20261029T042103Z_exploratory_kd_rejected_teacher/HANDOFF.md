# Exploratory rejected-teacher KD — launched

Observed 2026-10-29T04:30:50.854461+00:00 UTC. **Running epoch1/10, 750 successful CUDA updates** of220490; latest sampled loss13.0664558, supervised13.0200577, unweighted masked log KD0.231989324. Fixed total = supervised +0.2 KD. Finite-loss and successful-update checks passed through reported progress. AMP overflow retries so far3, current scale128.0; no skipped updates. Exact augmented teacher/student RGB parity verified on the first successful batch. This is an observation, not full-run completion.

Bundle: `/run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher`
Run outputs: `/run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher/run`
Frozen config: `/run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher/frozen/configs/exploratory_kd.yaml`
Authorization/lineage: `/run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher/authorization.json`
Active provenance: `/run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher/run/provenance.json`
Freeze SHA256: `a46adf2a3b576fada2503c4ccea231c920f9eee80f3efa882d126deca898ff2c`
Authorization SHA256: `ee35d009610b8478eb7634f882b5946e467cde5449583346f29d01259bc49fbf`
Log: `/run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher/training.log`
Status: `/run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher/status.json`
Service: `aerodepth-exploratory-kd-20261029t042103z.service`
Trainer PID: `313979` (separate inhibitor MainPID in startup_verified.json).

Original A10/B20 completion receipts and their best/last identities verified. Teacher is selected B epoch3 best, SHA256`6f9b1d27703ef0feba2f24441948a458650b239746632f9fc32d59ff0630b660`; original failed gate SHA256`10e749c38271d360c2c0dc50cde6ec5416f410532b8f6817166f6ac6d3e36b1f`. That gate still rejects B, and the unchanged ordinary full KD constructor was checked against that actual failed gate. It remains unchanged. This separately authorized exploratory arm is **not** the original accepted-teacher KD study, and its selected checkpoint is not automatically a deployment model. No teacher repair, near/selective objective, hyperparameter search, test evaluation or further arm is scheduled.

Student independently initialized from original epoch4 SHA256`b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6`, including all BN buffers, with fresh matched optimizer, scheduler, scaler, RNG, history and counters. Full initial state digest`e8dcc30b6b4149f0c07b68a26b97db0baafa426a90901eb22ab9e6885ebb643f`. Teacher eval/FP32/no-grad and outside optimizer; original teacher state digest matched. Training uses44,098 frames and unchanged4,410-frame FP32 validation each epoch. Batch2/seed42, AdamW1e-4/WD1e-4/clip1, exactly10-epoch cosine eta_min1e-6, original supervised objective plus0.2 masked absolute log-range KD, existing bounded same-batch AMP recovery. Best selected only by within-arm pooled FP32 validation AbsRel; per-trajectory metrics retained. All outcomes remain exploratory, including negative results.

13 focused checks passed, including strict authorization/resume identity, real original gate rejection, initialization/frozen teacher, zero-update serialization and unchanged epoch math. All97,016 explicit TRAIN/VAL image paths available with no dataset symlinks; guard rejected test/discovery probes before I/O. No reserved-test images opened. All136 protected existing files, including parent checkpoints and previous audit/probe evidence, retained exact hashes. Existing original source/configuration/results were not modified.

The requested HDD is still unmounted after recorded hardware failure. The user explicitly selected the AERODEPTH rescue USB, UUID`f31f2337-d69a-4c38-9352-b44d5562d9ac`. Approximately12.54GiB free after freezing. This does not claim the failed HDD was repaired. AC was connected; no competing GPU training was present.

Stop this run:

```bash
bash /run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher/stop.sh
```

Manual resume of this same experiment, only after its own complete `run/last.pt` exists:

```bash
bash /run/media/Ima/AERODEPTH/AeroDepth-rescue/exploratory-experiments/20261029T042103Z_exploratory_kd_rejected_teacher/resume.sh
```

Keep `run/best.pt`, `run/last.pt` and `run/previous_best.pt` when present together with the frozen bundle/input copies. Atomic checkpoints are written after each complete training epoch and FP32 validation. An unfinished epoch replays from the prior boundary. **No checkpoint yet at this observation**: interruption before the first epoch/validation completes cannot resume and does not authorize automatically starting a replacement experiment. Do not rerun start.sh to overwrite/restart. Strict resume includes exact exploratory arm/authorization/source/input/configuration identities and own pinned best. Whole experiments never retry automatically (`Restart=no`); errors stop.

The observed user-level inhibitor owns `sleep:handle-lid-switch` while active. Terminal disconnection is supported. `Linger=no`: full desktop logout may stop the user manager. KDE also handles lid events; physical lid behavior has not been verified, so keep the lid open. Forced sleep, shutdown, reboot, power failure and USB removal are not prevented; keep AC and USB connected. No privileged/permanent changes were made.
