# Corrected expanded training — live handoff, 26 October 2026

**Training is running; the experiment is not complete.** At the saved observation, stage A supervised MobileNet was in epoch 1/10, with 750/22,049 batches and 750 successful optimizer updates. Logged loss: 13.105461; AMP scale 128; overflow retries total 3. All completed batches pass finite prediction/loss/gradient checks before their optimizer update. Startup observation exceeded the requested first 100 batches. [Live-startup receipt](startup_observation.json).

## Bundle and frozen identities

Run bundle: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery`

Freeze SHA-256: `d49ab655ff5b1a94e7809e24bad60eac1cdba26bd7c9ce44853e924918e89d90`. The protocol, current recovery amendment, all frozen sources/configs/input metadata/initialization files/control scripts are indexed. Active arm configuration files are byte-identical to those in the original approved freeze; approved rates, schedules, losses, batch sizes, seeds, masks/range, budgets, validation selection and numerical teacher gate were not changed. The two gate function bodies also match the original freeze exactly.

The corrected source hashes match the existing numerical-recovery evidence. No diagnosis, test suite, benchmark, archive audit or scheduler search was repeated. Recovery integration was inspected and existing evidence checked: no optimizer/parameter update on a failed trial; gradients clear before retry; model buffers and RNG restore; scaler/growth/retry state checkpoint/resume; one finite optimizer/global-step update per batch; at most eight retries/nine attempts before hard failure. Teacher is frozen FP32/eval during KD and is not recomputed for retries.

Manifest uncompressed SHA: `67c6cad39767d847a2087a81d1ecc4de2ad8c9ecc0fa01e491f3950b25f8dd3a`. Epoch-4 initialization SHA: `b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6`. Both students start independently from those original weights and BatchNorm buffers, with fresh matched optimizer/scheduler/scaler/history/counters. The old failed run is not resumed.

The first launch's full TRAIN/VAL file-availability evidence was retained; a fresh limited first/middle/last metadata sample covered 132 image files across all 22 TRAIN/VAL realizations. No discovery/decoding audit was repeated. Preflight confirmed the same HDD UUID, AC connected, CUDA GTX 1650 and 683.6 GiB free externally.

## Automatic sequence and outputs

| Stage | Output below bundle | Budget | Behavior |
|---|---|---|---|
| A supervised MobileNet | `runs/A_supervised` | 10 epochs; 220,490 updates | Running, fresh epoch-4 initialization |
| B ResNet-50 teacher | `runs/B_teacher` | 20 epochs; 440,980 updates | Starts only after verified A completion |
| C frozen quality gate | `teacher_gate.json` | Saved best A/B FP32 validation | Must pass unchanged acceptance criteria |
| D KD MobileNet | `runs/D_kd` | 10 epochs; 220,490 updates | Starts only if gate passes; independent original epoch-4 initialization |

Numerical gate: teacher pooled validation AbsRel strictly below A; each validation path overall/near AbsRel and RMSE finite and no worse than A; identical valid/near pixel counts. Failed acceptance or any training/input/hash/checkpoint error stops the sequence. Restart=no; there is no automatic experiment retry, setting change or tuning. B/C/D have not started at this observation.

## Persistent session, logs and controls

User service: `aerodepth-expanded-20261026t114024-recovery.service`

Service/inhibitor MainPID: 170160. Supervisor PID: 170165. Current CUDA trainer PID: 170179. Actual current IDs/status are recorded in status.json and systemctl. No competing training workload was observed. The service is independent of this chat/terminal.

- Sequence log: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery/workflow.log`
- Current training log: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery/logs/A_supervised.log`
- Live status: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery/status.json`; transition history: `events.jsonl`
- Active configuration: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery/frozen/configs/expanded_supervised.yaml`
- Future checkpoint directories: the arm outputs listed above; each keeps atomic best.pt, last.pt and previous_best.pt when present.

Read progress:

```bash
RUN='/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery'
cat "$RUN/status.json"
tail -f "$RUN/logs/A_supervised.log"
```

Stop manually (ends the supervisor and its training/workers):

```bash
RUN='/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery'
bash "$RUN/stop.sh"
```

Resume manually **after an epoch-boundary last.pt exists**:

```bash
RUN='/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T114024_recovery'
bash "$RUN/resume.sh"
```

The resume script starts a new timestamped user unit, verifies completed-stage identities, resumes the current stage from its own last.pt, and continues through the unchanged gate. It never uses the failed bundle. Keep frozen source/configs/weights, best/last/previous-best, baseline/teacher checkpoints and gate together; strict source/arm/teacher/policy identities are enforced.

**No epoch checkpoint exists yet.** Stopping before epoch 1 finishes cannot resume this attempt and requires a fresh output destination; after later interruptions, work since the last complete epoch is replayed. Nothing is overwritten to work around this. Do not rerun start.sh or start a competing training process.

## Unattended limitations and preservation

The active user-level `sleep:handle-lid-switch` block inhibitor is registered and recorded in the receipt. KDE PowerDevil also owns lid handling, and physical desktop/lid behavior has not been tested; keep the lid open. AC was connected and battery 100% at preflight. Linger=no: terminal disconnection survives, but full user logout may stop the user manager. Forced sleep/shutdown, reboot, power loss and HDD disconnection are not prevented. No privileged or permanent system settings were changed.

The failed bundle `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420` and all of its indexed files remain unchanged. Original published weights and original main best/last identities were checked unchanged. Existing modified/untracked work was preserved. No test images/depth, cleanup, scheduler changes or Git publication occurred.
