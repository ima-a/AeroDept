# Expanded experiment launch handoff — 26 October 2026

**Launched, then automatically stopped on a numerical failure. No full arm completed.** Do not retry this attempt or alter the frozen settings without a separate diagnosis/decision.

## Failure and observed progress

Stage A, supervised expanded MobileNet, entered epoch 1/10 and exited at 10:54:53 IST with `FloatingPointError: Nonfinite unscaled gradients` in the verified `Experiment.step` after backward/unscale. The child was launched at 10:54:23 IST. Global finite prediction/loss checks precede that gradient check, but no numerical loss or successful-update count was persisted. Failure occurred before the first 250-batch progress record. Zero completed epochs; no best.pt or last.pt exists. No validation completed, teacher started, teacher gate executed or KD started.

The launcher checked the nonzero exit code and stopped. No retries, hyperparameter/scaler/scheduler changes, cleanup or later stages were attempted. The underlying numerical cause is unresolved; prior fixed-subset smoke checks do not establish full-inventory stability.

## Frozen protocol and locations

Bundle: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420`

Freeze identity (SHA-256 of freeze.json): `2b53a27cd4491843e92a7218a883158cbff6a6842ede83f966620497e7131de9`. All indexed frozen files still match after the failure. `protocol.json` freezes the numerical gate, budgets, initialization, precision and stop policy; `freeze.json` hashes it, active configs, source, input metadata, initialization files and control scripts. Original proposed protocol/configs remain unchanged in the repository and byte copies are under `frozen/proposal_history`. Active copies change only protocol_status metadata to frozen_authorized_20261026.

| Stage | Output | Frozen budget | State |
|---|---|---|---|
| A supervised | `runs/A_supervised` | 10 epochs, 220,490 updates | Failed during first epoch; provenance only |
| B teacher | `runs/B_teacher` | 20 epochs, 440,980 updates | Not created |
| C quality gate | `teacher_gate.json` | Validation-selected A/B best checkpoints | Not created |
| D KD | `runs/D_kd` | 10 epochs, 220,490 updates | Not created |

Batch 2, no accumulation. Student LR 1e-4; teacher encoder/decoder LR 1e-5/1e-4. Fixed cosine horizons 10/20; minimum LR 1e-6. KD weight 0.2, online frozen FP32 teacher, FP32 validation every epoch selecting minimum pooled AbsRel. Both student arms independently initialize original epoch-4 weights and BN buffers, with fresh matched training state; KD never initializes A's trained checkpoint. No early stopping or search.

Gate: teacher pooled validation AbsRel strictly lower than A; on each original validation trajectory, finite overall/near AbsRel and RMSE <= A, with exactly equal valid/near counts. Failure stops before KD. The supervisor verifies hashes/budgets at each completed stage before advancing, and performs no automatic retries.

Preflight confirmed manifest uncompressed SHA `67c6cad39767d847a2087a81d1ecc4de2ad8c9ecc0fa01e491f3950b25f8dd3a`, initialization SHA `b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6`, all 97,016 explicit TRAIN/VAL image files present/regular/nonempty (metadata only), CUDA GTX 1650 and approximately 684 GiB free externally before launch. No archive audits or entire test suite were repeated. New reserved test remains unacquired; original/new test image/depth inputs were excluded.

## Session, logs and controls

Persistent transient user unit: `aerodepth-expanded-20261026t104420.service`

Recorded inhibitor PID: 165114; supervisor PID: 165120; training PID: 165124. These are **historical PIDs, not active processes**. Post-failure unit state is failed, MainPID=0, empty control group. No Python CUDA workload remains. Restart policy is no.

- Full sequence log: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420/workflow.log`
- Training/error log: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420/logs/A_supervised.log`
- Current state: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420/status.json`; history: `events.jsonl`
- Active config: `/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420/frozen/configs/expanded_supervised.yaml`
- Exact launch argv and PID transitions are in status/events and preflight receipts.

Status command (read-only):

```bash
systemctl --user status aerodepth-expanded-20261026t104420.service --no-pager
```

Stop command for this bundle (already stopped):

```bash
bash '/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420/stop.sh'
```

The saved generic manual continuation command is below, **but it cannot resume this failed attempt**: no epoch-boundary last.pt exists, and it will refuse the existing incomplete output rather than overwrite or retry. After the numerical failure has been diagnosed, any newly authorized attempt needs its own frozen record and fresh output destination. Do not run this as a retry now.

```bash
bash '/run/media/Ima/Low storage/AeroDepth/expanded-experiments/20261026T104420/resume.sh'
```

For a later valid epoch-boundary interruption, that script starts a new named user service, verifies completed stage identities, resumes the current stage from its own last.pt, then continues through the same gate. best.pt/previous_best.pt/last.pt, configs, source, baseline and teacher/gate hashes must remain available.

## Unattended limits and preservation

ACAD reported online=1 and battery 100%. A temporary user service successfully acquired sleep:handle-lid-switch blockers; the training service requested the same blockers. The failed process has released them. KDE PowerDevil also owns lid handling: physical desktop/lid behavior was not tested, so no guarantee of safe lid closure is claimed. Keep the lid open for any future authorized run. Linger=no: user service survives terminal disconnection, but full logout may stop the user manager. Forced sleep/shutdown, reboot, power loss and HDD disconnect are not prevented. No permanent or privileged system settings were changed.

Original epoch-4 weights and original main best/last hashes remain unchanged. Original proposal/history/configurations, manifests, results and exposure ledgers were preserved. New launcher source is scripts/expanded_sequence.py; core verified training code was not modified. All work remains uncommitted.
