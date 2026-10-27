# Expanded teacher/student workflow — 26 October 2026

Current status, 27 October: full training was authorized and relaunched on 26 October with a new frozen bundle and the verified [same-batch numerical recovery](expanded_numerical_recovery.md). Supervised MobileNet completed 10 epochs; ResNet-50 teacher training is ongoing at the [dated snapshot](../results/expanded_status_20261027T150226Z/REPORT.md). The gate and KD are pending at that observation. Use the [current reproduction guide](expanded_reproduction.md) for portable paths, approved configurations and commands.

The first frozen full-run attempt failed during epoch 1 before saving a checkpoint. Its history remains unchanged. The original prelaunch wording of this document and numerical amendment is preserved at commit `21a73a2`; the recovery relaunch protocol and hashes are in [the launch receipt](../results/expanded_relaunch_20261026T114024/protocol.json). The approved budgets, rates, schedules, batch sizes, losses and gate were retained. Run order is supervised student A → teacher B → recorded validation quality gate → KD student C. A smoke teacher cannot authorize full KD. The original training, validation, scheduler and completed final-test study remain separate.

## Inputs and protection

Use `results/expanded_preparation_20261026/manifest_train_val.json.gz` explicitly, rooted at `/run/media/Ima/Low storage/AeroDepth/MidAir`. Its uncompressed SHA-256 is `67c6cad39767d847a2087a81d1ecc4de2ad8c9ecc0fa01e491f3950b25f8dd3a`. The manifest pins `reservations.json`; the runner also checks the exact independent groups before any image access.

| Assignment | Independent groups | Realizations | Frames |
|---|---|---:|---:|
| Train | Kite 0000,0001,0002,0005,0006,0021; PLE 0001,0002,0004,0005,0006,0007 | 20 | 44,098 |
| Unchanged validation | Kite 0003; PLE 0000 | 2 | 4,410 |
| New test, unacquired | Kite 0007,0008; PLE 0008,0009 | 4 reserved | 8,820 expected |
| Original completed test, quarantined | Kite 0004; PLE 0003 | 2 historical | 4,436 historical |

All climate equivalents of reserved groups are excluded from training. Pilot-exposed Kite 0003 and 0021 cannot become new test. Manifest test records remain empty. New training paths include the staging prefix documented in [preparation](../results/expanded_preparation_20261026/REPORT.md); original record paths remain unchanged.

The new runner never discovers images. It rejects a changed manifest/reservation hash, forbidden group, absolute/escaping/mismatched image path or dataset symlink before `Image.open`. A process audit hook permits only manifest-allowlisted image opens, rejects dataset writes and directory discovery, and excludes HDF5/archive opens. Dataset-level checks run in loader workers too. Bounded modes further restrict reads to a fixed 64-frame TRAIN subset; they permit no validation images. These protections cover this runner, not unrelated applications or an adversarial process.

## Models and objective

Teacher `resnet50_depth` uses torchvision ResNet-50 V1.5, `ResNet50_Weights.IMAGENET1K_V2`. Remove classification pooling/FC. Project /2,/4,/8,/16,/32 features to 16/32/64/128/128 channels, then use bilinear skip upsampling and depthwise-separable blocks with widths 128/64/32/16. Resize by actual skip dimensions and finally input dimensions. A 16→1 head followed by softplus + 0.1 produces positive camera range in metres. Encoder: 23,508,032 parameters; decoder including projections/head: 491,601; total: 23,999,633. No accuracy improvement is assumed.

Student architecture and state keys remain the published MobileNetV3-Small depth model: 927,008 encoder + 100,481 decoder = 1,027,489 parameters. Each student starts independently from `models/mobilenet_v3_small_epoch4.state_dict.pt`, SHA-256 `b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6`. Strict initialization loads all parameters and buffers, including BatchNorm. It imports no original optimizer or history. Training/data RNG resets after all model/teacher construction. Resume restores the complete new-run state instead of reinitializing.

The existing contract is unchanged: RGB [0,1], full-frame 192×320 bilinear resize (`align_corners=False`), ImageNet mean/std normalization exactly once inside either model, paired horizontal flip only during training. Decode MidAir uint16 PNG as float16 bit patterns in metres, finite inclusive [0.1,1000] GT mask before nearest target/mask resize; near means GT <10 m. Output has a 0.1 m floor and no upper clamp, fitted scale or classification crop.

Let V be the valid TRAIN GT pixels and S/T be student/teacher range on the **same augmented RGB tensor**:

`L = L1 + 0.2 L_absolute_log + 0.1 L_gradient + lambda_KD mean_V |log(S) - log(stop_gradient(T))|`.

Near-loss weight is zero. A uses lambda_KD=0; C uses a fixed proposed 0.2. No feature KD, temperature, output cache or coefficient search. Supervised and KD losses are reduced in FP32. Teacher inference is online FP32, eval/no_grad, with parameters outside the optimizer and no BatchNorm updates. Nonfinite/nonpositive predictions anywhere are fatal; they are never removed from the mask to improve loss. Empty masks produce a differentiable zero. Parameter/BatchNorm/gradient freezing is checked against a teacher state digest at checkpoints.

## Approved fixed protocol

| Setting | A supervised student | B teacher | C KD student |
|---|---|---|---|
| Config | `configs/expanded_supervised.yaml` | `configs/expanded_teacher.yaml` | `configs/expanded_kd.yaml` |
| Initialization | Published epoch 4 | ImageNet V2 encoder; new decoder | Identical published epoch 4 |
| AdamW learning rate | 1e-4 | Encoder 1e-5; decoder 1e-4 | 1e-4 |
| Weight decay / clip norm | 1e-4 / 1 | 1e-4 / 1 | 1e-4 / 1 |
| Physical/effective batch | 2 / 2 | 2 / 2 | 2 / 2 |
| Accumulation | None | None | None |
| Epochs | 10 | 20 | 10 |
| Updates/epoch | 22,049 | 22,049 | 22,049 |
| Total updates | 220,490 | 440,980 | 220,490 |
| Cosine horizon / minimum LR | 10 / 1e-6 | 20 / 1e-6 | 10 / 1e-6 |
| Training precision | AMP | AMP | AMP student; FP32 teacher |
| Validation | FP32 every epoch | FP32 every epoch | FP32 student every epoch |
| Checkpoint selection | Minimum pooled validation AbsRel | Same | Same |

Use seed 42, two loader workers, the same shuffled training order and paired flips for A/C, no drop-last, trainable BatchNorm in all trained models, no early stopping and no scheduler tuning. Scaler starts at 1024. The fixed epoch/update budget avoids unequal early stopping between students. Numerical failure stops a run; it does not silently alter the budget. A/C configurations are checked for equality in data/model/train/loss/seed/device at the full KD gate. If physical batch policy changes after review, change both students, recheck parity/feasibility and record new identities before full runs. Accumulation does not equalize BatchNorm.

Best and last checkpoints are atomic. Last checkpoints retain optimizer, scheduler, scaler, counters, history, Python/NumPy/CPU/CUDA and loader-generator RNG, complete config, input/initial-weight/source hashes, model/preprocessing contract, scope and arm. KD additionally pins teacher checkpoint/contract/FP32 precision and gate hash. Resume requires exact identity and the exact prior best hash beside its own `last.pt`; it cannot resume historical or bounded smoke/benchmark checkpoints as full training. Before promoting a new best, an atomic byte-identical `previous_best.pt` copy preserves the best pinned by the previous last checkpoint. Resume recovers only that exact hash if interruption occurred between best/last writes; an unrelated backup is rejected. Retain this recovery file. Epoch-boundary saves mean an interrupted incomplete epoch is replayed from the previous boundary. Resume requires unchanged source/config and compatible runtime; it does not promise bitwise identity across library/hardware changes.

### Numerical teacher acceptance

Use only each full run's FP32 validation-selected best checkpoint. Both must cover exactly `Kite_training/foggy/trajectory_2003` and `PLE_training/spring/trajectory_5000`. Correct valid/near pixel counts must match the frozen validation audit during each full epoch. Before C:

1. Teacher pooled validation AbsRel must be strictly below A's pooled AbsRel.
2. On **each** validation trajectory, teacher AbsRel, RMSE, near AbsRel and near RMSE must all be finite and ≤ A's corresponding value.
3. Valid/near GT pixel counts must be identical for teacher and A. Model predictions must have passed global finite/positive checks.

There is no numerical tolerance or test-informed adjustment to this proposed conservative rule. `gate` reads saved checkpoint metrics, hashes both checkpoints and records pass/fail. The gate and KD loader also require the same recorded implementation source hashes across baseline, teacher and current runner. Full KD recomputes the gate and rejects failed/modified gates or substituted teachers. If the gate fails, stop before KD and report the unsuitable teacher. Passing this familiar two-path validation gate does not establish final generalization. No gate was measured with the smoke teacher.

## Verification and laptop measurements

See [the implementation report](../results/expanded_workflow_20261026/REPORT.md) for exact measurements, compact receipts and preservation checks. Each real smoke uses two batches, saves, continues two batches, reconstructs/resumes and repeats those two continuation batches; all three final model state dicts matched with maximum absolute difference zero on this machine. Student initial state digests and both augmented RGB/order traces matched. Teacher and trained encoder/head update checks passed; KD teacher parameters/buffers remained frozen. These short runs do not measure model quality or long-run stability.

Benchmark warm-up is two batches, measurement eight batches (16 examples) per arm, CUDA synchronization, batch 2, AMP, external HDD and two workers, on the same 64 TRAIN frames. OS page cache is uncontrolled. Memory is PyTorch allocator peak during measured training, not total device/desktop usage. Four additional FP32 TRAIN batches provide an inference/validation-overhead proxy; no validation accuracy was measured. A real checkpoint write and an atomic recovery-copy/hash sequence are timed internally; full-run extrapolation conservatively includes two writes plus a recovery copy/hash sequence per epoch. Short timing does not establish full-dataset I/O, thermal or multi-day feasibility. Use the common larger student validation-overhead proxy when comparing A/C; raw per-arm startup/cache differences are retained in receipts.

## Commands for a later reviewed full run — NOT executed

Run from repository root with the mounted prepared dataset. These commands intentionally require `--confirm-full-training`; remove nothing from the guards. Keep code/config unchanged through the sequence and preserve fresh output names. `systemd-inhibit` blocks automatic sleep while each command runs; power, mount stability and storage remain operator prerequisites. The public ImageNet V2 cache is already under the ignored local cache, and full teacher checkpoints resume offline without it.

```bash
export TORCH_HOME="$PWD/artifacts/local/torch-cache"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
MIDAIR_EXPANDED_ROOT='/run/media/Ima/Low storage/AeroDepth/MidAir'
EXPANDED_PY="$PWD/.venv/bin/python"
EXPANDED_MANIFEST='results/expanded_preparation_20261026/manifest_train_val.json.gz'
EXPANDED_RESERVATIONS='results/expanded_preparation_20261026/reservations.json'

# A: independently fresh epoch-4 initialization, fresh optimizer/history.
systemd-inhibit --what=sleep --mode=block --why='Expanded supervised run' -- \
  "$EXPANDED_PY" -B scripts/expanded_training.py train \
  --config configs/expanded_supervised.yaml --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --manifest "$EXPANDED_MANIFEST" --reservations "$EXPANDED_RESERVATIONS" \
  --initialize models/mobilenet_v3_small_epoch4.state_dict.pt \
  --output runs/expanded_supervised_v1 --confirm-full-training

# B: only after A completes, on the same prepared train/validation inputs.
systemd-inhibit --what=sleep --mode=block --why='Expanded teacher run' -- \
  "$EXPANDED_PY" -B scripts/expanded_training.py train \
  --config configs/expanded_teacher.yaml --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --manifest "$EXPANDED_MANIFEST" --reservations "$EXPANDED_RESERVATIONS" \
  --output runs/expanded_teacher_v1 --confirm-full-training

# Gate: saved full-run validation metadata only; no model/image calls.
"$EXPANDED_PY" -B scripts/expanded_training.py gate \
  --baseline runs/expanded_supervised_v1/best.pt \
  --teacher runs/expanded_teacher_v1/best.pt \
  --output runs/expanded_teacher_v1/validation_gate.json

# Compute the exact selected full teacher hash; never use the smoke teacher.
read -r EXPANDED_TEACHER_SHA _ < <(sha256sum runs/expanded_teacher_v1/best.pt)

# C: only after reviewing a passing gate. Runner independently enforces it.
systemd-inhibit --what=sleep --mode=block --why='Expanded KD run' -- \
  "$EXPANDED_PY" -B scripts/expanded_training.py train \
  --config configs/expanded_kd.yaml --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --manifest "$EXPANDED_MANIFEST" --reservations "$EXPANDED_RESERVATIONS" \
  --initialize models/mobilenet_v3_small_epoch4.state_dict.pt \
  --teacher runs/expanded_teacher_v1/best.pt --teacher-sha256 "$EXPANDED_TEACHER_SHA" \
  --gate runs/expanded_teacher_v1/validation_gate.json \
  --output runs/expanded_kd_v1 --confirm-full-training
```

Resume each arm from its own last boundary, with no `--initialize`:

```bash
systemd-inhibit --what=sleep --mode=block --why='Resume expanded supervised run' -- \
  "$EXPANDED_PY" -B scripts/expanded_training.py train \
  --config configs/expanded_supervised.yaml --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --resume runs/expanded_supervised_v1/last.pt \
  --output runs/expanded_supervised_v1 --confirm-full-training

systemd-inhibit --what=sleep --mode=block --why='Resume expanded teacher run' -- \
  "$EXPANDED_PY" -B scripts/expanded_training.py train \
  --config configs/expanded_teacher.yaml --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --resume runs/expanded_teacher_v1/last.pt \
  --output runs/expanded_teacher_v1 --confirm-full-training

read -r EXPANDED_TEACHER_SHA _ < <(sha256sum runs/expanded_teacher_v1/best.pt)
systemd-inhibit --what=sleep --mode=block --why='Resume expanded KD run' -- \
  "$EXPANDED_PY" -B scripts/expanded_training.py train \
  --config configs/expanded_kd.yaml --dataset-root "$MIDAIR_EXPANDED_ROOT" \
  --resume runs/expanded_kd_v1/last.pt \
  --teacher runs/expanded_teacher_v1/best.pt --teacher-sha256 "$EXPANDED_TEACHER_SHA" \
  --gate runs/expanded_teacher_v1/validation_gate.json \
  --output runs/expanded_kd_v1 --confirm-full-training
```

Resume commands use the same default pinned manifest/reservations shown explicitly in launch commands. Preserve `best.pt`, `previous_best.pt` when present, `last.pt`, gate and baseline best together for full KD resume. Do not use `--resume` to change an arm, teacher, starting weights, horizon, data or source. Outputs/config names above are proposals; if already present, use a fresh version consistently rather than overwrite.

Bounded verification can be repeated with `smoke` or `benchmark` instead of `train`, fresh output directories under `artifacts/`, and without `--confirm-full-training`. For KD software checks only, pin the newly generated smoke teacher's `last.pt` hash and omit `--gate`; bounded scope cannot be promoted to a full teacher gate. The exact executed commands are retained in the local command receipt referenced by the report.

## Preservation and outstanding decisions

Original runners, configs, published weights, original best/last checkpoints, manifests and all historical study evidence are unchanged. New code extends only model registration/contracts and adds a separate loss/runner. Original MobileNet state keys and preprocessing remain compatible. The earlier plan's student initialization is corrected with its old text retained in local provenance.

The historical final-test verifier intentionally hashes historical source files. It will reject the evolved current `model.py`/`losses.py`; its historical hashes were not rewritten. Public commit `8f32f71` contains all matching historical source and byte-identical final-study evidence. Use the separate-checkout saved-result instructions in [final study](final_study.md). A private snapshot also remains under `artifacts/expanded_workflow_20261026T095357/historical_source`. New full checkpoints use their own source identities.

The launch freeze and operational checks are complete for the current laptop run; see the recovery relaunch receipt and dated snapshot. The commands above retain their original prelaunch example paths and proposal configurations; independent reproduction should use the approved configurations and portable commands in [the current guide](expanded_reproduction.md). Teacher acceptance is decided only after B finishes. New reserved test inputs remain unacquired at the snapshot and are unnecessary for this TRAIN/VAL workflow. A new test evaluation requires a separate frozen protocol; it is not authorized by these software checks or the current training run.
