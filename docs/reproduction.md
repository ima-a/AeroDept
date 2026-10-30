# Frozen validation and scheduler reproduction

Install as in README and supply the dataset root explicitly. `scripts/reproduce.py` is a versioned public wrapper around `aerodepth_lite.research`; it does not import the ignored original audit scripts or require a full training checkpoint. The inference export includes all required parameters/buffers and loads strictly without ImageNet downloads.

```bash
export MIDAIR_ROOT='/path/to/MidAir'
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/reproduce.py --dataset-root "$MIDAIR_ROOT" --study validation --device cpu --output artifacts/validation_repeat
python scripts/reproduce.py --dataset-root "$MIDAIR_ROOT" --study scheduler-v1 --device cuda --output artifacts/v1_repeat
python scripts/reproduce.py --dataset-root "$MIDAIR_ROOT" --study scheduler-v2 --device cuda --output artifacts/v2_repeat
```

A new output directory is required. Validation loads only the two validation trajectories. The full manifest is decompressed locally into the output directory and its original SHA-256 is checked. Each run writes complete per-frame policy logs, quality/event/call summaries, FP32 reference agreement, environment and input-preservation checks. Scheduler studies also write all searched configurations/results, the selected configuration, exact matched fixed positions, and actual selected/matched timing. Allow approximately 3 GB per run for derived FP32 prediction/GT/mask caches. Those generated caches are local and ignored. Old disposable validation arrays were removed during the 30 October cleanup; saved results remain under `results/`. Reproduction regenerates arrays into a fresh output directory. Retained private execution records are mapped in [archive locations](archive_locations.md). The original preparation/audit/full training checkpoints remain local; they are not prerequisites for inference or these public reproduction commands.

## Metric contract

FP32 eval mode, batch1 chronological replay, no augmentation; RGB float[0,1], full-frame bilinear resize to192×320, internal ImageNet mean/std once. Original MidAir float16-bit metre decoding, finite GT in[0.1,1000]m before nearest depth/mask resize. No crop, per-image scaling or upper prediction clamp. Depth is Euclidean camera range. Reused predictions are scored against EVERY current frame's GT without warping. AbsRel and squared errors pool valid pixels; delta thresholds are strict `<1.25**k`; near field is GT<10m. Pixel counts must match exactly.

The batch1 reference must match the saved audited batch4 FP32 results with absolute tolerance1e-5 for AbsRel/near AbsRel/deltas and1e-3m for RMSE/near RMSE. Material disagreement stops the study before policy search. CPU/GPU builds can differ; parameter equality does not promise bitwise inference equality across hardware.

GT and GT validity are offline scoring inputs only. The critical event is valid-GT full-frame p10 entering<8m or falling>=2m while<8m. State resets per trajectory. Missed-update rate is no new inference on the event frame, not failed hazard recognition. Delays end at the next inference in the same trajectory; unresolved events are right-censored. No events means undefined rates, not successful safety performance. Predicted p10 on the current GT-valid mask is a separate offline depth check, never a scheduler input.

## Bounded searches and timing

V1 reproduces all four initial policies, then loads exactly 12 saved configurations from `results/scheduler_v1/search_plan.json`. V2 loads exactly 8 configurations from `results/scheduler_v2/plan.json`. No new configurations are generated. Failed quality/freshness configurations remain in the results.

Both use per-trajectory eligibility: overall AbsRel/RMSE at most 5% worse than every-frame, near AbsRel/RMSE at most 10% worse; no censored critical updates and maximum observed delay<=2 frames. Among eligible, minimize worst trajectory call fraction, then worst normalized error across both trajectories/four errors, mean observed event delay, and config ID. Zero-event trajectories add no safety evidence. If no candidate passes, the least-violating diagnostic candidate is explicitly labelled ineligible.

Fixed matching uses each trajectory's selected validation call countK and lengthN, inferring at `ceil(k*N/K)` for k=0..K-1, including the first frame. Cadence is deterministic, approximately uniform and never GT/event-aligned. MAX_SKIP denotes skipped frames; MAX_SKIP=1 allows one skipped frame, giving a two-frame refresh interval.

Cached search uses counterfactual per-frame FP32 predictions. It is not a runtime benchmark. Final policy timings execute actual model calls, synchronize CUDA before/after each call, include H2D/D2H and CPU scheduler work, and exclude ten comparable warmups per policy. Total offline wall time also includes data loading/scoring/instrumentation; it is not live inference throughput. Frame age omits real-time queuing and sensor latency. Published timing is one hardware pass, not randomized repeated benchmarking, and supports no energy/battery claim.

V2's motion change uses causally held simulated GPS velocity and separately normalized accelerometer-change/gyro proxies, removing privileged GT velocity and the original maximum across mixed physical units. Separately, its intentional RGB design compares current RGB with the last inferred frame instead of the immediate predecessor. V2 evaluates these changes jointly; no isolated causal attribution is claimed. The original scheduler/replay implementation and negative results are preserved. The enhanced public runner handles undefined/censored event reporting; the legacy replay command remains for historical/synthetic workflows.

## Training, random initialization and resume

README gives the exact new-run and full-checkpoint resume commands. The recorded run used ImageNet initialization,50 epochs, 192×320,batch4,AdamW3e-4 and a fixed50-epoch cosine horizon; it took 16h 28m on GTX1650. A literal random-initialization run can be configured without altering the frozen original:

```bash
python - <<'PY'
from pathlib import Path
import yaml
c=yaml.safe_load(Path('models/training_config.yaml').read_text())
c['model']['pretrained']=False
Path('artifacts').mkdir(exist_ok=True)
Path('artifacts/random_init.yaml').write_text(yaml.safe_dump(c))
PY
aerodepth train --config artifacts/random_init.yaml --dataset-root "$MIDAIR_ROOT" --manifest artifacts/frozen_manifest.json --output runs/random_init
```

This command is documented, not executed as a new real-data experiment. It changes initialization and cannot reproduce the recorded model. Full resume needs trusted private `last.pt` plus the prior `best.pt` beside it, including optimizer/scaler/scheduler/RNG/history. The public inference state_dict cannot resume training. Synthetic tests exercise optimization and resume only on generated fixtures. Final-test evaluation remains intentionally absent.
