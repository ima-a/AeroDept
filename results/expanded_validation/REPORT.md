# Completed training and expanded validation audit

All 50 epochs completed without an exception; training and validation history is complete and finite. Training took 16h 28m on GTX1650. The original best.pt and last.pt loaded strictly; original checkpoints, history and complete audit evidence remain preserved locally. The compact inference-only export is public under models/.

Best epoch4 was selected by the lowest recorded AMP validation AbsRel 0.30312900253193636. The checkpoint was not reselected after FP32 evaluation. Full50-epoch history is in ../training/history.csv.

## Frozen FP32 validation

4,410 frames,2,205 per trajectory;192×320 full-frame resize; RGB bilinear, depth/mask nearest; internal normalization exactly once. Finite GT[0.1,1000]m, near GT<10m, strict delta thresholds<1.25^k, valid-pixel pooling. No upper prediction clamp, median scaling, crop or test evaluation. The original batch4 audit and chronological batch1 scheduler reference agree within declared tolerances.

| Scope | AbsRel | RMSE m | d1 | d2 | d3 | Near AbsRel | Near RMSE m | Valid pixels | Near pixels |

|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| Overall | 0.3015399584766882 | 127.33534411279983 | 0.4980362164860043 | 0.7671516443571086 | 0.8943295445144068 | 0.1961665879674402 | 3.362006079982595 | 212903814 | 32267840 |

| Kite_training/foggy/trajectory_2003 | 0.35749885870730297 | 165.19325760167007 | 0.39907319046653406 | 0.6849717337085571 | 0.857466091012965 | 0.4271206300049569 | 4.379247857046864 | 114412073 | 894287 |

| PLE_training/spring/trajectory_5000 | 0.2365357894653523 | 57.87631008381565 | 0.6129957536236464 | 0.8626152217169153 | 0.9371516541676322 | 0.1895833617261662 | 3.3284560744286638 | 98491741 | 31373553 |

## Observed limitations

Validation degrades after epoch4 while training loss falls. The last epoch recorded validation AbsRel 0.322685 versus best AMP0.303129. Foggy Kite near-field quality is poor and PLE contributes97.23% of near valid pixels, so pooled near error hides scene imbalance. Far300–1000m AbsRel 0.638864/RMSE388.07m is poor. Blurred boundaries, horizontal bands and missed geometry are visible in typical and poor examples.

The audited model produced5,108,332 floor predictions<=0.101m on valid pixels (2.399%) and28,734,482 more on invalid-GT pixels. All predictions were finite, no whole-frame constant collapse was found, and no hard upper-limit pile-up was found. Masking excludes21.423% of resized pixels. Structural camera-path/order checks and selected visuals found no obvious pairing misalignment; this is not proof of subpixel calibration.

## Evidence and reproduction

[Training curves](training_validation_curves.png), [near metrics/frame errors](near_metrics_and_frame_errors.png), [typical and poor predictions](prediction_contact_sheet.png), [full metrics](validation_metrics.json), [diagnostics](diagnostics.json), [audit verification](verification.json), [per-trajectory CSV](per_trajectory.csv). Selected figures derive only from development validation data; attribution/terms are in ../../NOTICE.md.

The original audit remains local and untouched. Public reproduction uses [the versioned runner](../../scripts/reproduce.py) and [portable instructions](../../docs/reproduction.md); it needs no ignored audit script, private full checkpoint or author cache. The original training checkpoint hash is c1d4279cdaf594d1c15bce1f4f1b6ae322c7ec3ef28349df1dd47d975818ebd8. The compressed relative manifest preserves the original uncompressed SHA-256 e7b093be69f1942f281a09563bd63efa479d4352de97da722fb0504208c1fbf5.
