# Exploratory KD epoch-9 inference export

This is the exact student selected on validation and evaluated as `KD_every` in the completed independent study. It has 1,027,489 parameters. The source checkpoint SHA-256 is `f776390515dee417c72e47e6ea41aae3710b2b9251edeaeee7a5fd7fb6fc0d45`.

The export contains only 304 model tensors, including all BatchNorm and RGB-normalization buffers. It excludes optimizer/scaler/scheduler/RNG state, training history, teacher weights and private paths. The teacher is not required for inference. Load with `torch.load(weights_only=True)` through the existing strict, checksum-verified inference loader.

All exported tensors match the evaluated checkpoint exactly. CPU FP32 outputs were bitwise equal for black, white, seeded random, odd-sized random and a synthetic PNG passed through the public image/inference pipeline. Metadata records the tensor-state digest, preprocessing, selection, training recipe, checkpoint/export hashes and parity checks. No research image was opened and no training or scientific evaluation was rerun for this export.

From the repository root:

```bash
aerodepth infer \
  --weights models/exploratory_kd_epoch9/exploratory_KD_epoch9.state_dict.pt \
  --metadata models/exploratory_kd_epoch9/exploratory_KD_epoch9.json \
  --image /path/to/image.jpg --output artifacts/prediction_KD.npy --device cpu
```

Input is RGB resized over the full frame to 192 × 320 with bilinear interpolation; ImageNet normalization occurs once inside the model. Output is a 192 × 320 FP32 map of positive Euclidean camera range in metres, with a 0.1 m floor and no upper clamp. No MidAir download or pretrained-weight download is needed for inference on your own image.

Verify this directory with `sha256sum -c SHA256SUMS`. See the project [NOTICE](../../NOTICE.md) for attribution and upstream terms.

KD remains **exploratory with a rejected teacher**. Its validation result underperforms A, and the independent pooled AbsRel improvement is accompanied by worse near RMSE. Publishing this export does not change the failed teacher gate, study outcomes, frozen protocols or the default demonstration model A. This export supports inference, not exact training resume.
