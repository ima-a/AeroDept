# ResNet-50 teacher epoch-3 inference export

This is the selected ResNet-50 V1.5 teacher used by the exploratory KD experiment. It has **23,999,633 parameters**. Epoch 3 was selected by minimum pooled FP32 validation AbsRel after a 20-epoch training budget. It failed the teacher acceptance gate and was not evaluated on the independent test.

The source checkpoint SHA-256 is `6f9b1d27703ef0feba2f24441948a458650b239746632f9fc32d59ff0630b660`. All exported tensors and the complete state digest match both that checkpoint and the saved validation audit. CPU FP32 outputs were bitwise equal on black, white, seeded random, odd-sized random and a synthetic PNG passed through the public inference pipeline. No research image was opened and no scientific evaluation or training was repeated.

The tensor-only export includes parameters and all BatchNorm/RGB-normalization buffers. Optimizer/scaler/scheduler/RNG state, private paths and full training states are excluded. The strict inference loader uses `torch.load(weights_only=True)` and constructs the model without downloading ImageNet initialization.

From the repository root:

```bash
aerodepth infer \
  --weights models/resnet50_teacher_epoch3/resnet50_teacher_epoch3.state_dict.pt \
  --metadata models/resnet50_teacher_epoch3/resnet50_teacher_epoch3.json \
  --image /path/to/image.jpg --output artifacts/prediction_teacher.npy --device cpu
```

Input uses full-frame bilinear RGB resize to 192 × 320 and one in-model ImageNet normalization. Output is an FP32 Euclidean camera-range map in metres, with a 0.1 m floor and no upper clamp. This is an inference artifact, not a resumable training checkpoint.

Verify with `sha256sum -c SHA256SUMS` from this directory. See [NOTICE](../../NOTICE.md) for MidAir and TorchVision/ImageNet attribution and terms.

The dashboard shows the saved teacher metrics only under **Validation** in Results and Model comparison. Its gate explanation includes the Kite near-field regressions. Teacher validation used a different training budget from A; these results are not a matched-budget architecture ranking. The depth viewer remains student-only; this export permits teacher inference on your own images through the CLI.
