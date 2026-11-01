# Published inference weights

All three evaluated student models are included as compact, inference-only exports. Each contains the model parameters and all BatchNorm/RGB-normalization buffers. Full optimizer, scheduler, AMP scaler, RNG and training states remain private.

| Model | Weights | Metadata |
|---|---|---|
| Original epoch 4 | [state dict](mobilenet_v3_small_epoch4.state_dict.pt) | [contract and provenance](mobilenet_v3_small_epoch4.json) |
| Supervised A epoch 9 | [state dict](presentation_A_epoch9_20261029T155521Z/supervised_A_epoch9.state_dict.pt) | [contract and provenance](presentation_A_epoch9_20261029T155521Z/supervised_A_epoch9.json) |
| Exploratory KD epoch 9 | [state dict](exploratory_kd_epoch9/exploratory_KD_epoch9.state_dict.pt) | [contract and parity checks](exploratory_kd_epoch9/exploratory_KD_epoch9.json) |

Each student has **1,027,489 parameters**. Use the strict checksum-verified loader in `aerodepth_lite.inference`; it loads with `torch.load(weights_only=True)`, constructs the architecture with `pretrained=False` and needs no ImageNet download. KD inference does not require the teacher or its full training checkpoint.

From the repository root, verify all three exports and metadata files:

```bash
(cd models && sha256sum -c SHA256SUMS)
```

Input is RGB in [0,1], resized over the full frame to 192 × 320 using bilinear interpolation. ImageNet normalization happens once inside the model. Output is a 192 × 320 FP32 map of Euclidean camera range in metres, with a 0.1 m floor and no upper clamp. There is no centre crop or fitted prediction scale. Evaluation uses finite ground truth in [0.1,1000] m and near ground truth below 10 m.

The Original checkpoint/export identities and parity checks remain in its metadata. A has its [own export guide](presentation_A_epoch9_20261029T155521Z/README.md). The [KD export guide](exploratory_kd_epoch9/README.md) records its lineage to the exact checkpoint used in the independent test and bitwise CPU FP32 parity checks. No research images or new scientific evaluation were needed to export KD.

A remains the default demonstration model. KD remains exploratory with a rejected teacher; publishing its weights does not change the gate or recorded negative findings. Inference exports support running the selected models, not resuming their original training. Datasets and playback caches must be acquired/prepared separately; see [the dashboard guide](../docs/presentation_dashboard.md) and [reproduction guide](../docs/expanded_reproduction.md).

The encoders originated from TorchVision MobileNetV3-Small ImageNet initialization and were finetuned on MidAir. See [NOTICE](../NOTICE.md) for dataset/framework attribution and upstream terms. The custom-CNN pilot used a different dataset/protocol and is not represented as a matched expanded-data baseline.
