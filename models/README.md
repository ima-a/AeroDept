# Published inference weights

`mobilenet_v3_small_epoch4.state_dict.pt` is the exact epoch-4 MobileNet depth model exported from the completed expanded-data run: **1,027,489 parameters**,4267069bytes, SHA-256 `b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6`.

It is a plain tensor state_dict, including BatchNorm and RGB-normalization buffers. Load with `torch.load(weights_only=True)` and strict loading through `aerodepth_lite.inference.load_inference_model`. The architecture is constructed with pretrained=False, so ImageNet cache/downloads are unnecessary. Optimizer, AMP scaler, RNG and training history are excluded. Original full checkpoints remain private/local.

Original checkpoint SHA-256: `c1d4279cdaf594d1c15bce1f4f1b6ae322c7ec3ef28349df1dd47d975818ebd8`. `mobilenet_v3_small_epoch4.json` contains original/export hashes, full frozen training config, manifest identity, contract, provenance, validation metrics and limitations. CPU FP32 equivalence on seed42 random, black, white and first validation RGB was bitwise identical (max absolute differences 0), against declared atol/rtol 1e-6.

Input: RGB[0,1] bilinearly resized to192×320; normalize inside the model once with mean[.485,.456,.406]/std[.229,.224,.225]. Output: positive Euclidean camera range in metres with floor.1m and no upper clamp. Evaluation alone uses finite GT[.1,1000]m and near GT<10m. No centre crop or prediction scale fitting.

The encoder began from TorchVision `MobileNet_V3_Small_Weights.IMAGENET1K_V1` and was finetuned on MidAir. See [NOTICE](../NOTICE.md) and linked upstream/data terms; this checkpoint does not introduce a blanket license grant. No custom-CNN weights are represented as an expanded-data baseline. The older custom-CNN pilot used a different dataset/protocol.

Verify:

```bash
(cd models && sha256sum -c SHA256SUMS)
```
