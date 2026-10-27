# Attribution and upstream terms

## MidAir

Training, validation and the relative scientific manifest derive from **Mid-Air: A multi-modal dataset for extremely low altitude drone flights**, Michael Fonder and Marc Van Droogenbroeck, CVPR Workshops 2019, University of Liège / Montefiore Institute. Dataset: https://midair.ulg.ac.be/ . Official terms: https://midair.ulg.ac.be/index.html . The dataset/benchmarks are published under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**: https://creativecommons.org/licenses/by-nc-sa/4.0/ . Respect attribution, noncommercial and share-alike terms for dataset-derived materials. Validation-derived figures retain that attribution; no final-test image is used. No raw dataset/archives are included. Scientific results use a custom subset/split, not an official benchmark score.

## TorchVision / ImageNet initialization

The encoder architecture and ImageNet-1K initialization come from TorchVision `MobileNet_V3_Small_Weights.IMAGENET1K_V1`; source implementation is BSD3-Clause, copyright Soumith Chintala 2016. Source/license: https://github.com/pytorch/vision/blob/main/LICENSE . Model provenance: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html . Initialization URL is recorded in model metadata. Pretrained weights can have dataset-derived terms distinct from the source-code license; retain upstream notices and consult the relevant ImageNet/TorchVision terms for downstream use. This repository does not assert that BSD alone licenses every training dataset or resulting weight use.

The project does not add or replace a blanket source/weight license as part of this packaging task. Existing rights and upstream terms remain in effect. Availability of a research checkpoint is not a new unrestricted commercial-use grant. No endorsement by dataset/framework authors is implied.

The separate expanded experiment uses a TorchVision ResNet-50 V1.5 encoder with `ResNet50_Weights.IMAGENET1K_V2` and a project-specific metric-range decoder. The same TorchVision source and weight/data-term distinctions above apply. Its ImageNet initialization and private training checkpoints are not bundled here. Expanded TRAIN-derived figures retain the MidAir attribution and terms above; no new test image is included.
