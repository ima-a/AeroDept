# Optional historical Kaggle workflow

The published model was trained on the laptop, not Kaggle. See [current public reproduction](reproduction.md). This optional runner remains maintained but full hosted Kaggle GPU execution is unverified.

# MobileNet training handoff

The main candidate is `mobilenet_v3_small_depth`. The custom `aerodepth_tiny`
baseline and its configurations remain available. No accuracy improvement or
Raspberry Pi latency improvement has been measured. Actual Kaggle GPU execution
remains unverified. Synthetic runs establish software integration only.

## Architecture and candidate configuration

Torchvision MobileNetV3-Small uses `IMAGENET1K_V1` ImageNet weights for its full
encoder; its pooling and classification head are discarded. Skip features at
indices 0, 1, 3, and 8 supply 16, 16, 24, and 48 channels. The final 576-channel
feature is projected to 128 channels. Four depthwise-separable decoder blocks
produce 64, 32, 24, and 16 channels, followed by a one-channel convolution.
Bilinear interpolation uses exact skip sizes and then the RGB input size, so odd
dimensions work. Softplus plus the minimum depth produces positive Euclidean
camera range in metres. Predictions are not clamped to the valid-label maximum.

Actual trainable parameters: **1,027,489**, comprising **927,008 encoder** and
**100,481 decoder/head** parameters. FP32 weights occupy about 4.110 MB, excluding
buffers, optimizer state, and checkpoint metadata. Counts were measured with
torchvision 0.29.1+cpu and PyTorch 2.14.1+cpu.

`configs/mobilenet.yaml` is a separate candidate, not a tuned final protocol:

| Choice | Candidate |
|---|---|
| Resolution | 192 x 320 (height x width), full-frame paired resize |
| Initialization | IMAGENET1K_V1 encoder, random decoder; all parameters fine-tuned |
| RGB | Dataset supplies [0,1]; model applies mean [0.485,0.456,0.406], std [0.229,0.224,0.225] |
| Spatial preprocessing | RGB bilinear, depth/mask nearest; paired horizontal flip during training; no crop |
| Valid labels | Finite range from 0.1 to 1000 metres, **provisional** |
| Loss | Existing masked L1 1.0, log 0.2, gradient 0.1; near-field weight 0.0 |
| Optimizer | AdamW, learning rate 3e-4, weight decay 1e-4, gradient clip 1.0 |
| Batch / workers | 16 / 2; revise for measured GPU memory and I/O |
| LR schedule | Cosine, T_max 50 epochs, minimum 3e-6, **provisional** |
| Budget | 50 epochs; review against measured train and validation throughput |
| Selection | Lowest validation AbsRel (`best.pt`); `last.pt` is latest completed epoch |

The data valid range, near-field reporting threshold, optimizer schedule, replay
scheduler, and critical-event thresholds require review. Replay placeholders are
explicitly provisional and are not the pilot's frozen/tuned scheduler. Training
does not execute adaptive replay. Review depth limits using training/validation
only; do not tune using final test data.

Normalization lives inside the model. This preserves raw [0,1] RGB for the replay
visual-change scheduler and plotting while consistently normalizing training and
inference. Label masking and paired transforms remain in the existing dataset.
The checkpoint records this architecture/preprocessing contract, full config,
manifest hash, optimizer, learning-rate scheduler, AMP scaler, history, and RNG
states (including loader generators). Resume rejects changed preprocessing,
manifest, loss, seed, optimizer options, or explicitly configured LR schedule.
Worker counts/library versions should stay fixed for reproducibility; CUDA can
still introduce nondeterminism. Old baseline checkpoints remain loadable.

## Dependencies and installation

Python >=3.10, torch >=2.2, and a **matching** torchvision >=0.17 installation
are required. Other project dependencies are numpy >=1.26, Pillow >=10,
PyYAML >=6, matplotlib >=3.8; real MidAir needs h5py >=3.10. Editable installation
also needs setuptools >=77 and wheel; the bootstrap validates the environment
and installs only missing utility packages with `--no-deps`. The notebook does
not replace torch or torchvision. If their imports fail, choose a compatible
Kaggle image before running this workflow. It does not repair CUDA drivers.

Pretrained initialization needs Internet once or the official torchvision
ImageNet weight file already cached under PyTorch's hub/checkpoints cache. The
bundle deliberately contains no pretrained weights or download links. Offline
automated tests use `pretrained=False`; resume and checkpoint inference also
skip pretrained downloads.

## Exact Kaggle steps

1. Upload the separate `artifacts/kaggle/aerodepth_kaggle_code.zip` as a private
   Kaggle dataset and attach it alongside the verified MidAir dataset. Do not
   replace or repackage the MidAir upload. Import `notebooks/kaggle_train.ipynb`
   into a Kaggle notebook.
2. Select a GPU accelerator and enable Internet for initial ImageNet weights
   unless cached. Run the environment and snapshot setup cells. An extracted
   project is supported; only small allowlisted code files are copied to working.
   If the ZIP is intact, paste its sidecar SHA-256 into `EXPECTED_BUNDLE_SHA256`.
3. Set an explicit `DATASET_ROOT` if discovery finds multiple trees. Set
   `EXPECTED_TRAJECTORIES` or `EXPECTED_TRAJECTORY_COUNT` from the verified upload
   inventory when available. Run validation and review the discovered inventory.
   Without an external inventory, internal alignment cannot prove completeness.
4. Generate the deterministic grouped manifest once or attach an existing one.
   All weather variants of the same environment/path share a split: MidAir's
   climate digit is removed (IDs 0003/1003/2003/3003 become path 0003). All variants
   of the pilot's Kite paths 0003 and 0021 are excluded from final test. Add any other
   inspected groups to `ADDITIONAL_INSPECTED_PATHS` before generation. Unknown
   layouts conservatively group by numeric path and exclude pilot path numbers.
   Retain the manifest. This is a custom held-out path split, not the official
   MidAir robustness benchmark protocol.
5. Run the two-epoch, three-batch-per-split GPU smoke. Review memory use and the
   throughput estimate. It measures train and validation separately after the
   first warm-up epoch; the duration estimate needs extra allowance for setup,
   checkpoint writing, session limits, and I/O variability.
6. Review depth limits, resolution, batch size, loss/schedule, and epoch budget.
   Edit `CONFIG` in the launch cell; set the depth-range and budget review fields
   to true and `RUN_FULL_TRAINING=True`. Keep `t_max` fixed across interruptions.
   Resume from `last.pt`, with the prior `best.pt` beside it, using the same data,
   manifest, and configuration. Checkpoints are saved at epoch boundaries.
7. Run the validation-only cell to save metrics, loss curve, and prediction/error
   plots. Download the run directory, preparation directory, and code-bundle
   receipt before session expiration. The notebook has no automatic test cell.

Generated outputs use `/kaggle/working`; images/HDF5 are read directly from the
attached input. Do not copy the full dataset into working. Multi-GPU parallel
training is not configured; this runner uses the default single CUDA device.

For a new session, attach the prior outputs, select its saved manifest in
`EXISTING_MANIFEST`, and point `RESUME_CHECKPOINT` at its `last.pt`. The trainer
copies the companion best checkpoint into the new run directory if needed.
Resume history is retained. An interrupted epoch is repeated. Save outputs during
long sessions; Kaggle's actual session/storage limits must be checked in the UI.

## Bundle provenance and local verification

Run `python scripts/package_kaggle.py` to create the small allowlisted ZIP plus
`.zip.sha256` and `.provenance.json` sidecars. `SOURCE_PROVENANCE.json` inside the
ZIP records the base revision, dirty-worktree status, read-only Git status, and
SHA-256 for each bundled file. A dirty snapshot is not claimed to equal its base
commit. The bundle README is generated without dataset download links. No Git
metadata, datasets, environments, credentials, old experiments, or checkpoints
are included. The extracted-source setup verifies each file hash; the intact-ZIP
setup can additionally verify the original ZIP checksum.

Local checks cover normal/odd output sizes, finite positive range, masked-loss
backpropagation, offline initialization, checkpoint round-trip and resumed versus
continuous training, the baseline regressions, replay compatibility, grouped
splits/pilot exclusion, synchronization, bundle setup, and notebook syntax.
The opt-in real-data integration test is deliberately skipped unless requested.
No final-test evaluation or long CPU training is performed during preparation.
