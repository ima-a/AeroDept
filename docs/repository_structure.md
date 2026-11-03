# Repository structure

The repository contains the runnable project, software checks, user documentation and curated evidence. Local evaluation material and operational archives are excluded from Git.

| Folder | Purpose |
|---|---|
| `src/aerodepth_lite/` | Actual Python package: models, losses, metrics, data, training, inference, scheduling and dashboard helpers |
| `scripts/` | Supported command-line entry points listed below |
| `tests/` | Checks for model loading, calculations, preprocessing, scheduling, checkpoint behaviour and dashboard views |
| `configs/` | `dashboard.json` for current inference exports, frozen experiment configurations and ignored private local inputs |
| `docs/` | User guides, scientific methods, results and reproduction instructions |
| `models/` | Four allowlisted inference exports (three students and the ResNet-50 teacher), metadata and checksums; full training states stay private |
| `results/`, `evidence/` | Frozen protocols, relative manifests, recorded metrics, scientific figures and provenance |

`src` means source code. `pyproject.toml` installs the package from this directory and exposes the `aerodepth` command. The dashboard and scripts import it, so it is required. Modules named `presentation.py` support dashboard/replay functionality; they are not PowerPoint files. The internal `kaggle.py` module remains because training uses its grouped-path utilities; the optional Kaggle notebook and upload helpers have been removed.

`tests/` contains software checks, not the held-out research dataset. Most checks use synthetic inputs or saved artifacts. Real-data tests are marked `integration`. Running software checks is separate from rerunning a research experiment.

## Supported scripts

| Script | Purpose |
|---|---|
| `dashboard.py` | Launch the Streamlit research dashboard |
| `expanded_training.py` | Expanded supervised/teacher training and the quality gate |
| `exploratory_kd_training.py` | Separate rejected-teacher distillation workflow |
| `presentation.py` | Validation replay, cache and timing stages using fresh outputs |
| `reproduce.py` | Original validation and scheduler reproduction |
| `smoke.py` | Dataset-free synthetic end-to-end software check |
| `prepare_independent_test.py` | Guarded independent-data inventory and extraction |
| `independent_test.py` | Frozen independent-study runner; completed exposure is not repeatable in place |
| `verify_independent_test.py` | Read-only audit of completed independent-study artifacts |
| `final_study.py` | Historical Original-model study entry point; use its documented source revision |
| `verify_final_study.py` | Read-only historical-study audit with the matching source revision |

The former host-specific supervisor, laptop launcher, Kaggle bootstrap/packager, superseded test proposal runner and one-off export/report/cache-check scripts have been retired. Their historical versions remain in Git history and the local operational archive. Core implementation, frozen scientific results and the current independent-study runner are unchanged.

## Root files

| File | Why it remains |
|---|---|
| `README.md` | Project overview, setup, results and navigation |
| `pyproject.toml` | Package metadata, dependencies, CLI registration and pytest settings |
| `MANIFEST.in` | Controls files included in a source distribution, including scripts, configurations and model exports |
| `requirements-tested.txt` | Recorded dependency versions for reproducible setup |
| `Makefile` | Installation, software-check and synthetic-workflow commands used by manual CI |
| `NOTICE.md` | Dataset, TorchVision and initialization attribution/upstream terms |

## Local-only files

`CODE_GUIDE.md` and `docs/VIVA_GUIDE.md` remain on the prepared machine, ignored by Git and excluded from source distributions. The current slides are `presentations/project_evaluation/AeroDepth_Lite_Project_Evaluation.pptx` with a companion shared-layout PDF. These local paths are not included in a GitHub clone; public readers can use the [final report](FINAL_REPORT.md), [training method](expanded_training.md) and the implementation in `src/aerodepth_lite/`.

The `notebooks/` folder has been deleted. Datasets, full checkpoints, caches and environments remain ignored. Retired internal documents and helper scripts are retained under the ignored `artifacts/archive/repository_cleanup_20261101/` for local historical reference. Published results and their recorded identities remain in place.
