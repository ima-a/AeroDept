# Documentation

Start with the guide for the task you want to perform. Training and the independent 8,820-frame study are complete. Dated evidence under `results/` records the original experiments; current interpretation is in the final report.

| Need | Guide |
|---|---|
| Install and run the project | [Project README](../README.md) and [installation](installation.md) |
| Use the dashboard and external inputs | [Dashboard guide](presentation_dashboard.md) |
| Understand results and limitations | [Final report](FINAL_REPORT.md) |
| Understand folders, scripts and root files | [Repository structure](repository_structure.md) |
| Obtain data and understand its formats | [Dataset guide](dataset.md) |
| Verify the completed independent study | [Independent test](independent_final_test.md) |
| Reproduce original validation and scheduling | [Validation reproduction](reproduction.md) |
| Reproduce expanded training or resume old checkpoints | [Expanded reproduction](expanded_reproduction.md) |
| Understand the model, losses and teacher gate | [Expanded training method](expanded_training.md) |
| Understand exploratory distillation | [Exploratory KD](exploratory_kd.md) |
| Understand AMP overflow recovery | [Numerical recovery](expanded_numerical_recovery.md) |
| Reproduce the historical Original-model study | [Historical final study](final_study.md) |

A and exploratory KD selected epoch 9. Teacher B failed its acceptance gate, so original gated KD did not run. The separate exploratory KD arm retained that rejection. The independent study evaluated seven frozen comparisons on four reserved paths and passed saved-evidence verification.

All three evaluated students and the ResNet-50 teacher have [published inference weights](../models/README.md). Select **Model comparison → Validation** in the dashboard to see teacher metrics and its failed-gate explanation; the teacher has no independent-test result. KD inference needs no teacher or private checkpoint; training resume and historical study execution have separate requirements. The current dashboard uses `configs/dashboard.json`.

The main evidence directories are `results/student_validation_comparison_20261029T152648Z/`, `results/presentation_validation_20261029T155521Z/` and `results/independent_test_20261030/`.

Evaluation slides, `CODE_GUIDE.md` and `docs/VIVA_GUIDE.md` are local-only. Internal agent handoffs, publication checklists, local maintenance receipts and design-debug captures have been removed from this documentation tree. The notebook workflow and its launch/packaging helpers have been retired.
