# Documentation index

Use these guides for the current project. Dated reports under `results/` describe what happened at that time; earlier progress snapshots are not the present training status.

| Need | Start here |
|---|---|
| Run inference or the dashboard | [Project README](../README.md) |
| Explain and locate code in a viva | [CODE_GUIDE.md](../CODE_GUIDE.md) and [VIVA_GUIDE.md](VIVA_GUIDE.md) |
| Verify the new independent test | [independent_final_test.md](independent_final_test.md) |
| Present the project | [Editable PowerPoint and preview](../presentations/aerodepth_lite/README.md) |
| Explain findings and limitations | [FINAL_REPORT.md](FINAL_REPORT.md) |
| Install dependencies | [installation.md](installation.md) |
| Understand data formats and split versions | [dataset.md](dataset.md) |
| Use or reproduce the A dashboard | [presentation_dashboard.md](presentation_dashboard.md) |
| Reproduce original validation / scheduler runs | [reproduction.md](reproduction.md) |
| Reproduce expanded training | [expanded_reproduction.md](expanded_reproduction.md) |
| Read the supervised / teacher / KD method | [expanded_training.md](expanded_training.md) |
| Understand rejected-teacher exploration | [exploratory_kd.md](exploratory_kd.md) |
| Understand same-batch AMP recovery | [expanded_numerical_recovery.md](expanded_numerical_recovery.md) |
| Locate retained local files or see cleanup details | [archive_locations.md](archive_locations.md) |

## Current status

Original student training completed 50 epochs, A completed 10, teacher B completed 20, and the separate exploratory KD run completed 10. The teacher failed the original gate. Original gated KD did not run. A and exploratory KD select epoch 9. The reserved 8,820-frame independent study completed once under its frozen protocol, with all seven comparisons and saved-evidence verification passing.

Canonical comparison evidence is in `results/student_validation_comparison_20261029T152648Z/` and `results/presentation_validation_20261029T155521Z/`. These paths, model exports, split identities and recorded results survive the cleanup unchanged.

## Historical and optional workflows

- [final_study.md](final_study.md): the original model's completed 4,436-frame test, with its own source revision; separate from the new reserved study.
- [local_training.md](local_training.md): original laptop workflow and dated observations. Local historical output paths may now resolve through the archive mapping.
- [kaggle_training.md](kaggle_training.md): optional Kaggle workflow.
- [publication.md](publication.md) and [GIT_READY_CHECKLIST.md](GIT_READY_CHECKLIST.md): publication history and source-delivery scope.
- [expanded_experiment_20261025/PLAN.md](expanded_experiment_20261025/PLAN.md): historical acquisition/experiment plan.

## Source identity after adding code markers

The current code includes one-word section comments for presentation. Its AST is unchanged, but comments change file hashes. Historical strict-source checkpoint resume and sealed protocols must use their matching source revision, not a rewritten receipt. The pre-annotation expanded sources are available at commit `fb1b4aedd15280abdb1126293e022d78cdcbf4e5`. The original final study instead uses `8f32f71` as documented in its guide.

Fresh training can use current sources and record their new identities. Do not mix checkpoints from different source identities within a run. Exported-weight inference and saved dashboard views remain supported by the annotated code.
