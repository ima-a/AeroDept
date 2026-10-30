# AeroDepth-Lite project presentation

- **AeroDepth_Lite_Project_Presentation.pptx** — the 13-slide presentation. Text, tables and the workflow shapes/connectors are editable. The training curves are images plotted from saved experimental values.
- **AeroDepth_Lite_Preview.pdf** — companion layout preview, rendered from the same layout definitions; not a PowerPoint-exported PDF.
- **slide_overview.jpg** — overview of all 13 slides.
- **slide_content_and_notes.json** — slide text and speaker notes, including evidence paths and literature links.

Replace the bracketed team names, roll numbers, department, college and guide fields on slide 1. Speaker notes provide the details behind the concise slide content. The presentation uses the supplied slide order, a 16:9 layout, and Noto Sans / Arial fonts.

## Evidence used

The consolidated `docs/FINAL_REPORT.md` and the completed 29 October 2026 student and presentation-validation reports are the primary sources. Exact model metrics are read from `results/student_validation_comparison_20261029T152648Z/comparison.csv`; curves use `results/presentation_validation_20261029T155521Z/training_curves.json`. Dataset counts and duplicate-path checking use the frozen expanded train/validation manifest. No new model training, inference, test-set access or experiment was performed.

The deck distinguishes original training (17,635 frames) from expanded training (44,098), and familiar validation (4,410) from the reserved independent test (8,820 expected; not run). The historical 4,436-frame test is not presented as new-model generalization. Classification metrics are marked inapplicable to dense range regression. Negative teacher/KD and near-field results are retained.

Literature sources: [Eigen et al.](https://arxiv.org/abs/1406.2283), [Howard et al.](https://arxiv.org/abs/1905.02244), [He et al.](https://arxiv.org/abs/1512.03385), [Hinton et al.](https://arxiv.org/abs/1503.02531), and [Fonder & Van Droogenbroeck](https://orbi.uliege.be/handle/2268/234665). [MidAir range definition](https://midair.ulg.ac.be/faq.html) and [download source](https://midair.ulg.ac.be/download.html).

The architecture diagram is constructed with native vector shapes and connectors, not an image-generation tool.

## Build and checks

`build_presentation.py` requires python-pptx and Pillow. In the creation environment:

```bash
PYTHONPATH=/tmp/aerodepth_pptx_deps python presentations/aerodepth_lite/build_presentation.py
```

The script checks the slide count, ZIP integrity, XML parsing and calculated text bounds. The companion previews were visually inspected. `layout_check.json` records these checks. A PowerPoint application render was not available in this environment, so native font substitution can alter the layout slightly.
