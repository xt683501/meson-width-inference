# Model-ready training table

`meson-train.csv` has 370 rows, one per training meson at its central mass. `width_mev` is the width label in MeV; `split` is `train` for every row. All other feature columns follow [INPUT_FORMAT.md](../INPUT_FORMAT.md). This compact table is suitable for the illustrative `train_example.py`; it is not the 501-mass-draw augmentation used in the study.

The table was extracted from the project's corrected training workbook (`0902_500_train_corrected_v1.xlsx`, SHA-256 `33dcc74619b1f70f913f084aac84d7d047c9e2e47b9eec237970bdc5f61882e6`). The extracted CSV's SHA-256 is `f202c3b93b144fd0bfa0f85618034280c38bda0e4b7bfe94799007b8bab6da13`. It includes corrected width labels for ten training particle names; for detailed primary measurements consult the [Particle Data Group](https://pdg.lbl.gov/). See the [paper](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/43/2/020201) for the study's data and method context.
