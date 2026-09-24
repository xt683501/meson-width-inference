# Meson total-width prediction

An FT-Transformer for meson total decay widths, accompanying [*Meson Properties and Symmetry Emergence Based on the Deep Neural Network*](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/43/2/020201), *Chinese Physics Letters* **43**, 020201 (2026). [arXiv version](https://arxiv.org/abs/2509.17093).

## Weights and data

Download the two checkpoints from the [v1.3.0 release](https://github.com/xt683501/meson-width-inference/releases/tag/v1.3.0) into `weights/`:

| File | Description | SHA-256 |
|---|---|---|
| `paper-model.pth` | Checkpoint used for the paper's predictions | `164059cedd335ae7c525fde9e10af974bb01534b4680dccaca6429ac0e3e37d9` |
| `revised-model.pth` | Later paper-derived checkpoint with lower error on the corrected training data | `a45b25a8d8d6b67fe54c279f5f5a0a827178915ee13e147f39b6cef653ec2275` |

Check the downloads with `cd weights && sha256sum -c SHA256SUMS`. Model-ready data are in [`data/meson-train.csv`](data/meson-train.csv) (370 training mesons) and [`data/meson-other-particles.csv`](data/meson-other-particles.csv) (47 records outside the training table, including 12 alternate encodings). See the [data notes](data/README.md) for label types; more detailed measured properties are available from the [Particle Data Group](https://pdg.lbl.gov/).

## Predict a width

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python predict.py examples/encoding-cases.csv predictions-paper.csv
python predict.py examples/encoding-cases.csv predictions-revised.csv --weights weights/revised-model.pth
```

The output CSV adds `predicted_width_mev` in MeV. Provide the particle's encoded quantum numbers, mass, and quark-content hypothesis—not just its name. The four rows in `examples/encoding-cases.csv` show different encoding cases; [INPUT_FORMAT.md](INPUT_FORMAT.md) explains every field and the project's categorical codes. `examples/paper-table1.csv` provides four published-prediction regression examples. The default weight is `paper-model.pth`.

For a readable two-stage cross-scale regression example, see [`train_example.py`](train_example.py): first fit log-width with LogCosh, then refine using squared relative error in physical width. It reads the published training table; it is an illustrative implementation, not the script that produced either released checkpoint.

Predictions for proposed or newly found particles can be compared with experimental width measurements as they become available.

## License

Original code and documentation use the [MIT License](LICENSE). `model.py` adapts code from [lucidrains/tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch); its MIT notice is retained in `THIRD_PARTY_LICENSE`.
