# Meson total-width inference

Inference-only code for the FT-Transformer model in Xin Tong *et al.*, [*Chinese Physics Letters* **43**, 020201 (2026)](https://doi.org/10.1088/0256-307X/43/2/020201). The full dataset and training code are not included.

## Get the weights

Download both files from the [v1.0.0 release page](https://github.com/xt683501/meson-width-inference/releases/tag/v1.0.0) into `weights/`.

| File | Description | SHA-256 |
|---|---|---|
| `paper-model.pth` | Checkpoint used for the paper's predictions | `164059cedd335ae7c525fde9e10af974bb01534b4680dccaca6429ac0e3e37d9` |
| `revised-model.pth` | Later checkpoint derived from the paper weight, with lower error on corrected training data | `a45b25a8d8d6b67fe54c279f5f5a0a827178915ee13e147f39b6cef653ec2275` |

Check the downloaded files with `cd weights && sha256sum -c SHA256SUMS`. Better fit on the training data does **not** guarantee better predictions for newly discovered particles.

## Run inference

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python predict.py examples/paper-table1.csv predictions-paper.csv
python predict.py examples/paper-table1.csv predictions-revised.csv --weights weights/revised-model.pth
```

The output CSV adds `predicted_width_mev` in MeV. The four example rows demonstrate the required input format; they are not the full dataset. `predict.py` uses the paper weight by default and refuses to overwrite an existing output file.

## Input format

Provide a CSV with the columns shown in `examples/paper-table1.csv`: categorical codes `P`, `C`, `G`; `I`, `I3`, `J`, `Mass`; known-value masks `Ik`, `I3k`, `Jk`, `Mk`; and quark-content columns `u`, `ubar`, `d`, `dbar`, `s`, `sbar`, `c`, `cbar`, `b`, `bbar`. `Mass` is in GeV. `N` and `Nk` may be omitted: the CLI derives `N = 0.1 × Mass / 0.134976828` and sets `Nk = 1`. `name` is optional. A particle name alone is not enough to run inference: supply its encoded features.

The output is a model prediction, not a measurement.

## License

Original code and documentation are under the [MIT License](LICENSE). `model.py` adapts code from [lucidrains/tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch); its MIT notice is retained in `THIRD_PARTY_LICENSE`.
