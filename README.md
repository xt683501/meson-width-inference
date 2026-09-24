# Meson total-width inference

Inference-only code for the FT-Transformer model in Xin Tong *et al.*, [*Chinese Physics Letters* **43**, 020201 (2026)](https://doi.org/10.1088/0256-307X/43/2/020201). This repository does not include the full dataset or training code.

## Weights

| Release asset | Description | SHA-256 |
|---|---|---|
| `paper-model.pth` | Weight used for the paper's Table 1 predictions | `164059cedd335ae7c525fde9e10af974bb01534b4680dccaca6429ac0e3e37d9` |
| `revised-model.pth` | v6: separate paper-derived fine-tune; lower error on the corrected training labels | `a45b25a8d8d6b67fe54c279f5f5a0a827178915ee13e147f39b6cef653ec2275` |

The v6 weight is not the paper weight. It was fine-tuned from the paper-derived lineage using corrected training labels and two validation-particle width-interval constraints. Those two particles are therefore **not independent validation evidence** for v6. Its lower training error is **not** an improvement claim on validation or other particles. The two files are available on the [v1.0.0 release page](https://github.com/xt683501/meson-width-inference/releases/tag/v1.0.0) rather than in Git because each exceeds GitHub's normal 100 MB file limit. Download both assets into `weights/`; verify them with `cd weights && sha256sum -c SHA256SUMS`.

## Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python predict.py examples/paper-table1.csv predictions-paper.csv
python predict.py examples/paper-table1.csv predictions-revised.csv --weights weights/revised-model.pth
```

The output CSV adds `predicted_width_mev` (MeV). The example contains four encoded inputs solely to demonstrate the format. `predict.py` defaults to `weights/paper-model.pth` and refuses to overwrite an existing output file.

## Input format

Supply a CSV with the same columns as `examples/paper-table1.csv`: categorical codes `P`, `C`, `G`; numerical `I`, `I3`, `J`, `Mass`; their known-value masks `Ik`, `I3k`, `Jk`, `Mk`; and quark-content columns `u`, `ubar`, `d`, `dbar`, `s`, `sbar`, `c`, `cbar`, `b`, `bbar`. `Mass` is in GeV. `N` and `Nk` may be omitted: the CLI derives `N = 0.1 × Mass / 0.134976828` and sets `Nk = 1`. `name` is optional metadata. A particle name alone is not a model-ready input; feature values and category codes must be supplied correctly.

The model outputs a predicted total width, not an experimental measurement. The original weight is preserved for paper-result replication; choose the revised weight explicitly for its separate inference output.

## Attribution

The repository's original code and documentation are released under the [MIT License](LICENSE). `model.py` adapts code from [lucidrains/tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch); its original MIT notice is retained in `THIRD_PARTY_LICENSE`.
