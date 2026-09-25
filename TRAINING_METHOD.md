# Training across a wide width range

The widths in the released 370-row training table run from about `1.29e-14` to `625` MeV—a ratio of about `10^16.7`. Fitting raw MeV widths with ordinary squared error would let the wide states dominate. The study instead predicts **log width**: the target is `y = ln(Γ / MeV)` and the model output is `ŷ`. Inference converts back with `Γ̂ = exp(ŷ)` MeV. The FT-Transformer takes the [encoded categorical quantum numbers, masked numerical features, and numerical quark-content features](INPUT_FORMAT.md); its architecture is shared by both phases. The historical training data also sampled masses around each central value, whereas the compact public table keeps one central-mass row per particle.

## The two-stage idea in the original training code

1. **Coarse fit in log space.** The original trainer can choose `LogCoshLoss`, `mean(log(cosh(ŷ − y)))`. The logarithm compresses the width range, while LogCosh is smooth near zero and less sensitive than a quadratic loss to large initial residuals. `train_example.py` evaluates this with an algebraically equivalent, overflow-resistant formula.
2. **Switch when the fit is ready, not after a mandatory epoch count.** The original loop monitors the training-set `LogMAPE = mean(|ŷ − y| / |y|) × 100%` (excluding `|y|` near zero) after each epoch. If it falls below a configured threshold, the **next** epoch uses stage 2. This is a switching diagnostic in *log coordinates*; it is **not** a physical-width percentage error. The threshold is a training choice, not a universal performance benchmark. The example defaults to `3.3%` to illustrate the original threshold mechanism; its one-row-per-particle data differ from the augmented training runs, so tune the threshold rather than assuming a fixed switching epoch. If it is never met, training stays in stage 1. An optional `--phase1-epochs` supplies a maximum stage-1 duration for experiments.
3. **Refine relative error in physical units.** The original `MSPE_from_Log_Loss` converts both output and target with `exp`, then minimizes `mean(((Γ̂ − Γ) / (Γ + 1e-14 MeV))²)`. This shifts attention from absolute MeV differences to proportional width errors across scales. The `1e-14` stabilizer matters for the very narrowest widths: this training loss is therefore not numerically identical to the paper's exact per-particle relative-error equation `|Γ̂ − Γ| / Γ`. Report evaluation metrics separately from the loss. The example logs training physical-width RMS relative error using the exact denominator `Γ`.

The original program records both log-space and physical-space diagnostics, uses AdamW and a plateau learning-rate scheduler, and can save the best checkpoint separately in each stage. **The phase losses, threshold, and hyperparameters are configurable.** This example defaults to the paper's conceptual LogCosh-to-relative-error order, but `--phase1-loss` and `--phase2-loss` can also select the reverse order found in another archived configuration. A configured second stage need not actually be reached. Do not assume that either released checkpoint followed the defaults here: `train_example.py` is a cleaned-up explanation of the mechanism, not the script that generated `paper-model.pth` or `revised-model.pth`.

## Data and example

The original study perturbed each particle mass to make multiple related rows; those rows are **not independent particles**. This repository's [`data/meson-train.csv`](data/meson-train.csv) instead has one central-mass input for each of 370 training particles, keeping the example short and the unit of evaluation clear. The [other-particle table](data/meson-other-particles.csv) is not read by the training example.

After installing `requirements.txt` and downloading the checkpoints, a small CPU check is:

```bash
python train_example.py --smoke --output /tmp/meson-smoke.pth
```

For a full illustrative run, choose your own output path (the script refuses to overwrite an existing file or write in `weights/`):

```bash
python train_example.py --output /tmp/meson-example.pth \
  --epochs 300 --switch-log-mape-pct 3.3 --lr 1e-4 --batch-size 64
```

You may add `--init weights/paper-model.pth` to start from that checkpoint **without modifying it**. A full run has not been benchmarked or claimed to reproduce either released weight. For the physical context and reported results, see the [journal article](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/43/2/020201) or [arXiv version](https://arxiv.org/abs/2509.17093).
