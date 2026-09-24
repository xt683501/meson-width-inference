# Model input format

`predict.py` reads a UTF-8 CSV containing **encoded model features**, one particle or one physical hypothesis per row. It predicts total width in MeV. The input is not a list of particle names: `name` is optional metadata and does not affect the model.

Start with [`examples/encoding-cases.csv`](examples/encoding-cases.csv), which shows four **actual encoded rows** from the released training table. `examples/paper-table1.csv` is useful for reproducing four paper predictions but, by itself, does not show all category codes, unknown-value masks, or mixed-flavour entries. Copy an existing row only when its quantum numbers and flavour hypothesis truly match the particle you want to study.

## Required columns

| Columns | Meaning and input |
|---|---|
| `P`, `C`, `G` | **Project-specific category codes**, not raw parity values. Use the mapping below. |
| `I`, `I3`, `J` | Isospin, its third component, and spin. Supply a finite number even when unknown; the corresponding mask determines whether it is used. |
| `Mass` | Positive mass in **GeV**. Use the mass estimate for the particle or hypothesis you are evaluating. |
| `Ik`, `I3k`, `Jk`, `Mk` | Known-value masks: `1` = known, `0` = unknown. `Mass` is required and must be positive even if `Mk=0`; the released data use `Mk=1`. |
| `u`, `ubar`, `d`, `dbar`, `s`, `sbar`, `c`, `cbar`, `b`, `bbar` | Ten numerical quark-content features in exactly this order; see below. |

Optional columns: `name` (carried through to the output), `N`, `Nk`, and any additional metadata. The program derives `N = 0.1 × Mass / 0.134976828` and sets `Nk=1` when they are omitted. If supplied, `N` must agree with the derived value (within the program's numerical tolerance) and `Nk` must equal 1. The neutral-pion reference mass and factor are the constants in `model-config.json`; this is the **inference** convention. The historical mass-augmented training rows kept `N` fixed at the central-mass value while perturbing `Mass`, so do not pass those augmented rows directly through this CLI.

## Project-specific `P`, `C`, `G` encoding

| Code in model CSV | `P` original value | `C` original value | `G` original value |
|---:|---:|---:|---:|
| `0` | `−1` | `−1` | `−1` |
| `1` | `+1` | `+1` | `+1` |
| `2` | original `−222` marker | original `−333` marker | original `−333` marker |

Code `2` combines the source's unavailable/not-applicable/undetermined cases; the model has **one** third category for each of these columns. All three codes are looked up as separate learned category embeddings; `P`, `C`, and `G` have no additional known-value masks. The model cannot distinguish different reasons for code `2` after encoding. Do not submit the raw `−1`, `+1`, `−222`, or `−333` values to `predict.py` as category codes. In particular, `C=2` or `G=2` is appropriate when that quantum number was not assigned in the project's encoding; it is **not** a third physical parity value. If a quantum number is unresolved, do not guess 0 or 1 from a particle name; use code `2` only when that matches the project's unassigned-category convention. A blank or `NaN` category cell is invalid.

## Masks and missing numerical values

Each numerical feature is paired with a known mask: `I/Ik`, `I3/I3k`, `J/Jk`, `Mass/Mk`, `N/Nk`. `1` means known. When a mask is `0`, the model replaces that feature's numerical embedding with a learned unknown embedding; the mask itself is still supplied to the model. The CSV still needs a **finite** placeholder. The released training rows use `−222` for masked `I`, `I3`, or `J`; `0` also works as a masked-out placeholder. Neither is a measured quantum number, and leaving a cell blank or writing `NaN` is rejected by the CLI. In the released 370-row table, `Ik=0` occurs 7 times, `I3k=0` 9 times, and `Jk=0` 44 times; `Mk` and `Nk` are always `1`.

`predict.py` accepts `Mk=0`, but it **still requires a positive `Mass` and derives `N` from it with `Nk=1`**. Thus `Mk=0` masks the direct Mass feature, not all mass information: the derived `N` remains available to the model. The published examples use `Mk=1`; use that setting for an ordinary mass-based prediction. The CLI does not represent a wholly missing mass.

## Quark-content numbers

The ten quark columns are **numerical encoding weights, not ten independent yes/no flags**. They have no missing-value masks: `0` is an encoded zero contribution, not a missing-value marker. Simple flavour assignments often use 0 and 1. The released data also contain rounded magnitudes such as `0.408` (`1/√6`), `0.577` (`1/√3`), `0.707` (`1/√2`), and `0.816` (`2/√6`) for flavour mixtures; `2` appears in some multi-quark encodings. These entries represent the project's chosen flavour-content hypothesis; they do not by themselves resolve mixing angles, signs/phases, or a particle's true internal structure. `predict.py` checks that these columns contain finite numbers, but it cannot decide whether a new hypothesis is physically appropriate. Use the encoded rows and the [paper](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/43/2/020201) as guides rather than inferring an exotic state's vector from its name.

## Worked rows

The four rows in `examples/encoding-cases.csv` are copied from `data/meson-train.csv` with `N` and `Nk` deliberately omitted so the CLI derives them:

- `B+`: ordinary heavy-flavour entry; `P=0`, while unassigned `C` and `G` are code `2`; bottom-antiquark column is nonzero.
- `K(L)0`: `C=1` and `I3k=0`, showing a masked numerical feature and fractional flavour entries.
- `b_1(1235)0`: `G=1` and fractional light-flavour entries.
- `T_cc(3875)+`: code `2` for all three categorical features, three masked numerical features, and a non-binary quark-content value. This row is an **example of the recorded encoding**, not a claim that every state with a similar name has the same composition.

Run either weight on the example:

```bash
python predict.py examples/encoding-cases.csv encoding-paper.csv
python predict.py examples/encoding-cases.csv encoding-revised.csv --weights weights/revised-model.pth
```

The output adds `predicted_width_mev`. It is a model estimate, not an experimental measurement. More detailed measured masses and widths are available from the [Particle Data Group](https://pdg.lbl.gov/); the model and feature definitions are discussed in the [journal paper](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/43/2/020201) and its [arXiv version](https://arxiv.org/abs/2509.17093).
