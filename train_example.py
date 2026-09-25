"""Illustrative two-stage training example for the meson-width FT-Transformer.

Purpose
-------
This is a *pedagogical* implementation of the two-stage cross-scale protocol
described in the paper (Xin Tong *et al.*, *Chinese Physics Letters* **43**,
020201 (2026), arXiv:2509.17093, Eqs. 10-11):

  * Phase 1: LogCosh loss on log-width. Monitor training log-width MAPE;
    switch at the requested threshold, as in the original training loop.
  * Phase 2: mean squared relative error on the physical MeV width, per species,
    with the original code's 1e-14 MeV denominator stabilizer.

        e_i   = |predicted_i - reference_i| / reference_i     (paper Eq. 10)
        Loss  = mean(e_i^2)                                    (paper Eq. 11)

This is a compact example of the two-stage method, not the script used to
produce the released checkpoints.

Data contract
-------------
The default CSV (``data/meson-train.csv``) holds exactly one row per training
species (370 species, ``split=train`` only; no rows from the other-particles table).
Each row sits at the species' exact central mass, i.e.
``Mass == N * 0.134976828 / 0.1`` (the paper's deterministic-inference mass),
with a positive finite ``width_mev`` label.  The columns are the model-ready
feature set of :mod:`predict` plus ``width_mev`` and ``split``.

Safety
------
* The released checkpoints (``weights/paper-model.pth``,
  ``weights/revised-model.pth``) are treated as read-only.  ``--output`` is
  rejected if the file already exists, if it resolves inside ``weights/``, or
  if it carries a reserved released-weight name.
* ``--init`` optionally loads a state dict (for example the paper weight) as
  the starting point; it is opened read-only and never modified.

Environment
-----------
Like :mod:`predict`, this module needs the runtime dependencies from
``requirements.txt`` (torch, einops, hyper-connections).  Run from the
repository root:

    # tiny CPU smoke test (two epochs, 64 species, a few seconds)
    python train_example.py --smoke --output /tmp/smoke-weights.pth

    # full illustrative run on CPU (this is an example, not a benchmark)
    python train_example.py --output /tmp/example-weights.pth \
        --epochs 300 --switch-log-mape-pct 3.3 --lr 1e-4 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

try:
    from model import FTTransformer
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"{exc} -- install the runtime dependencies first "
        f"(pip install -r requirements.txt)"
    ) from exc

from predict import read_inputs

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "model-config.json"
DEFAULT_DATA = ROOT / "data" / "meson-train.csv"
WEIGHTS_DIR = ROOT / "weights"
PROTECTED_WEIGHT_NAMES = {"paper-model.pth", "revised-model.pth"}
EXPECTED_TRAIN_SPECIES = 370
LOG2 = math.log(2.0)

ILLUSTRATIVE_BANNER = (
    "Two-stage cross-scale example. Phase losses and switching threshold are configurable; "
    "this does not reproduce either released checkpoint."
)


def logcosh(x: torch.Tensor) -> torch.Tensor:
    """Elementwise ``log(cosh(x))``, stable for large |x|.

    Uses ``log cosh x = |x| + log1p(exp(-2|x|)) - log 2``; for large |x|
    the exponential underflows to 0 and the expression approaches |x|-log 2.
    """
    ax = x.abs()
    return ax + torch.nn.functional.softplus(-2.0 * ax) - LOG2


def phase1_loss(log_pred: torch.Tensor, log_true: torch.Tensor) -> torch.Tensor:
    """Phase 1: mean LogCosh loss on the log-width target."""
    return logcosh(log_pred - log_true).mean()


def msre_mev_loss(log_pred: torch.Tensor, width_true: torch.Tensor,
                  epsilon: float = 1e-14) -> torch.Tensor:
    """Phase 2 (paper Eqs. 10/11): mean squared relative error in MeV.

    Follows the original MSPE_from_Log_Loss: ``exp(log_pred)`` converts back
    to MeV and ``1e-14`` stabilizes the denominator. This is close to, but
    not identical with, Eq. 11 for the narrowest widths.
    """
    relative = (log_pred.exp() - width_true) / (width_true + epsilon)
    return relative.square().mean()


def log_mape_pct(log_pred: torch.Tensor, log_true: torch.Tensor) -> float:
    """Original switch metric: MAPE of log-width, not physical-width MAPE."""
    mask = log_true.abs() > 1e-8
    if not bool(mask.any()):
        return 0.0
    return float(((log_pred[mask] - log_true[mask]).abs() /
                  log_true[mask].abs()).mean() * 100.0)


def configured_loss(name: str, log_pred: torch.Tensor, log_true: torch.Tensor,
                    width_true: torch.Tensor) -> torch.Tensor:
    """Select the phase loss by name, following the original config pattern."""
    if name == "logcosh":
        return phase1_loss(log_pred, log_true)
    if name == "mspe":
        return msre_mev_loss(log_pred, width_true)
    raise ValueError(f"unknown phase loss: {name}")


def build_model(config: dict) -> FTTransformer:
    """Instantiate the FT-Transformer from a model-config.json mapping."""
    features = config["features"]
    return FTTransformer(
        categorical_cols=features["categorical_cols"],
        categorical_cardinalities=config["categorical_cardinalities"],
        combinable_numerical_cols=features["combinable_numerical_cols"],
        is_known_col_map=features["is_known_col_map"],
        quark_cols=features["quark_cols"],
        **config["model_params"],
    )


def read_train_csv(path: Path, config: dict,
                   expected_species: int | None = EXPECTED_TRAIN_SPECIES
                   ) -> tuple[dict[str, torch.Tensor], list[str], list[float]]:
    """Load the model-ready training CSV and return (tensors, names, widths).

    Feature tensors are built with :func:`predict.read_inputs` (the same code
    path used for inference, so the CSV is model-ready by construction).
    Additional contract: every row has ``split == "train"``, species names
    are unique (one row per species at the exact central mass), and
    ``width_mev`` is positive and finite.
    """
    rows, tensors, fields = read_inputs(path, config)
    for column in ("width_mev", "split", "name"):
        if column not in fields:
            raise ValueError(f"training CSV is missing required column {column!r}")
    names: list[str] = []
    widths: list[float] = []
    for line, row in enumerate(rows, start=2):
        if row["split"] != "train":
            raise ValueError(f"row {line}: split must be 'train', got {row['split']!r}")
        width = float(row["width_mev"])
        if not math.isfinite(width) or width <= 0:
            raise ValueError(f"row {line}: width_mev must be positive and finite")
        names.append(row["name"])
        widths.append(width)
    if len(set(names)) != len(names):
        raise ValueError("training CSV must contain one row per species (unique names)")
    if expected_species is not None and len(names) != expected_species:
        raise ValueError(
            f"training CSV must contain {expected_species} species, got {len(names)}"
        )
    return tensors, names, widths


def validate_output_path(output: Path) -> Path:
    """Refuse overwrites and anything that would touch the released weights."""
    out = output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    if WEIGHTS_DIR.resolve() in out.parents:
        raise ValueError(
            f"refusing to write inside the released-weights directory: {out}"
        )
    if out.name in PROTECTED_WEIGHT_NAMES:
        raise ValueError(
            f"refusing to write a checkpoint named {out.name!r}: "
            f"that name is reserved for the released weights"
        )
    return out


def _train_metrics(model: FTTransformer,
                   tensors: dict[str, torch.Tensor],
                   width_true: torch.Tensor, log_true: torch.Tensor,
                   batch_size: int) -> tuple[float, float]:
    """Return (training log-MAPE %, physical-width RMSRE %) for all rows."""
    model.eval()
    with torch.inference_mode():
        parts = []
        for start in range(0, len(width_true), batch_size):
            batch = {k: v[start:start + batch_size] for k, v in tensors.items()}
            parts.append(model(x=batch).reshape(-1))
        log_pred = torch.cat(parts)
        err = (log_pred.exp() - width_true).abs() / width_true
        return log_mape_pct(log_pred, log_true), float(math.sqrt(torch.mean(err.square()).item()) * 100.0)


def train(data_path: Path = DEFAULT_DATA,
          config_path: Path = DEFAULT_CONFIG,
          output_path: Path | None = None,
          *,
          init_weights: Path | None = None,
          epochs: int = 300,
          phase1_epochs: int | None = None,
          switch_log_mape_pct: float = 3.3,
          phase1_loss_name: str = "logcosh",
          phase2_loss_name: str = "mspe",
          lr: float = 1e-4,
          weight_decay: float = 1e-3,
          batch_size: int = 64,
          seed: int = 48,
          device: str = "cpu",
          smoke: bool = False,
          smoke_rows: int = 64,
          log_every: int = 10,
          expected_species: int | None = EXPECTED_TRAIN_SPECIES,
          ) -> dict:
    """Run the two-stage protocol and return a small summary dict.

    Batching is at the species level: the CSV contains exactly one row per
    species (at the exact central mass), so a batch of ``batch_size`` rows is
    a batch of ``batch_size`` distinct species.  All randomness (initial
    weights and per-epoch shuffling) is driven by ``seed``.
    """
    if smoke:
        device = "cpu"
        epochs = min(epochs, 2)
        phase1_epochs = 1  # exercise both phases in the tiny smoke run
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if phase1_epochs is not None and not 0 <= phase1_epochs < epochs:
        raise ValueError("phase1_epochs must satisfy 0 <= phase1_epochs < epochs")
    if not math.isfinite(switch_log_mape_pct) or switch_log_mape_pct <= 0:
        raise ValueError("switch_log_mape_pct must be positive and finite")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if phase1_loss_name not in {"logcosh", "mspe"} or phase2_loss_name not in {"logcosh", "mspe"}:
        raise ValueError("phase loss names must be logcosh or mspe")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not lr > 0.0:
        raise ValueError("lr must be positive")
    out = validate_output_path(output_path) if output_path is not None else None

    torch.manual_seed(seed)
    random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = build_model(config).to(device)
    if init_weights is not None:
        state = torch.load(init_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"initialized from {init_weights}")
    else:
        print("initialized from random weights (no --init given)")

    tensors, names, widths = read_train_csv(data_path, config, expected_species)
    if smoke:
        tensors = {k: v[:smoke_rows] for k, v in tensors.items()}
        names, widths = names[:smoke_rows], widths[:smoke_rows]
    log_true = torch.tensor([math.log(w) for w in widths], dtype=torch.float32, device=device)
    width_true = torch.tensor(widths, dtype=torch.float32, device=device)
    n = len(names)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=20)

    print(ILLUSTRATIVE_BANNER)
    print(f"training: n_species={n} epochs={epochs} switch_log_mape_pct={switch_log_mape_pct} "
          f"phase1={phase1_loss_name} phase2={phase2_loss_name} "
          f"phase1_max_epochs={phase1_epochs} lr={lr} weight_decay={weight_decay} "
          f"batch_size={batch_size} seed={seed} device={device} smoke={smoke}")

    summary: dict = {
        "epochs_run": 0,
        "phase2_entered_at_epoch": None,
        "last_loss": None,
        "final_rmsre_pct": None,
        "final_log_mape_pct": None,
        "output": str(out) if out is not None else None,
    }
    phase2_active = phase1_epochs == 0
    for epoch in range(1, epochs + 1):
        phase = 2 if phase2_active else 1
        if phase == 2 and summary["phase2_entered_at_epoch"] is None:
            summary["phase2_entered_at_epoch"] = epoch
        model.train()
        order = torch.randperm(n, generator=generator).tolist()
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch = {k: v[idx].to(device) for k, v in tensors.items()}
            log_pred = model(x=batch).reshape(-1)
            loss_name = phase1_loss_name if phase == 1 else phase2_loss_name
            loss = configured_loss(loss_name, log_pred, log_true[idx], width_true[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)
        summary["epochs_run"] = epoch
        # The original loop checked the training log-MAPE after each epoch;
        # a threshold crossing affects the following epoch, not this one.
        log_mape, rmsre = _train_metrics(model, tensors, width_true, log_true, batch_size)
        mean_loss = total_loss / n
        summary.update(last_loss=mean_loss, final_rmsre_pct=rmsre,
                       final_log_mape_pct=log_mape)
        if epoch % log_every == 0 or epoch == epochs or (phase == 1 and log_mape < switch_log_mape_pct):
            print(f"epoch {epoch:4d} phase={phase} loss={mean_loss:.6g} "
                  f"train_log_mape={log_mape:.4f}% train_rmsre={rmsre:.4f}%")
        if phase == 1 and (log_mape < switch_log_mape_pct or
                           (phase1_epochs is not None and epoch >= phase1_epochs)):
            phase2_active = True
            scheduler.best = float("inf")
            scheduler.num_bad_epochs = 0
        scheduler.step(mean_loss)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out)
        print(f"saved state dict -> {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Illustrative two-stage training example (NOT an exact paper/v6 recipe).",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="model-ready training CSV (default: %(default)s)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="model config JSON (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=None,
                        help="NEW state-dict output path; must not exist and must "
                             "not be inside weights/ (released weights are read-only)")
    parser.add_argument("--init", type=Path, default=None, dest="init_weights",
                        help="optional state dict to initialize from (opened read-only)")
    parser.add_argument("--epochs", type=int, default=300, help="total epochs (default: %(default)s)")
    parser.add_argument("--phase1-epochs", type=int, default=None,
                        help="optional maximum phase-1 epochs; normally switch by training log-MAPE")
    parser.add_argument("--switch-log-mape-pct", type=float, default=3.3,
                        help="switch to phase 2 after training log-width MAPE falls below this percent")
    parser.add_argument("--phase1-loss", choices=("logcosh", "mspe"), default="logcosh")
    parser.add_argument("--phase2-loss", choices=("logcosh", "mspe"), default="mspe")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: %(default)s)")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="species per batch (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=48, help="determinism seed (default: %(default)s)")
    parser.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")
    parser.add_argument("--smoke", action="store_true",
                        help="CPU tiny dry-run: at most 2 epochs on the first 64 species")
    args = parser.parse_args()
    if args.output is None:
        parser.error("--output is required (a new file; released weights are never overwritten)")
    summary = train(
        data_path=args.data,
        config_path=args.config,
        output_path=args.output,
        init_weights=args.init_weights,
        epochs=args.epochs,
        phase1_epochs=args.phase1_epochs,
        switch_log_mape_pct=args.switch_log_mape_pct,
        phase1_loss_name=args.phase1_loss,
        phase2_loss_name=args.phase2_loss,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        smoke=args.smoke,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
