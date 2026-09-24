"""Inference-only CLI for the model associated with CPL 43, 020201 (2026)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from model import FTTransformer

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "model-config.json"
DEFAULT_WEIGHTS = ROOT / "weights" / "paper-model.pth"
PREDICTION_COLUMN = "predicted_width_mev"


def _number(value: str, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {column} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {column} must be finite")
    return result


def read_inputs(path: Path, config: dict) -> tuple[list[dict], dict[str, torch.Tensor]]:
    feature_cfg = config["features"]
    cats = feature_cfg["categorical_cols"]
    nums = feature_cfg["combinable_numerical_cols"]
    masks = list(feature_cfg["is_known_col_map"].values())
    quarks = feature_cfg["quark_cols"]
    required = set(cats + nums + masks + quarks) - {"N", "Nk"}
    # Mass is in GeV; this model uses N = 0.1 * Mass / m(pi0).
    pion_mass = config["pion0_mass_gev"]
    n_scale = config["n_scale"]
    if pion_mass <= 0 or n_scale <= 0:
        raise ValueError("invalid mass normalization in model config")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("input CSV has no header")
        if PREDICTION_COLUMN in reader.fieldnames:
            raise ValueError(f"input already has output column {PREDICTION_COLUMN}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"missing required CSV columns: {', '.join(missing)}")
        rows = list(reader)
        fields = reader.fieldnames
    if not rows:
        raise ValueError("input CSV contains no particles")
    values = {column: [] for column in cats + nums + masks + quarks}
    for line, row in enumerate(rows, start=2):
        for column in values:
            if column == "N":
                mass = _number(row["Mass"], "Mass", line)
                expected = n_scale * mass / pion_mass
                if row.get("N", "") not in ("", None):
                    supplied = _number(row["N"], "N", line)
                    if not math.isclose(supplied, expected, rel_tol=1e-5, abs_tol=1e-7):
                        raise ValueError(f"row {line}: N disagrees with 0.1 * Mass / m(pi0)")
                value = expected
            elif column == "Nk":
                value = 1.0
                if row.get("Nk", "") not in ("", None) and _number(row["Nk"], "Nk", line) != 1:
                    raise ValueError(f"row {line}: Nk must be 1 for derived N")
            else:
                value = _number(row[column], column, line)
            if column in cats:
                if value != int(value) or not 0 <= value < 3:
                    raise ValueError(f"row {line}: {column} must be an integer code 0, 1, or 2")
            elif column in masks and value not in (0, 1):
                raise ValueError(f"row {line}: {column} mask must be 0 or 1")
            elif column == "Mass" and value <= 0:
                raise ValueError(f"row {line}: Mass must be positive and in GeV")
            values[column].append(value)
    tensors = {column: torch.tensor(data, dtype=torch.long if column in cats else torch.float32)
               for column, data in values.items()}
    return rows, tensors, fields


def verify_sha256(path: Path) -> str:
    checksum_path = path.parent / "SHA256SUMS"
    if not checksum_path.exists():
        return "not verified (SHA256SUMS absent)"
    expected = None
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == path.name:
            expected = parts[0]
            break
    if expected is None:
        return "not verified (weight not listed)"
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}")
    return actual


def predict(input_path: Path, output_path: Path, weights_path: Path, config_path: Path,
            batch_size: int = 16) -> None:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output CSV paths must differ")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows, inputs, fields = read_inputs(input_path, config)
    digest = verify_sha256(weights_path)
    feature_cfg = config["features"]
    model = FTTransformer(
        categorical_cols=feature_cfg["categorical_cols"],
        categorical_cardinalities=config["categorical_cardinalities"],
        combinable_numerical_cols=feature_cfg["combinable_numerical_cols"],
        is_known_col_map=feature_cfg["is_known_col_map"],
        quark_cols=feature_cfg["quark_cols"],
        **config["model_params"],
    ).to("cpu")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = {key: value[start:start + batch_size] for key, value in inputs.items()}
            log_widths = model(x=batch).reshape(-1).tolist()
            result.extend(math.exp(value) for value in log_widths)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("non-finite model output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields + [PREDICTION_COLUMN])
        writer.writeheader()
        for row, width in zip(rows, result):
            writer.writerow({**row, PREDICTION_COLUMN: format(width, ".9g")})
    print(f"Predicted {len(rows)} particle(s); output: {output_path}; weights SHA256: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict total meson width (MeV) from encoded CSV features")
    parser.add_argument("input", type=Path, help="encoded input CSV; Mass is in GeV")
    parser.add_argument("output", type=Path, help="prediction output CSV")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    predict(args.input, args.output, args.weights, args.config, args.batch_size)


if __name__ == "__main__":
    main()
