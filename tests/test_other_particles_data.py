"""Check the published central-mass records outside the training table."""
from __future__ import annotations

import csv
import json
import math
import unittest
from collections import Counter
from pathlib import Path

from predict import ROOT, read_inputs


class OtherParticlesDataTest(unittest.TestCase):
    def test_splits_and_label_roles(self) -> None:
        path = ROOT / "data/meson-other-particles.csv"
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 47)
        self.assertEqual(Counter(r["split"] for r in rows),
                         {"val1": 8, "val2": 6, "val3": 21, "vale": 12})
        self.assertEqual(Counter(r["label_type"] for r in rows),
                         {"point": 34, "upper_limit": 8, "range": 5})
        self.assertEqual(Counter(r["record_role"] for r in rows),
                         {"main": 35, "alternate": 12})
        self.assertEqual(len({(r["split"], r["name"]) for r in rows}), 47)
        with (ROOT / "data/meson-train.csv").open(newline="", encoding="utf-8") as file:
            train_names = {r["name"] for r in csv.DictReader(file)}
        self.assertTrue(train_names.isdisjoint(r["name"] for r in rows))
        for row in rows:
            self.assertGreater(float(row["archived_width_mev"]), 0)
            self.assertTrue(math.isfinite(float(row["archived_width_mev"])))
            self.assertLess(abs(float(row["Mass"]) - float(row["N"]) * 0.134976828 / 0.1), 1e-8)
            if row["record_role"] == "alternate":
                self.assertEqual(row["split"], "vale")
                self.assertEqual(row["alias_of"], row["name"].rstrip("`"))
            else:
                self.assertEqual(row["alias_of"], "")

    def test_rows_are_model_ready(self) -> None:
        config = json.loads((ROOT / "model-config.json").read_text(encoding="utf-8"))
        rows, tensors, _ = read_inputs(ROOT / "data/meson-other-particles.csv", config)
        self.assertEqual(len(rows), 47)
        self.assertEqual(tensors["Mass"].shape[0], 47)


if __name__ == "__main__":
    unittest.main()
