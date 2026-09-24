"""Regression tests for selected rows of Table 1, CPL 43 020201 (2026)."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from predict import ROOT, predict, read_inputs
import json


class PaperExamplesTest(unittest.TestCase):
    def test_table1_rounded_values(self) -> None:
        expected = {
            "K^*(1410)-": 231.3,
            "K_2^*(1430)+": 99.8,
            "D_2(2740)+": 88.1,
            "psi(4360)": 110.6,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "predictions.csv"
            predict(ROOT / "examples/paper-table1.csv", output,
                    ROOT / "weights/paper-model.pth", ROOT / "model-config.json")
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), len(expected))
            for row in rows:
                self.assertAlmostEqual(float(row["predicted_width_mev"]),
                                       expected[row["name"]], delta=0.051)

    def test_revised_weight_is_separate_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "revised.csv"
            predict(ROOT / "examples/paper-table1.csv", output,
                    ROOT / "weights/revised-model.pth", ROOT / "model-config.json")
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(float(rows[3]["predicted_width_mev"]),
                                   109.647835, delta=0.0001)
            self.assertNotAlmostEqual(float(rows[3]["predicted_width_mev"]),
                                      110.6, delta=0.05)

    def test_n_must_match_mass(self) -> None:
        config = json.loads((ROOT / "model-config.json").read_text(encoding="utf-8"))
        rows, _, _ = read_inputs(ROOT / "examples/paper-table1.csv", config)
        self.assertEqual(len(rows), 4)
        with tempfile.TemporaryDirectory() as temp_dir:
            broken = Path(temp_dir) / "bad.csv"
            with (ROOT / "examples/paper-table1.csv").open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                fields, data = reader.fieldnames, list(reader)
            data[0]["N"] = "999"
            with broken.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
            with self.assertRaisesRegex(ValueError, "N disagrees"):
                read_inputs(broken, config)

    def test_bad_category_is_rejected(self) -> None:
        config = json.loads((ROOT / "model-config.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            broken = Path(temp_dir) / "bad.csv"
            with (ROOT / "examples/paper-table1.csv").open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                fields, data = reader.fieldnames, list(reader)
            data[0]["P"] = "3"
            with broken.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
            with self.assertRaisesRegex(ValueError, "P must be an integer code"):
                read_inputs(broken, config)

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "existing.csv"
            output.write_text("keep this", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                predict(ROOT / "examples/paper-table1.csv", output,
                        ROOT / "weights/paper-model.pth", ROOT / "model-config.json")
            self.assertEqual(output.read_text(encoding="utf-8"), "keep this")


if __name__ == "__main__":
    unittest.main()
