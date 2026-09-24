"""Tests for the illustrative two-stage training example (train_example.py).

Covers:
  * loss formulas: LogCosh (phase 1) and mean squared relative error in MeV
    (phase 2, paper Eqs. 10/11) against hand-computed values;
  * the data/meson-train.csv schema: 370 unique train species, one exact
    central-mass row each (Mass == N * 0.134976828 / 0.1), positive finite
    widths, model-ready columns per predict.read_inputs, split=train only;
  * training-table names are disjoint from the archived particles outside
    that table (source split codes val1/val2/val3/vale);
  * a tiny deterministic CPU trainer smoke run with output protection
    (never overwrites existing files, never writes into weights/).

The trainer smoke tests use a tiny model config so they run in seconds on
CPU; they exercise the same code path as a full run.
"""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from predict import read_inputs
import train_example
from train_example import (ROOT, EXPECTED_TRAIN_SPECIES, build_model, logcosh,
                           msre_mev_loss, phase1_loss, read_train_csv, train)

CSV_PATH = ROOT / "data" / "meson-train.csv"
CONFIG_PATH = ROOT / "model-config.json"
EXPECTED_HEADER = ["name", "P", "C", "G", "N", "I", "I3", "J", "Mass",
                   "Nk", "Ik", "I3k", "Jk", "Mk",
                   "u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar", "b", "bbar",
                   "width_mev", "split"]

# Frozen 2026-09-24 from 0902_500_train_corrected_v1.xlsx (SHA-256
# 33dcc74619b1f70f913f084aac84d7d047c9e2e47b9eec237970bdc5f61882e6):
# 8 val1 + 6 val2 + 21 val3 + 12 vale records (source codes retained;
# backtick names are alternate encodings).
OTHER_PARTICLE_NAMES = (
    'Bbar_s2^*(5840)0',
    'D_1(2420)+',
    'D_2(2740)+',
    'D_s0(2590)-',
    'D_s0^*(2317)+',
    'D_s0^*(2317)+`',
    'D_s0^*(2317)-',
    'D_s0^*(2317)-`',
    'D_s1(2460)+',
    'D_s1(2460)+`',
    'D_s1(2460)-',
    'D_s1(2460)-`',
    'D_s1^*(2860)-',
    'K^*(1410)-',
    'K_0^*(1950)0',
    'K_2(1580)+',
    'K_2(1770)-',
    'K_2^*(1430)+',
    'K_3(2320)-',
    'Kbar(1830)0',
    'Kbar_0^*(700)0',
    'Kbar_3^*(1780)0',
    'a_0(1710)0',
    'a_0(980)+`',
    'a_0(980)-',
    'a_0(980)-`',
    'a_0(980)0`',
    'a_1(1260)0',
    'a_1(1640)+',
    'a_4(1970)0',
    'chi_c1(3872)',
    'chi_c1(3872)`',
    'eta(1405)0',
    'f_0(1370)0',
    'f_0(1770)0',
    'f_0(980)0',
    'f_0(980)0`',
    'f_2(1640)0',
    'f_2(1950)0',
    'pi_2(1880)-',
    'psi(4230)',
    'psi(4230)`',
    'psi(4360)',
    'psi(4360)`',
    'psi(4660)',
    'psi(4660)`',
    'rho(1900)-',
)


class LossFormulaTests(unittest.TestCase):
    """Hand-computed checks of the two phase losses."""

    def test_logcosh_values(self) -> None:
        # exact arithmetic in float64
        got = logcosh(torch.tensor([0.0, 1.0, -1.0, 20.0], dtype=torch.float64))
        self.assertAlmostEqual(got[0].item(), 0.0, places=12)           # log cosh 0 = 0
        self.assertAlmostEqual(got[1].item(), math.log(math.cosh(1.0)), places=12)
        self.assertAlmostEqual(got[2].item(), got[1].item(), places=12)  # symmetric
        # large-|x| asymptote: log cosh x = |x| - log 2 + O(exp(-2|x|))
        self.assertAlmostEqual(got[3].item(), 20.0 - math.log(2.0), places=12)
        # the float32 path (used by the trainer) stays within float32 rounding
        got32 = logcosh(torch.tensor([0.0, 1.0, 20.0]))
        self.assertLess(got32[0].item(), 1e-6)
        self.assertAlmostEqual(got32[1].item(), got[1].item(), delta=1e-6)
        self.assertAlmostEqual(got32[2].item(), got[3].item(), delta=1e-5)

    def test_phase1_is_mean_logcosh_of_delta(self) -> None:
        torch.manual_seed(0)
        log_true = torch.randn(5)
        delta = torch.randn(5)
        expected = logcosh(delta).mean()
        self.assertAlmostEqual(phase1_loss(log_true + delta, log_true).item(),
                               expected.item(), places=9)

    def test_msre_hand_computed(self) -> None:
        # widths [1, 4] MeV, predictions [2, 2] MeV -> e = [1, 0.5]
        width = torch.tensor([1.0, 4.0])
        pred = torch.tensor([2.0, 2.0])
        loss = msre_mev_loss(torch.log(pred), width)
        self.assertAlmostEqual(loss.item(), (1.0 ** 2 + 0.5 ** 2) / 2.0, places=6)
        # exact predictions -> zero loss
        self.assertAlmostEqual(msre_mev_loss(torch.log(width), width).item(), 0.0, places=12)

    def test_msre_is_relative(self) -> None:
        # scaling both width and prediction leaves the relative error unchanged
        width = torch.tensor([1.0, 4.0])
        pred = torch.tensor([2.0, 2.0])
        scale = 137.035999084  # MeV per GeV
        self.assertAlmostEqual(
            msre_mev_loss(torch.log(pred * scale), width * scale).item(),
            msre_mev_loss(torch.log(pred), width).item(), places=6)

    def test_losses_are_differentiable(self) -> None:
        log_pred = torch.randn(4, requires_grad=True)
        width = torch.exp(torch.rand(4))
        phase1_loss(log_pred, torch.log(width)).backward()
        grad_p1 = log_pred.grad.clone()
        self.assertTrue(torch.isfinite(grad_p1).all())
        log_pred.grad = None
        msre_mev_loss(log_pred, width).backward()
        self.assertTrue(torch.isfinite(log_pred.grad).all())
        # same prediction, different phases: losses differ (cross-scale design)
        self.assertNotAlmostEqual(phase1_loss(log_pred.detach(), torch.log(width)).item(),
                                  msre_mev_loss(log_pred.detach(), width).item(), places=3)


class DatasetSchemaTests(unittest.TestCase):
    """Contract of data/meson-train.csv: 370 unique train species, exact
    central mass, positive widths, model-ready columns, no other-particle rows."""

    @classmethod
    def setUpClass(cls) -> None:
        with CSV_PATH.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            cls.fields = list(reader.fieldnames or [])
            cls.rows = list(reader)

    def test_header_matches_predict_plus_label_and_split(self) -> None:
        self.assertEqual(self.fields, EXPECTED_HEADER)

    def test_370_unique_species_split_train_only(self) -> None:
        self.assertEqual(len(self.rows), EXPECTED_TRAIN_SPECIES)
        names = [row["name"] for row in self.rows]
        self.assertEqual(len(set(names)), EXPECTED_TRAIN_SPECIES)
        self.assertEqual({row["split"] for row in self.rows}, {"train"})

    def test_widths_positive_and_finite(self) -> None:
        for line, row in enumerate(self.rows, start=2):
            width = float(row["width_mev"])
            self.assertGreater(width, 0.0, f"row {line}: width not positive")
            self.assertTrue(math.isfinite(width), f"row {line}: width not finite")

    def test_exact_central_mass_contract(self) -> None:
        # Audit-justified derivation: Mass == N * 0.134976828 / 0.1
        # (PAPER_METHOD_REVIEW: exact central row for all 417 (split,name)
        # entries; 501 mass draws collapse to one physical species).
        max_rel = 0.0
        for row in self.rows:
            n_val = float(row["N"])
            mass = float(row["Mass"])
            max_rel = max(max_rel, abs(n_val - 0.1 * mass / 0.134976828) / n_val)
        self.assertLess(max_rel, 1e-9)

    def test_model_ready_via_predict(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        rows, tensors, fields = read_inputs(CSV_PATH, config)
        self.assertEqual(len(rows), EXPECTED_TRAIN_SPECIES)
        for column, tensor in tensors.items():
            self.assertEqual(tensor.shape, (EXPECTED_TRAIN_SPECIES,), column)

    def test_no_other_particle_names_in_training_table(self) -> None:
        train_names = {row["name"] for row in self.rows}
        overlap = train_names & set(OTHER_PARTICLE_NAMES)
        self.assertEqual(overlap, set(),
                         f"training CSV contains names outside its source split: {sorted(overlap)}")
        self.assertEqual(len(OTHER_PARTICLE_NAMES), 47)

    def test_config_instantiates_model(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        model = build_model(config)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 10_000_000)  # ~35M for the full 512-dim config


class TrainerSmokeTests(unittest.TestCase):
    """Tiny CPU runs of train(): two phases, determinism, output protection."""

    def _tiny_config(self, path: Path) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["model_params"] = {"dim": 32, "depth": 1, "heads": 2,
                                  "dim_head": 16, "attn_dropout": 0.0,
                                  "ff_dropout": 0.0}
        path.write_text(json.dumps(config), encoding="utf-8")

    def _mini_csv(self, path: Path, n_rows: int = 8) -> None:
        lines = CSV_PATH.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[: n_rows + 1]) + "\n", encoding="utf-8")

    def _run(self, td: Path, *, epochs: int, phase1_epochs: int,
             output: Path | None = None, **kwargs) -> dict:
        cfg, data = td / "config.json", td / "mini.csv"
        self._tiny_config(cfg)
        self._mini_csv(data)
        return train(data_path=data, config_path=cfg, output_path=output,
                     epochs=epochs, phase1_epochs=phase1_epochs,
                     lr=1e-3, batch_size=4, seed=7, expected_species=None, **kwargs)

    def test_tiny_two_phase_training_saves_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tiny.pth"
            summary = self._run(Path(td), epochs=2, phase1_epochs=1, output=out)
            self.assertEqual(summary["epochs_run"], 2)
            self.assertEqual(summary["phase2_entered_at_epoch"], 2)
            self.assertTrue(math.isfinite(summary["final_rmsre_pct"]))
            self.assertTrue(math.isfinite(summary["last_loss"]))
            self.assertTrue(out.exists())
            state = torch.load(out, weights_only=True)
            self.assertTrue(all(torch.isfinite(v).all() for v in state.values()))
            config = json.loads((Path(td) / "config.json").read_text(encoding="utf-8"))
            model = build_model(config)
            model.load_state_dict(state, strict=True)  # reloadable by the repo model

    def test_same_seed_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a, b = td / "a.pth", td / "b.pth"
            self._run(td, epochs=2, phase1_epochs=1, output=a)
            self._run(td, epochs=2, phase1_epochs=1, output=b)
            sa, sb = torch.load(a, weights_only=True), torch.load(b, weights_only=True)
            self.assertEqual(set(sa), set(sb))
            for key in sa:
                self.assertTrue(torch.equal(sa[key], sb[key]), key)

    def test_phase_selection_counts(self) -> None:
        calls = {"p1": 0, "p2": 0}
        real_p1, real_p2 = train_example.phase1_loss, train_example.msre_mev_loss

        def p1(*args, **kwargs):
            calls["p1"] += 1
            return real_p1(*args, **kwargs)

        def p2(*args, **kwargs):
            calls["p2"] += 1
            return real_p2(*args, **kwargs)

        train_example.phase1_loss, train_example.msre_mev_loss = p1, p2
        try:
            with tempfile.TemporaryDirectory() as td:
                self._run(Path(td), epochs=3, phase1_epochs=1)
        finally:
            train_example.phase1_loss, train_example.msre_mev_loss = real_p1, real_p2
        # 8 species / batch 4 = 2 batches per epoch: epoch 1 -> phase 1,
        # epochs 2-3 -> phase 2
        self.assertEqual(calls["p1"], 2)
        self.assertEqual(calls["p2"], 4)

    def test_init_weights_loads_strictly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            first = td / "first.pth"
            self._run(td, epochs=1, phase1_epochs=0, output=first)
            second = td / "second.pth"
            summary = self._run(td, epochs=1, phase1_epochs=0, output=second,
                                init_weights=first)
            self.assertEqual(summary["epochs_run"], 1)
            self.assertTrue(second.exists())

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            existing = td / "keep.pth"
            existing.write_text("keep this", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self._run(td, epochs=1, phase1_epochs=0, output=existing)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep this")

    def test_refuses_output_inside_weights_dir(self) -> None:
        refused = ROOT / "weights" / "__test_refused__.pth"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self._run(Path(td), epochs=1, phase1_epochs=0, output=refused)
        self.assertFalse(refused.exists())

    def test_refuses_reserved_weight_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            reserved = td / "paper-model.pth"
            with self.assertRaises(ValueError):
                self._run(td, epochs=1, phase1_epochs=0, output=reserved)
            self.assertFalse(reserved.exists())


if __name__ == "__main__":
    unittest.main()
