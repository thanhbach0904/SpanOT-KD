"""WP1: w_eff_valid averages omega only over the L1 support [:min(s, t)].

Run: python -m unittest tests.test_w_eff_valid -v   (from Multi-Level-OT/)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _stubs / make_span_snapshot

import unittest

import torch

from _stubs import make_batch, make_loss  # noqa: F401  (also sets sys.path)
from train.span_ot import effective_weight_valid


class TestWEffValid(unittest.TestCase):
    def test_hand_computed_plan_example(self):
        # B=2, T_max=5, sizes (3, 2). Valid rows: [1, .1, .1] and [1, 1].
        # w_eff_valid = mean(0.4, 1.0) = 0.7; plain .mean() = (3.2 + 5.0) / 10 = 0.82.
        w = torch.tensor([[1.0, 0.1, 0.1, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0]])  # [2, 5]
        got = effective_weight_valid(w, student_sizes=[3, 4], teacher_sizes=[5, 2])
        self.assertAlmostEqual(got, 0.7, places=6)
        self.assertAlmostEqual(float(w.mean()), 0.82, places=6)
        self.assertNotAlmostEqual(got, float(w.mean()), places=3)

    def test_skips_empty_samples_and_defaults(self):
        w = torch.tensor([[0.1, 0.1], [0.5, 0.5]])
        self.assertAlmostEqual(effective_weight_valid(w, [0, 2], [2, 2]), 0.5, places=6)
        self.assertEqual(effective_weight_valid(None, [3], [3]), 1.0)
        self.assertEqual(effective_weight_valid(w, [0, 0], [2, 2]), 1.0)

    def test_loss_emits_w_eff_valid(self):
        s_out, t_out, s_lab, t_lab, _ = make_batch()
        diag = make_loss("entropy")(0, s_out, t_out, s_lab, t_lab, step=0)[-1]
        self.assertIn("w_eff_valid", diag)
        self.assertTrue(0.0 < diag["w_eff_valid"] <= 1.0)
        diag_off = make_loss(span=False)(0, s_out, t_out, s_lab, t_lab, step=0)[-1]
        self.assertEqual(diag_off["w_eff_valid"], 1.0)


if __name__ == "__main__":
    unittest.main()
