"""Unit tests for the random-span control in train.span_ot.

Run: python -m unittest tests.test_span_random_select -v   (from Multi-Level-OT/)

Toy example (hand-computed). Both tokenisers split the answer into 4
single-character tokens, so there are 4 aligned spans, each with one teacher
and one student position. Teacher rows are one-hot (entropy 0). Student rows:
    row 0: [0.5, 0.5]  H = ln 2    = 0.6931  -> gap 0.6931
    row 1: [0.9, 0.1]  H = 0.3251            -> gap 0.3251
    row 2: [0.6, 0.4]  H = 0.6730            -> gap 0.6730
    row 3: [1.0, 0.0]  H = 0                 -> gap 0 (not > 0)
S+ = {0, 2, 1} ranked by gap as 0, 2, 1. With r=0.5, num_high = ceil(0.5*3) = 2,
so the entropy path gives omega = [1, d, 1, d] (d = 0.1) and target token mass 2.
"""
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.span_ot import _sample_spans_matching_mass, compute_position_weights_one_sample

OFFSETS = [(0, 1), (1, 2), (2, 3), (3, 4)]
DELTA = 0.1


def _probs():
    student = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.6, 0.4], [1.0, 0.0]])  # [T_max=4, k=2]
    teacher = torch.tensor([[1.0, 0.0]] * 4)  # [4, 2]
    return student, teacher


def _weights(mode, seed=0, pool="active", student=None):
    s, t = _probs()
    if student is not None:
        s = student
    return compute_position_weights_one_sample(
        student_probs=s, teacher_probs=t, student_size=4, teacher_size=4,
        student_offsets=OFFSETS, teacher_offsets=OFFSETS,
        top_r=0.5, low_delta=DELTA, select_mode=mode,
        rng=random.Random(seed) if mode == "random" else None, random_pool=pool,
    )


class TestSampleSpansMatchingMass(unittest.TestCase):
    def test_zero_target_selects_nothing(self):
        self.assertEqual(_sample_spans_matching_mass([[0], [1]], 0, random.Random(0)), [])

    def test_unit_spans_hit_target_exactly(self):
        for seed in range(50):
            chosen = _sample_spans_matching_mass([[0], [1], [2], [3]], 2, random.Random(seed))
            self.assertEqual(sum(len(c) for c in chosen), 2)

    def test_fallback_when_every_span_too_long(self):
        chosen = _sample_spans_matching_mass([[0, 1, 2]], 1, random.Random(0))
        self.assertEqual(chosen, [[0, 1, 2]])

    def test_selection_is_random_and_reproducible(self):
        picks = {tuple(map(tuple, _sample_spans_matching_mass([[0], [1], [2], [3]], 2, random.Random(s))))
                 for s in range(50)}
        self.assertGreater(len(picks), 1)  # not a fixed choice
        a = _sample_spans_matching_mass([[0], [1], [2], [3]], 2, random.Random(7))
        b = _sample_spans_matching_mass([[0], [1], [2], [3]], 2, random.Random(7))
        self.assertEqual(a, b)


class TestRandomModeWeights(unittest.TestCase):
    def test_entropy_mode_unchanged(self):
        self.assertTrue(torch.allclose(_weights("entropy"), torch.tensor([1.0, DELTA, 1.0, DELTA])))

    def test_random_active_matches_mass(self):
        # 2 tokens at 1.0 and 2 at delta -> sum = 2.2 for every seed.
        for seed in range(50):
            w = _weights("random", seed)
            self.assertAlmostEqual(float(w.sum()), 2 * 1.0 + 2 * DELTA, places=6)
            self.assertTrue(set(w.tolist()) <= {1.0, float(torch.tensor(DELTA))})

    def test_random_pos_pool_never_picks_gap_leq_zero_span(self):
        for seed in range(50):
            w = _weights("random", seed, pool="pos")
            self.assertAlmostEqual(float(w[3]), DELTA, places=6)  # span 3 has gap 0
            self.assertAlmostEqual(float(w.sum()), 2.2, places=6)

    def test_random_active_actually_moves_off_entropy_choice(self):
        entropy = _weights("entropy")
        moved = any(not torch.allclose(_weights("random", s), entropy) for s in range(50))
        self.assertTrue(moved)

    def test_no_positive_gap_sample_is_all_delta_in_both_modes(self):
        s, _ = _probs()
        teacher_like = torch.tensor([[1.0, 0.0]] * 4)  # student == teacher -> every gap == 0
        for mode in ("entropy", "random"):
            w = _weights(mode, 0, student=teacher_like)
            self.assertTrue(torch.allclose(w, torch.full((4,), DELTA)))


if __name__ == "__main__":
    unittest.main()
