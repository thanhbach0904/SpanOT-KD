"""WP5: pin what top_r = 1.0 actually does (it is NOT MultiLevelOT).

Run: python -m unittest tests.test_span_r1_semantics -v   (from Multi-Level-OT/)

Same toy as test_span_random_select: 4 unit spans, teacher one-hot, student
gaps [ln2, 0.325, 0.673, 0]. Span 3 has gap 0 (not > 0).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _stubs / make_span_snapshot

import unittest

import torch

import _stubs  # noqa: F401  (sets sys.path)
from train.span_ot import compute_position_weights_one_sample

OFF4 = [(0, 1), (1, 2), (2, 3), (3, 4)]
STUDENT = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.6, 0.4], [1.0, 0.0]])  # [4, 2]
TEACHER = torch.tensor([[1.0, 0.0]] * 4)  # [4, 2]


def _w(top_r, low_delta, student=STUDENT):
    return compute_position_weights_one_sample(
        student_probs=student, teacher_probs=TEACHER, student_size=4, teacher_size=4,
        student_offsets=OFF4, teacher_offsets=OFF4, top_r=top_r, low_delta=low_delta,
    )


class TestR1Semantics(unittest.TestCase):
    def test_r1_is_positive_gap_filter(self):
        # All positive-gap spans at 1.0, the gap-0 span at delta.
        self.assertTrue(torch.allclose(_w(1.0, 0.1), torch.tensor([1.0, 1.0, 1.0, 0.1])))

    def test_r1_no_positive_gap_downweights_everything(self):
        self.assertTrue(torch.allclose(_w(1.0, 0.1, student=TEACHER.clone()), torch.full((4,), 0.1)))

    def test_delta1_is_true_mlot_collapse(self):
        self.assertTrue(torch.equal(_w(1.0, 1.0), torch.ones(4)))
        self.assertTrue(torch.equal(_w(0.3, 1.0), torch.ones(4)))


if __name__ == "__main__":
    unittest.main()
