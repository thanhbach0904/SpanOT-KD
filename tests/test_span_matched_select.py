"""WP3: matched-weight control (select_mode="matched").

Run: python -m unittest tests.test_span_matched_select -v   (from Multi-Level-OT/)

Toy example (hand-computed). 4 single-char tokens on both sides -> 4 spans,
one position each. Teacher rows one-hot (H=0). Student rows:
    row 0: [0.5, 0.5]  gap ln2 > 0
    row 1: [0.6, 0.4]  gap 0.673 > 0
    row 2: [1.0, 0.0]  gap 0  -> delta
    row 3: [1.0, 0.0]  gap 0  -> delta
top_r=1.0 keeps both positive spans: entropy omega = [1, 1, .1, .1],
mean over n=4 is 0.55, so matched = [.55, .55, .55, .55].
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _stubs / make_span_snapshot

import json
import random
import unittest
from pathlib import Path

import torch

from _stubs import make_batch, make_loss
from make_span_snapshot import run_case
from train.span_ot import compute_position_weights_one_sample

OFF4 = [(0, 1), (1, 2), (2, 3), (3, 4)]
STUDENT = torch.tensor([[0.5, 0.5], [0.6, 0.4], [1.0, 0.0], [1.0, 0.0]])  # [T_max=4, k=2]
TEACHER = torch.tensor([[1.0, 0.0]] * 4)  # [4, 2]


def _w(mode, student=STUDENT, teacher=TEACHER, s_size=4, t_size=4, s_off=OFF4, t_off=OFF4, top_r=1.0):
    return compute_position_weights_one_sample(
        student_probs=student, teacher_probs=teacher, student_size=s_size, teacher_size=t_size,
        student_offsets=s_off, teacher_offsets=t_off, top_r=top_r, low_delta=0.1, select_mode=mode,
    )


class TestMatchedHandComputed(unittest.TestCase):
    def test_plan_example(self):
        self.assertTrue(torch.allclose(_w("entropy"), torch.tensor([1.0, 1.0, 0.1, 0.1])))
        self.assertTrue(torch.allclose(_w("matched"), torch.full((4,), 0.55)))

    def test_no_spans_fallback_all_ones(self):
        w = _w("matched", s_off=[], t_off=[])
        self.assertTrue(torch.equal(w, torch.ones(4)))

    def test_no_positive_gap_sample_stays_all_delta(self):
        # Every gap == 0 -> entropy returns early with all-delta; matched
        # (mean of a constant) must agree. Regression check only: since [:n]
        # is already uniform here, this does not distinguish a matched branch
        # that skips the early return (that only differs when inactive 1.0
        # positions sit inside [:n]); the mean-equality test covers that path.
        w = _w("matched", student=TEACHER.clone())
        self.assertTrue(torch.allclose(w, torch.full((4,), 0.1)))

    def test_positions_beyond_n_keep_entropy_value(self):
        # Student: 2 tokens "ab"|"cd"; teacher: 4 tokens "a"|"b"|"c"|"d".
        # Spans (0,2): s[0], t[0,1]  gap ln2 > 0 -> 1.0
        #       (2,4): s[1], t[2,3]  gap 0      -> 0.1
        # entropy omega = [1, 1, .1, .1]; n = min(2, 4) = 2.
        # matched: [:2] = mean(1, 1) = 1; positions 2, 3 are outside the L1
        # support but inside KL/Sinkhorn's, so they must stay 0.1 (resetting
        # them to 1.0 would give matched MORE KL/Sinkhorn mass than entropy).
        s_off, t_off = [(0, 2), (2, 4)], [(0, 1), (1, 2), (2, 3), (3, 4)]
        student = torch.tensor([[0.5, 0.5], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # [4, 2]
        kw = dict(student=student, s_size=2, t_size=4, s_off=s_off, t_off=t_off, top_r=0.5)
        e, m = _w("entropy", **kw), _w("matched", **kw)
        self.assertTrue(torch.allclose(e, torch.tensor([1.0, 1.0, 0.1, 0.1])))
        self.assertTrue(torch.allclose(m, torch.tensor([1.0, 1.0, 0.1, 0.1])))


class TestMatchedMeanEqualsEntropy(unittest.TestCase):
    def test_random_inputs(self):
        s_off = [(0, 2), (2, 5), (5, 6), (6, 9), (9, 12)]
        t_off = [(0, 1), (1, 2), (2, 5), (5, 8), (8, 9), (9, 10), (10, 12)]
        for seed in range(20):
            g = torch.Generator().manual_seed(seed)
            s = torch.softmax(torch.randn(8, 5, generator=g) * 2, -1)  # [8, 5]
            t = torch.softmax(torch.randn(8, 5, generator=g) * 2, -1)  # [8, 5]
            for top_r in (0.3, 0.5, 0.7):
                kw = dict(student=s, teacher=t, s_size=5, t_size=7, s_off=s_off, t_off=t_off, top_r=top_r)
                e, m = _w("entropy", **kw), _w("matched", **kw)
                n = 5
                self.assertAlmostEqual(float(m[:n].mean()), float(e[:n].mean()), places=6)
                self.assertAlmostEqual(float(m.mean()), float(e.mean()), places=6)  # full T_max too
                self.assertLess(float(m[:n].max() - m[:n].min()), 1e-7)  # uniform on [:n]


class TestExistingArmsUnchanged(unittest.TestCase):
    """Snapshot taken from the pre-edit code by tests/make_span_snapshot.py."""

    def test_entropy_and_random_bit_identical(self):
        path = Path(__file__).resolve().parent / "fixtures" / "span_weights_snapshot.json"
        cases = json.loads(path.read_text())
        self.assertGreater(len(cases), 0)
        for c in cases:
            self.assertEqual(run_case(c, "entropy"), c["entropy"])
            self.assertEqual(run_case(c, "random", rng_seed=c["seed"], pool="active"), c["random_active"])
            self.assertEqual(run_case(c, "random", rng_seed=c["seed"], pool="pos"), c["random_pos"])


class TestMatchedGradients(unittest.TestCase):
    """Dead-gradient check: every component must send finite, nonzero grads
    to the student in matched mode (real loss, real weight path, stub tokenisers)."""

    def test_each_component_has_live_finite_grad(self):
        for idx, name in ((3, "l1"), (4, "kl"), (5, "sinkhorn")):
            s_out, t_out, s_lab, t_lab, W = make_batch(seed=5)
            out = make_loss("matched")(0, s_out, t_out, s_lab, t_lab, step=0)
            comp, diag = out[idx], out[-1]
            self.assertTrue(torch.isfinite(comp), name)
            self.assertLess(diag["w_eff_valid"], 1.0, "matched weights are trivial; test is vacuous")
            comp.backward()
            self.assertIsNotNone(W.grad, name)
            self.assertTrue(torch.isfinite(W.grad).all(), name)
            self.assertGreater(float(W.grad.abs().sum()), 0.0, name)

    def test_matched_and_entropy_share_w_eff_in_loss(self):
        s_out, t_out, s_lab, t_lab, _ = make_batch(seed=5, requires_grad=False)
        d_e = make_loss("entropy")(0, s_out, t_out, s_lab, t_lab, step=0)[-1]
        d_m = make_loss("matched")(0, s_out, t_out, s_lab, t_lab, step=0)[-1]
        self.assertAlmostEqual(d_e["w_eff_valid"], d_m["w_eff_valid"], places=6)


if __name__ == "__main__":
    unittest.main()
