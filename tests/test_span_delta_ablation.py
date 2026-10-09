"""Delta ablation: pin what span_low_delta does across the grid {0, .1, .25, .5, .75, 1}.

Run: python -m unittest tests.test_span_delta_ablation -v   (from Multi-Level-OT/)

Toy (same as test_span_r1_semantics): 4 unit spans, teacher one-hot (H_t = 0),
student rows give gaps [ln2, 0.325, 0.673, 0]. With top_r=0.5 there are 3
positive-gap spans, num_high = ceil(0.5 * 3) = 2 -> spans 0 and 2 at 1.0, span 1
(rank 3) and span 3 (gap 0) at delta. So omega = [1, d, 1, d] and
w_eff = (2 + 2d) / 4 = 0.5 + 0.5 d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _stubs

import unittest

import torch

from _stubs import make_batch, make_loss
from train.span_ot import compute_position_weights_one_sample, effective_weight_valid

DELTAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
OFF4 = [(0, 1), (1, 2), (2, 3), (3, 4)]
STUDENT = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.6, 0.4], [1.0, 0.0]])  # [4, 2]
TEACHER = torch.tensor([[1.0, 0.0]] * 4)  # [4, 2]


def _peaked(logits: torch.Tensor) -> torch.Tensor:
    """Teacher logits with one spike per row.

    The stub teacher (randn * 2, V=60) is less confident than the student, so no
    span has a positive gap. Scaling it does not help: DistillationLoss divides
    logits by their std (`normalize`), which is scale-invariant. A spike keeps
    its shape after that, giving H_t << H_s and a non-trivial top-r partition.
    """
    out = 0.01 * logits  # [B, L, V]
    out[..., 0] = 10.0
    return out


def _w(low_delta, top_r=0.5):
    return compute_position_weights_one_sample(
        student_probs=STUDENT, teacher_probs=TEACHER, student_size=4, teacher_size=4,
        student_offsets=OFF4, teacher_offsets=OFF4, top_r=top_r, low_delta=low_delta,
    )  # [4]


class TestDeltaHandComputed(unittest.TestCase):
    def test_weights_per_delta(self):
        for d in DELTAS:
            self.assertTrue(torch.allclose(_w(d), torch.tensor([1.0, d, 1.0, d])), d)

    def test_w_eff_is_affine_in_delta(self):
        for d in DELTAS:
            w_eff = effective_weight_valid(_w(d).unsqueeze(0), [4], [4])
            self.assertAlmostEqual(w_eff, 0.5 + 0.5 * d, places=6, msg=d)


class TestDeltaOnFullLoss(unittest.TestCase):
    """Real DistillationLoss.forward with stub tokenisers (see _stubs.py)."""

    def _run(self, d, requires_grad=False, peaked_teacher=True):
        s_out, t_out, s_lab, t_lab, W = make_batch(seed=5, requires_grad=requires_grad)
        if peaked_teacher:
            t_out.logits = _peaked(t_out.logits)
        out = make_loss("entropy", low_delta=d)(0, s_out, t_out, s_lab, t_lab, step=0)
        return out, W

    def test_no_positive_gap_sample_gets_no_kd_at_delta0(self):
        # Unscaled stub teacher has HIGHER entropy than the student on every
        # span -> all gaps <= 0 -> every aligned position gets delta. At d=0
        # such a sample receives zero KD weight on its whole L1 support.
        out, _ = self._run(0.0, peaked_teacher=False)
        self.assertEqual(out[-1]["w_eff_valid"], 0.0)

    def test_selection_does_not_depend_on_delta(self):
        # Entropy gaps are computed before delta is used, so the set of
        # positions at 1.0 must be the same for every delta < 1.
        from train import span_ot
        captured = {}
        real = span_ot.compute_batch_position_weights

        def spy(**kw):
            w = real(**kw)
            captured[kw["low_delta"]] = w.clone()  # [B, T]
            return w

        span_ot.compute_batch_position_weights = spy
        try:
            for d in DELTAS:
                self._run(d)
        finally:
            span_ot.compute_batch_position_weights = real
        high0 = captured[0.0] == 1.0  # [B, T]
        self.assertTrue(bool((~high0).any()), "no low-priority position; test is vacuous")
        for d in DELTAS[:-1]:
            self.assertTrue(torch.equal(captured[d] == 1.0, high0), d)
        self.assertTrue(torch.equal(captured[1.0], torch.ones_like(captured[1.0])))

    def test_each_component_affine_in_delta(self):
        # Every component is sum(omega * term) / const, so it is A + d * B with
        # A from the 1.0 positions and B from the delta positions. The ablation
        # interpolates between "high spans only" (d=0) and MLOT (d=1).
        c0 = [float(x) for x in self._run(0.0)[0][3:6]]
        c1 = [float(x) for x in self._run(1.0)[0][3:6]]
        for d in DELTAS:
            got = [float(x) for x in self._run(d)[0][3:6]]
            for name, g, a, b in zip(("l1", "kl", "sinkhorn"), got, c0, c1):
                expect = a + d * (b - a)
                self.assertAlmostEqual(g, expect, delta=1e-5 * max(1.0, abs(expect)), msg=(name, d))

    def test_delta1_equals_vanilla(self):
        s_out, t_out, s_lab, t_lab, _ = make_batch(seed=5, requires_grad=False)
        t_out.logits = _peaked(t_out.logits)
        van = make_loss("entropy", span=False)(0, s_out, t_out, s_lab, t_lab, step=0)
        one = make_loss("entropy", low_delta=1.0)(0, s_out, t_out, s_lab, t_lab, step=0)
        for idx in (3, 4, 5):
            self.assertAlmostEqual(float(van[idx]), float(one[idx]), places=7)

    def test_delta0_grads_live_and_finite(self):
        # d=0 zeroes the low spans; the high spans must still carry gradient.
        for idx, name in ((3, "l1"), (4, "kl"), (5, "sinkhorn")):
            out, W = self._run(0.0, requires_grad=True)
            comp, diag = out[idx], out[-1]
            self.assertTrue(torch.isfinite(comp), name)
            self.assertGreater(diag["w_eff_valid"], 0.0, name)
            self.assertLess(diag["w_eff_valid"], 1.0, name)
            comp.backward()
            self.assertTrue(torch.isfinite(W.grad).all(), name)
            self.assertGreater(float(W.grad.abs().sum()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
