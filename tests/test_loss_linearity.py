"""WP2: is each distillation component linear in omega?

If loss(c * 1) == c * loss(1) for a constant omega = c, no component
normalises by sum(omega) (or anything omega-dependent), so a global scalar
on distil_factor is a valid "same mean weight" control for that component.

Run: python -m unittest tests.test_loss_linearity -v   (from Multi-Level-OT/)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _stubs / make_span_snapshot

import unittest
from unittest import mock

import torch

from _stubs import make_batch, make_loss
from models.distillation_model import KL_wo, Sinkhorn_seq


def _components(c):
    """(l1, kl, sinkhorn) from the real forward, with omega forced to c * ones."""
    s_out, t_out, s_lab, t_lab, _ = make_batch(seed=3, requires_grad=False)
    loss = make_loss("entropy")
    if c is None:
        loss.span_kd_enabled = False  # vanilla path: position_weights = None
        out = loss(0, s_out, t_out, s_lab, t_lab, step=0)
    else:
        def fake_weights(student_probs, teacher_probs, **_):
            B, T, _k = teacher_probs.shape
            return torch.full((B, T), c)  # [B, T]
        with mock.patch("train.span_ot.compute_batch_position_weights", fake_weights):
            out = loss(0, s_out, t_out, s_lab, t_lab, step=0)
    return [float(x) for x in out[3:6]]


class TestComponentLinearity(unittest.TestCase):
    def test_kl_wo_scales_linearly(self):
        g = torch.Generator().manual_seed(0)
        y_s, y_t = torch.randn(1, 3, 5, generator=g), torch.randn(1, 3, 5, generator=g)  # [B=1, T=3, k=5]
        base = KL_wo(y_s, y_t)
        for c in (0.5, 0.3):
            got = KL_wo(y_s, y_t, position_weights=torch.full((1, 3), c))
            self.assertTrue(torch.allclose(got, c * base, rtol=1e-6, atol=0), (c, got, base))

    def test_sinkhorn_scales_linearly(self):
        g = torch.Generator().manual_seed(1)
        x = torch.softmax(torch.randn(3, 5, generator=g), -1)  # [T=3, k=5]
        y = torch.softmax(torch.randn(3, 5, generator=g), -1)  # [T=3, k=5]
        sk = Sinkhorn_seq()
        base = sk.sinkhorn_loss(x, y)
        for c in (0.5, 0.3):
            got = sk.sinkhorn_loss(x, y, row_weights=torch.full((3,), c))
            self.assertTrue(torch.allclose(got, c * base, rtol=1e-6, atol=0), (c, got, base))

    def test_full_forward_each_component_linear(self):
        base = _components(None)
        ones = _components(1.0)
        # omega = 1 must reproduce the vanilla numbers exactly.
        for b, o in zip(base, ones):
            self.assertAlmostEqual(b, o, places=7)
        for c in (0.5, 0.3):
            got = _components(c)
            for name, g_, b in zip(("l1", "kl", "sinkhorn"), got, base):
                self.assertAlmostEqual(g_, c * b, delta=1e-6 * max(1.0, abs(b)), msg=(name, c))


if __name__ == "__main__":
    unittest.main()
